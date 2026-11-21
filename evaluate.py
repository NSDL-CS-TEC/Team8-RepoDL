"""
Entry point for evaluation. It trains nothing and only reads results/.

Steps:
  1. summary: mean and spread of balanced accuracy and AUROC per model, over fold x seed,
     from every results/*_folds.csv -> results/resumen_modelos.csv
  2. lr:      learning-rate selection from the lr-tuning runs (src/elegir_tasa_aprendizaje.py)
  3. stats:   permutation test against real chance, paired subject bootstrap between every
              pair of models, and per-subject failure analysis (src/analisis_final.py)
              -> results/permutacion.csv, comparacion_pareada.csv, analisis_por_sujeto.csv
                 figures/permutacion.png, analisis_por_sujeto.png

Usage:
  python evaluate.py              run all three steps
  python evaluate.py summary      run only the steps you name
  python evaluate.py stats --n-permutations 200
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from train import ROOT, base_env, load_config, run_script

STEPS = ("summary", "lr", "stats")


def summary(results_dir):
    rows = []
    for f in sorted(results_dir.glob("*_folds.csv")):
        if "_quick" in f.stem:
            continue
        df = pd.read_csv(f)
        if "modelo" not in df or "bal_acc" not in df:
            continue
        for model, g in df.groupby("modelo", sort=False):
            rows.append({
                "modelo": model,
                "archivo": f.name,
                "n_corridas": len(g),
                "semillas": g["seed"].nunique() if "seed" in g else None,
                "bal_acc_media": g["bal_acc"].mean(),
                "bal_acc_de": g["bal_acc"].std(ddof=1),
                "auroc_media": g["auroc"].mean() if "auroc" in g else None,
            })
    if not rows:
        print("No *_folds.csv files found in", results_dir)
        return
    out = pd.DataFrame(rows).sort_values("bal_acc_media", ascending=False)
    out.to_csv(results_dir / "resumen_modelos.csv", index=False)
    with pd.option_context("display.width", 140, "display.max_rows", 100,
                           "display.float_format", "{:.3f}".format):
        print(out.drop(columns="archivo").to_string(index=False))
    print("\nBalanced accuracy over fold x seed runs. Use comparacion_pareada.csv (per-subject")
    print("bootstrap) for the hypothesis test, not this table. Saved: results/resumen_modelos.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help=f"any of {', '.join(STEPS)} (default: all)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--n-permutations", type=int)
    args = ap.parse_args()
    bad = [s for s in args.steps if s not in STEPS]
    if bad:
        ap.error(f"unknown step(s): {', '.join(bad)}. Choose from {', '.join(STEPS)}")

    cfg = load_config(args.config)
    ev = cfg.get("evaluate", {}) or {}
    env = base_env(cfg, quick=False)
    env.pop("EEG_QUICK", None)
    results_dir = Path(env["EEG_DATA_DIR"]).parent / "results"

    for step in args.steps or STEPS:
        if step == "summary":
            print("\n=== summary")
            summary(results_dir)
        elif step == "lr":
            run_script("elegir_tasa_aprendizaje", env, "lr")
        elif step == "stats":
            e = dict(env)
            e["EEG_NPERM"] = str(args.n_permutations or ev.get("n_permutations", 2000))
            if ev.get("pair"):
                e["EEG_PAR"] = ",".join(ev["pair"])
            if ev.get("models"):
                e["EEG_MODELOS"] = ",".join(ev["models"])
            run_script("analisis_final", e, "stats")


if __name__ == "__main__":
    sys.exit(main())
