"""
Entry point for data preparation and training.

The pipeline lives in src/ as one script per stage. Each script takes its settings from
EEG_* environment variables. This file reads config.yaml, sets those variables, and runs
the script in a subprocess, so every run is driven from a single configuration file.

Usage:
  python train.py --list                 list the configured experiments
  python train.py prepare                inventory -> windows -> folds/baselines -> images
  python train.py eegnet resnet18_cwt    run one or more experiments by name
  python train.py all                    run every experiment in config.yaml's `all` list
  python train.py resnet18_cwt --quick   smoke test (1 fold, 3 epochs, files end in _quick)
  python train.py eegnet --device cpu    override the configured device

Outputs go to results/ and figures/ (see README.md).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# config key -> (environment variable, formatter)
_ENV_KEYS = {
    "lr": ("EEG_LR", lambda v: f"{float(v):g}"),
    "seeds": ("EEG_SEEDS", lambda v: ",".join(str(int(s)) for s in v)),
    "repr": ("EEG_REPR", str),
    "pretrained": ("EEG_PRETRAIN", lambda v: "1" if v else "0"),
    "channels": ("EEG_CANALES", lambda v: ",".join(v) if isinstance(v, list) else str(v)),
    "band": ("EEG_BANDA", lambda v: "-".join(str(x) for x in v) if isinstance(v, list) else str(v)),
    "tag": ("EEG_ETIQUETA", str),
    "compare": ("EEG_COMPARE", str),
    "fold": ("EEG_FOLD", lambda v: str(int(v))),
    "seed": ("EEG_SEMILLA", lambda v: str(int(v))),
    "force": ("EEG_FORZAR", lambda v: "1" if v else "0"),
    "device": ("EEG_DEVICE", str),
    "epochs": ("EEG_EPOCAS", lambda v: str(int(v))),
}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def base_env(cfg, device=None, quick=None):
    """Environment shared by every stage: data path, device, quick mode."""
    env = dict(os.environ)
    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    env["EEG_DATA_DIR"] = str(data_dir)
    dev = device or cfg.get("device", "auto")
    if dev and dev != "auto":
        env["EEG_DEVICE"] = dev
    if quick if quick is not None else cfg.get("quick", False):
        env["EEG_QUICK"] = "1"
    # Lets operations that MPS does not implement fall back to the CPU instead of failing.
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def run_script(script, env, label):
    path = SRC / f"{script}.py"
    if not path.exists():
        sys.exit(f"Script not found: {path}")
    shown = {k: v for k, v in env.items() if k.startswith("EEG_") and os.environ.get(k) != v}
    print(f"\n=== {label}: src/{script}.py  {shown}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", str(path)], env=env, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"{label} failed (exit code {r.returncode})")
    print(f"=== {label} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def experiment_env(spec, env, device_override):
    env = dict(env)
    for key, value in spec.items():
        if key == "script" or value is None:
            continue
        if key not in _ENV_KEYS:
            sys.exit(f"Unknown key in config.yaml experiment: {key}")
        if key == "device" and device_override:
            continue   # the command-line --device wins over the experiment's device
        var, fmt = _ENV_KEYS[key]
        env[var] = fmt(value)
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="prepare, all, or experiment names")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--quick", action="store_true", help="1 fold, 3 epochs, outputs end in _quick")
    ap.add_argument("--list", action="store_true", help="list experiments and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    experiments = cfg["experiments"]

    if args.list or not args.targets:
        print("prepare:", ", ".join(cfg["prepare"]))
        print("experiments:")
        for name, spec in experiments.items():
            extra = {k: v for k, v in spec.items() if k != "script"}
            print(f"  {name:24s} src/{spec['script']}.py  {extra or ''}")
        print("all:", ", ".join(cfg["all"]))
        return

    device = None if args.device in (None, "auto") else args.device
    env = base_env(cfg, device=device, quick=True if args.quick else None)

    targets = []
    for t in args.targets:
        targets.extend(cfg["all"] if t == "all" else [t])

    for t in targets:
        if t == "prepare":
            for script in cfg["prepare"]:
                run_script(script, env, f"prepare/{script}")
        elif t in experiments:
            spec = experiments[t]
            run_script(spec["script"], experiment_env(spec, env, device), t)
        else:
            sys.exit(f"Unknown target '{t}'. Use --list to see the options.")


if __name__ == "__main__":
    main()
