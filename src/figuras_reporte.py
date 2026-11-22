"""
English figures and the main results table for the conference report.

Reads only results/ (no training). Run after `python evaluate.py`.
Outputs:
  figures/fig_resultados.pdf/.png      per-subject balanced accuracy with 95% subject bootstrap
                                       CI for every model + paired delta of the main comparison
  figures/fig_diagnostico.pdf/.png     EEGNet channel/band diagnostic (paired delta, 95% CI)
  figures/fig_interpretacion.pdf/.png  mean scalogram and Grad-CAM, slow minus fast
  results/tabla_principal.csv          numbers behind the report's main table
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
N_BOOT, RNG = 2000, np.random.default_rng(0)

MAIN_A = os.environ.get("EEG_MAIN_A", "resnet18_cwt_mps")
MAIN_B = os.environ.get("EEG_MAIN_B", "eegnet_lr0.003_final")

# (file stem, label, group) in table order
MODELS = [
    ("control_permutado", "Shuffled labels (chance control)", "control"),
    ("indice_ensayo", "Trial position (drift control)", "control"),
    ("potencia_bandas_zsesion", "Band power + LR, session z-score", "classical"),
    ("potencia_bandas", "Band power + LR", "classical"),
    ("resnet18_cwt_scratch", "ResNet18, CWT, no pretraining", "ablation"),
    ("resnet18_stft_mps", "ResNet18, STFT, ImageNet", "ablation"),
    ("resnet18_cwt", "ResNet18, CWT, ImageNet (CUDA replication)", "replication"),
    ("eegnet", "EEGNet, lr 1e-3, 40 epochs (initial run)", "replication"),
    ("resnet18_cwt_mps", "ResNet18, CWT, ImageNet (main)", "main"),
    ("eegnet_lr0.003_final", "EEGNet, raw signal (main baseline)", "main"),
]
COLORS = {"control": "#9aa5b1", "classical": "#7b8794", "ablation": "#2f9e8f",
          "replication": "#c3a3e0", "main": "#2b6cb0"}


def boot_ci(v):
    v = np.asarray(v, float)
    idx = RNG.integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def per_subject(stem):
    if stem in ("control_permutado", "indice_ensayo", "potencia_bandas", "potencia_bandas_zsesion"):
        d = pd.read_csv(RES / "baselines_por_sujeto.csv")
        d = d[d.modelo == stem]
    else:
        f = RES / f"{stem}_por_sujeto.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f)
    return d.set_index("subject")["bal_acc"]


def folds(stem):
    if stem in ("control_permutado", "indice_ensayo", "potencia_bandas", "potencia_bandas_zsesion"):
        d = pd.read_csv(RES / "baselines_folds.csv")
        return d[d.modelo == stem]
    f = RES / f"{stem}_folds.csv"
    return pd.read_csv(f) if f.exists() else None


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    rows, subj = [], {}
    for stem, label, group in MODELS:
        ps, fo = per_subject(stem), folds(stem)
        if ps is None or fo is None:
            print(f"skip {stem}: no results")
            continue
        subj[stem] = ps
        lo, hi = boot_ci(ps.values)
        rows.append(dict(modelo=stem, label=label, group=group, runs=len(fo),
                         bal_acc=fo.bal_acc.mean(), bal_acc_sd=fo.bal_acc.std(ddof=1),
                         auroc=fo.auroc.mean(), subj_mean=ps.mean(), subj_lo=lo, subj_hi=hi,
                         above_chance=f"{int((ps > 0.5).sum())}/{len(ps)}"))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "tabla_principal.csv", index=False)
    print(tab.drop(columns="label").round(3).to_string(index=False))

    # ---- Figure: results ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(tab))
    for i, r in tab.iterrows():
        a1.errorbar(r.subj_mean, i, xerr=[[r.subj_mean - r.subj_lo], [r.subj_hi - r.subj_mean]],
                    fmt="o", color=COLORS[r.group], ms=5, capsize=3, lw=1.2)
    a1.set_yticks(y, tab.label, fontsize=8)
    a1.axvline(0.5, color="#52606d", ls=":", lw=1)
    a1.set_xlabel("balanced accuracy per subject (mean, 95% bootstrap CI)")
    a1.set_title("(a) Unseen-subject performance, 24 subjects", fontsize=10)
    a1.grid(axis="x", alpha=0.3)

    if MAIN_A in subj and MAIN_B in subj:
        d = (subj[MAIN_A] - subj[MAIN_B]).dropna().sort_values()
        lo, hi = boot_ci(d.values)
        # use the CI of the official paired analysis (analisis_final.py) when available
        cp = pd.read_csv(RES / "comparacion_pareada.csv")
        r = cp[(cp.modelo_a == MAIN_A) & (cp.modelo_b == MAIN_B)]
        if len(r):
            lo, hi = r.ic95_bajo.iloc[0], r.ic95_alto.iloc[0]
        else:
            r = cp[(cp.modelo_a == MAIN_B) & (cp.modelo_b == MAIN_A)]
            if len(r):
                lo, hi = -r.ic95_alto.iloc[0], -r.ic95_bajo.iloc[0]
        cols = np.where(d.values > 0, "#2f9e8f", "#c05621")
        a2.barh(np.arange(len(d)), d.values, color=cols)
        a2.set_yticks(np.arange(len(d)), [f"s{int(s)}" for s in d.index], fontsize=7)
        a2.axvline(0, color="#1f2933", lw=0.8)
        a2.axvspan(lo, hi, color="#2b6cb0", alpha=0.15)
        a2.axvline(d.mean(), color="#2b6cb0", lw=1.2)
        a2.text(0.98, 0.02, f"mean Δ {d.mean():+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]",
                transform=a2.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(fc="white", ec="#9aa5b1", lw=0.6))
        a2.xaxis.set_major_locator(MaxNLocator(5))
        a2.set_xlabel("Δ balanced accuracy (ResNet18 CWT − EEGNet)")
        a2.set_title("(b) Paired difference per subject", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_resultados")

    # ---- Figure: EEGNet diagnostic ----
    dg = pd.read_csv(RES / "diagnostico_eegnet.csv").dropna(subset=["delta_vs_ref"])
    names = {"solo Fz": "Fz only", "solo Pz": "Pz only", "solo Oz": "Oz only",
             "sin delta (4 a 50 Hz)": "no delta (4–50 Hz)", "solo delta (1 a 4 Hz)": "delta only (1–4 Hz)",
             "theta y alfa (4 a 13 Hz)": "theta+alpha (4–13 Hz)", "solo Fz sin delta": "Fz only, 4–50 Hz"}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    yy = np.arange(len(dg))[::-1]
    ax.errorbar(dg.delta_vs_ref, yy, xerr=[dg.delta_vs_ref - dg.ic95_bajo, dg.ic95_alto - dg.delta_vs_ref],
                fmt="o", color="#2b6cb0", capsize=3)
    ax.set_yticks(yy, [names.get(e, e) for e in dg.entrada], fontsize=8)
    for yi, b in zip(yy, dg.por_sujeto):
        ax.text(0.004, yi, f"{b:.3f}", va="center", fontsize=7, color="#52606d")
    ax.axvline(0, color="#1f2933", lw=0.8)
    ax.set_xlabel("Δ per-subject balanced accuracy vs. 3 channels, 1–50 Hz")
    ax.set_xlim(None, 0.035)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_diagnostico")

    # ---- Figure: interpretation (class means per channel on all windows + Grad-CAM) ----
    d = np.load(RES / "ventanas.npz")
    keep = np.load(RES / "mascara_artefactos.npy")
    y = d["y"][keep].astype(bool)
    imgs = np.load(RES / "imagenes_cwt.npy", mmap_mode="r")
    acc = {0: np.zeros((3, 128, 128)), 1: np.zeros((3, 128, 128))}
    for i0 in range(0, len(y), 2000):
        blk = np.asarray(imgs[i0:i0 + 2000], dtype=np.float64) / 255 * 6 - 3   # back to z units
        yb = y[i0:i0 + 2000]
        acc[1] += blk[yb].sum(0)
        acc[0] += blk[~yb].sum(0)
    diff = acc[1] / y.sum() - acc[0] / (~y).sum()
    z = np.load(RES / "saliencia_cwt.npz")
    ext = [-3, 0, 0, 127]
    freqs = np.geomspace(1, 40, 128)
    ticks = [1, 4, 8, 13, 30]
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
    v = np.abs(diff).max()
    for k, ch in enumerate(("Fz", "Pz", "Oz")):
        im = axs[k].imshow(diff[k], origin="lower", aspect="auto", extent=ext, cmap="RdBu_r",
                           vmin=-v, vmax=v)
        axs[k].set_title(f"{ch}: mean scalogram, slow − fast", fontsize=9)
    fig.colorbar(im, ax=axs[2], fraction=0.046, label="z units")
    cam = z["cam_lenta"] - z["cam_rapida"]
    c = np.abs(cam).max()
    im2 = axs[3].imshow(cam, origin="lower", aspect="auto", extent=ext, cmap="PuOr_r", vmin=-c, vmax=c)
    axs[3].set_title("Grad-CAM, slow − fast (1 model)", fontsize=9)
    fig.colorbar(im2, ax=axs[3], fraction=0.046)
    for ax in axs:
        ax.set_xlabel("time to deviation (s)", fontsize=8)
    axs[0].set_yticks([np.argmin(np.abs(freqs - t)) for t in ticks], ticks)
    axs[0].set_ylabel("frequency (Hz)")
    fig.tight_layout()
    save(fig, "fig_interpretacion")

    # ---- Figure: per-subject failure analysis ----
    ps = pd.read_csv(RES / "analisis_por_sujeto.csv").dropna(subset=[f"bal_acc_{MAIN_B}"])
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for stem, lab, col in ((MAIN_B, "EEGNet", "#2b6cb0"), (MAIN_A, "ResNet18 CWT", "#c05621")):
        b1.scatter(ps.pct_lenta, ps[f"bal_acc_{stem}"], s=18, color=col, label=lab, alpha=0.85)
        b2.scatter(ps[f"tnr_{stem}"], ps[f"tpr_{stem}"], s=18, color=col, label=lab, alpha=0.85)
    for _, r in ps.iterrows():
        b1.annotate(str(int(r.subject)), (r.pct_lenta, r[f"bal_acc_{MAIN_B}"]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points", color="#52606d")
    b1.axhline(0.5, color="#52606d", ls=":", lw=1)
    b1.set_xlabel("proportion of slow windows in subject")
    b1.set_ylabel("balanced accuracy")
    b1.set_title("(a) Performance vs. subject base rate", fontsize=9)
    b1.legend(fontsize=7, frameon=False)
    b2.plot([0, 1], [1, 0], color="#9aa5b1", lw=0.8, ls="--")
    b2.set_xlabel("true-negative rate (fast correct)")
    b2.set_ylabel("true-positive rate (slow correct)")
    b2.set_title("(b) Per-class rates per subject", fontsize=9)
    b2.set_xlim(0.3, 1.02)
    b2.set_ylim(0.3, 1.02)
    fig.tight_layout()
    save(fig, "fig_fallos")

    # ---- Figure: the three representations of one slow and one fast window ----
    X = d["X"][keep]
    stft = np.load(RES / "imagenes_stft.npy", mmap_mode="r")
    sess = d["session"][keep]
    s0 = sess[np.argmax(y)]                       # a session with slow windows
    i_slow = np.where((sess == s0) & y)[0][0]
    i_fast = np.where((sess == s0) & ~y)[0][0]
    t = np.arange(750) / 250 - 3
    fig, axs = plt.subplots(2, 3, figsize=(10, 4.2))
    for row, (i, lab) in enumerate(((i_fast, "fast"), (i_slow, "slow"))):
        for k, ch in enumerate(("Fz", "Pz", "Oz")):
            axs[row, 0].plot(t, X[i, k] - 60 * k, lw=0.5, color=("#2b6cb0", "#2f9e8f", "#c05621")[k])
        axs[row, 0].set_yticks([0, -60, -120], ["Fz", "Pz", "Oz"])
        axs[row, 0].set_title(f"Raw signal (µV), {lab} trial", fontsize=9)
        for col, arr, name in ((1, imgs, "CWT"), (2, stft, "STFT")):
            rgb = np.asarray(arr[i]).transpose(1, 2, 0)
            axs[row, col].imshow(rgb, origin="lower", aspect="auto", extent=[-3, 0, 0, 127])
            axs[row, col].set_title(f"{name} image (R=Fz, G=Pz, B=Oz), {lab}", fontsize=9)
            axs[row, col].set_yticks([])
        for ax in axs[row]:
            ax.tick_params(labelsize=7)
    for ax in axs[1]:
        ax.set_xlabel("time to deviation (s)", fontsize=8)
    axs[0, 1].set_ylabel("1 → 40 Hz (log)", fontsize=8)
    axs[0, 2].set_ylabel("1 → 40 Hz (linear)", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_representaciones")

    print("Saved figures/fig_resultados, fig_diagnostico, fig_interpretacion, fig_fallos, fig_representaciones and results/tabla_principal.csv")


if __name__ == "__main__":
    main()
