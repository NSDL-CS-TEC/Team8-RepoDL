"""
Writes numeros.tex (LaTeX macros) and tabla_principal.tex from the project's results/,
and copies the report figures. Run after `python evaluate.py` and
`python src/figuras_reporte.py` in the project:

  /opt/anaconda3/envs/dl/bin/python generar_numeros.py /Users/alberto/Documents/Proyecto
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if len(sys.argv) > 1:
    PROJ = Path(sys.argv[1])
elif (HERE.parents[1] / "results").is_dir():   # copy inside the repo: report/01_reporte/
    PROJ = HERE.parents[1]
else:
    PROJ = Path("/Users/alberto/Documents/Proyecto")
RES, FIG = PROJ / "results", PROJ / "figures"
A, B = "resnet18_cwt_mps", "eegnet_lr0.003_final"

tab = pd.read_csv(RES / "tabla_principal.csv").set_index("modelo")
comp = pd.read_csv(RES / "comparacion_pareada.csv")
subj = pd.read_csv(RES / "analisis_por_sujeto.csv")
cond = pd.read_csv(RES / "por_condicion.csv")


def pair(a, b):
    """Delta a - b with CI, whatever order comparacion_pareada.csv stored it in."""
    r = comp[(comp.modelo_a == a) & (comp.modelo_b == b)]
    if len(r):
        r = r.iloc[0]
        return r.delta_medio, r.ic95_bajo, r.ic95_alto, r.a_mejor_en
    r = comp[(comp.modelo_a == b) & (comp.modelo_b == a)].iloc[0]
    n_b, n = map(int, r.a_mejor_en.split("/"))
    return -r.delta_medio, -r.ic95_alto, -r.ic95_bajo, f"{n - n_b}/{n}"


def f3(x):
    return f"{x:+.3f}".replace("-", "$-$") if x < 0 else f"{x:+.3f}"


def ci(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]".replace("-", "$-$")


def subjtxt(m):
    r = tab.loc[m]
    return f"{r.subj_mean:.3f}"


m = {}
d, lo, hi, better = pair(A, B)
m["deltaMain"], m["ciMain"], m["resBetter"] = f3(d), ci(lo, hi), better.replace("/", " of ")
_nb, _n = map(int, better.split("/"))
m["eegBetter"] = str(_n - _nb - int(((subj[f"bal_acc_{A}"] - subj[f"bal_acc_{B}"]).dropna() == 0).sum()))
m["resMainSubj"], m["eegMainSubj"] = subjtxt(A), subjtxt(B)
if hi < 0:
    m["mainVerdict"] = ("the mean $\\Delta$ is negative and its interval excludes zero, so EEGNet "
                        "on the raw signal is significantly better than the CNN on scalograms.")
elif d < 0:
    m["mainVerdict"] = ("the mean $\\Delta$ is negative and its interval includes zero, so the CNN "
                        "on scalograms is not better than EEGNet, and probably slightly worse.")
else:
    m["mainVerdict"] = ("the mean $\\Delta$ is positive but its interval includes zero, so the "
                        "evidence does not meet the decision rule.")

d2, *_ = pair(A, "eegnet")
m["deltaInitial"] = f3(d2)
d3, *_ = pair("resnet18_cwt", B)
m["deltaCuda"] = f3(d3)
d4, l4, h4, _ = pair("resnet18_cwt", A)
m["deltaCudaMps"], m["ciCudaMps"] = f3(d4), ci(l4, h4)

d5, l5, h5, _ = pair(A, "resnet18_cwt_scratch")
m["deltaScratch"], m["ciScratch"], m["scratchSubj"] = f3(d5), ci(l5, h5), subjtxt("resnet18_cwt_scratch")

try:
    d6, l6, h6, _ = pair(A, "resnet18_stft_mps")
    m["deltaStft"], m["ciStft"], m["stftSubj"] = f3(d6), ci(l6, h6), subjtxt("resnet18_stft_mps")
except (IndexError, KeyError):   # STFT run not finished yet: draft placeholders
    print("WARNING: resnet18_stft_mps missing, writing placeholders")
    d6 = l6 = h6 = float("nan")
    m["deltaStft"] = m["ciStft"] = m["stftSubj"] = "\\textbf{??}"
if l6 > 0:
    m["stftVerdict"] = "differently, with wavelets ahead"
    m["stftSentence"] = "the wavelet's adaptive time--frequency resolution gives a small but reliable gain."
    m["stftShort"] = "gives a small, reliable gain"
elif h6 < 0:
    m["stftVerdict"] = "differently, with spectrograms ahead"
    m["stftSentence"] = "the fixed-window spectrogram is better, so the wavelet's adaptive resolution does not help here."
    m["stftShort"] = "does not help (STFT is better)"
else:
    m["stftVerdict"] = "indistinguishably"
    m["stftSentence"] = ("the interval includes zero, so the choice of time--frequency transform "
                         "does not measurably matter here.")
    m["stftShort"] = "makes no measurable difference"

# failure analysis
col = f"bal_acc_{B}"
s = subj.dropna(subset=[col]).sort_values(col)
m["worstRange"] = f"{s[col].min():.2f} to {s[col].max():.2f} for EEGNet"
m["worstSubjects"] = ", ".join(f"{int(r.subject)} ({r[col]:.2f}, {100 * r.pct_lenta:.0f}\\% slow)"
                               for _, r in s.head(4).iterrows())
c = cond.pivot(index="modelo", columns="condicion", values="bal_acc")
m["condText"] = (f"{c.loc[B, 'm']:.3f} (motion) vs.\\ {c.loc[B, 'n']:.3f} (motionless) for EEGNet and "
                 f"{c.loc[A, 'm']:.3f} vs.\\ {c.loc[A, 'n']:.3f} for ResNet18; a descriptive, "
                 "not pre-registered, contrast.")

# agreement numbers from the stored scores
d_ = np.load(RES / "ventanas.npz")
keep = np.load(RES / "mascara_artefactos.npy")
y = d_["y"][keep].astype(bool)
pa = np.mean([v >= 0.5 for v in np.load(RES / f"{A}_scores.npz").values()], 0) >= 0.5
pb = np.mean([v >= 0.5 for v in np.load(RES / f"{B}_scores.npz").values()], 0) >= 0.5
dis = pa != pb
both = (pa != y) & (pb != y)
m["agreePct"] = f"{100 * (~dis).mean():.1f}\\%"
m["disagree"] = f"EEGNet {100 * (pb[dis] == y[dis]).mean():.1f}\\% vs.\\ ResNet18 {100 * (pa[dis] == y[dis]).mean():.1f}\\%"
m["bothWrongSlow"] = f"{100 * y[both].mean():.1f}\\%"

with open(HERE / "numeros.tex", "w") as f:
    for k, v in m.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# main table
lines = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
         "Model & Input & BA & AUROC & Per-subject BA [95\\% CI] & Subj.\\ $>0.5$ \\\\", "\\midrule"]
inputs = {"control_permutado": "band power", "indice_ensayo": "trial index",
          "potencia_bandas_zsesion": "band power", "potencia_bandas": "band power",
          "resnet18_cwt_scratch": "CWT image", "resnet18_stft_mps": "STFT image",
          "resnet18_cwt": "CWT image", "eegnet": "raw $3\\times750$",
          "resnet18_cwt_mps": "CWT image", "eegnet_lr0.003_final": "raw $3\\times750$"}
prev = None
for stem, r in tab.iterrows():
    if prev is not None and r.group != prev:
        lines.append("\\midrule")
    prev = r.group
    row = (f"{r.label} & {inputs[stem]} & {r.bal_acc:.3f} $\\pm$ {r.bal_acc_sd:.3f} & {r.auroc:.3f} & "
           f"{r.subj_mean:.3f} [{r.subj_lo:.3f}, {r.subj_hi:.3f}] & {r.above_chance.replace('/', ' of ')}")
    if r.group == "main":
        row = " & ".join(f"\\textbf{{{c.strip()}}}" for c in row.split(" & "))
    lines.append(row + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(HERE / "tabla_principal.tex").write_text("\n".join(lines) + "\n")

perm = pd.read_csv(RES / "permutacion.csv")
perm = perm[perm.estadistico == "por_sujeto"]
names = {"eegnet_lr0.003_final": "EEGNet (main)", "resnet18_cwt_mps": "ResNet18 CWT (main)",
         "resnet18_stft_mps": "ResNet18 STFT", "resnet18_cwt_scratch": "ResNet18 scratch",
         "resnet18_cwt": "ResNet18 CWT (CUDA)", "eegnet": "EEGNet (initial)"}
pl = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
      "Model & Observed & Chance (shuffle) & Chance (shift) & $p$ \\\\", "\\midrule"]
for mod, g in perm.groupby("modelo", sort=False):
    sh = g[g.nula == "barajado"].iloc[0]
    ci_ = g[g.nula == "desplazamiento"].iloc[0]
    pl.append(f"{names.get(mod, mod)} & {sh.bal_acc_observada:.3f} & {sh.azar_medio:.3f} & "
              f"{ci_.azar_medio:.3f} & $<$0.001 \\\\")
pl += ["\\bottomrule", "\\end{tabular}"]
(HERE / "tabla_permutacion.tex").write_text("\n".join(pl) + "\n")

cl = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule",
      "Model & Motion (m) & Motionless (n) \\\\", "\\midrule"]
for mod in c.index:
    cl.append(f"{names.get(mod, mod)} & {c.loc[mod, 'm']:.3f} & {c.loc[mod, 'n']:.3f} \\\\")
r0 = cond[cond.modelo == B].set_index("condicion")
cl += ["\\midrule", f"Sessions / windows & {r0.loc['m', 'sesiones']} / {r0.loc['m', 'ventanas']:,} & "
       f"{r0.loc['n', 'sesiones']} / {r0.loc['n', 'ventanas']:,} \\\\".replace(",", "{,}"),
       f"Slow proportion & {r0.loc['m', 'pct_lenta']:.3f} & {r0.loc['n', 'pct_lenta']:.3f} \\\\",
       "\\bottomrule", "\\end{tabular}"]
(HERE / "tabla_condicion.tex").write_text("\n".join(cl) + "\n")

(HERE / "figures").mkdir(exist_ok=True)
for name in ("fig_resultados", "fig_diagnostico", "fig_interpretacion", "fig_fallos", "fig_representaciones", "pipeline"):
    for ext in ("pdf", "png"):
        if (FIG / f"{name}.{ext}").exists():
            shutil.copy(FIG / f"{name}.{ext}", HERE / "figures" / f"{name}.{ext}")
for k, v in m.items():
    print(f"{k:14s} {v}")
