"""
Elige la tasa de aprendizaje de cada modelo usando SOLO la validacion interna.

Lee los archivos *_folds.csv de results/ y, para cada familia (eegnet, resnet18_cwt,
resnet18_stft), compara las tasas de aprendizaje probadas con la AUROC de validacion
interna (columna auroc_val) promediada sobre los 5 folds de la semilla 0. La validacion
interna usa sujetos del conjunto de entrenamiento, asi que el fold de prueba nunca
interviene en la eleccion. Por eso la tabla no muestra metricas de prueba.

Familia eegnet: archivo eegnet_folds.csv (lr 1e-3 por defecto) y eegnet_lr*_folds.csv
Familia resnet18_cwt: resnet18_cwt_folds.csv (lr 1e-4 por defecto) y resnet18_cwt_lr*_folds.csv
Las variantes _scratch no entran en la comparacion.
"""
import os
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get(
    "EEG_DATA_DIR",
    Path.home() / "Documents" / "TEC" / "Neural systems" / "Proyecto" / "6427334",
))
OUT_DIR = DATA_DIR.parent / "results"

DEFAULT_LR = {"eegnet": 1e-3, "resnet18_cwt": 1e-4, "resnet18_stft": 1e-4}
NAME_RE = re.compile(r"^(eegnet|resnet18_cwt|resnet18_stft)(?:_lr(.+))?$")


def main():
    rows = []
    for path in sorted(OUT_DIR.glob("*_folds.csv")):
        m = NAME_RE.match(path.name[: -len("_folds.csv")])
        if not m:
            continue
        family = m.group(1)
        lr = float(m.group(2)) if m.group(2) else DEFAULT_LR[family]
        s0 = pd.read_csv(path).query("seed == 0")
        rows.append(dict(familia=family, lr=lr, archivo=path.name, folds_semilla0=len(s0),
                         auroc_val=s0.auroc_val.mean(), epoca_mejor_mediana=s0.epoca_mejor.median()))
    if not rows:
        print(f"No encontre archivos *_folds.csv en {OUT_DIR}")
        return
    tab = pd.DataFrame(rows).sort_values(["familia", "lr"])
    pd.set_option("display.width", 200)
    print("===== AUROC de validacion interna, semilla 0 =====")
    shown = tab.copy()
    shown["lr"] = shown.lr.map(lambda v: f"{v:g}")
    print(shown.round(4).to_string(index=False))
    for family, g in tab.groupby("familia"):
        g = g[g.folds_semilla0 == 5]
        if g.empty:
            continue
        best = g.loc[g.auroc_val.idxmax()]
        print(f"\n{family}: mejor lr = {best.lr:g}  (AUROC val {best.auroc_val:.4f}, "
              f"{len(g)} tasas comparadas)")
        if best.lr == DEFAULT_LR[family]:
            print("   Es la tasa por defecto: ya existen sus 15 corridas, no hay que repetir nada.")
        else:
            print(f"   Corrida final con semillas 0,1,2:  %env EEG_LR={best.lr:g}   y   %env EEG_SEEDS=0,1,2")


if __name__ == "__main__":
    main()
