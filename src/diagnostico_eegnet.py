"""
Etapa 5 (diagnostico): que senales usa EEGNet, por canal y por banda de frecuencia.

Vuelve a entrenar EEGNet con el mismo protocolo (mismos folds, semillas, perdida, tope
de epocas y parada temprana) cambiando una sola cosa por variante: los canales de
entrada o la banda de frecuencia. Cada variante se corre llamando a entrenar_eegnet.py
con EEG_CANALES o EEG_BANDA, asi que la implementacion del entrenamiento es una sola.

La pregunta concreta que motiva esto: en los escalogramas medios, las ventanas lentas
tienen menos potencia por debajo de 4 Hz en Fz. Eso puede ser actividad cerebral o puede
ser movimiento de ojos o de cuerpo. Si al quitar lo de abajo de 4 Hz el desempeno no
baja, la diferencia de Fz no era lo que sostenia la decision; si baja mucho, el modelo
dependia de una senal que probablemente no es neural.

Variantes (la referencia es eegnet, con los 3 canales y la banda 1 a 50 Hz de la etapa 2):
  por canal : solo Fz, solo Pz, solo Oz
  por banda : 4 a 50 Hz (sin delta), 1 a 4 Hz (solo delta), 4 a 13 Hz (theta y alfa)
  cruce     : solo Fz sin delta, para separar canal y banda

Aviso sobre el filtro: la banda extra se aplica a la ventana de 3 s ya cortada, no a la
grabacion continua, asi que deja transitorios en los bordes (ver band_limit en
entrenar_eegnet.py). Es un diagnostico, no la tuberia principal.

Variables de entorno:
  EEG_FORZAR  1: vuelve a entrenar aunque ya existan los resultados de la variante.
  EEG_SOLO    subconjunto de variantes a correr, separadas por coma (por sus nombres).

Salidas en results/: eegnet_ch*_folds.csv y eegnet_banda*_folds.csv (y sus por_sujeto y
scores), mas diagnostico_eegnet.csv con la tabla comparativa.
Salidas en figures/: diagnostico_eegnet.png
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisis_final import DATA_DIR, FIG_DIR, OUT_DIR, boot_ci, sign_flip_p  # noqa: E402

REFERENCIA = "eegnet"
FORZAR = os.environ.get("EEG_FORZAR") == "1"

# (nombre del modelo, variables de entorno, descripcion para la tabla)
VARIANTES = [
    ("eegnet_chFz", {"EEG_CANALES": "Fz"}, "solo Fz"),
    ("eegnet_chPz", {"EEG_CANALES": "Pz"}, "solo Pz"),
    ("eegnet_chOz", {"EEG_CANALES": "Oz"}, "solo Oz"),
    ("eegnet_banda4-50", {"EEG_BANDA": "4-50"}, "sin delta (4 a 50 Hz)"),
    ("eegnet_banda1-4", {"EEG_BANDA": "1-4"}, "solo delta (1 a 4 Hz)"),
    ("eegnet_banda4-13", {"EEG_BANDA": "4-13"}, "theta y alfa (4 a 13 Hz)"),
    ("eegnet_chFz_banda4-50", {"EEG_CANALES": "Fz", "EEG_BANDA": "4-50"},
     "solo Fz sin delta"),
]


def run_variant(name, env_extra):
    """Entrena una variante llamando a entrenar_eegnet.py. Devuelve True si corrio."""
    if (OUT_DIR / f"{name}_folds.csv").exists() and not FORZAR:
        print(f"[salto] {name}: ya existe (EEG_FORZAR=1 para repetir)")
        return False
    env = dict(os.environ, EEG_DATA_DIR=str(DATA_DIR), **env_extra)
    env.pop("EEG_QUICK", None)
    print(f"[corre] {name}: {env_extra}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u",
                        str(Path(__file__).resolve().parent / "entrenar_eegnet.py")],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"{name} fallo con codigo {r.returncode}")
    for line in r.stdout.splitlines():
        if line.startswith(("bal_acc", "por sujeto")):
            print("        " + line)
    print(f"        listo en {(time.time() - t0) / 60:.1f} min", flush=True)
    return True


def collect(nombres):
    """Tabla comparativa: desempeno de cada variante y su diferencia con la referencia."""
    ref_sub = pd.read_csv(OUT_DIR / f"{REFERENCIA}_por_sujeto.csv")[["subject", "bal_acc"]]
    rows = []
    for name, _, desc in nombres:
        f = OUT_DIR / f"{name}_folds.csv"
        g = OUT_DIR / f"{name}_por_sujeto.csv"
        if not (f.exists() and g.exists()):
            continue
        tab = pd.read_csv(f)
        sub = pd.read_csv(g)[["subject", "bal_acc"]]
        m = sub.merge(ref_sub, on="subject", suffixes=("", "_ref"))
        d = m.bal_acc - m.bal_acc_ref
        (lo, hi), _ = boot_ci(d)
        rows.append(dict(variante=name, entrada=desc, bal_acc=tab.bal_acc.mean(),
                         auroc=tab.auroc.mean(), por_sujeto=sub.bal_acc.mean(),
                         delta_vs_ref=d.mean(), ic95_bajo=lo, ic95_alto=hi,
                         p_signos=sign_flip_p(d), n_sujetos=len(d),
                         epoca_mejor=tab.epoca_mejor.median()))
    return pd.DataFrame(rows)


def figura(tab, ref_row):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   (matplotlib no esta instalado; se omite la figura)")
        return
    t = pd.concat([ref_row, tab], ignore_index=True)
    y = np.arange(len(t))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(t) + 2.2))

    ax = axes[0]
    colors = ["tab:green"] + ["tab:blue"] * len(tab)
    ax.barh(y, t.bal_acc, color=colors)
    ax.axvline(0.5, color="black", ls=":", lw=1)
    ax.axvline(float(ref_row.bal_acc.iloc[0]), color="tab:green", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(t.entrada, fontsize=9)
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("balanced accuracy (media de 15 corridas)")
    ax.set_title("EEGNet por canal y por banda", fontsize=11)

    ax = axes[1]
    d = t.delta_vs_ref.fillna(0.0)
    ax.barh(y, d, xerr=[d - t.ic95_bajo.fillna(0.0), t.ic95_alto.fillna(0.0) - d],
            color=colors, error_kw=dict(lw=1, capsize=3))
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("diferencia por sujeto frente a los 3 canales, 1 a 50 Hz")
    ax.set_title("Diferencia pareada, IC 95% sobre sujetos", fontsize=11)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "diagnostico_eegnet.png", dpi=130)
    plt.close(fig)


def main():
    if not (OUT_DIR / f"{REFERENCIA}_folds.csv").exists():
        print(f"Falta {REFERENCIA}_folds.csv (la referencia). Corre antes entrenar_eegnet.py.")
        return
    solo = os.environ.get("EEG_SOLO")
    pedidas = [v for v in VARIANTES if not solo or v[0] in solo.split(",")]
    print(f"Variantes pedidas: {len(pedidas)}   referencia: {REFERENCIA}   "
          f"forzar: {FORZAR}\n")
    for name, env_extra, _ in pedidas:
        run_variant(name, env_extra)

    tab = collect(pedidas)
    if tab.empty:
        print("\nNo hay resultados que resumir.")
        return
    ref_tab = pd.read_csv(OUT_DIR / f"{REFERENCIA}_folds.csv")
    ref_sub = pd.read_csv(OUT_DIR / f"{REFERENCIA}_por_sujeto.csv")
    ref_row = pd.DataFrame([dict(
        variante=REFERENCIA, entrada="3 canales, 1 a 50 Hz (referencia)",
        bal_acc=ref_tab.bal_acc.mean(), auroc=ref_tab.auroc.mean(),
        por_sujeto=ref_sub.bal_acc.mean(), delta_vs_ref=np.nan,
        ic95_bajo=np.nan, ic95_alto=np.nan, p_signos=np.nan,
        n_sujetos=len(ref_sub), epoca_mejor=ref_tab.epoca_mejor.median())])

    out = pd.concat([ref_row, tab], ignore_index=True)
    out.to_csv(OUT_DIR / "diagnostico_eegnet.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n===== DIAGNOSTICO DE EEGNET POR CANAL Y POR BANDA =====")
    print(out.round(4).to_string(index=False))
    print("\nbal_acc y auroc: media de los 15 pares fold x semilla.")
    print("delta_vs_ref: diferencia por sujeto frente a la referencia, IC 95% bootstrap")
    print("remuestreando sujetos, y p de la permutacion de signos pareada.")
    figura(tab, ref_row)
    print(f"\nGuardado en: {OUT_DIR / 'diagnostico_eegnet.csv'}\n"
          f"Figura en: {FIG_DIR / 'diagnostico_eegnet.png'}")


if __name__ == "__main__":
    main()
