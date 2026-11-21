"""
Etapa 2: ventanas de EEG previas a cada desviacion, con sus etiquetas.

Lee ensayos.csv (etapa 1) y, para cada sesion, el .set correspondiente. Aplica
las reglas acordadas, recorta los 3 s de EEG justo antes del inicio de la
desviacion en Fz, Pz y Oz, y guarda todo en un solo archivo:

  results/ventanas.npz        : X, y y metadatos (ver abajo)
  results/ventanas_meta.csv   : los metadatos en tabla, para revisar en Excel

Reglas acordadas
  - Se excluye la sesion duplicada s04_051130m.set (misma grabacion que s05_051120m).
  - Ensayo valido: 0.1 s <= TR <= 60 s y sin solape (hueco previo >= 3 s).
  - Referencia por sesion: percentil 5 del TR de los ensayos con TR valido.
  - rapida (y=0): TR < 1.5 x ref ; lenta (y=1): TR > 2.5 x ref ; el resto se descarta.
  - Senal: pasa-banda 1 a 50 Hz (Butterworth orden 4, fase cero) sobre la grabacion
    continua, y luego se toma una muestra de cada dos (500 Hz -> 250 Hz).
  - Ventana: 3 s inmediatamente antes de la desviacion = 750 muestras, canales Fz, Pz, Oz.

Contenido de ventanas.npz
  X       (N, 3, 750) float32, microvoltios, canales en orden Fz, Pz, Oz
  y       (N,) 0 = rapida, 1 = lenta
  subject (N,) numero de sujeto
  session (N,) nombre del archivo .set
  trial   (N,) indice del ensayo dentro de la sesion (sirve para revisar deriva en el tiempo)
  rt      (N,) tiempo de reaccion en segundos
  rt_ref  (N,) referencia de la sesion en segundos
  maxabs  (N,) amplitud maxima absoluta de la ventana (para decidir rechazo de artefactos)
"""
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

# ------------------------- Configuracion -------------------------
DATA_DIR = Path(os.environ.get(
    "EEG_DATA_DIR",
    Path.home() / "Documents" / "TEC" / "Neural systems" / "Proyecto" / "6427334",
))
OUT_DIR = DATA_DIR.parent / "results"

EXCLUDE_FILES = {"s04_051130m.set"}
CHANNELS = ("FZ", "PZ", "OZ")

MIN_RT, MAX_RT = 0.1, 60.0
WIN_SEC = 3.0
REF_PCT = 5
K_FAST, K_SLOW = 1.5, 2.5

BAND = (1.0, 50.0)
DECIM = 2                      # 500 Hz -> 250 Hz

SUBJECT_RE = re.compile(r"^s(\d+)_", re.IGNORECASE)


# ------------------------- Etiquetado -------------------------
def label_session(df):
    """Aplica las reglas acordadas a los ensayos de UNA sesion.
    Devuelve el mismo DataFrame con rt_ref, usable y label (0, 1 o -1 = descartado)."""
    df = df.copy()
    df["overlap"] = df.gap_prev < WIN_SEC
    df["rt_ok"] = (df.rt >= MIN_RT) & (df.rt <= MAX_RT)
    df["usable"] = df.rt_ok & ~df.overlap
    ref = np.percentile(df.loc[df.rt_ok, "rt"], REF_PCT) if df.rt_ok.any() else np.nan
    df["rt_ref"] = ref
    df["label"] = -1
    df.loc[df.usable & (df.rt < K_FAST * ref), "label"] = 0
    df.loc[df.usable & (df.rt > K_SLOW * ref), "label"] = 1
    return df


# ------------------------- Lectura y filtrado -------------------------
def load_channels(path):
    """Lee el .set y devuelve (srate, senales de FZ, PZ, OZ como float64 (3, n))."""
    mat = loadmat(path, squeeze_me=True, struct_as_record=False,
                  variable_names=["EEG"])
    eeg = mat["EEG"]
    srate = float(eeg.srate)
    labels = [str(c.labels).upper() for c in np.atleast_1d(eeg.chanlocs)]
    idx = [labels.index(c) for c in CHANNELS]
    data = np.asarray(eeg.data[idx, :], dtype=np.float64)
    return srate, data


def bandpass(data, srate):
    sos = butter(4, BAND, btype="bandpass", fs=srate, output="sos")
    return sosfiltfilt(sos, data, axis=1)


def cut_windows(sig, onsets, srate):
    """Recorta, para cada desviacion, los WIN_SEC antes del inicio.
    sig: (3, n) filtrada a srate. Devuelve (k, 3, n_win) y la mascara de los validos."""
    n_raw = int(round(WIN_SEC * srate))
    wins, valid = [], []
    for lat in onsets:
        end = int(round(lat)) - 1          # latencias de EEGLAB empiezan en 1
        start = end - n_raw
        if start < 0 or end > sig.shape[1]:
            wins.append(np.zeros((sig.shape[0], n_raw // DECIM)))
            valid.append(False)
            continue
        wins.append(sig[:, start:end:DECIM])
        valid.append(True)
    return np.stack(wins), np.array(valid)


# ------------------------- Programa principal -------------------------
def main():
    ens_path = OUT_DIR / "ensayos.csv"
    if not ens_path.exists():
        print(f"No encontre {ens_path}. Corre primero inventario_sesiones.py.")
        return
    ens = pd.read_csv(ens_path)
    files = [p for p in sorted(DATA_DIR.glob("*.set"))
             if p.name not in EXCLUDE_FILES and p.name in set(ens.archivo)]
    print(f"Carpeta de datos: {DATA_DIR}\nSesiones a procesar: {len(files)}\n")

    X, meta = [], []
    t0 = time.time()
    for i, path in enumerate(files, 1):
        df = label_session(ens[ens.archivo == path.name].reset_index(drop=True))
        sel = df[df.label >= 0]
        print(f"[{i}/{len(files)}] {path.name}: {len(sel)} ventanas "
              f"({int((sel.label == 0).sum())} rapida / {int((sel.label == 1).sum())} lenta)",
              end="", flush=True)
        if sel.empty:
            print()
            continue
        srate, data = load_channels(path)
        if srate != 500.0:
            raise ValueError(f"{path.name}: srate {srate}, se esperaba 500")
        sig = bandpass(data, srate)
        wins, valid = cut_windows(sig, sel.onset.to_numpy(), srate)
        sel = sel[valid]
        wins = wins[valid]
        X.append(wins.astype(np.float32))
        m = pd.DataFrame(dict(
            subject=int(SUBJECT_RE.match(path.name).group(1)),
            session=path.name, trial=sel.trial.to_numpy(), y=sel.label.to_numpy(),
            rt=sel.rt.to_numpy(), rt_ref=sel.rt_ref.to_numpy(),
            maxabs=np.abs(wins).max(axis=(1, 2)),
        ))
        meta.append(m)
        print(f"   ({time.time() - t0:.0f} s)")

    X = np.concatenate(X)
    meta = pd.concat(meta, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "ventanas.npz", X=X, y=meta.y.to_numpy().astype(np.int8),
        subject=meta.subject.to_numpy(), session=meta.session.to_numpy().astype(str),
        trial=meta.trial.to_numpy(), rt=meta.rt.to_numpy(),
        rt_ref=meta.rt_ref.to_numpy(), maxabs=meta.maxabs.to_numpy(),
    )
    meta.to_csv(OUT_DIR / "ventanas_meta.csv", index=False, encoding="utf-8-sig")

    # ------------------------- Resumen en pantalla -------------------------
    print("\n===== RESUMEN =====")
    print(f"Ventanas: {len(meta)}   forma de X: {X.shape}   NaN: {int(np.isnan(X).sum())}")
    print(f"Sujetos: {meta.subject.nunique()}   sesiones: {meta.session.nunique()}")
    print(f"rapida (0): {int((meta.y == 0).sum())}   lenta (1): {int((meta.y == 1).sum())}")
    print("Amplitud maxima por ventana (uV), percentiles 50/90/99/100:",
          np.percentile(meta.maxabs, [50, 90, 99, 100]).round(1))
    print(f"\nGuardado en: {OUT_DIR}  (ventanas.npz, ventanas_meta.csv)")


if __name__ == "__main__":
    main()
