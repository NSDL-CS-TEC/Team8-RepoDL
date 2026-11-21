# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:20:42 2026

@author: Juan Pablo
"""

"""
Etapa 1: inventario de sesiones del dataset de Cao et al. (2019).

Lee cada archivo .set (formato MATLAB v5 de EEGLAB) de uno en uno y registra,
por sesion: duracion, numero de ensayos, distribucion de tiempos de reaccion (TR)
y solape de ventanas. No modifica los datos originales.

Genera dos CSV en la carpeta "results" (junto a la carpeta de datos):
  - inventario_sesiones.csv : una fila por sesion
  - ensayos.csv             : una fila por ensayo (marcas, TR, solape, etiqueta provisional)

Marcadores: 251/252 = inicio de desviacion (izq/der), 253 = inicio de respuesta,
254 = fin de respuesta.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

# ------------------------- Configuracion -------------------------
# Ajusta DATA_DIR si tu ruta es distinta (puedes fijarla con la variable de
# entorno EEG_DATA_DIR).
DATA_DIR = Path(os.environ.get(
    "EEG_DATA_DIR",
    Path.home() / "Documents" / "TEC" / "Neural systems" / "Proyecto" / "6427334",
))
OUT_DIR = DATA_DIR.parent / "results"

ONSET = (251, 252)   # inicio de desviacion
RESP_ON = 253        # inicio de respuesta
RESP_OFF = 254       # fin de respuesta

MIN_RT = 0.1         # s; TR menores se tratan como artefactos de marcado
WIN_SEC = 3.0        # s; ventana de EEG previa al desvio
REF_PCT = 5          # percentil del TR de referencia (se calcula por sesion)
K_FAST, K_SLOW = 1.5, 2.5   # rapida: TR < 1.5*ref ; lenta: TR > 2.5*ref
# Limite superior de TR: aun sin decidir. Se deja sin limite y se reporta
# cuantos TR pasan de 5 s y de 10 s para decidirlo viendo los datos.
MAX_RT = np.inf

NAME_RE = re.compile(r"^s(\d+)_(\d{6})(?:_(\d+))?([mn])\.set$", re.IGNORECASE)


# ------------------------- Lectura -------------------------
def parse_name(fname):
    """Saca sujeto, fecha (AAMMDD), sub-sesion y condicion (m/n) del nombre."""
    m = NAME_RE.match(fname)
    if not m:
        return dict(subject=np.nan, date=None, sub_session=None, condition=None)
    return dict(subject=int(m.group(1)), date=m.group(2),
                sub_session=m.group(3), condition=m.group(4).lower())


def read_set(path):
    """Lee solo lo necesario de un .set (MAT v5). Devuelve metadatos y eventos."""
    mat = loadmat(path, squeeze_me=True, struct_as_record=False,
                  variable_names=["EEG"])
    eeg = mat["EEG"]
    srate = float(eeg.srate)
    data = eeg.data
    if isinstance(data, np.ndarray) and data.ndim == 2:
        n_chan, n_samp = data.shape
        data_in_file = True
    else:  # los datos estan en un .fdt aparte
        n_chan, n_samp = int(eeg.nbchan), int(eeg.pnts)
        data_in_file = False
    events = []
    for e in np.atleast_1d(eeg.event):
        try:
            events.append((int(str(e.type)), float(e.latency)))
        except (ValueError, AttributeError):
            continue  # marcas que no son numericas (p. ej. "boundary")
    labels = [str(c.labels).upper() for c in np.atleast_1d(eeg.chanlocs)]
    return srate, n_chan, n_samp, data_in_file, events, labels


# ------------------------- Ensayos -------------------------
def build_trials(events):
    """Agrupa los marcadores en ensayos completos: desviacion -> inicio -> fin
    de respuesta. Devuelve (lista de ensayos, numero de ensayos incompletos)."""
    trials, cur, n_incomplete = [], None, 0
    for typ, lat in events:
        if typ in ONSET:
            if cur is not None:      # llego otra desviacion sin cerrar el ensayo
                n_incomplete += 1
            cur = dict(side=typ, onset=lat, resp_on=None, resp_off=None)
        elif typ == RESP_ON and cur is not None and cur["resp_on"] is None:
            cur["resp_on"] = lat
        elif typ == RESP_OFF and cur is not None and cur["resp_on"] is not None:
            cur["resp_off"] = lat
            trials.append(cur)
            cur = None
    if cur is not None:
        n_incomplete += 1
    return trials, n_incomplete


def trials_table(trials, srate):
    """Tabla de ensayos con TR, hueco previo, solape y etiqueta provisional."""
    df = pd.DataFrame(trials)
    if df.empty:
        return df, np.nan
    df["rt"] = (df.resp_on - df.onset) / srate
    df["resp_dur"] = (df.resp_off - df.resp_on) / srate
    # Hueco entre el fin de la respuesta anterior y esta desviacion. Para el
    # primer ensayo es el tiempo desde el inicio de la grabacion.
    df["gap_prev"] = (df.onset - df.resp_off.shift(1)) / srate
    df.loc[df.index[0], "gap_prev"] = df.onset.iloc[0] / srate
    df["overlap"] = df.gap_prev < WIN_SEC   # la ventana de 3 s no queda limpia
    df["rt_ok"] = (df.rt >= MIN_RT) & (df.rt <= MAX_RT)
    usable = df.rt_ok & ~df.overlap

    # Referencia por sesion: percentil 5 de los TR sin artefactos
    ref = np.percentile(df.loc[df.rt_ok, "rt"], REF_PCT) if df.rt_ok.any() else np.nan

    df["label"] = "excluido"
    df.loc[usable, "label"] = "intermedia"
    df.loc[usable & (df.rt < K_FAST * ref), "label"] = "rapida"
    df.loc[usable & (df.rt > K_SLOW * ref), "label"] = "lenta"
    return df, ref


# ------------------------- Resumen por sesion -------------------------
def summarize(fname, size_mb, srate, n_chan, n_samp, data_in_file, labels,
              n_markers, n_incomplete, df, ref):
    info = dict(archivo=fname, **parse_name(fname))
    info.update(size_mb=round(size_mb, 1), srate=srate, n_chan=n_chan,
                dur_min=round(n_samp / srate / 60, 1), data_in_file=data_in_file,
                tiene_Fz_Pz_Oz=all(c in labels for c in ("FZ", "PZ", "OZ")),
                n_marcadores=n_markers, n_ensayos=len(df),
                n_incompletos=n_incomplete)
    if df.empty:
        return info
    rt = df.rt
    info.update(
        rt_p5=rt.quantile(0.05), rt_p25=rt.quantile(0.25), rt_mediana=rt.median(),
        rt_p75=rt.quantile(0.75), rt_p95=rt.quantile(0.95), rt_max=rt.max(),
        frac_rt_menor_01=(rt < MIN_RT).mean(),
        frac_rt_mayor_5=(rt > 5).mean(), frac_rt_mayor_10=(rt > 10).mean(),
        frac_solape=df.overlap.mean(),
        n_usables=int((df.rt_ok & ~df.overlap).sum()),
        rt_ref=ref,
        n_rapida=int((df.label == "rapida").sum()),
        n_lenta=int((df.label == "lenta").sum()),
        n_intermedia=int((df.label == "intermedia").sum()),
    )
    info["n_etiquetados"] = info["n_rapida"] + info["n_lenta"]
    return info


# ------------------------- Programa principal -------------------------
def main():
    files = sorted(DATA_DIR.glob("*.set"))
    if not files:
        print(f"No encontre archivos .set en: {DATA_DIR}\n"
              "Revisa DATA_DIR al inicio del script.")
        return
    print(f"Carpeta de datos: {DATA_DIR}\nArchivos .set: {len(files)}\n")

    sessions, all_trials = [], []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        size_mb = path.stat().st_size / 1e6
        try:
            srate, n_chan, n_samp, in_file, events, labels = read_set(path)
            trials, n_inc = build_trials(events)
            df, ref = trials_table(trials, srate)
            sessions.append(summarize(path.name, size_mb, srate, n_chan, n_samp,
                                      in_file, labels, len(events), n_inc, df, ref))
            if not df.empty:
                df.insert(0, "archivo", path.name)
                df.insert(1, "trial", np.arange(len(df)))
                all_trials.append(df)
        except Exception as exc:  # un archivo con problemas no detiene el resto
            print(f"   ERROR: {exc}")
            sessions.append(dict(archivo=path.name, **parse_name(path.name),
                                 size_mb=round(size_mb, 1), error=str(exc)))

    ses = pd.DataFrame(sessions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ses.to_csv(OUT_DIR / "inventario_sesiones.csv", index=False, encoding="utf-8-sig")
    if all_trials:
        pd.concat(all_trials).to_csv(OUT_DIR / "ensayos.csv", index=False,
                                     encoding="utf-8-sig")

    # ------------------------- Resumen en pantalla -------------------------
    print("\n===== RESUMEN =====")
    print(f"Archivos leidos: {len(ses)}   con error: "
          f"{ses['error'].notna().sum() if 'error' in ses else 0}")
    print(f"Sujetos distintos: {ses['subject'].nunique()}")
    print("Sesiones por condicion:\n", ses['condition'].value_counts(dropna=False))
    print("Sesiones por sujeto:\n", ses['subject'].value_counts().sort_index())
    if "n_ensayos" in ses:
        print(f"\nEnsayos totales: {int(ses['n_ensayos'].sum())}   "
              f"usables: {int(ses['n_usables'].sum())}   "
              f"etiquetados (rapida/lenta): {int(ses['n_etiquetados'].sum())}")
    print(f"\nCSV guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()