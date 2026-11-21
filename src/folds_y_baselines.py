"""
Etapa 3: limpieza de artefactos, folds por sujeto y baselines simples.

Lee results/ventanas.npz (etapa 2) y hace tres cosas:
  1. Marca las ventanas con amplitud maxima > 150 uV como artefacto y guarda la
     mascara de las que se conservan (results/mascara_artefactos.npy, alineada con
     ventanas.npz). Los demas scripts deben usar esa misma mascara.
  2. Arma folds por sujeto (todas las sesiones de un sujeto en el mismo fold),
     equilibrando el numero de ventanas por fold, para 3 semillas. Se guardan en
     results/folds.csv para usar exactamente los mismos folds en todos los modelos.
  3. Corre referencias baratas con validacion cruzada por sujeto (5 folds x 3 semillas):
       - indice_ensayo: posicion del ensayo dentro de la sesion (revisa deriva en el tiempo)
       - potencia_bandas: log de potencia theta, alfa y beta en Fz, Pz y Oz
       - potencia_bandas_zsesion: lo mismo, normalizando cada sesion con sus propias ventanas
       - control_permutado: potencia_bandas_zsesion con etiquetas barajadas dentro de cada
         sesion; debe dar cerca de 0.5, y si no, hay un error en la tuberia

Nota sobre la normalizacion por sesion: usa estadisticas de las ventanas de la sesion
que se evalua (sin etiquetas). Equivale a una calibracion con datos no etiquetados del
mismo sujeto, y hay que declararlo en el reporte.

Salidas en results/: mascara_artefactos.npy, folds.csv, baselines_folds.csv,
baselines_por_sujeto.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ------------------------- Configuracion -------------------------
DATA_DIR = Path(os.environ.get(
    "EEG_DATA_DIR",
    Path.home() / "Documents" / "TEC" / "Neural systems" / "Proyecto" / "6427334",
))
OUT_DIR = DATA_DIR.parent / "results"

MAXABS_UV = 150.0              # ventanas con amplitud maxima mayor se rechazan
N_FOLDS = 5
SEEDS = (0, 1, 2)
FS = 250
BANDS = {"theta": (4, 8), "alfa": (8, 13), "beta": (13, 30)}
MIN_PER_CLASS = 30             # minimo por clase para calcular la metrica de un sujeto
N_BOOT = 2000


# ------------------------- Folds -------------------------
def make_folds(sizes, seed):
    """Asigna cada sujeto a un fold. Se ordenan los sujetos de mayor a menor tamano
    (con un ruido aleatorio que depende de la semilla) y cada uno va al fold que
    tenga menos ventanas hasta ese momento."""
    rng = np.random.default_rng(seed)
    subs = sizes.index.to_numpy()
    n = sizes.to_numpy().astype(float)
    order = np.argsort(-(n * rng.uniform(0.7, 1.3, len(n))))
    load = np.zeros(N_FOLDS)
    fold = {}
    for k in order:
        f = int(np.argmin(load))
        fold[int(subs[k])] = f
        load[f] += n[k]
    return fold


# ------------------------- Caracteristicas -------------------------
def band_features(X):
    """Log10 de la potencia media por banda y canal. Devuelve (N, 9)."""
    f, P = welch(X.astype(np.float64), fs=FS, nperseg=FS, axis=-1)
    feats = []
    for lo, hi in BANDS.values():
        m = (f >= lo) & (f < hi)
        feats.append(np.log10(P[:, :, m].mean(axis=2)))
    return np.concatenate(feats, axis=1)


def zscore_by_session(F, session):
    out = np.empty_like(F)
    for s in np.unique(session):
        i = session == s
        out[i] = (F[i] - F[i].mean(axis=0)) / (F[i].std(axis=0) + 1e-9)
    return out


def trial_position(trial, session):
    """Posicion relativa (0 a 1) del ensayo dentro de su sesion, y su cuadrado."""
    pos = np.empty(len(trial), dtype=np.float64)
    for s in np.unique(session):
        i = session == s
        t = trial[i].astype(np.float64)
        pos[i] = (t - t.min()) / (t.max() - t.min() + 1e-9)
    return np.stack([pos, pos ** 2], axis=1)


# ------------------------- Evaluacion -------------------------
def cv_eval(F, y, subject, all_folds, name):
    """Validacion cruzada por sujeto. Devuelve tabla por fold y las probabilidades
    de prueba de cada semilla (cada ventana se predice cuando su sujeto esta en prueba)."""
    rows, scores = [], {}
    for seed, fmap in all_folds.items():
        fold = np.array([fmap[int(s)] for s in subject])
        score = np.zeros(len(y))
        for k in range(N_FOLDS):
            te, tr = fold == k, fold != k
            sc = StandardScaler().fit(F[tr])
            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(sc.transform(F[tr]), y[tr])
            p = clf.predict_proba(sc.transform(F[te]))[:, 1]
            score[te] = p
            rows.append(dict(modelo=name, seed=seed, fold=k, n_test=int(te.sum()),
                             bal_acc=balanced_accuracy_score(y[te], (p >= 0.5).astype(int)),
                             auroc=roc_auc_score(y[te], p)))
        scores[seed] = score
    return pd.DataFrame(rows), scores


def per_subject(y, subject, scores, name):
    """Balanced accuracy por sujeto (promedio de las semillas), solo sujetos con
    al menos MIN_PER_CLASS ventanas de cada clase."""
    rows = []
    for s in np.unique(subject):
        i = subject == s
        if min((y[i] == 0).sum(), (y[i] == 1).sum()) < MIN_PER_CLASS:
            continue
        ba = np.mean([balanced_accuracy_score(y[i], (sc[i] >= 0.5).astype(int))
                      for sc in scores.values()])
        rows.append(dict(modelo=name, subject=int(s), n=int(i.sum()), bal_acc=ba))
    return pd.DataFrame(rows)


def boot_ci(values, n_boot=N_BOOT, seed=0):
    """IC 95% bootstrap de la media, remuestreando sujetos."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


# ------------------------- Programa principal -------------------------
def main():
    path = OUT_DIR / "ventanas.npz"
    if not path.exists():
        print(f"No encontre {path}. Corre primero crear_ventanas.py.")
        return
    d = np.load(path)
    keep = d["maxabs"] <= MAXABS_UV
    np.save(OUT_DIR / "mascara_artefactos.npy", keep)
    print(f"Ventanas totales: {len(keep)}   rechazadas (>{MAXABS_UV:.0f} uV): "
          f"{int((~keep).sum())}   conservadas: {int(keep.sum())}")

    X = d["X"][keep]
    y = d["y"][keep].astype(int)
    subject = d["subject"][keep]
    session = d["session"][keep]
    trial = d["trial"][keep]
    print(f"Conservadas: rapida {int((y == 0).sum())}   lenta {int((y == 1).sum())}   "
          f"sujetos {len(np.unique(subject))}\n")

    # Folds por sujeto
    sizes = pd.Series(subject).value_counts().sort_index()
    all_folds = {seed: make_folds(sizes, seed) for seed in SEEDS}
    folds_df = pd.DataFrame({"subject": sizes.index, "n_ventanas": sizes.to_numpy()})
    for seed, fmap in all_folds.items():
        folds_df[f"fold_seed{seed}"] = [fmap[int(s)] for s in sizes.index]
    folds_df.to_csv(OUT_DIR / "folds.csv", index=False)
    for seed, fmap in all_folds.items():
        fold = np.array([fmap[int(s)] for s in subject])
        print(f"Semilla {seed}: ventanas por fold {np.bincount(fold, minlength=N_FOLDS).tolist()}"
              f"   lentas por fold {np.bincount(fold, weights=y, minlength=N_FOLDS).astype(int).tolist()}")

    # Caracteristicas
    F_band = band_features(X)
    F_band_z = zscore_by_session(F_band, session)
    F_pos = trial_position(trial, session)
    rng = np.random.default_rng(123)
    y_perm = y.copy()
    for s in np.unique(session):
        i = np.where(session == s)[0]
        y_perm[i] = rng.permutation(y[i])

    jobs = [("indice_ensayo", F_pos, y), ("potencia_bandas", F_band, y),
            ("potencia_bandas_zsesion", F_band_z, y),
            ("control_permutado", F_band_z, y_perm)]
    folds_all, subj_all, summary = [], [], []
    for name, F, yy in jobs:
        tab, scores = cv_eval(F, yy, subject, all_folds, name)
        ps = per_subject(yy, subject, scores, name)
        folds_all.append(tab)
        subj_all.append(ps)
        lo, hi = boot_ci(ps.bal_acc)
        summary.append(dict(
            modelo=name,
            bal_acc_media=tab.bal_acc.mean(), bal_acc_sd=tab.bal_acc.std(),
            auroc_media=tab.auroc.mean(),
            por_sujeto_media=ps.bal_acc.mean(), por_sujeto_IC95=f"[{lo:.3f}, {hi:.3f}]",
            sujetos_sobre_0_5=f"{int((ps.bal_acc > 0.5).sum())}/{len(ps)}"))

    pd.concat(folds_all).to_csv(OUT_DIR / "baselines_folds.csv", index=False)
    pd.concat(subj_all).to_csv(OUT_DIR / "baselines_por_sujeto.csv", index=False)

    # ------------------------- Resumen en pantalla -------------------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n===== RESUMEN (5 folds x 3 semillas, validacion por sujeto) =====")
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print("\nbal_acc_media y auroc_media: sobre los 15 pares fold x semilla.")
    print("por_sujeto: balanced accuracy por sujeto (solo con >= "
          f"{MIN_PER_CLASS} ventanas por clase), media e IC 95% bootstrap sobre sujetos.")
    print(f"\nGuardado en: {OUT_DIR}")


if __name__ == "__main__":
    main()
