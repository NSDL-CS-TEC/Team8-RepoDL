"""
Etapa 4a: EEGNet sobre la senal cruda (baseline de la comparacion principal).

Entrena y evalua EEGNet (Lawhern et al., 2018) con validacion cruzada por sujeto,
usando exactamente los mismos folds (results/folds.csv), las mismas ventanas y la
misma mascara de artefactos que los baselines de la etapa 3.

Entrada : ventanas de 3 canales (Fz, Pz, Oz) x 750 muestras (3 s a 250 Hz), normalizadas
          por sesion y por canal (media y desviacion de la sesion). Esto usa datos sin
          etiquetas de la sesion evaluada, igual que el baseline potencia_bandas_zsesion.
Entrenamiento: AdamW, perdida binaria ponderada por clase, hasta N_EPOCHS epocas con
          parada temprana sobre una validacion interna formada por sujetos del conjunto
          de entrenamiento (el fold de prueba nunca interviene en la parada).
Salidas en results/:
  eegnet_folds.csv        una fila por fold x semilla (mismo formato que baselines_folds.csv)
  eegnet_por_sujeto.csv   balanced accuracy por sujeto (mismo formato que baselines_por_sujeto.csv)
  eegnet_scores.npz       probabilidades de prueba por semilla, alineadas con las ventanas
                          conservadas, para el analisis posterior

Variables de entorno (se fijan en la consola de Spyder con %env NOMBRE=valor):
  EEG_QUICK  1: prueba rapida (un solo fold, 3 epocas). Los archivos salen con el sufijo
             _quick, para no sobreescribir los resultados de la corrida completa.
  EEG_DEVICE forzar el dispositivo (cuda, mps o cpu). Por defecto se elige el mejor.
  EEG_LR     tasa de aprendizaje (por defecto 1e-3). Si se fija, los archivos de salida
             llevan la etiqueta, por ejemplo eegnet_lr0.0003_folds.csv.
  EEG_EPOCAS tope de epocas (por defecto 40; la corrida final usa 80, igual que el CNN).
  EEG_ETIQUETA sufijo extra para el nombre de salida, por ejemplo final.
  EEG_SEEDS  semillas separadas por coma (por defecto 0,1,2). Para el ajuste de la tasa
             de aprendizaje se usa solo 0.
  EEG_CANALES  subconjunto de canales, por ejemplo Fz o Fz,Pz (por defecto Fz,Pz,Oz).
  EEG_BANDA    banda extra "lo-hi" en Hz, por ejemplo 4-50 para quitar lo de abajo de 4 Hz.
             EEG_CANALES y EEG_BANDA son el diagnostico de que senales llevan la
             informacion, y agregan al nombre los sufijos _chXX y _bandaLO-HI.
Los archivos de salida se llaman NOMBRE_folds.csv, NOMBRE_por_sujeto.csv y NOMBRE_scores.npz.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# ------------------------- Configuracion -------------------------
DATA_DIR = Path(os.environ.get(
    "EEG_DATA_DIR",
    Path.home() / "Documents" / "TEC" / "Neural systems" / "Proyecto" / "6427334",
))
OUT_DIR = DATA_DIR.parent / "results"

_lr_env = os.environ.get("EEG_LR")
LR = float(_lr_env) if _lr_env else 1e-3
MODEL_NAME = f"eegnet_lr{LR:g}" if _lr_env else "eegnet"

FS = 250
CHANNELS = ("Fz", "Pz", "Oz")          # orden de los canales en ventanas.npz

# Diagnostico por canal y por banda: que senales llevan la informacion.
_ch_env = os.environ.get("EEG_CANALES")
USE_CH = tuple(c.strip() for c in _ch_env.split(",")) if _ch_env else CHANNELS
_band_env = os.environ.get("EEG_BANDA")
BAND = tuple(float(v) for v in _band_env.split("-")) if _band_env else None
if _ch_env:
    MODEL_NAME += "_ch" + "".join(USE_CH)
if _band_env:
    MODEL_NAME += f"_banda{_band_env}"

N_FOLDS = 5
SEEDS = tuple(int(x) for x in os.environ.get("EEG_SEEDS", "0,1,2").split(","))
MIN_PER_CLASS = 30
N_BOOT = 2000

# Hiperparametros (se mantienen fijos; la misma disciplina se aplica al CNN con imagenes)
N_EPOCHS = int(os.environ.get("EEG_EPOCAS", 40))
if os.environ.get("EEG_ETIQUETA"):
    MODEL_NAME += f"_{os.environ['EEG_ETIQUETA']}"
PATIENCE = 8
BATCH = 128
WEIGHT_DECAY = 1e-2
N_VAL_SUBJECTS = 4             # sujetos de entrenamiento reservados para la parada temprana
F1, D, F2 = 8, 2, 16
KERN_T = 125                   # media segundo a 250 Hz
KERN_SEP = 32
DROPOUT = 0.5

QUICK = os.environ.get("EEG_QUICK") == "1"
if QUICK:
    SEEDS, N_EPOCHS = (0,), 3
    MODEL_NAME += "_quick"     # la prueba rapida nunca sobreescribe los resultados reales

def pick_device():
    """cuda, mps (Apple Silicon) o cpu. Se puede forzar con EEG_DEVICE."""
    forced = os.environ.get("EEG_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()


# ------------------------- Datos -------------------------
def band_limit(X, band):
    """Limita la banda de las ventanas ya cortadas (Butterworth orden 4, fase cero).

    A diferencia del filtro de crear_ventanas.py, que corre sobre la grabacion continua,
    este se aplica a la ventana de 3 s y por eso deja transitorios en los bordes. Es un
    diagnostico, no la tuberia principal, y hay que declararlo asi en el reporte."""
    lo, hi = band
    nyq = FS / 2
    if lo > 0 and hi < nyq:
        sos = butter(4, (lo, hi), btype="bandpass", fs=FS, output="sos")
    elif lo > 0:
        sos = butter(4, lo, btype="highpass", fs=FS, output="sos")
    else:
        sos = butter(4, hi, btype="lowpass", fs=FS, output="sos")
    return np.ascontiguousarray(sosfiltfilt(sos, X, axis=-1), dtype=np.float32)


def zscore_by_session_channel(X, session):
    """Normaliza cada canal con la media y desviacion de toda su sesion."""
    out = np.empty_like(X)
    for s in np.unique(session):
        i = session == s
        mu = X[i].mean(axis=(0, 2), keepdims=True)
        sd = X[i].std(axis=(0, 2), keepdims=True) + 1e-8
        out[i] = (X[i] - mu) / sd
    return out


# ------------------------- Modelo -------------------------
class EEGNet(nn.Module):
    """EEGNet v4 con una salida (logit de la clase lenta)."""

    def __init__(self, n_ch=3, n_t=750):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, KERN_T), padding="same", bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(DROPOUT))
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, KERN_SEP), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(DROPOUT))
        with torch.no_grad():
            n_flat = self._features(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.dense = nn.Linear(n_flat, 1)

    def _features(self, x):
        x = self.block1(x)
        x = self.depthwise(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(1)

    def forward(self, x):                 # x: (B, 1, canales, tiempo)
        return self.dense(self._features(x)).squeeze(1)

    def apply_max_norm(self):
        """Restricciones de norma maxima del articulo original."""
        with torch.no_grad():
            self.depthwise.weight.copy_(torch.renorm(self.depthwise.weight, 2, 0, 1.0))
            self.dense.weight.copy_(torch.renorm(self.dense.weight, 2, 0, 0.25))


# ------------------------- Entrenamiento -------------------------
@torch.no_grad()
def predict(model, Xg, idx):
    model.eval()
    out = []
    for i in range(0, len(idx), 512):
        b = idx[i:i + 512]
        out.append(torch.sigmoid(model(Xg[b].unsqueeze(1))).cpu())
    return torch.cat(out).numpy()


def pick_val_subjects(y, subject, train_mask, seed, fold):
    """Elige sujetos de validacion entre los de entrenamiento que tengan ambas clases."""
    rng = np.random.default_rng(1000 * seed + fold)
    cands = []
    for s in np.unique(subject[train_mask]):
        i = (subject == s) & train_mask
        if min((y[i] == 0).sum(), (y[i] == 1).sum()) >= MIN_PER_CLASS:
            cands.append(s)
    return rng.choice(cands, size=N_VAL_SUBJECTS, replace=False)


def train_one(Xg, yg, y, subject, train_mask, seed, fold):
    """Entrena un EEGNet con los sujetos de train_mask. Devuelve el modelo con la
    mejor AUROC de validacion y el numero de epocas usadas."""
    torch.manual_seed(1000 * seed + fold)
    val_subj = pick_val_subjects(y, subject, train_mask, seed, fold)
    is_val = np.isin(subject, val_subj)
    tr_idx = torch.as_tensor(np.where(train_mask & ~is_val)[0], device=DEVICE)
    va_idx = torch.as_tensor(np.where(train_mask & is_val)[0], device=DEVICE)
    y_va = y[va_idx.cpu().numpy()]

    n_pos = float(yg[tr_idx].sum())
    pos_weight = torch.tensor((len(tr_idx) - n_pos) / n_pos, device=DEVICE)
    model = EEGNet(Xg.shape[1], Xg.shape[2]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state, best_ep, wait = -1.0, None, 0, 0
    for ep in range(1, N_EPOCHS + 1):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=DEVICE)]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xg[b].unsqueeze(1)), yg[b])
            loss.backward()
            opt.step()
            model.apply_max_norm()
        auc = roc_auc_score(y_va, predict(model, Xg, va_idx))
        if auc > best_auc:
            best_auc, best_ep, wait = auc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_ep, best_auc


# ------------------------- Metricas por sujeto -------------------------
def per_subject(y, subject, scores, name):
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
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


# ------------------------- Programa principal -------------------------
def main():
    d = np.load(OUT_DIR / "ventanas.npz")
    keep = np.load(OUT_DIR / "mascara_artefactos.npy")
    folds = pd.read_csv(OUT_DIR / "folds.csv").set_index("subject")
    X = d["X"][keep]
    y = d["y"][keep].astype(int)
    subject = d["subject"][keep]
    session = d["session"][keep]
    X = X[:, [CHANNELS.index(c) for c in USE_CH]]
    if BAND is not None:
        X = band_limit(X, BAND)
    X = zscore_by_session_channel(X, session)
    print(f"Modelo: {MODEL_NAME}   dispositivo: {DEVICE}   ventanas: {len(y)}   "
          f"sujetos: {len(np.unique(subject))}   modo rapido: {QUICK}")
    print(f"Canales: {', '.join(USE_CH)}   banda extra: "
          f"{_band_env or 'ninguna (1-50 Hz de la etapa 2)'}   forma de X: {X.shape}\n")

    Xg = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    yg = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)

    rows, scores = [], {}
    t0 = time.time()
    for seed in SEEDS:
        fold_of = folds[f"fold_seed{seed}"].reindex(subject).to_numpy()
        score = np.full(len(y), np.nan)
        for k in range(1 if QUICK else N_FOLDS):
            te = fold_of == k
            model, best_ep, best_auc = train_one(Xg, yg, y, subject, ~te, seed, k)
            p = predict(model, Xg, torch.as_tensor(np.where(te)[0], device=DEVICE))
            score[te] = p
            ba = balanced_accuracy_score(y[te], (p >= 0.5).astype(int))
            auc = roc_auc_score(y[te], p)
            rows.append(dict(modelo=MODEL_NAME, lr=LR, seed=seed, fold=k, n_test=int(te.sum()),
                             bal_acc=ba, auroc=auc, epoca_mejor=best_ep,
                             auroc_val=best_auc))
            print(f"semilla {seed} fold {k}: bal_acc {ba:.3f}  auroc {auc:.3f}  "
                  f"(mejor epoca {best_ep}, auroc val {best_auc:.3f})  [{time.time() - t0:.0f} s]",
                  flush=True)
        scores[seed] = score

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_DIR / f"{MODEL_NAME}_folds.csv", index=False)
    np.savez_compressed(OUT_DIR / f"{MODEL_NAME}_scores.npz",
                        **{f"seed{s}": sc for s, sc in scores.items()})
    if QUICK:
        print("\nModo rapido: no se calculan resumenes por sujeto.")
        return
    ps = per_subject(y, subject, scores, MODEL_NAME)
    ps.to_csv(OUT_DIR / f"{MODEL_NAME}_por_sujeto.csv", index=False)
    lo, hi = boot_ci(ps.bal_acc)

    print(f"\n===== RESUMEN {MODEL_NAME} (lr {LR:g}, semillas {SEEDS}, validacion por sujeto) =====")
    print(f"bal_acc: {tab.bal_acc.mean():.3f} +- {tab.bal_acc.std():.3f}   "
          f"auroc: {tab.auroc.mean():.3f}")
    print(f"por sujeto: media {ps.bal_acc.mean():.3f}  IC95 [{lo:.3f}, {hi:.3f}]   "
          f"sujetos sobre 0.5: {int((ps.bal_acc > 0.5).sum())}/{len(ps)}")
    print(f"epoca mejor (mediana): {tab.epoca_mejor.median():.0f}   "
          f"tiempo total: {(time.time() - t0) / 60:.1f} min")
    print(f"criterio para elegir la tasa de aprendizaje, AUROC de validacion interna "
          f"(semilla 0, media de folds): {tab[tab.seed == 0].auroc_val.mean():.4f}")
    print(f"\nGuardado en: {OUT_DIR}")


if __name__ == "__main__":
    main()
