"""
Etapa 4b (parte 2): CNN preentrenado (ResNet18) sobre imagenes tiempo-frecuencia.

Mismo protocolo que EEGNet (entrenar_eegnet.py): mismas ventanas conservadas, mismos folds
por sujeto (results/folds.csv), mismas 3 semillas, mismos sujetos de validacion interna
para la parada temprana, misma perdida ponderada por clase y el mismo presupuesto de
epocas y paciencia. Solo cambia la entrada (imagenes de crear_imagenes.py) y el modelo.

Variables de entorno (se fijan en la consola de Spyder con %env NOMBRE=valor):
  EEG_REPR      cwt (por defecto) o stft: que imagenes usar. Es la ablacion de representacion.
  EEG_PRETRAIN  1 (por defecto): pesos de ImageNet y ajuste fino de toda la red.
                0: la misma red entrenada desde cero (ablacion opcional del preentrenamiento).
  EEG_QUICK     1: un solo fold, 3 epocas, y reporta el tiempo por epoca. Los archivos
                salen con el sufijo _quick, para no sobreescribir la corrida completa.
  EEG_DEVICE    forzar el dispositivo (cuda, mps o cpu). Por defecto se elige el mejor.
                En mps se entrena en float32 (sin precision mixta).
  EEG_LR        tasa de aprendizaje (por defecto 1e-4). Si se fija, los archivos de salida
                llevan la etiqueta, por ejemplo resnet18_cwt_lr3e-05_folds.csv.
  EEG_SEEDS     semillas separadas por coma (por defecto 0,1,2). Para el ajuste de la tasa
                de aprendizaje se usa solo 0.
  EEG_COMPARE   nombre del modelo EEGNet contra el que se compara (por defecto eegnet).
  EEG_ETIQUETA  sufijo extra para el nombre de salida, por ejemplo mps. Sirve para repetir
                una corrida en otra maquina sin sobreescribir la original.

Salidas en results/ (NOMBRE = resnet18_cwt, resnet18_stft o resnet18_cwt_scratch ...):
  NOMBRE_folds.csv, NOMBRE_por_sujeto.csv, NOMBRE_scores.npz
Al final compara contra EEGNet (si existen sus resultados) con la regla de decision:
diferencia por sujeto (CNN menos EEGNet) y su IC 95% bootstrap remuestreando sujetos.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_eegnet import boot_ci, per_subject, pick_device, pick_val_subjects  # noqa: E402

# ------------------------- Configuracion -------------------------
DATA_DIR = Path(os.environ.get(
    "EEG_DATA_DIR",
    Path.home() / "Documents" / "TEC" / "Neural systems" / "Proyecto" / "6427334",
))
OUT_DIR = DATA_DIR.parent / "results"

REPR = os.environ.get("EEG_REPR", "cwt")
PRETRAINED = os.environ.get("EEG_PRETRAIN", "1") == "1"
QUICK = os.environ.get("EEG_QUICK") == "1"
_lr_env = os.environ.get("EEG_LR")
LR = float(_lr_env) if _lr_env else 1e-4
COMPARE = os.environ.get("EEG_COMPARE", "eegnet")
MODEL_NAME = (f"resnet18_{REPR}" + ("" if PRETRAINED else "_scratch")
              + (f"_lr{LR:g}" if _lr_env else "")
              + (f"_{os.environ['EEG_ETIQUETA']}" if os.environ.get("EEG_ETIQUETA") else "")
              + ("_quick" if QUICK else ""))   # la prueba rapida no toca los resultados reales

N_FOLDS = 5
SEEDS = tuple(int(x) for x in os.environ.get("EEG_SEEDS", "0,1,2").split(","))
# Presupuesto igual al de EEGNet (entrenar_eegnet.py): mantener estos tres valores sincronizados
N_EPOCHS = 80
PATIENCE = 8
BATCH = 64
WEIGHT_DECAY = 1e-2
if QUICK:
    SEEDS, N_EPOCHS = (0,), 3

DEVICE = pick_device()
USE_AMP = DEVICE.type == "cuda"      # en mps se entrena en float32
# channels_last solo acelera en cuda; en mps rompe el paso hacia atras
MEM_FORMAT = torch.channels_last if DEVICE.type == "cuda" else torch.contiguous_format
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


# ------------------------- Modelo -------------------------
def build_model():
    import torchvision
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m.to(DEVICE).to(memory_format=MEM_FORMAT)


def prep(imgs, b):
    """uint8 (0 a 255) -> float normalizado con las estadisticas de ImageNet."""
    x = (imgs[b].float() / 255.0 - MEAN) / STD
    return x.contiguous(memory_format=MEM_FORMAT)


@torch.no_grad()
def predict(model, imgs, idx):
    model.eval()
    out = []
    for i in range(0, len(idx), 256):
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(prep(imgs, idx[i:i + 256]))
        out.append(torch.sigmoid(logits.float().squeeze(1)).cpu())
    return torch.cat(out).numpy()


def train_one(imgs, yg, y, subject, train_mask, seed, fold):
    torch.manual_seed(1000 * seed + fold)
    val_subj = pick_val_subjects(y, subject, train_mask, seed, fold)   # los mismos que EEGNet
    is_val = np.isin(subject, val_subj)
    tr_idx = torch.as_tensor(np.where(train_mask & ~is_val)[0], device=DEVICE)
    va_idx = torch.as_tensor(np.where(train_mask & is_val)[0], device=DEVICE)
    y_va = y[va_idx.cpu().numpy()]

    n_pos = float(yg[tr_idx].sum())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr_idx) - n_pos) / n_pos, device=DEVICE))
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_auc, best_state, best_ep, wait, t_ep = -1.0, None, 0, 0, []
    for ep in range(1, N_EPOCHS + 1):
        t0 = time.time()
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=DEVICE)]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2:                         # BatchNorm necesita al menos 2 ejemplos
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(prep(imgs, b)).squeeze(1)
            loss = lossf(logits.float(), yg[b])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        auc = roc_auc_score(y_va, predict(model, imgs, va_idx))
        t_ep.append(time.time() - t0)
        if auc > best_auc:
            best_auc, best_ep, wait = auc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_ep, best_auc, float(np.mean(t_ep))


# ------------------------- Comparacion con EEGNet -------------------------
def compare_with_eegnet(tab, ps):
    f, g = OUT_DIR / f"{COMPARE}_folds.csv", OUT_DIR / f"{COMPARE}_por_sujeto.csv"
    if not (f.exists() and g.exists()):
        print(f"\n(No hay resultados de {COMPARE} para comparar.)")
        return
    if len(SEEDS) != 3:
        print("\n(Comparacion omitida: se corrio con una sola semilla, ajuste de tasa de aprendizaje.)")
        return
    m = tab.merge(pd.read_csv(f), on=["seed", "fold"], suffixes=("", "_eeg"))
    d_fold = m.bal_acc - m.bal_acc_eeg
    ms = ps.merge(pd.read_csv(g), on="subject", suffixes=("", "_eeg"))
    d_sub = ms.bal_acc - ms.bal_acc_eeg
    lo, hi = boot_ci(d_sub)
    print(f"\n===== {MODEL_NAME} contra {COMPARE} =====")
    print(f"diferencia de balanced accuracy por fold x semilla: media {d_fold.mean():+.3f}   "
          f"{MODEL_NAME} mejor en {int((d_fold > 0).sum())}/{len(d_fold)}")
    print(f"diferencia por sujeto (n = {len(d_sub)}): media {d_sub.mean():+.3f}   "
          f"IC95 bootstrap [{lo:+.3f}, {hi:+.3f}]   el IC excluye 0: {bool(lo > 0 or hi < 0)}")


# ------------------------- Programa principal -------------------------
def main():
    d = np.load(OUT_DIR / "ventanas.npz")
    keep = np.load(OUT_DIR / "mascara_artefactos.npy")
    folds = pd.read_csv(OUT_DIR / "folds.csv").set_index("subject")
    y = d["y"][keep].astype(int)
    subject = d["subject"][keep]
    arr = np.load(OUT_DIR / f"imagenes_{REPR}.npy")
    if len(arr) != len(y):
        raise ValueError(f"imagenes_{REPR}.npy tiene {len(arr)} imagenes y hay {len(y)} ventanas")
    imgs = torch.as_tensor(arr, device=DEVICE)
    del arr
    yg = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    print(f"Modelo: {MODEL_NAME}   dispositivo: {DEVICE}   imagenes: {tuple(imgs.shape)}   "
          f"preentrenado: {PRETRAINED}   lr: {LR:g}   semillas: {SEEDS}   modo rapido: {QUICK}\n")

    rows, scores = [], {}
    t0 = time.time()
    for seed in SEEDS:
        fold_of = folds[f"fold_seed{seed}"].reindex(subject).to_numpy()
        score = np.full(len(y), np.nan)
        for k in range(1 if QUICK else N_FOLDS):
            te = fold_of == k
            model, best_ep, best_auc, s_ep = train_one(imgs, yg, y, subject, ~te, seed, k)
            p = predict(model, imgs, torch.as_tensor(np.where(te)[0], device=DEVICE))
            score[te] = p
            ba = balanced_accuracy_score(y[te], (p >= 0.5).astype(int))
            auc = roc_auc_score(y[te], p)
            rows.append(dict(modelo=MODEL_NAME, lr=LR, seed=seed, fold=k, n_test=int(te.sum()),
                             bal_acc=ba, auroc=auc, epoca_mejor=best_ep, auroc_val=best_auc,
                             seg_por_epoca=s_ep))
            print(f"semilla {seed} fold {k}: bal_acc {ba:.3f}  auroc {auc:.3f}  "
                  f"(mejor epoca {best_ep}, auroc val {best_auc:.3f}, {s_ep:.1f} s/epoca)  "
                  f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
            del model
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
    print(f"bal_acc: {tab.bal_acc.mean():.3f} +- {tab.bal_acc.std():.3f}   auroc: {tab.auroc.mean():.3f}")
    print(f"por sujeto: media {ps.bal_acc.mean():.3f}  IC95 [{lo:.3f}, {hi:.3f}]   "
          f"sujetos sobre 0.5: {int((ps.bal_acc > 0.5).sum())}/{len(ps)}")
    print(f"epoca mejor (mediana): {tab.epoca_mejor.median():.0f}   "
          f"tiempo total: {(time.time() - t0) / 60:.1f} min")
    print(f"criterio para elegir la tasa de aprendizaje, AUROC de validacion interna "
          f"(semilla 0, media de folds): {tab[tab.seed == 0].auroc_val.mean():.4f}")
    compare_with_eegnet(tab, ps)
    print(f"\nGuardado en: {OUT_DIR}")


if __name__ == "__main__":
    main()
