"""
Etapa 5 (cualitativo): mapas de saliencia Grad-CAM del CNN sobre los escalogramas.

Entrena un solo ResNet18 con el protocolo de siempre (semilla 0, fold 0, mismos sujetos
de validacion interna) y calcula Grad-CAM sobre los sujetos de prueba de ese fold, que
la red no vio. El modelo se guarda en results/, asi que la figura se puede volver a
hacer sin reentrenar.

Grad-CAM se toma en layer4 (resolucion 4 x 4, interpolada a 128 x 128). Los 3 canales de
EEG entran como canales de color y se mezclan en la primera convolucion, asi que el mapa
es uno solo sobre el plano tiempo-frecuencia, no uno por canal. Se promedia sobre las
ventanas que la red clasifico bien, por clase verdadera.

Lo que esto NO establece: que las regiones marcadas sean causales. Es interpretacion,
no evidencia, y el reporte lo dice asi. Para la comparacion cuantitativa estan
analisis_final.py y diagnostico_eegnet.py.

Variables de entorno:
  EEG_FOLD    fold de prueba (por defecto 0).
  EEG_SEMILLA semilla (por defecto 0).
  EEG_FORZAR  1: reentrena aunque ya exista el modelo guardado.

Salidas:
  results/resnet18_cwt_fold{F}_seed{S}.pt   pesos del modelo usado para la figura
  figures/saliencia_cwt.png                 Grad-CAM medio por clase y escalograma medio
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entrenar_cnn as cnn  # noqa: E402
from crear_imagenes import CLIP, FMAX, FMIN, IMG  # noqa: E402

OUT_DIR = cnn.OUT_DIR
FIG_DIR = cnn.OUT_DIR.parent / "figures"
FOLD = int(os.environ.get("EEG_FOLD", 0))
SEMILLA = int(os.environ.get("EEG_SEMILLA", 0))
FORZAR = os.environ.get("EEG_FORZAR") == "1"
CKPT = OUT_DIR / f"resnet18_cwt_fold{FOLD}_seed{SEMILLA}.pt"
FREQ_TICKS = (1, 4, 8, 13, 30, 40)


def freq_rows():
    """Fila de la imagen que corresponde a cada frecuencia (eje logaritmico 1 a 40 Hz)."""
    freqs = np.logspace(np.log10(FMIN), np.log10(FMAX), IMG)
    return [(hz, int(np.argmin(np.abs(freqs - hz)))) for hz in FREQ_TICKS]


def grad_cam(model, imgs, idx, clase, batch=64):
    """Grad-CAM medio en layer4 para la clase pedida (1 = lenta, 0 = rapida).

    La red tiene una sola salida, asi que el logit es la evidencia de la clase lenta:
    para la clase rapida se usa su negativo."""
    acts, grads = {}, {}
    h1 = model.layer4.register_forward_hook(
        lambda m, i, o: acts.__setitem__("a", o))
    h2 = model.layer4.register_full_backward_hook(
        lambda m, gi, go: grads.__setitem__("g", go[0]))
    signo = 1.0 if clase == 1 else -1.0
    model.eval()
    acc = np.zeros((IMG, IMG), dtype=np.float64)
    n = 0
    try:
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            model.zero_grad(set_to_none=True)
            logits = model(cnn.prep(imgs, b)).squeeze(1)
            (signo * logits).sum().backward()
            a, g = acts["a"], grads["g"]
            w = g.mean(dim=(2, 3), keepdim=True)              # peso por canal
            cam = F.relu((w * a).sum(dim=1, keepdim=True))     # (B, 1, 4, 4)
            cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)
            acc += cam[:, 0].detach().float().cpu().numpy().sum(axis=0)
            n += len(b)
    finally:
        h1.remove()
        h2.remove()
    return acc / max(n, 1)


def mean_image(imgs, idx, batch=512):
    """Escalograma medio (promediado sobre los 3 canales) en unidades z."""
    acc = np.zeros((IMG, IMG), dtype=np.float64)
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        acc += imgs[b].float().mean(dim=1).sum(dim=0).cpu().numpy()
    return acc / len(idx) / 255 * 2 * CLIP - CLIP


def figura(cams, medias, info):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   (matplotlib no esta instalado; se omite la figura)")
        return
    rows = freq_rows()
    cam_lim = max(np.abs(c).max() for c in cams.values())
    med_lim = max(np.abs(m).max() for m in medias.values())
    paneles = [
        ("Grad-CAM, clase lenta", cams[1], "magma", 0, cam_lim),
        ("Grad-CAM, clase rapida", cams[0], "magma", 0, cam_lim),
        ("Grad-CAM, lenta menos rapida", cams[1] - cams[0], "RdBu_r", -cam_lim / 2, cam_lim / 2),
        ("Escalograma medio, lenta", medias[1], "RdBu_r", -med_lim, med_lim),
        ("Escalograma medio, rapida", medias[0], "RdBu_r", -med_lim, med_lim),
        ("Escalograma medio, lenta menos rapida", medias[1] - medias[0], "RdBu_r",
         -np.abs(medias[1] - medias[0]).max(), np.abs(medias[1] - medias[0]).max()),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey=True)
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes.ravel(), paneles):
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, extent=[-3, 0, 0, IMG - 1])
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_yticks([r for _, r in rows])
        ax.set_yticklabels([f"{hz:g}" for hz, _ in rows], fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("tiempo respecto al inicio de la desviacion (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("frecuencia (Hz)")
    fig.suptitle(f"Saliencia del CNN sobre escalogramas. {info}\n"
                 "Las regiones marcadas son interpretacion, no evidencia de causalidad.",
                 fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "saliencia_cwt.png", dpi=130)
    plt.close(fig)


def main():
    d = np.load(OUT_DIR / "ventanas.npz")
    keep = np.load(OUT_DIR / "mascara_artefactos.npy")
    y = d["y"][keep].astype(int)
    subject = d["subject"][keep]
    folds = pd.read_csv(OUT_DIR / "folds.csv").set_index("subject")
    fold_of = folds[f"fold_seed{SEMILLA}"].reindex(subject).to_numpy()
    te = fold_of == FOLD

    imgs = torch.as_tensor(np.load(OUT_DIR / "imagenes_cwt.npy"), device=cnn.DEVICE)
    yg = torch.as_tensor(y, dtype=torch.float32, device=cnn.DEVICE)
    print(f"Dispositivo: {cnn.DEVICE}   fold de prueba {FOLD}, semilla {SEMILLA}   "
          f"ventanas de prueba: {int(te.sum())}   sujetos de prueba: "
          f"{sorted(np.unique(subject[te]).tolist())}\n")

    model = cnn.build_model()
    if CKPT.exists() and not FORZAR:
        model.load_state_dict(torch.load(CKPT, map_location=cnn.DEVICE))
        print(f"Modelo cargado de {CKPT.name} (EEG_FORZAR=1 para reentrenar).")
        info_entren = "modelo guardado"
    else:
        t0 = time.time()
        model, best_ep, best_auc, s_ep = cnn.train_one(imgs, yg, y, subject, ~te,
                                                       SEMILLA, FOLD)
        torch.save(model.state_dict(), CKPT)
        print(f"Entrenado en {(time.time() - t0) / 60:.1f} min (mejor epoca {best_ep}, "
              f"auroc val {best_auc:.3f}). Guardado en {CKPT.name}.")
        info_entren = f"mejor epoca {best_ep}"

    p = cnn.predict(model, imgs, torch.as_tensor(np.where(te)[0], device=cnn.DEVICE))
    pred = (p >= 0.5).astype(int)
    yt = y[te]
    from sklearn.metrics import balanced_accuracy_score
    ba = balanced_accuracy_score(yt, pred)
    print(f"balanced accuracy del fold: {ba:.3f}")

    te_idx = np.where(te)[0]
    cams, medias, conteo = {}, {}, {}
    for clase in (0, 1):
        ok = te_idx[(yt == clase) & (pred == clase)]      # solo las bien clasificadas
        conteo[clase] = len(ok)
        gi = torch.as_tensor(ok, device=cnn.DEVICE)
        cams[clase] = grad_cam(model, imgs, gi, clase)
        medias[clase] = mean_image(imgs, gi)
        print(f"  clase {'lenta' if clase else 'rapida'}: {len(ok)} ventanas bien "
              f"clasificadas de {int((yt == clase).sum())}")

    info = (f"fold {FOLD}, semilla {SEMILLA}, bal acc {ba:.3f}, {info_entren}. "
            f"Promedio sobre {conteo[1]} lentas y {conteo[0]} rapidas bien clasificadas.")
    figura(cams, medias, info)
    np.savez_compressed(OUT_DIR / "saliencia_cwt.npz",
                        cam_lenta=cams[1], cam_rapida=cams[0],
                        media_lenta=medias[1], media_rapida=medias[0])
    print(f"\nFigura en: {FIG_DIR / 'saliencia_cwt.png'}\n"
          f"Mapas en: {OUT_DIR / 'saliencia_cwt.npz'}")


if __name__ == "__main__":
    main()
