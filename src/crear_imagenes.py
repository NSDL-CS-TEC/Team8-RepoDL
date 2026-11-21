"""
Etapa 4b (parte 1): imagenes tiempo-frecuencia a partir de las ventanas de EEG.

Para cada ventana conservada (results/ventanas.npz + mascara_artefactos.npy) genera dos
representaciones, cada una de 3 canales (Fz, Pz, Oz) x 128 x 128, que se apilan como los
tres canales de color de una imagen:

  cwt  : escalograma. Transformada wavelet continua con wavelet de Morlet (6 ciclos),
         128 frecuencias espaciadas en escala logaritmica de 1 a 40 Hz.
  stft : espectrograma. Transformada de Fourier de tiempo corto (ventana Hann de 1 s,
         salto de 24 ms), 1 a 40 Hz en escala lineal, llevado a 128 x 128.

Ambas son el logaritmo de la potencia. Luego se normaliza por sesion, canal y frecuencia
(media y desviacion de las ventanas de esa sesion, sin usar etiquetas), se recorta a
+-3 desviaciones y se guarda como uint8 (0 a 255). El eje horizontal es el tiempo
(los 3 s previos a la desviacion) y el vertical es la frecuencia.

La normalizacion por sesion usa datos sin etiquetas de la sesion evaluada, igual que en
EEGNet y en el baseline potencia_bandas_zsesion, y hay que declararla en el reporte.
Para evitar efectos de borde, la senal se extiende por reflexion antes de la wavelet.

Salidas:
  results/imagenes_cwt.npy, results/imagenes_stft.npy   (N, 3, 128, 128) uint8, alineadas
        con las ventanas conservadas (ventanas.npz filtrado con la mascara)
  figures/medias_cwt.png, figures/medias_stft.png      mapas medios por clase y su diferencia

Modo de prueba rapida: EEG_QUICK=1 usa solo 256 ventanas y no escribe los archivos finales.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_eegnet import pick_device  # noqa: E402

# ------------------------- Configuracion -------------------------
DATA_DIR = Path(os.environ.get(
    "EEG_DATA_DIR",
    Path.home() / "Documents" / "TEC" / "Neural systems" / "Proyecto" / "6427334",
))
OUT_DIR = DATA_DIR.parent / "results"
FIG_DIR = DATA_DIR.parent / "figures"

FS = 250
N_T = 750
IMG = 128
FMIN, FMAX = 1.0, 40.0
N_CYCLES = 6
PAD = N_T // 2                       # extension por reflexion a cada lado
STFT_NFFT, STFT_WIN, STFT_HOP = 1024, 250, 6
BATCH = 64
CLIP = 3.0
CHANNELS = ("Fz", "Pz", "Oz")

QUICK = os.environ.get("EEG_QUICK") == "1"
DEVICE = pick_device()


# ------------------------- Transformadas -------------------------
def morlet_bank():
    """Banco de wavelets de Morlet en el dominio de la frecuencia. Devuelve (freqs, H)."""
    n_fft = N_T + 2 * PAD
    freqs = torch.logspace(np.log10(FMIN), np.log10(FMAX), IMG, device=DEVICE)
    nu = torch.fft.fftfreq(n_fft, d=1.0 / FS).to(DEVICE)
    sigma = (freqs / N_CYCLES)[:, None]
    H = torch.exp(-0.5 * ((nu[None, :] - freqs[:, None]) / sigma) ** 2)
    H = H * (nu[None, :] > 0)        # senal analitica: solo frecuencias positivas
    return freqs, H


@torch.no_grad()
def cwt_log_power(x, H):
    """x: (B, 3, 750) -> (B, 3, 128, 128) log10 de la potencia."""
    xp = F.pad(x, (PAD, PAD), mode="reflect")
    X = torch.fft.fft(xp, dim=-1)
    Y = torch.fft.ifft(X[:, :, None, :] * H[None, None], dim=-1)
    P = (Y.real ** 2 + Y.imag ** 2)[..., PAD:PAD + N_T]
    P = F.adaptive_avg_pool2d(P, (IMG, IMG))
    return torch.log10(P + 1e-8)


@torch.no_grad()
def stft_log_power(x, window):
    B, C, L = x.shape
    S = torch.stft(x.reshape(B * C, L), n_fft=STFT_NFFT, hop_length=STFT_HOP,
                   win_length=STFT_WIN, window=window, center=True,
                   pad_mode="reflect", return_complex=True)
    P = S.real ** 2 + S.imag ** 2
    df = FS / STFT_NFFT
    lo, hi = int(np.ceil(FMIN / df)), int(np.floor(FMAX / df))
    P = P[:, lo:hi + 1, :]
    P = F.interpolate(P[:, None], size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0]
    return torch.log10(P + 1e-8).reshape(B, C, IMG, IMG)


# ------------------------- Construccion y normalizacion -------------------------
def build(fn, X, session):
    """Aplica la transformada por lotes y normaliza por sesion, canal y frecuencia."""
    L = torch.empty((len(X), 3, IMG, IMG), dtype=torch.float16, device=DEVICE)
    for i in range(0, len(X), BATCH):
        xb = torch.as_tensor(X[i:i + BATCH], dtype=torch.float32, device=DEVICE)
        L[i:i + BATCH] = fn(xb).half()
    out = np.empty(tuple(L.shape), dtype=np.uint8)
    for s in np.unique(session):
        idx = torch.as_tensor(np.where(session == s)[0], device=DEVICE)
        sub = L[idx].float()
        m = sub.mean(dim=(0, 3), keepdim=True)
        sd = sub.std(dim=(0, 3), keepdim=True) + 1e-6
        z = ((sub - m) / sd).clamp(-CLIP, CLIP)
        out[idx.cpu().numpy()] = ((z + CLIP) / (2 * CLIP) * 255).round().byte().cpu().numpy()
    return out


# ------------------------- Figuras -------------------------
def class_means(imgs, y):
    means = {}
    for c in (0, 1):
        idx = np.where(y == c)[0]
        acc = np.zeros(imgs.shape[1:], dtype=np.float64)
        for i in range(0, len(idx), 1000):
            acc += imgs[idx[i:i + 1000]].astype(np.float64).sum(axis=0)
        means[c] = acc / len(idx) / 255 * 2 * CLIP - CLIP        # de vuelta a unidades z
    return means


def save_figure(imgs, y, kind, freq_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   (matplotlib no esta instalado; se omite la figura)")
        return
    means = class_means(imgs, y)
    diff = means[1] - means[0]
    lim = max(np.abs(means[0]).max(), np.abs(means[1]).max())
    dlim = np.abs(diff).max()
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    for r, ch in enumerate(CHANNELS):
        for c, (title, arr, v) in enumerate([("rapida (media)", means[0][r], lim),
                                              ("lenta (media)", means[1][r], lim),
                                              ("lenta - rapida", diff[r], dlim)]):
            ax = axes[r, c]
            im = ax.imshow(arr, origin="lower", aspect="auto", cmap="RdBu_r",
                           vmin=-v, vmax=v, extent=[-3, 0, 0, IMG - 1])
            ax.set_title(f"{ch}, {title}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_yticks([row for _, row in freq_rows])
            ax.set_yticklabels([f"{hz:g}" for hz, _ in freq_rows], fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("tiempo respecto al inicio de la desviacion (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("frecuencia (Hz)")
    fig.suptitle(f"Mapas medios (unidades z) por clase, {kind}")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"medias_{kind}.png", dpi=130)
    plt.close(fig)


# ------------------------- Programa principal -------------------------
def main():
    d = np.load(OUT_DIR / "ventanas.npz")
    keep = np.load(OUT_DIR / "mascara_artefactos.npy")
    X, y, session = d["X"][keep], d["y"][keep].astype(int), d["session"][keep]
    if QUICK:
        X, y, session = X[:256], y[:256], session[:256]
    print(f"Dispositivo: {DEVICE}   ventanas: {len(X)}   modo rapido: {QUICK}\n")

    freqs, H = morlet_bank()
    window = torch.hann_window(STFT_WIN, device=DEVICE)
    cwt_rows = [(hz, int(torch.argmin(torch.abs(freqs - hz)))) for hz in (1, 4, 8, 13, 30, 40)]
    stft_rows = [(hz, int(round((hz - FMIN) / (FMAX - FMIN) * (IMG - 1)))) for hz in (1, 4, 8, 13, 30, 40)]

    jobs = [("cwt", lambda xb: cwt_log_power(xb, H), cwt_rows),
            ("stft", lambda xb: stft_log_power(xb, window), stft_rows)]
    for kind, fn, rows in jobs:
        t0 = time.time()
        imgs = build(fn, X, session)
        print(f"{kind}: forma {imgs.shape}, media de pixeles {imgs.mean():.1f}, "
              f"min {imgs.min()}, max {imgs.max()}   ({time.time() - t0:.0f} s)")
        if QUICK:
            continue
        np.save(OUT_DIR / f"imagenes_{kind}.npy", imgs)
        save_figure(imgs, y, kind, rows)
        del imgs
    if not QUICK:
        print(f"\nImagenes en: {OUT_DIR}\nFiguras en: {FIG_DIR}")


if __name__ == "__main__":
    main()
