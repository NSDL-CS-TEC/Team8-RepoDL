<<<<<<< HEAD
# Decoding a subject-relative state from pre-event EEG: time-frequency images vs. raw signal

Course final project by Juan Pablo and Alberto. From the 3 s of EEG (Fz, Pz, Oz) before a car drifts out of its lane, we predict whether the driver's response will be **fast or slow**, always testing on **unseen subjects**. The main question is whether a ResNet18 pretrained on ImageNet, applied to wavelet scalograms, generalizes better than EEGNet on the raw signal.

**Hypothesis.** The ResNet18 on CWT scalograms reaches a higher balanced accuracy on unseen subjects than EEGNet on the raw signal, with the same compute and tuning budget.

**Decision rule.** Compute Δ = balanced accuracy of the CNN minus that of EEGNet, per subject and averaged over seeds. The hypothesis holds if the mean Δ is positive and its 95% bootstrap interval (resampling subjects) excludes 0.

**Current result: the hypothesis is not supported.** EEGNet is equal to or better than the ResNet18 (see `results/comparacion_pareada.csv`).

The full protocol, decisions and status are in `report/Final project protocol ... .docx`.

## Repository layout

```
.
├── README.md
├── config.yaml          every run setting, plus the fixed protocol for reference
├── train.py             data preparation and training (wraps src/)
├── evaluate.py          summary table, lr selection, permutation test, paired bootstrap
├── requirements.txt
├── src/                 one script per pipeline stage (see below)
├── results/             CSV/NPZ outputs, logs/, respaldo_cuda_jp/ (JP's CUDA runs)
├── figures/             PNG figures
├── report/              report, protocol document, slides
└── 6427334/             dataset (.set files), not versioned
```

| Stage | Script in `src/` | Main outputs |
|---|---|---|
| 1 Inventory | `inventario_sesiones.py` | `inventario_sesiones.csv`, `ensayos.csv` |
| 2 Windows and labels | `crear_ventanas.py` | `ventanas.npz`, `ventanas_meta.csv` |
| 3 Folds and classical baselines | `folds_y_baselines.py` | `mascara_artefactos.npy`, `folds.csv`, `baselines_*.csv` |
| 4 Images (CWT, STFT) | `crear_imagenes.py` | `imagenes_cwt.npy`, `imagenes_stft.npy`, `figures/medias_*.png` |
| 4a EEGNet | `entrenar_eegnet.py` | `eegnet*_folds.csv`, `*_por_sujeto.csv`, `*_scores.npz` |
| 4b ResNet18 | `entrenar_cnn.py` | `resnet18_*_folds.csv`, `*_por_sujeto.csv`, `*_scores.npz` |
| 5 EEGNet diagnostic | `diagnostico_eegnet.py` | `diagnostico_eegnet.csv`, `figures/diagnostico_eegnet.png` |
| 5 Grad-CAM | `saliencia_cnn.py` | `saliencia_cwt.npz`, `figures/saliencia_cwt.png` |
| 5 lr selection | `elegir_tasa_aprendizaje.py` | printed table |
| 5 Statistics | `analisis_final.py` | `permutacion.csv`, `comparacion_pareada.csv`, `analisis_por_sujeto.csv` |

## Setup

```bash
conda create -n dl python=3.13 && conda activate dl
pip install -r requirements.txt     # on CUDA, install torch/torchvision for your CUDA version first
```

Download the dataset from figshare (DOI [10.6084/m9.figshare.6427334.v2](https://doi.org/10.6084/m9.figshare.6427334.v2)) and unzip it into `6427334/` at the project root. If it lives elsewhere, set `data_dir` in `config.yaml`. `results/` and `figures/` are always written next to the data folder.

## Reproduction

```bash
python train.py --list                # experiments defined in config.yaml
python train.py prepare               # stages 1-4: inventory, windows, folds + baselines, images
python train.py eegnet resnet18_cwt   # main comparison (15 runs each: 5 folds x 3 seeds)
python train.py all                   # main comparison + lr check + ablation + diagnostics
python evaluate.py                    # summary table, lr selection, permutation + bootstrap
```

- `python train.py <experiment> --quick` runs one fold for 3 epochs. Its outputs end in `_quick`, so the full results are not overwritten.
- `--device cuda|mps|cpu` overrides `device` in `config.yaml`.
- Running `evaluate.py` on the current `results/` reproduces `permutacion.csv`, `comparacion_pareada.csv` and `analisis_por_sujeto.csv` exactly. All resampling uses fixed seeds.
- The scripts in `src/` can also be run directly. They then need `EEG_DATA_DIR` set; the other `EEG_*` variables are documented at the top of each script.

## Experimental details

**Data.** Cao, Chuang, King and Lin (2019), *Multi-channel EEG recordings during a sustained-attention driving task*, Scientific Data 6:19. Figshare v2, CC BY 4.0. The dataset has 27 subjects and 62 sessions of 32-channel EEG at 500 Hz. We drop session `s04_051130m`, a duplicate of `s05_051120m`, which leaves 26 subjects and 61 sessions. In session names, `n` means motionless and `m` means motion (dataset tutorial).

**Preprocessing.**
- Band-pass 1–50 Hz (zero-phase Butterworth, order 4) on the continuous recording, then decimation to 250 Hz.
- A 3 s window ending at the lane-deviation onset (markers 251/252), so the input is 3 × 750 samples.
- Each window is normalized per session and channel with that session's statistics. No labels are used, but this is an unlabeled calibration on the test subject, and the report must declare it.
- No data augmentation.

**Labels.** The reaction time (RT) runs from deviation onset to response onset (253). The reference is the 5th percentile of the session's RTs. RT < 1.5 × reference is *fast* and RT > 2.5 × reference is *slow*. Intermediate trials are discarded.

**Exclusions.**
- RT < 0.1 s or > 60 s.
- Windows that overlap the previous response.
- Windows with a peak amplitude > 150 µV.

The final set is **18,087 windows**: 13,094 fast and 4,993 slow (27.6% slow).

**Split.** 5-fold cross-validation by subject: all sessions of a subject fall in the same fold, and the folds are stored in `results/folds.csv`. Every model uses the same folds with seeds 0, 1 and 2. For early stopping, 4 training subjects are held out as an internal validation set. The test fold never takes part in training or stopping.

**Images.** One 128 × 128 image per channel, from 1 to 40 Hz. The CWT is a Morlet wavelet with 6 cycles on log-spaced frequencies; the STFT uses linear frequencies. The 3 channels become the 3 color channels.

| Model | Input | Parameters | Training |
|---|---|---|---|
| EEGNet v4 (F1=8, D=2, F2=16, kernel 125), from scratch | 1 × 3 × 750 | 2,265 | AdamW lr 1e-3, wd 0.01, batch 128, ≤40 epochs, patience 8, max-norm constraints |
| ResNet18, ImageNet weights, all layers fine-tuned, 1-logit head | 3 × 128 × 128 | 11,177,025 | AdamW lr 1e-4, wd 0.01, batch 64, ≤80 epochs, patience 8, AMP on CUDA only |
| Logistic regression on log band power (θ, α, β × 3 channels) | 9 features | 10 | scikit-learn, class-balanced |

- **Objective.** Binary cross-entropy on the logit of the *slow* class, weighted by the class ratio of the training set.
- **Learning rates.** We compared 3 values per model on seed 0 using internal-validation AUROC (`python evaluate.py lr`). For EEGNet, 3e-3 scored marginally higher than the default 1e-3, but the reported EEGNet runs use 1e-3.
- **Metrics.**
  - The primary metric is balanced accuracy on unseen subjects. AUROC is a secondary metric.
  - For the hypothesis test, balanced accuracy is computed per subject, only for subjects with at least 30 windows of each class (24 of 26 subjects). The 95% bootstrap interval uses 2,000 subject resamples.
  - The permutation test permutes labels within each session while the stored predictions stay fixed. It does not retrain the models, and the report states this.
- **Controls and ablations.**
  - ResNet18 without pretraining (`resnet18_cwt_scratch`).
  - STFT instead of CWT (`resnet18_stft`, pending).
  - EEGNet by channel and by frequency band (`diagnostico_eegnet`).
  - Classical baselines, a trial-index baseline, and a shuffled-label control (`baselines_*.csv`).

**Hardware.**
- JP: NVIDIA RTX 3080 (CUDA, mixed precision). About 11 s per ResNet18 epoch.
- Alberto: Apple M1 Pro (MPS, float32). About 23 s per ResNet18 epoch.

`results/respaldo_cuda_jp/` keeps JP's original CUDA outputs. The runs in `results/logs/` (EEGNet, its diagnostics, `resnet18_cwt_mps` and `resnet18_cwt_scratch`) ran on MPS. CUDA and MPS runs are not bitwise identical, which is why `resnet18_cwt` (CUDA) and `resnet18_cwt_mps` are reported separately.

## Storage policy

The dataset (`6427334/`, about 18 GB, plus its zip) is not versioned; download it from figshare. The following are also not versioned, since `train.py` regenerates them:
- `results/ventanas.npz` (146 MB)
- `results/imagenes_*.npy` (848 MB each)
- `results/*.pt` checkpoints (43 MB, only the Grad-CAM model is kept)

Everything else in `results/` (CSV, `*_scores.npz` test probabilities, logs) is small, and `evaluate.py` needs it, so it is kept.
=======
# Team 8 - Deep Learning
>>>>>>> origin/main
