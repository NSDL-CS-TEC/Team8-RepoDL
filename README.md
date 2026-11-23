# Decoding a subject-relative state from pre-event EEG: time-frequency images vs. raw signal

*Team 8 – Deep Learning (final project)*

Course final project by Juan Pablo Acevedo Robles and Alberto Martinez Hernandez. From the 3 s of EEG (Fz, Pz, Oz) before a car drifts out of its lane, we predict whether the driver's response will be **fast or slow**, always testing on **unseen subjects**. The main question is whether a ResNet18 pretrained on ImageNet, applied to wavelet scalograms, generalizes better than EEGNet on the raw signal.

**Hypothesis.** The ResNet18 on CWT scalograms reaches a higher balanced accuracy on unseen subjects than EEGNet on the raw signal, with the same compute and tuning budget.

**Decision rule.** Compute Δ = balanced accuracy of the CNN minus that of EEGNet, per subject and averaged over seeds. The hypothesis holds if the mean Δ is positive and its 95% bootstrap interval (resampling subjects) excludes 0.

**Result: the hypothesis is not supported.** On 24 subjects, EEGNet reaches 0.774 per-subject balanced accuracy and the ResNet18 0.750: Δ = −0.024, 95% CI [−0.037, −0.009]. EEGNet is better in 19 of 24 subjects (`results/comparacion_pareada.csv`, `results/tabla_principal.csv`).

Sensitivity to the CNN learning rate. Juan Pablo's CUDA runs extend the learning-rate grid of the ResNet18 to 1e-5, 3e-5, 1e-4, 3e-4 and 3e-3. On the same validation rule, 3e-4 scores highest (AUROC 0.883 against 0.868 for 1e-4). At 3e-4 the per-subject difference is +0.000, 95% CI [−0.015, +0.017]. The hypothesis is not supported in either case: EEGNet is slightly better at the default CNN learning rate, and the gap closes at the best rate on validation.

**Main runs.** The ResNet18 is `resnet18_cwt_mps` and EEGNet is `eegnet_lr0.003_final`. Both ran on the same hardware and precision (Apple MPS, float32), as did every ablation, so each paired comparison changes one factor only. Juan Pablo's CUDA run (`resnet18_cwt`) and the initial EEGNet (`eegnet`, lr 1e-3, 40 epochs) are reported as replications and lead to the same conclusion.

The report, slides and one-page summary (PDF and LaTeX source) are in `report/` (see `report/README.md`). The original protocol document (16 Nov) is archived in `report/archive/`; its numbers predate the final runs.

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
└── data/                dataset (.set files), not versioned
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
| 5 Statistics | `analisis_final.py` | `permutacion.csv`, `comparacion_pareada.csv`, `analisis_por_sujeto.csv`, `por_condicion.csv` |
| Report figures | `figuras_reporte.py`, `figura_pipeline.py` | `figures/fig_*.{png,pdf}`, `figures/pipeline.*`, `tabla_principal.csv` |

## Setup

```bash
conda create -n dl python=3.13 && conda activate dl
pip install -r requirements.txt     # on CUDA, install torch/torchvision for your CUDA version first
```

Download the dataset from figshare (DOI [10.6084/m9.figshare.6427334.v2](https://doi.org/10.6084/m9.figshare.6427334.v2)) and unzip it into `data/` at the project root. If it lives elsewhere, set `data_dir` in `config.yaml`. `results/` and `figures/` are always written next to the data folder.

## Reproduction

```bash
python train.py --list                # experiments defined in config.yaml
python train.py prepare               # stages 1-4: inventory, windows, folds + baselines, images
python train.py eegnet_final resnet18_cwt_mps   # main comparison (15 runs each: 5 folds x 3 seeds)
python train.py all                   # main comparison + lr check + ablations + diagnostics
python evaluate.py                    # summary table, lr selection, permutation + bootstrap
python src/figuras_reporte.py         # English report figures and the main table
python src/figura_pipeline.py         # pipeline diagram
```

- `python train.py <experiment> --quick` runs one fold for 3 epochs. Its outputs end in `_quick`, so the full results are not overwritten.
- `--device cuda|mps|cpu` overrides `device` in `config.yaml`.
- Running `evaluate.py` on the current `results/` reproduces `permutacion.csv`, `comparacion_pareada.csv` and `analisis_por_sujeto.csv` exactly. All resampling uses fixed seeds.
- The scripts in `src/` can also be run directly. They then need `EEG_DATA_DIR` set; the other `EEG_*` variables are documented at the top of each script.

## Experimental details

**Data.** Cao, Chuang, King and Lin (2019), *Multi-channel EEG recordings during a sustained-attention driving task*, Scientific Data 6:19. Figshare v2, CC BY 4.0. The dataset has 27 subjects and 62 sessions of 32-channel EEG at 500 Hz. We drop session `s04_051130m` as a duplicate of `s05_051120m`, which leaves 26 subjects and 61 sessions. The files differ byte-wise (different SHA-256), but all 32 EEG channels and all 1,083 events are sample-identical. Only the vehicle-position channel differs. In session names, `n` means motionless and `m` means motion (dataset tutorial).

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
| EEGNet v4 (F1=8, D=2, F2=16, kernel 125), from scratch | 1 × 3 × 750 | 2,265 | AdamW lr 3e-3, wd 0.01, batch 128, ≤80 epochs, patience 8, max-norm constraints |
| ResNet18, ImageNet weights, all layers fine-tuned, 1-logit head | 3 × 128 × 128 | 11,177,025 | AdamW lr 1e-4, wd 0.01, batch 64, ≤80 epochs, patience 8, AMP on CUDA only |
| Logistic regression on log band power (θ, α, β × 3 channels) | 9 features | 10 | scikit-learn, class-balanced |

- **Objective.** Binary cross-entropy on the logit of the *slow* class, weighted by the class ratio of the training set.
- **Learning rates.** We compared 3 values per model on seed 0 using internal-validation AUROC (`python evaluate.py lr`). The selections were EEGNet 3e-3 (AUROC 0.883) and, within the three rates run on MPS, ResNet18 1e-4 (0.868). The extended CUDA grid selects 3e-4 for the ResNet18 (0.883); those runs are reported as a sensitivity analysis (results/respaldo_cuda_jp/). The final EEGNet (`eegnet_final`) uses 3e-3 and the same 80-epoch cap as ResNet18. The earlier `eegnet` run (1e-3, 40 epochs) is kept, because the channel/band diagnostics were run with that configuration and are compared against it.
- **No learning-rate schedule.** The rate is constant and there is no augmentation; the decision threshold is 0.5.
- **Metrics.**
  - The primary metric is balanced accuracy on unseen subjects. AUROC is a secondary metric.
  - For the hypothesis test, balanced accuracy is computed per subject, only for subjects with at least 30 windows of each class (24 of 26 subjects). The 95% bootstrap interval uses 2,000 subject resamples.
  - The permutation test permutes labels within each session while the stored predictions stay fixed. It does not retrain the models, and the report states this.
- **Controls and ablations.**
  - ResNet18 without pretraining (`resnet18_cwt_scratch`).
  - STFT instead of CWT (`resnet18_stft_mps`). CWT minus STFT is −0.001, CI [−0.010, +0.007], so the transform makes no measurable difference. Pretrained minus scratch is +0.015, CI [+0.004, +0.025], so pretraining helps.
  - EEGNet by channel and by frequency band (`diagnostico_eegnet`).
  - Classical baselines, a trial-index baseline, and a shuffled-label control (`baselines_*.csv`).

**Hardware.**
- Juan Pablo: NVIDIA RTX 3080 (CUDA, mixed precision). About 11 s per ResNet18 epoch.
- Alberto: Apple M1 Pro (MPS, float32). About 23 s per ResNet18 epoch; the final EEGNet takes 12 min for 15 trainings.

`results/respaldo_cuda_jp/` keeps Juan Pablo's original CUDA outputs, including the lr-check runs. The runs in `results/logs/` ran on MPS: EEGNet, its diagnostics, `resnet18_cwt_mps`, `resnet18_cwt_scratch`, `eegnet_lr0.003_final` and `resnet18_stft_mps`. CUDA and MPS runs are not bitwise identical, which is why `resnet18_cwt` (CUDA) and `resnet18_cwt_mps` are reported separately.

## Storage policy

The dataset (`data/`, about 18 GB, plus its zip) is not versioned; download it from figshare. The following are also not versioned, since `train.py` regenerates them:
- `results/ventanas.npz` (146 MB)
- `results/imagenes_*.npy` (848 MB each)
- `results/*.pt` checkpoints (43 MB, only the Grad-CAM model is kept)

Everything else in `results/` (CSV, `*_scores.npz` test probabilities, logs) is small, and `evaluate.py` needs it, so it is kept.
