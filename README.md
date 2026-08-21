# Conditional Diffusion–Augmented Cascaded Multi-View Attention BiLSTM for Non-Invasive Blood Glucose Estimation from Photoplethysmography

This package provides the reproducibility materials associated with the manuscript. The framework integrates conditional diffusion-based data augmentation, cascaded multi-view attention BiLSTM regression, and moth–flame optimization (MFO) for fold-specific hyperparameter selection.

## Dataset

The original PPG dataset used in this study is publicly available from Mendeley Data, Version 3:

**DOI:** 10.17632/37pm7jk7jn.3

The original dataset is not redistributed in this package.

## Pipeline Overview

1. The raw PPG recordings are identified by participant ID and recording ID. BGL labels are converted from mg/dL to mmol/L, recordings are cropped to 10 s, filtered using a fourth-order 0.30–15 Hz Butterworth band-pass filter in SOS form with `sosfiltfilt`, divided into two non-overlapping 5-s segments, and independently standardized using segment-wise z-score normalization.
2. A one-dimensional conditional diffusion model (CDM) is used to generate PPG segments conditioned on BGL, heart rate (HR), and body mass index (BMI). The CDM uses a 64/128/256/512 conditional U-Net with GroupNorm, SiLU, residual connections, four-head cross-attention, 1000-step DDPM diffusion, and a hybrid MSE + SSIM loss. For each real training segment, five synthetic candidates are generated and two are randomly retained, producing a 1:2 real-to-synthetic augmentation ratio.
3. A cascaded multi-view attention BiLSTM (CMAB) network is used for BGL regression. The network combines a four-stage Conv1D encoder, channel attention, four-head temporal self-attention, multiplicative attention fusion, three cascaded BiLSTM layers, temporal mean pooling, cross-layer Hadamard feature interaction, and a fully connected regression head. MFO is used independently within each outer fold to optimize the learning rate, BiLSTM hidden-unit counts, dropout rate, and attention-fusion coefficient.

No additional signal-quality rejection criterion was applied. All 67 cropped recordings were retained for subsequent segmentation and analysis.

## Package Contents

- `data.py`: data loading, participant/recording ID parsing, BGL unit conversion, 10-s cropping, 5-s segmentation, and participant-wise fold construction.
- `preprocessing.py`: Butterworth band-pass filtering and segment-wise z-score normalization.
- `traditional_augmentation.py`: time warping, scaling, Gaussian-noise augmentation, five-candidate generation, and two-candidate retention.
- `cdm.py`: conditional 1D U-Net, DDPM diffusion process, physiological cross-attention, MSE + SSIM loss, and CDM training utilities.
- `synthetic_generation.py`: fold-safe synthetic PPG generation using only outer-training participants.
- `cmab.py`: CMAB regression network with convolutional encoding, multi-view attention, cascaded BiLSTM, and hierarchical feature fusion.
- `mfo.py`: fold-specific moth–flame optimization for CMAB hyperparameters.
- `evaluation.py`: RMSE, MAE, MARD, Clarke Error Grid, recording-level out-of-fold aggregation, and participant-clustered paired bootstrap analysis.
- `fold_assignments.csv`: participant-level outer-fold assignments for the reported partition seeds.
- `requirements.txt`: Python package requirements for the reproducibility code.

## Validation Protocol

Experiments were conducted on a public dataset containing 67 original 10-s PPG recordings from 23 participants. Subject-wise five-fold cross-validation was used so that all recordings from the same participant remained within a single fold. The two 5-s out-of-fold predictions belonging to each original recording were averaged to obtain one recording-level prediction, after which all 67 recording-level predictions were pooled for the primary evaluation.

The primary random seed was 42. Participant-partition sensitivity analyses used seeds 2026 and 3407.

Across the three participant-level fold assignments, the reported recording-level performance was RMSE 0.60 ± 0.01 mmol/L, MAE 0.39 ± 0.01 mmol/L, MARD 5.52 ± 0.04%, Clarke Zone A 96.8 ± 0.2%, and Zones A+B 100 ± 0.0%.

## Reproducibility Materials Requested During Peer Review

The files in this package correspond to the reproducibility materials requested during peer review:

- Participant-level fold assignments: `fold_assignments.csv`
- Data loading and fold construction: `data.py`
- Signal preprocessing: `preprocessing.py`
- Conventional augmentation: `traditional_augmentation.py`
- Conditional diffusion model: `cdm.py`
- Synthetic PPG generation: `synthetic_generation.py`
- CMAB regression model: `cmab.py`
- MFO implementation and search settings: `mfo.py`
- RMSE, MAE, MARD, Clarke error-grid evaluation, recording-level aggregation, and participant-clustered bootstrap: `evaluation.py`

Synthetic PPG waveforms are not redistributed as a separate dataset. The synthetic-generation procedure and random-seed settings are provided in the accompanying code and manuscript.
