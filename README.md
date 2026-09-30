# CS3807 – Deep Learning Laboratory | Experiment 7

**End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders**

Shiv Nadar University Chennai · B.Tech AI & Data Science · Semester V · AY 2026–27

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR-NOTEBOOK-ID)

---

## Overview

This experiment builds four related models on **MNIST** under one controlled protocol and compares them on reconstruction quality, denoising ability and generative behaviour:

| # | Model | Latent representation |
|---|-------|-----------------------|
| 1 | Fully Connected Autoencoder (FC-AE) | 16-d vector |
| 2 | Convolutional Autoencoder (CAE) | 7×7×64 feature map |
| 3 | Denoising CAE (DAE) | 7×7×64 feature map |
| 4 | Variational Autoencoder (VAE) | 2-d Gaussian (μ, log σ²) |

Evaluation uses **MSE, MAE and SSIM** on a held-out test set. Extra studies cover latent dimension, noise level, latent-space visualisation, random generation, interpolation and per-image error analysis.

## Setup

| Item | Value |
|------|-------|
| Dataset | MNIST (28×28×1, scaled to [0, 1]) |
| Split | 9,000 train / 1,000 validation / 2,000 test |
| Optimizer | Adam, lr = 1e-3 |
| Batch size / epochs | 128 / 20 |
| Loss | Binary cross-entropy (VAE: BCE + KL) |
| Seed | 42 |
| Framework | TensorFlow 2.20 / Keras 3.13 (Colab GPU) |

Labels are used only for visualisation, never for training.

## Repository Structure

```
.
├── DL_LAB7.ipynb              # Full Colab notebook (run top to bottom)
├── Experiment_7_Report.tex    # LaTeX report source
├── Experiment_7_Report.pdf    # Compiled report
├── figures/                   # All plots produced by the notebook
└── README.md
```

## How to Run

1. Open `DL_LAB7.ipynb` in Google Colab (use the badge above).
2. Set the runtime to **GPU** (Runtime → Change runtime type).
3. Run **Runtime → Run all**.
4. Figures and CSV tables are saved to `figures/` and zipped as `exp7_outputs.zip`.

Dependencies (preinstalled on Colab): `tensorflow`, `numpy`, `pandas`, `matplotlib`, `scikit-image`.

To build the report, compile `Experiment_7_Report.tex` with `pdflatex` (keep the `figures/` folder next to it).

## Results

### Reconstruction comparison (test set)

| Model | MSE | MAE | SSIM | Parameters | Time (s) |
|-------|-----|-----|------|-----------|----------|
| FC Autoencoder | 0.02047 | 0.05513 | 0.75865 | 211,040 | 12.2 |
| Conv. Autoencoder | 0.00259 | 0.01508 | 0.97371 | 74,497 | 20.7 |
| Denoising CAE (Gaussian σ = 0.2) | 0.00437 | 0.02057 | 0.94717 | 74,497 | 29.3 |
| VAE (z = μ) | 0.04476 | 0.10552 | 0.48419 | 485,957 | 24.9 |

### Denoising (metrics vs. clean images)

| Noise | Level | SSIM (noisy input) | SSIM (denoised) |
|-------|-------|--------------------|-----------------|
| Gaussian | 0.1 | 0.679 | 0.962 |
| Gaussian | 0.2 | 0.594 | 0.947 |
| Gaussian | 0.3 | 0.509 | 0.911 |
| Salt & Pepper | 0.05 | 0.708 | 0.957 |
| Salt & Pepper | 0.10 | 0.579 | 0.945 |
| Salt & Pepper | 0.20 | 0.447 | 0.902 |

### Latent-dimension study (FC-AE)

| d_z | MSE | SSIM | Parameters |
|-----|-----|------|-----------|
| 2 | 0.04634 | 0.44814 | 210,130 |
| 8 | 0.02731 | 0.68355 | 210,520 |
| 16 | 0.02095 | 0.75195 | 211,040 |
| 32 | 0.01966 | 0.76689 | 212,080 |

### VAE losses (per image, test set)

| Reconstruction (BCE) | KL | Total |
|----------------------|----|-------|
| 151.317 | 5.951 | 157.268 |

## Key Findings

- **CAE ≫ FC-AE:** about 8× lower MSE and SSIM 0.974 vs 0.759, with ~65% fewer parameters, because convolutions preserve spatial locality. (The CAE's 3136-value latent map is also a much weaker bottleneck than the 16-d code.)
- **Denoising works:** the DAE lifts SSIM from 0.45–0.71 (noisy) to 0.90–0.96 across both noise types and all levels.
- **Latent size has diminishing returns:** 2 → 8 cuts MSE by ~41%, while 16 → 32 cuts only ~6%.
- **VAE trades sharpness for a usable latent space:** lower SSIM (0.484) from the 2-d bottleneck and KL regulariser, but it supports random generation and smooth interpolation.
- **Hardest images:** digits with loops and crossings (8, 2) and unusual thick or slanted styles give the highest reconstruction error.

## Figures

| File | Content |
|------|---------|
| `plot1_original_vs_recon.png` | Original vs FC-AE reconstruction + error map |
| `plot2_fc_ae_loss.png` | FC-AE training/validation loss |
| `plot3_fc_vs_cae.png` | Original → FC-AE → CAE |
| `plot4_denoise_*.png` | Clean vs noisy vs denoised |
| `plot5_noise_vs_{mse,mae,ssim}.png` | Noise level vs metrics |
| `plot6_vae_latent_space.png` | 2-D VAE latent space by digit |
| `plot7_vae_generated.png` | 25 randomly generated digits |
| `plot8_interpolation.png` | Latent interpolation (1 → 7) |
| `plot9_vae_loss.png` | VAE total / reconstruction / KL loss |
| `plot10_latent_dim_vs_mse.png` | Latent dimension vs MSE |
| `plot_error_hist_*.png`, `high_error_*.png` | Error distributions and five worst samples |
| `bonus_vae_manifold.png` | Decoded 2-D latent manifold |

## References

1. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
2. D. P. Kingma, M. Welling, "Auto-Encoding Variational Bayes," ICLR, 2014.
3. P. Vincent et al., "Stacked Denoising Autoencoders," JMLR, 2010.
4. Y. LeCun, C. Cortes, C. Burges, "MNIST Handwritten Digit Database."
5. TensorFlow, Keras and scikit-image documentation.

## Author

**Sai Hari** · B.Tech AI & Data Science, Shiv Nadar University Chennai
