# Attention Type and Attention Position in a Pretrained ResNet50

**CS3807 – Deep Learning Laboratory · Experiments 8 & 9**
Shiv Nadar University Chennai · B.Tech AI & Data Science · Semester V

A controlled study of how different attention mechanisms, and *where* they are inserted, change the behaviour of an ImageNet-pretrained **ResNet50** on five-class flower classification (`tf_flowers`).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR-NOTEBOOK-ID)

---

## Highlights

- Ten models compared under one identical protocol: same split, preprocessing, augmentation, optimizer, epochs and seed.
- Attention types: **channel (SE-style)**, **spatial**, **sequential channel→spatial**, **lightweight self-attention**, a **combined** module, plus **ECA** and **CBAM**.
- Positions studied: **P2** (after conv3_x), **P3** (after conv4_x), **P4** (after conv5_x), and the multi-stage P3+P4 combination.
- Evaluation covers accuracy, macro precision/recall/F1, parameter counts, inference time, confusion matrices, feature maps, attention maps and Grad-CAM-style overlays.

**Main finding:** self-attention at P3 (+1.44 pp), the combined module (+1.62 pp) and channel→spatial attention at P4 (+1.26 pp) improved on the baseline. The sigmoid-gated modules at P2/P3 stayed within about ±1 pp of it, and stacking attention at P3+P4 was *worse* than P4 alone.

---

## Dataset

| Item | Value |
|---|---|
| Dataset | TensorFlow Flowers (`tf_flowers`), 3,670 RGB images |
| Classes | dandelion, daisy, tulips, sunflowers, roses |
| Split | Stratified 70 : 15 : 15 → 2,567 train / 547 val / 556 test |
| Seed | 42 (the same index arrays are reused for every model) |
| Input | 224 × 224 × 3, `resnet50.preprocess_input` |
| Augmentation (train only) | horizontal flip, rotation 0.05, zoom 0.10, translation 0.05, contrast 0.10 |

---

## Method

**Backbone:** `ResNet50(include_top=False, weights="imagenet")` → Global Average Pooling → Dropout (0.3) → Dense(5, softmax).

**Attention insertion points**

| Position | After | Feature shape |
|---|---|---|
| P2 | conv3_x | 28 × 28 × 512 |
| P3 | conv4_x | 14 × 14 × 1024 |
| P4 | conv5_x | 7 × 7 × 2048 |

Every module preserves the tensor shape, so it can be dropped between two ResNet stages without changing the rest of the network.

**Training protocol (identical for all models)**

| Phase | Epochs | Optimizer | Trainable |
|---|---|---|---|
| 1 – frozen backbone | 3 | Adam, lr 1e-3 | attention + classifier head |
| 2 – fine-tuning | 2 | Adam, lr 1e-5 | + conv5 block (BatchNorm frozen) |

Batch size 32 · loss: categorical cross-entropy · each configuration trained once.

---

## Results

All values are test-set percentages on the same 556 images (macro-averaged).

| Model | Position | Acc. | Prec. | Recall | F1 | Total params | ms/img |
|---|---|---|---|---|---|---|---|
| Baseline | – | 92.09 | 92.09 | 92.08 | 92.04 | 23,597,957 | 10.04 |
| Channel (SE) | P3 | 92.27 | 92.52 | 92.10 | 92.20 | 23,730,117 | 7.48 |
| Spatial | P3 | 91.55 | 91.56 | 91.42 | 91.38 | 23,598,055 | 7.15 |
| Channel→Spatial | P2 | 91.55 | 91.45 | 91.45 | 91.43 | 23,631,367 | 9.18 |
| Channel→Spatial | P3 | 91.37 | 91.35 | 91.34 | 91.25 | 23,730,215 | 7.53 |
| Channel→Spatial | P4 | 93.35 | 93.41 | 93.34 | 93.34 | 24,124,519 | 7.75 |
| Self-Attention | P3 | 93.53 | 93.56 | 93.45 | 93.46 | 24,122,246 | 10.33 |
| **Combined (C+S+Self)** | P3 | **93.71** | **93.86** | **93.59** | **93.65** | 24,254,504 | 10.16 |
| ECA | P3 | 91.19 | 91.05 | 91.14 | 91.08 | 23,597,960 | 9.48 |
| CBAM | P3 | 91.73 | 92.00 | 91.40 | 91.60 | 23,730,215 | 10.56 |
| Channel→Spatial, multi-stage | P3 + P4 | 91.19 | 91.29 | 91.16 | 91.05 | 24,256,777 | 7.99 |

### Key observations

- **Attention type matters at P3.** Only the self-attention models clearly beat the baseline. Self-attention is residual with a learnable scale initialised to 0, so it starts as an identity mapping and leaves the pretrained features intact.
- **Position matters.** The same channel→spatial module gives 91.55 % (P2), 91.37 % (P3) and 93.35 % (P4). The gain appears at the deepest stage, where the module feeds directly into the newly trained classifier.
- **More attention is not better.** P3+P4 (91.19 %) is about 2 pp below P4 alone and costs more parameters.
- **Parameters do not predict accuracy.** All modules add at most 2.8 % parameters. ECA (+3) and Spatial (+98) give no gain, while SE and CBAM (+132k) give almost none.
- **Per class.** Self-attention improves F1 for all five classes (largest for daisy +2.6 pp and dandelion +2.0 pp). Channel→Spatial at P3 helps dandelion and roses but loses on tulips and sunflowers.
- **Qualitative maps are weakly selective.** The learned spatial gates are nearly flat, the top-20 channel weights are all ≈ 1.0 (a saturated gate), and the gradient-based overlays look alike for the baseline and attention models.

<p align="center">
  <img src="figures/accuracy_comparison.png" width="48%" alt="Accuracy comparison"/>
  <img src="figures/accuracy_vs_parameters.png" width="48%" alt="Accuracy vs parameters"/>
</p>

---

## Repository structure

```
.
├── CS3807_Experiments_8_9_your_version_final.ipynb   # full Colab notebook (training + evaluation)
├── report/
│   ├── main.tex                                      # LaTeX report (compile on Overleaf)
│   └── figures/                                      # plots used by the report
├── figures/                                          # figures used in this README
└── README.md
```

> Adjust the paths above to match how you organise the repository.

---

## Reproducing the experiment

1. Open the notebook in Google Colab (button above) and select a **GPU** runtime.
2. Run the cells top to bottom. The notebook downloads `tf_flowers` via TensorFlow Datasets and builds the fixed stratified split (`numpy.random.default_rng(42)`).
3. Models train in two phases (3 frozen + 2 fine-tuning epochs). Weights and history are saved after every epoch, so an interrupted run resumes from its last finished epoch.
4. To fill the results that were not stored, re-run the cells for the attention-order experiment (`order_Spatial_Channel`) and the spatial-kernel experiments (`spatial_kernel_3`, `spatial_kernel_5`).

**Requirements:** Python 3, TensorFlow 2.x / Keras, TensorFlow Datasets, NumPy, pandas, scikit-learn, matplotlib, seaborn.

### Building the report

Upload the `report/` folder to [Overleaf](https://www.overleaf.com/) (or compile locally with `pdflatex`). Set your register number and links in the `EDIT THESE` block at the top of `main.tex`. Keep the `figures/` folder next to `main.tex`.

---

## References

1. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.
2. J. Hu, L. Shen, S. Albanie, G. Sun, E. Wu, "Squeeze-and-Excitation Networks," IEEE TPAMI, 2020.
3. S. Woo, J. Park, J.-Y. Lee, I. S. Kweon, "CBAM: Convolutional Block Attention Module," ECCV, 2018.
4. Q. Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks," CVPR, 2020.
5. A. Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.
6. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," ICCV, 2017.
7. TensorFlow Datasets: https://www.tensorflow.org/datasets

---

**Author:** Sai Hari · B.Tech AI & Data Science, Shiv Nadar University Chennai
