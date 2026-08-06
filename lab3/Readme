# Experiment 3 — Convolutional Neural Networks for Image Classification (CIFAR-10)

## Objective
Design, implement, train, and evaluate a Convolutional Neural Network (CNN) using TensorFlow/Keras for multi-class image classification on the CIFAR-10 dataset, and study the effect of key CNN hyperparameters (pooling strategy, optimizer, kernel size) on model performance.

## Dataset
- **Name:** CIFAR-10 (`tf.keras.datasets.cifar10`)
- **Size:** 60,000 images total — 50,000 train / 10,000 test
- **Image dimensions:** 32 × 32 × 3 (RGB)
- **Classes (10):** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

## Model Architecture
Sequential CNN:

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3 kernel, ReLU |
| Flatten | — |
| Dense | 64 units, ReLU |
| Dense | 10 units, Softmax |

**Total trainable parameters:** 122,570
**Optimizer:** Adam · **Loss:** Sparse Categorical Cross-Entropy · **Batch size:** 64 · **Epochs:** 5

## Results (Base Model)

| Metric | Value |
|---|---|
| Test Loss | 0.9259 |
| Test Accuracy | 67.82% |
| Precision (weighted) | 0.6837 |
| Recall (weighted) | 0.6782 |
| F1-score (weighted) | 0.6776 |

## Hyperparameter Comparison

| Model Variant | Validation Accuracy | Validation Loss |
|---|---|---|
| Base (Max Pooling, 3×3, Adam) | 67.82% | 0.9259 |
| Average Pooling | 65.63% | 0.9695 |
| SGD Optimizer | 46.72% | 1.4630 |
| Larger Kernel (5×5) | 67.39% | 0.9379 |

## What This Experiment Covers
- Building blocks of a CNN: convolution, activation, pooling, flattening, dense layers
- Effect of hyperparameters: kernel size, pooling strategy, optimizer, filters, batch size, epochs
- Visualization of intermediate feature maps across convolutional layers
- Model evaluation: accuracy, precision, recall, F1-score, confusion matrix, classification report

## Files
- `dl_lab_3.ipynb` — Google Colab notebook with full implementation
- `Experiment_3.tex` — LaTeX lab report (Overleaf)
- Generated plots (sample images, class distribution, accuracy/loss curves, confusion matrix, feature maps, etc.)

## Links
- **Google Colab:** [add link here]
- **GitHub Repo:** [add link here]

## Tools Used
Python · TensorFlow / Keras · NumPy · Matplotlib · Seaborn · scikit-learn
