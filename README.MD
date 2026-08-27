# CS3807 – Deep Learning Laboratory

## Experiment 5: Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning

---

## 📌 Overview

This repository contains the implementation of **Experiment 4** for the **CS3807 – Deep Learning Laboratory**.

The experiment focuses on studying the evolution of Deep Convolutional Neural Network (CNN) architectures and understanding **Transfer Learning** and **Fine-Tuning**.

---

## 🎯 Objectives

- Study the evolution of deep CNN architectures.
- Compare LeNet-5, AlexNet, VGG16, GoogleNet and ResNet.
- Understand Transfer Learning.
- Fine-tune pretrained CNN models.
- Compare classification performance using different evaluation metrics.

---

## 📊 Dataset

The experiment uses the **CIFAR-10 dataset**.

| Property | Details |
|---|---|
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Number of Classes | 10 |
| Image Size | 32 × 32 × 3 |
| Image Type | RGB |

### CIFAR-10 Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

---

## 🧠 CNN Architectures Studied

| Architecture | Year | Major Contribution |
|---|---:|---|
| LeNet-5 | 1998 | First practical CNN |
| AlexNet | 2012 | ReLU, Dropout and GPU training |
| VGG16 | 2014 | Uniform 3 × 3 convolution filters |
| GoogleNet | 2014 | Inception modules |
| ResNet | 2015 | Residual learning using skip connections |

---

## 🔄 Transfer Learning

Transfer Learning uses a pretrained CNN model that has already learned general image features from a large dataset such as ImageNet.

The following steps are performed:

1. Load a pretrained model.
2. Remove the original classifier.
3. Freeze the convolutional base.
4. Add a Global Average Pooling layer.
5. Add a Dense layer with ReLU activation.
6. Add a Softmax output layer.
7. Train the new classifier.
8. Fine-tune selected convolutional layers.

---

## 🏗️ Pretrained CNN Models

The experiment considers the following pretrained models:

- VGG16
- ResNet50
- MobileNetV2
- EfficientNetB0 (Optional)

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 10–20 |
| Loss Function | Categorical Cross Entropy |
| Evaluation Metric | Accuracy |

---

## 🔧 Fine-Tuning

For fine-tuning, the last convolutional block of the pretrained model is unfrozen and trained for additional epochs.

Fine-tuning allows the pretrained features to adapt to the target classification task.

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## 📊 Results and Plots

The experiment includes the following plots:

1. Sample CIFAR-10 Images
2. Training Accuracy
3. Validation Accuracy
4. Training Loss
5. Validation Loss
6. Confusion Matrix
7. Misclassified Images (Optional)

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

---

## 📁 Repository Structure

```text
CS3807-Experiment-4/
│
├── Experiment_4.ipynb
├── README.md
└── results/
    ├── sample_images.png
    ├── accuracy_plot.png
    ├── loss_plot.png
    └── confusion_matrix.png
