# Experiment 4 – Comparative Study of Deep CNN Architectures Using Transfer Learning

## CS3807 – Deep Learning Laboratory

**Shiv Nadar University Chennai**  
**B.Tech Artificial Intelligence & Data Science – Semester V**  
**Academic Year: 2026–27**

---

## 📌 Aim

To study the evolution of Deep Convolutional Neural Network (CNN) architectures and implement transfer learning using a pretrained VGG16 model on the CIFAR-10 dataset.

The experiment includes transfer learning, fine-tuning, model evaluation, hyperparameter study, and CNN architecture comparison.

---

## 🎯 Objectives

- Study the evolution of deep CNN architectures.
- Compare LeNet-5, AlexNet, VGG16, GoogleNet, and ResNet.
- Understand transfer learning.
- Implement transfer learning using VGG16.
- Fine-tune selected convolutional layers.
- Evaluate the model using different performance metrics.

---

## 🗂️ Dataset – CIFAR-10

CIFAR-10 contains:

- **Training images:** 50,000
- **Testing images:** 10,000
- **Number of classes:** 10
- **Image size:** 32 × 32 × 3

### Classes

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

## 🔄 Transfer Learning

A pretrained **VGG16 model with ImageNet weights** was used.

### Workflow

```text
CIFAR-10
    ↓
Pretrained VGG16
    ↓
Remove Original Classifier
    ↓
Freeze Convolutional Base
    ↓
Global Average Pooling
    ↓
Dense Layer + ReLU
    ↓
Softmax Output
    ↓
10-Class Prediction
