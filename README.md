# Experiment 6 — Sequence Learning & Video Understanding

> **CS3807 — Deep Learning Laboratory**  
> **B.Tech Artificial Intelligence & Data Science**  
> **Shiv Nadar University Chennai**

---

## Overview

This experiment explores **sequence learning using recurrent neural networks** and extends the same concepts to **video understanding**.

Three recurrent architectures are implemented and compared:

- Vanilla RNN
- LSTM
- GRU

The experiment is then extended to:

- Video action recognition using **MobileNetV2 + LSTM**
- Encoder–decoder **Sequence-to-Sequence (Seq2Seq)** learning

The experiments were implemented using **Python, TensorFlow/Keras and Google Colab**.

---

## Objectives

The experiment aims to:

1. Understand how recurrent neural networks process sequential data.
2. Implement a Vanilla RNN for human activity recognition.
3. Implement and compare LSTM and GRU architectures.
4. Analyze training and validation performance.
5. Study the effect of sequence length on model performance.
6. Analyze classification using confusion matrices.
7. Extract spatial features from video frames using a CNN.
8. Model temporal relationships between video frames using an LSTM.
9. Implement an encoder–decoder sequence-to-sequence model.
10. Understand teacher forcing and sequence generation.

---

## Datasets

### 1. UCI Human Activity Recognition

The primary experiment uses the **UCI Human Activity Recognition Using Smartphones (HAR)** dataset.

The experiment uses six activity classes:

| Class | Activity |
|---|---|
| 0 | WALKING |
| 1 | WALKING_UPSTAIRS |
| 2 | WALKING_DOWNSTAIRS |
| 3 | SITTING |
| 4 | STANDING |
| 5 | LAYING |

Each sequence contains:

```text
128 time steps × 9 sensor features
