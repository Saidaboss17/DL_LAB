#!/bin/bash
# generate_readme.sh
# Creates a README.md for Experiment 6 (RNN/LSTM/GRU + Video Understanding)

cat > README.md << 'EOF'
# Experiment 6 — End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

**Course:** CS3807 — Deep Learning Laboratory
**Program:** B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai

## Overview
This experiment builds an end-to-end understanding of recurrent sequence
learning by implementing and comparing Vanilla RNN, LSTM and GRU models,
extending the pipeline to video understanding using CNN-extracted features,
and demonstrating an encoder–decoder sequence-to-sequence model.

## Datasets
- **UCI HAR (Human Activity Recognition Using Smartphones):** raw inertial
  signals, windowed as `128 x 9` sequences across 6 activity classes.
- **UCF101 (subset):** 3 action classes (Basketball, Biking, WalkingWithDog),
  10 videos each, used for the CNN–LSTM video pipeline.
- **Synthetic reversal task:** 5000 sequences of length 4 for the
  encoder–decoder seq2seq demo.

## Pipeline
1. Preprocessing — windowing, normalization, train/val/test split (70/15/15)
2. Temporal data visualization
3. RNN / LSTM / GRU training and evaluation on HAR
4. Confusion matrix and per-class analysis
5. Effect of sequence length (T = 32, 64, 128)
6. CNN (MobileNetV2) + LSTM pipeline for video action recognition
7. Encoder–decoder LSTM for sequence reversal

## Results

### HAR Classification
| Model | Accuracy (%) | Macro F1 (%) | Parameters | Training Time (s) |
|-------|--------------|--------------|------------|--------------------|
| RNN   | 68.148       | 68.070       | 1,974      | 35.14              |
| LSTM  | 90.000       | 89.983       | 6,006      | 63.30              |
| GRU   | 93.333       | 93.326       | 4,758      | 86.26              |

### Sequence Length vs. Test F1 (%)
| T   | RNN    | LSTM   | GRU    |
|-----|--------|--------|--------|
| 32  | 75.138 | 88.929 | 89.638 |
| 64  | 74.612 | 93.335 | 92.962 |
| 128 | 69.315 | 94.408 | 93.021 |

### CNN–LSTM Video Classification
Accuracy / Precision / Recall / F1: **100.00%** (5-video held-out test set,
3 classes: Basketball, Biking, WalkingWithDog)

### Sequence-to-Sequence (Reversal Task)
- Token Accuracy: **99.95%**
- Sequence Accuracy: **99.80%**

## Repository Structure
