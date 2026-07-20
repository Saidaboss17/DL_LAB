Single Layer Perceptron — Banknote Authentication

Implementation of a Single Layer Perceptron from scratch (no ML framework) for binary classification, trained and evaluated on the UCI Banknote Authentication dataset. Built for CS3807 (Deep Learning Laboratory), B.Tech AI & Data Science, Shiv Nadar University Chennai.

Overview

A perceptron is trained to distinguish authentic from forged banknotes using four statistical features extracted from wavelet-transformed banknote images. The model — weight updates, bias updates, and the step activation function — is implemented manually using only NumPy, without sklearn.linear_model.Perceptron or any deep learning library.

How It Works


Load & explore the data — 1372 banknote samples, 4 numerical features (Variance, Skewness, Curtosis, Entropy), binary target (0 = Authentic, 1 = Forged), no missing values.
EDA — histograms, correlation heatmap, pairwise scatter plots, and boxplots are generated to understand feature distributions and class separability.
Preprocessing — features are standardized with StandardScaler (fit on the training set only), and the data is split 80/20 into train/test sets using a stratified split (random_state=42).
Perceptron implementation — a Perceptron class is built from scratch:

Weights and bias are initialized to zero.
For each training sample, the weighted sum z = w·x + b is passed through a step activation function: f(z) = 1 if z ≥ 0 else 0.
Weights and bias are updated using the perceptron learning rule: w ← w + η(y − ŷ)x, b ← b + η(y − ŷ).
Training runs for a fixed number of epochs (learning_rate = 0.01, epochs = 20), tracking misclassifications, weights, and bias at every epoch.



Evaluation — the trained model is evaluated on the held-out test set using Accuracy, Precision, Recall, F1-score, and a Confusion Matrix (sklearn.metrics).


Over 20 epochs, misclassifications dropped from 51 to 15, and the final model achieved 98.55% accuracy on the test set.

Repository Structure

.
├── exp_1_deep_learning.ipynb    # Full implementation (EDA → training → evaluation)
├── Experiment_1.tex             # LaTeX lab report source
├── Experiment_1.pdf             # Compiled lab report
├── Experiment_1_images/         # Plots generated during EDA/training/evaluation
└── README.md

Getting Started

Requirements

bashpip install pandas numpy matplotlib seaborn scikit-learn

Run the notebook

Download data_banknote_authentication.txt from the UCI Banknote Authentication Dataset and place it in the working directory, then:

bashjupyter notebook exp_1_deep_learning.ipynb

Run the cells top to bottom — EDA plots, training logs, and evaluation metrics will be generated in order.

Rebuild the report

The .tex file expects the images folder alongside it as images/ — rename Experiment_1_images/ to images/ (or update the \includegraphics paths), then:

bashpdflatex Experiment_1.tex
