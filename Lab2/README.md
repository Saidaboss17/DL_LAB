# Experiment 2 — Multi-Layer Perceptron (MLP) for Multi-Class Image Classification

CS3807 – Deep Learning Laboratory, B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai.

## Objective

Implement a Multi-Layer Perceptron using TensorFlow/Keras for multi-class image classification on the **Fashion-MNIST** dataset, covering image preprocessing, model construction, training, evaluation, and automated hyperparameter optimization.

## Dataset

- **Fashion-MNIST**: 60,000 training images, 10,000 testing images, 10 classes, 28×28 grayscale.
- Classes: T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot.

## Contents

- **Preprocessing:** flattening 28×28 images to 784-length vectors, pixel normalization to [0,1], one-hot label encoding.
- **Baseline model:** `Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`, compiled with Adam and categorical cross-entropy, trained for 20 epochs (batch size 32).
- **Evaluation:** accuracy, precision, recall, F1-score, confusion matrix, and per-class classification report.
- **Hyperparameter optimization:**
  - Keras Tuner (`RandomSearch`) over hidden layers, neurons, activation, dropout, learning rate, and optimizer.
  - scikit-learn `RandomizedSearchCV` with cross-validation over an `MLPClassifier`, followed by retraining the best configuration and evaluating it on the test set.
- **Comparison:** baseline vs. optimized model performance and training time.

## Key Results

| Metric      | Baseline (TensorFlow MLP) | Optimized (scikit-learn MLP) |
|-------------|---------------------------|-------------------------------|
| Accuracy    | 0.8844                    | 0.8778                        |
| Precision   | 0.8836                    | 0.8792                        |
| Recall      | 0.8844                    | 0.8778                        |
| F1-score    | 0.8835                    | 0.8778                        |

The baseline model slightly outperforms the optimized one, most likely due to a smaller training budget in the search (single hidden layer, fewer iterations) rather than the chosen hyperparameters being genuinely worse.

## Tech Stack

Python, NumPy, Pandas, Matplotlib, Seaborn, TensorFlow/Keras, Keras Tuner, scikit-learn.

## How to Run

1. Open `Experiment_2_Implementation_of_a_Multi_Layer_Perceptron.ipynb` in Google Colab.
2. Run all cells top to bottom — Fashion-MNIST loads directly via `tensorflow.keras.datasets`.

## Report

The full LaTeX report (`Experiment_2.tex`) documents the objective, theory, methodology, complete output, all mandatory plots with inference, hyperparameter search results, and discussion.

## Author

Sai Hari — B.Tech AI & Data Science, Shiv Nadar University Chennai
