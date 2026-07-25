# DL_LAB — Deep Learning Laboratory

Coursework repository for **CS3807 – Deep Learning Laboratory**, B.Tech Artificial Intelligence & Data Science, Shiv Nadar University Chennai.

Contains lab experiments implementing core neural network concepts from scratch and with TensorFlow/Keras and scikit-learn, along with the corresponding lab reports.

---

## Repository Structure

```
DL_LAB/
├── Experiment_1_Perceptron/
│   ├── exp_1_deep_learning.ipynb
│   └── Experiment_1_Report.tex
├── Experiment_2_MLP/
│   ├── Experiment_2_Implementation_of_a_Multi_Layer_Perceptron.ipynb
│   ├── Experiment_2.tex
│   └── figures/
└── README.md
```

> Adjust the folder/file names above to match your actual repo layout if they differ.

---

## Experiment 1 — Perceptron Learning Algorithm

**Objective:** Implement the perceptron learning algorithm from scratch and study its behaviour on both a real dataset and classic logic gates.

**Contents:**
- Binary classification on the **Banknote Authentication** dataset (variance, skewness, curtosis, entropy → authentic/forged), including EDA, correlation analysis, train/test split, feature scaling, and evaluation (accuracy, precision, recall, F1, confusion matrix).
- Effect of learning rate on convergence speed.
- From-scratch perceptron vs. scikit-learn's `Perceptron`.
- **Logic gates (AND, OR, NOT):** weights logged after every individual update, with the decision boundary plotted at each step until convergence.
- **XOR gate:** demonstrates that a single-layer perceptron cannot learn a non-linearly-separable function — the weight vector is shown to enter a repeating cycle instead of converging, with an explicit pattern analysis of why.

**Key result:** AND, OR, and NOT converge in a handful of updates; XOR never converges and instead cycles through a fixed set of weight states indefinitely, illustrating the perceptron convergence theorem's requirement of linear separability.

---

## Experiment 2 — Multi-Layer Perceptron (MLP) for Image Classification

**Objective:** Implement an MLP using TensorFlow/Keras for multi-class image classification on **Fashion-MNIST**, then optimize it with automated hyperparameter search.

**Contents:**
- Dataset exploration and preprocessing (flattening 28×28 images to 784-length vectors, normalization, one-hot encoding).
- MLP architecture: `Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`, trained with Adam for 20 epochs.
- Evaluation via accuracy, precision, recall, F1-score, confusion matrix, and per-class classification report.
- Hyperparameter optimization with **Keras Tuner** (`RandomSearch`) and **scikit-learn's `RandomizedSearchCV`**, followed by retraining and evaluating the optimized model against the baseline.

**Key result:** Baseline MLP reaches **88.44% test accuracy**. The optimized `MLPClassifier` from the completed `RandomizedSearchCV` search reached 87.78% — slightly below baseline, attributed to a smaller search budget (single hidden layer, fewer training iterations) rather than a genuinely worse configuration.

---

## Tech Stack

- Python 3
- NumPy, Pandas
- Matplotlib, Seaborn
- TensorFlow / Keras, Keras Tuner
- scikit-learn

---

## How to Run

1. Open the notebook for the relevant experiment in Google Colab (or Jupyter locally).
2. Run all cells top to bottom.
3. Datasets are loaded directly via `tensorflow.keras.datasets` (Fashion-MNIST) or uploaded manually (Banknote Authentication `.zip`).

---

## Reports

Each experiment's LaTeX report (compiled with Overleaf) documents the objective, theory, methodology, full output/results, plots with inference, and discussion. See the `.tex` files in each experiment folder.

---

## Author

Sai Hari — B.Tech AI & Data Science, Shiv Nadar University Chennai
