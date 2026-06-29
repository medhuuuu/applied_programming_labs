# Neural Essay Score Prediction (Backpropagation)

A neural regression model built with PyTorch to predict essay quality scores
from linguistic features, compared against a scikit-learn baseline.

**Course:** Applied Programming (LIS050), Stockholm University

---

## Overview

This project implements essay score prediction using two approaches:

1. **scikit-learn** LinearRegression — baseline model
2. **PyTorch** — custom neural network with manual SGD and backpropagation

Both models predict essay scores from two linguistic features:
- Essay length: n^(1/4) (4th root of token count)
- Lexical diversity: OVIX score

---

## Contents

- `aes_backprop.py` — full implementation including data loading,
  feature extraction, scikit-learn baseline, and PyTorch neural network

---

## Model Details

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Training data:** Essay Set 1
- **Validation data:** Essay Set 2
- All scores standardized to zero mean, unit variance

---

## Key Concepts

- Error backpropagation
- Stochastic gradient descent
- Feature engineering for NLP
- Model comparison (classical ML vs neural network)

---

## How to Run

```bash
python3 aes_backprop.py
```

> Note: Requires essay dataset files. Uses PyTorch and scikit-learn.

---

## Tech Stack

Python · PyTorch · scikit-learn · NumPy · Backpropagation · SGD