# Titanic Survival Prediction with Custom k-Nearest Neighbors (k-NN)

This repository contains two Jupyter notebooks implementing  
a custom-built k-Nearest Neighbors (k-NN) classifier (using only NumPy and Pandas)  
to predict passenger survival on the Titanic dataset.

---

##  Files in This Repository

| File                                | Description                                                                                      |
|-------------------------------------|--------------------------------------------------------------------------------------------------|
| `_Titanic Survival Prediction Using Custom k-Nearest Neighbors (k-NN)`           | Initial implementation of a simple k-NN classifier (single nearest neighbor, Euclidean distance). |
| `Titanic Survival Prediction with Optimized k-Nearest Neighbors (k-NN)`       | Improved and fine-tuned k-NN model with hyperparameter tuning (varied k-values, distance metrics). |

---

##  Project Overview

The goal is to:
- Predict the Titanic `Survived` target variable (binary outcome) using k-NN.
- Build the algorithm **from scratch** using only core Python tools (no scikit-learn).
- Compare performance across different:
    - k-values (`1, 3, 5, 7`)
    - Distance metrics (Euclidean, Manhattan, Hamming)

---

##  Key Features

 Data cleaning and preprocessing  
 Feature scaling (Min-Max normalization)  
 Euclidean, Manhattan, and Hamming distance functions  
 Majority voting across k neighbors  
 Accuracy comparisons and best model selection

---

##  Best Model Results

| Best Configuration | Accuracy   |
|---------------------|-----------|
| k = 5, Euclidean    | ~84.8%   |

This optimized model significantly improves over the baseline NN approach (~72%)  
and demonstrates the impact of careful hyperparameter tuning.

---

##  How to Use

1. Clone or download the repository.
2. Open the notebooks (`.ipynb` files) in Jupyter Notebook or Jupyter Lab.
3. Run all cells to reproduce the full analysis, model building, and evaluation.

---

##  Next Steps

Future improvements could include:
- Cross-validation for more robust accuracy estimates.
- Weighted k-NN (giving closer neighbors more influence).
- Trying alternative machine learning models for comparison.

---

##  License

This project is shared for educational and learning purposes.

