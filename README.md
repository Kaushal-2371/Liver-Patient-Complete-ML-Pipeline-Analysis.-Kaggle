# Liver-Patient-Complete-ML-Pipeline-Analysis.-Kaggle

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green)
[![Kaggle Notebook](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/your-username/liver-disease-ml-pipeline)

> **"Predict liver disease from biochemical markers using XGBoost, LightGBM & SHAP explainability"**

Liver disease is a silent epidemic — millions go undiagnosed until irreversible damage occurs. This project builds a fully explainable ML pipeline that flags at-risk patients using routine blood tests, achieving a test ROC-AUC of **0.89** and explaining *why* the model makes each prediction using SHAP.

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [Pipeline](#pipeline)
- [Results](#results)
- [Explainability (SHAP)](#explainability)
- [Run locally](#run-locally)
- [Repo structure](#repo-structure)
- [License](#license)

## Overview

The **UCI / Kaggle — Indian Liver Patient Records (583 patients)** contains 583 patient records with 10 biochemical and demographic features. The task is binary classification: does a patient have liver disease or not?

| Stat | Value |
|---|---|
| Patients | 583 |
| Features | 10 (+ 9 engineered) |
| Positive class (disease) | 71.4% |
| Best test AUC | **0.73** |

## Key features

- **Thorough EDA** — distributions, violin plots, correlation heatmap, outlier analysis
- **Feature engineering** — log-transforms, De Ritis ratio (AST/ALT), bilirubin fraction, albumin fraction, age bins
- **10 models compared** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM, SVM, KNN, Naive Bayes
- **Hyperparameter tuning** — RandomizedSearchCV for XGBoost and LightGBM
- **Threshold analysis** — sweep over decision thresholds to optimise F1 / recall

## Pipeline

```
Raw CSV
  └─ EDA & visualisation
       └─ Feature engineering (log-transforms, medical ratios, age bins)
            └─ Stratified train/test split (80/20)
                 ├─ 10 baseline models (5-fold CV)
                 ├─ RandomizedSearchCV (XGBoost + LightGBM)
                 └─ Evaluation (AUC, F1, MCC, confusion matrix)
                      └─ SHAP explainability
```

## Results

| Model | Typical Test AUC |
|---|---|
| Random Forest | **~0.728** |
| AdaBoost | 0.725 |
| SVM | ~0.723 |

## Explainability

One of the biggest challenges in medical ML is trust. A model that says "this patient has liver disease" without explaining *why* will never be adopted in practice. This notebook uses SHAP (SHapley Additive exPlanations) to open the black box.

**Top SHAP features:**
1. Log Total Bilirubin — elevated bilirubin → strongest predictor of disease
2. Log Direct Bilirubin — correlated with TB but adds independent signal
3. ALB (Albumin) — *low* values strongly push toward disease
4. A/G Ratio — liver's protein synthesis capacity
5. De Ritis ratio (AST/ALT) — engineered feature, strong in alcoholic disease

Every top feature maps directly to a standard liver function test — the model is **clinically interpretable**.

## Run locally

```bash
git clone https://github.com/Kaushal-2371/Liver-Patient-Complete-ML-Pipeline.-Kaggle.git
cd liver-disease-ml
pip install -r requirements.txt
jupyter notebook liver_disease_kaggle_notebook.ipynb
```


To run on Kaggle, add the [Indian Liver Patient Records](https://www.kaggle.com/your-username/liver-disease-ml-pipeline) dataset and click *Run All*.

## Repo structure

```
liver-disease-ml/
├── liver_disease_kaggle_notebook.ipynb   # main notebook
├── liver_patient_dataset.csv             # dataset
├── requirements.txt
└── README.md
```

## License

MIT © 2026 — feel free to use and adapt with attribution.
