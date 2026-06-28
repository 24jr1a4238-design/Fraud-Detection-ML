# Fraud-Detection-ML


A machine learning project to detect fraudulent credit card transactions using classification algorithms.

## Problem Statement
Financial fraud causes massive losses every year. This project builds a model that can accurately flag fraudulent transactions from a highly imbalanced dataset, where genuine transactions vastly outnumber fraud cases.

## Dataset
- Source: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions, only 492 fraudulent (~0.17%)
- Features: Time, Amount, V1-V28 (PCA-anonymized), Class (target)
- Download `creditcard.csv` and place it in the project folder before running.

## Tools Used
- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest)
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## Approach
1. Data preprocessing and feature scaling
2. Handling class imbalance using SMOTE
3. Training Logistic Regression and Random Forest models
4. Evaluating using Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix

## How to Run
pip install pandas scikit-learn imbalanced-learn
python fraud_detection.py

## Output
Prints model evaluation metrics for both models and saves the trained Random Forest model as `fraud_detection_model.pkl`.
