# Fraud Detection using Machine Learning
# Dataset: Credit Card Fraud Detection (Kaggle - mlg-ulb/creditcardfraud)
# Download dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the same folder before running.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("creditcard.csv")
print("Dataset shape:", df.shape)
print(df['Class'].value_counts())  # 0 = genuine, 1 = fraud

# -------------------------------
# 2. Preprocessing
# -------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])

# -------------------------------
# 3. Train/Test Split (stratified)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# 4. Handle Class Imbalance (SMOTE on training data only)
# -------------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("After SMOTE:", y_train_bal.value_counts())

# -------------------------------
# 5. Train Models
# -------------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_bal, y_train_bal)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)

# -------------------------------
# 6. Evaluate Models
# -------------------------------
def evaluate(model, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"\n--- {name} ---")
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1-Score:", f1_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

evaluate(log_reg, "Logistic Regression")
evaluate(rf, "Random Forest")

# -------------------------------
# 7. (Optional) Save the best model
# -------------------------------
import joblib
joblib.dump(rf, "fraud_detection_model.pkl")
print("\nModel saved as fraud_detection_model.pkl")
