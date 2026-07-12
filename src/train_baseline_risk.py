"""
train_baseline_risk.py
Purpose: Train a simple baseline model (Logistic Regression) on the Risk
dataset (Diagnosis target), so later we can COMPARE it against the Deep
Learning model. This proves whether deep learning actually performs better.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import joblib
import os

os.makedirs('models/risk', exist_ok=True)

# Step 1: Load the already-preprocessed data (created by preprocess_risk.py)
data = np.load('data/processed/risk_module_data.npz')
X_train, X_test = data['X_train'], data['X_test']
ydiag_train, ydiag_test = data['ydiag_train'], data['ydiag_test']

# Step 2: Train Logistic Regression on the Diagnosis target (Benign/Malignant)
# class_weight='balanced' -> handles the imbalance (Benign >> Malignant) automatically
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, ydiag_train)

# Step 3: Evaluate on the TEST set (data the model never saw during training)
probs = model.predict_proba(X_test)[:, 1]   # probability of Malignant
preds = model.predict(X_test)

auroc = roc_auc_score(ydiag_test, probs)
f1 = f1_score(ydiag_test, preds)

print("===== BASELINE MODEL (Logistic Regression) - Risk/Diagnosis =====")
print(f"AUROC: {auroc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nDetailed report:")
print(classification_report(ydiag_test, preds, target_names=['Benign', 'Malignant']))

# Step 4: Save the model so we can compare it later against the deep learning model
joblib.dump(model, 'models/risk/baseline_logistic_regression.pkl')
print("\nBaseline model saved to models/risk/baseline_logistic_regression.pkl")