"""
train_baseline_recurrence.py
Purpose: Train a simple baseline model (Logistic Regression) on the
Recurrence dataset, for later comparison against the deep learning model.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import joblib
import os

os.makedirs('models/recurrence', exist_ok=True)

# Step 1: Load the already-preprocessed data (created by preprocess_recurrence.py)
data = np.load('data/processed/recurrence_module_data.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Step 2: Train Logistic Regression
# (SMOTE already balanced the training data in preprocessing, so no class_weight needed here)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 3: Evaluate on the TEST set
probs = model.predict_proba(X_test)[:, 1]   # probability of Recurred = Yes
preds = model.predict(X_test)

auroc = roc_auc_score(y_test, probs)
f1 = f1_score(y_test, preds)

print("===== BASELINE MODEL (Logistic Regression) - Recurrence =====")
print(f"AUROC: {auroc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nDetailed report:")
print(classification_report(y_test, preds, target_names=['No Recurrence', 'Recurred']))

# Step 4: Save the model for comparison later
joblib.dump(model, 'models/recurrence/baseline_logistic_regression.pkl')
print("\nBaseline model saved to models/recurrence/baseline_logistic_regression.pkl")