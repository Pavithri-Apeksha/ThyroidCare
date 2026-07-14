"""
train_xgboost_recurrence.py
Purpose: Train XGBoost as a stronger baseline for the Recurrence dataset.
"""
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import joblib
import os

os.makedirs('models/recurrence', exist_ok=True)

# Load preprocessed data (already SMOTE-balanced from preprocessing)
data = np.load('data/processed/recurrence_module_data.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Train XGBoost
# Smaller max_depth here because the dataset is small (364 rows) -> prevents overfitting
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

#  Evaluate on test set
probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

auroc = roc_auc_score(y_test, probs)
f1 = f1_score(y_test, preds)

print("===== XGBOOST MODEL - Recurrence =====")
print(f"AUROC: {auroc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nDetailed report:")
print(classification_report(y_test, preds, target_names=['No Recurrence', 'Recurred']))

# Save the model
joblib.dump(model, 'models/recurrence/xgboost_model.pkl')
print("\nModel saved to models/recurrence/xgboost_model.pkl")