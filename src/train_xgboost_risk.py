"""
train_xgboost_risk.py
Purpose: Train XGBoost (a strong tree-based ML algorithm) as a STRONGER baseline than Logistic Regression, for the Risk dataset (Diagnosis target). Tabular data often works very well with XGBoost, so this gives a fair, tough comparison point for our deep learning model later.
"""
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import joblib
import os

os.makedirs('models/risk', exist_ok=True)

# Load preprocessed data
data = np.load('data/processed/risk_module_data.npz')
X_train, X_test = data['X_train'], data['X_test']
ydiag_train, ydiag_test = data['ydiag_train'], data['ydiag_test']

# Handle class imbalance
# scale_pos_weight tells XGBoost how much more to "care" about the minority class (Malignant)
scale_pos_weight = (ydiag_train == 0).sum() / (ydiag_train == 1).sum()

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,       # number of trees
    max_depth=6,             # how deep each tree can grow
    learning_rate=0.1,       # how fast the model learns
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, ydiag_train)

# Evaluate on test set
probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

auroc = roc_auc_score(ydiag_test, probs)
f1 = f1_score(ydiag_test, preds)

print("===== XGBOOST MODEL - Risk/Diagnosis =====")
print(f"AUROC: {auroc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nDetailed report:")
print(classification_report(ydiag_test, preds, target_names=['Benign', 'Malignant']))

# Save the model
joblib.dump(model, 'models/risk/xgboost_model.pkl')
print("\nModel saved to models/risk/xgboost_model.pkl")