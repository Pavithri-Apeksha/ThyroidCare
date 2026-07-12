"""
explain_shap_recurrence.py
Purpose: Generate SHAP explainability plots for the Recurrence model.
Shows WHICH features drive recurrence risk predictions.
"""
import numpy as np
import torch
import torch.nn as nn
import shap
import json
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(42)
os.makedirs('reports', exist_ok=True)

# ---- Step 1: Load data + feature names ----
data = np.load('data/processed/recurrence_module_data.npz')
X_train, X_test = data['X_train'], data['X_test']
with open('models/recurrence/feature_columns.json') as f:
    meta = json.load(f)
feature_names = meta['feature_cols']

# ---- Step 2: Rebuild model architecture (confirmed from dl_model.pt state_dict) ----
# net.0 = Linear(51, 32)
# net.1 = ReLU
# net.2 = Dropout
# net.3 = Linear(32, 16)
# net.4 = ReLU
# net.5 = Linear(16, 1)
# (No BatchNorm in this checkpoint)
class RecurrenceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = RecurrenceMLP(X_train.shape[1])
model.load_state_dict(torch.load('models/recurrence/dl_model.pt'))
model.eval()

def predict_recur(x_numpy):
    with torch.no_grad():
        xb = torch.tensor(x_numpy, dtype=torch.float32)
        logits = model(xb)
        probs = torch.sigmoid(logits).numpy()
    return probs

# ---- Step 3: SHAP explainer ----
background = X_train[np.random.choice(len(X_train), min(50, len(X_train)), replace=False)]
explainer = shap.KernelExplainer(predict_recur, background)

# Small test set (55 rows) - explain all of it
sample = X_test
print(f"Computing SHAP values for {len(sample)} patients... this takes a minute.")
shap_values = explainer.shap_values(sample, nsamples=100)

# ---- Step 4: Global feature importance ----
plt.figure()
shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('reports/shap_summary_recurrence.png', dpi=150)
plt.close()
print("Saved: reports/shap_summary_recurrence.png")

# ---- Step 5: Individual patient explanation ----
plt.figure()
shap.waterfall_plot(
    shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                      data=sample[0], feature_names=feature_names),
    show=False
)
plt.tight_layout()
plt.savefig('reports/shap_patient_example_recurrence.png', dpi=150)
plt.close()
print("Saved: reports/shap_patient_example_recurrence.png")

# ---- Step 6: Ranked feature importance ----
mean_abs_shap = np.abs(shap_values).mean(axis=0)
ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])
print("\nTop 10 most important features for Recurrence prediction:")
for name, val in ranking[:10]:
    print(f"  {name}: {val:.4f}")