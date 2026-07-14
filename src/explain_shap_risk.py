"""
explain_shap_risk.py
Purpose: Generate SHAP explainability plots for the Risk (Diagnosis) model. Shows WHICH features drive predictions - required by the proposal fortransparent, trustworthy clinical decision support.
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

# Load data + feature names
data = np.load('data/processed/risk_module_data.npz')
X_train, X_test = data['X_train'], data['X_test']
with open('models/risk/feature_columns.json') as f:
    meta = json.load(f)
feature_names = meta['feature_cols']

# Rebuild model architecture and load trained weights
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.diag_head = nn.Linear(32, 1)
        self.risk_head = nn.Linear(32, 3)
    def forward(self, x):
        h = self.shared(x)
        return self.diag_head(h).squeeze(-1), self.risk_head(h)

model = MultiTaskMLP(X_train.shape[1])
model.load_state_dict(torch.load('models/risk/dl_model_v2.pt'))
model.eval()

# Wrap the model so SHAP sees a simple function: numpy in -> probability out
def predict_diag(x_numpy):
    with torch.no_grad():
        xb = torch.tensor(x_numpy, dtype=torch.float32)
        diag_logits, _ = model(xb)
        probs = torch.sigmoid(diag_logits).numpy()
    return probs

# Set up SHAP explainer 
# background = a small representative sample used as the "reference point"
background = X_train[np.random.choice(len(X_train), 50, replace=False)]
explainer = shap.KernelExplainer(predict_diag, background)

# sample = the patients we actually want to explain (300 for speed)
sample_idx = np.random.choice(len(X_test), 300, replace=False)
sample = X_test[sample_idx]

print("Computing SHAP values... this takes a few minutes.")
shap_values = explainer.shap_values(sample, nsamples=100)

# Global feature importance (summary plot)
plt.figure()
shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('reports/shap_summary_risk.png', dpi=150)
plt.close()
print("Saved: reports/shap_summary_risk.png")

# Individual patient explanation (first patient in sample) 
plt.figure()
shap.waterfall_plot(
    shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                      data=sample[0], feature_names=feature_names),
    show=False
)
plt.tight_layout()
plt.savefig('reports/shap_patient_example_risk.png', dpi=150)
plt.close()
print("Saved: reports/shap_patient_example_risk.png")

# Print ranked feature importance (mean absolute SHAP value)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])
print("\nTop 10 most important features for Diagnosis prediction:")
for name, val in ranking[:10]:
    print(f"  {name}: {val:.4f}")