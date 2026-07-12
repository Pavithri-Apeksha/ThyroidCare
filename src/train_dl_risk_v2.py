"""
train_dl_risk.py
Purpose: Train a Deep Learning model (Multi-Task Neural Network) on the
Risk dataset. ONE model predicts BOTH targets at once:
  - Diagnosis (Benign/Malignant)  -> primary, strong task
  - Thyroid_Cancer_Risk (Low/Medium/High) -> secondary task

NOTE: The 'Medium' risk class is very hard to separate from 'Low' using
the available features (confirmed true for XGBoost too, not a bug) -
this is a documented dataset limitation, discussed in the report.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import os

torch.manual_seed(42)
os.makedirs('models/risk', exist_ok=True)

# ---- Step 1: Load preprocessed data ----
data = np.load('data/processed/risk_module_data.npz')
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
ydiag_train, ydiag_val, ydiag_test = data['ydiag_train'], data['ydiag_val'], data['ydiag_test']
yrisk_train, yrisk_val, yrisk_test = data['yrisk_train'], data['yrisk_val'], data['yrisk_test']

Xtr = torch.tensor(X_train, dtype=torch.float32)
ydiag_tr = torch.tensor(ydiag_train, dtype=torch.float32)
yrisk_tr = torch.tensor(yrisk_train, dtype=torch.long)
Xval = torch.tensor(X_val, dtype=torch.float32)
Xtest = torch.tensor(X_test, dtype=torch.float32)

# ---- Step 2: Define the Multi-Task Neural Network ----
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.diag_head = nn.Linear(32, 1)   # binary output: Malignant probability
        self.risk_head = nn.Linear(32, 3)   # 3-class output: Low/Medium/High

    def forward(self, x):
        h = self.shared(x)
        return self.diag_head(h).squeeze(-1), self.risk_head(h)

model = MultiTaskMLP(X_train.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ---- Step 3: Loss functions ----
diag_pos_weight = torch.tensor((ydiag_train == 0).sum() / (ydiag_train == 1).sum())
diag_loss_fn = nn.BCEWithLogitsLoss(pos_weight=diag_pos_weight)

# Moderate (sqrt-scaled) class weighting for risk head - softer than full
# balancing, which was found to overcorrect and destroy the Low class instead.
counts = np.bincount(yrisk_train)
risk_class_weights = np.sqrt(counts.sum() / (3 * counts))
risk_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(risk_class_weights, dtype=torch.float32))

# ---- Step 4: Training loop with validation-based best model tracking ----
batch_size = 1024
n = Xtr.shape[0]
epochs = 30
best_score = -1
best_state = None

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(n)
    total_loss = 0
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb, ydb, yrb = Xtr[idx], ydiag_tr[idx], yrisk_tr[idx]
        opt.zero_grad()
        diag_logits, risk_logits = model(xb)
        loss = diag_loss_fn(diag_logits, ydb) + risk_loss_fn(risk_logits, yrb)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        diag_logits, risk_logits = model(Xval)
        diag_probs = torch.sigmoid(diag_logits).numpy()
        risk_preds = torch.argmax(risk_logits, dim=1).numpy()
        val_diag_auroc = roc_auc_score(ydiag_val, diag_probs)
        val_risk_f1 = f1_score(yrisk_val, risk_preds, average='macro')
        combined_score = val_diag_auroc + val_risk_f1

    if combined_score > best_score:
        best_score = combined_score
        best_state = model.state_dict()

    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:2d} | Loss: {total_loss:.2f} | Diag AUROC: {val_diag_auroc:.4f} | Risk macro-F1: {val_risk_f1:.4f}")

print(f"\nBest combined validation score: {best_score:.4f}")

# ---- Step 5: Load best model, evaluate on TEST set ----
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    diag_logits, risk_logits = model(Xtest)
    diag_probs = torch.sigmoid(diag_logits).numpy()
    diag_preds = (diag_probs > 0.5).astype(int)
    risk_preds = torch.argmax(risk_logits, dim=1).numpy()

test_auroc = roc_auc_score(ydiag_test, diag_probs)
test_f1 = f1_score(ydiag_test, diag_preds)

print("\n===== DEEP LEARNING MODEL (Multi-Task MLP) - Risk Module =====")
print(f"Diagnosis - Test AUROC: {test_auroc:.4f}")
print(f"Diagnosis - Test F1: {test_f1:.4f}")
print("\nDiagnosis Detailed Report:")
print(classification_report(ydiag_test, diag_preds, target_names=['Benign', 'Malignant']))
print("\nRisk Level Detailed Report:")
print(classification_report(yrisk_test, risk_preds, target_names=['Low', 'Medium', 'High'], zero_division=0))
print("\nNOTE: 'Medium' class recall is expected to be low - both XGBoost and")
print("this DL model struggle to separate it from 'Low' using available features.")
print("This is a documented dataset limitation (see research report discussion).")

# ---- Step 6: Save the trained model ----
torch.save(model.state_dict(), 'models/risk/dl_model_v2.pt')
print("\nModel saved to models/risk/dl_model_v2.pt")