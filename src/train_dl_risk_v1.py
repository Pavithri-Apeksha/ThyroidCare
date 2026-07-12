"""
train_dl_risk.py
Purpose: Train a Deep Learning model (Multi-Task Neural Network) on the
Risk dataset. ONE model predicts BOTH targets at once:
  - Diagnosis (Benign/Malignant)
  - Thyroid_Cancer_Risk (Low/Medium/High)
This is the CORE deep learning component of the research.
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
# "Shared" layers learn general patterns from the data.
# Then it SPLITS into two "heads" - one for each prediction task.
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

# ---- Step 3: Loss functions (handle class imbalance) ----
diag_pos_weight = torch.tensor((ydiag_train == 0).sum() / (ydiag_train == 1).sum())
diag_loss_fn = nn.BCEWithLogitsLoss(pos_weight=diag_pos_weight)
risk_loss_fn = nn.CrossEntropyLoss()

# ---- Step 4: Training loop with validation-based best model tracking ----
batch_size = 1024
n = Xtr.shape[0]
epochs = 30
best_auroc = 0
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

    # Check performance on validation set (data not used for training)
    model.eval()
    with torch.no_grad():
        diag_logits, _ = model(Xval)
        diag_probs = torch.sigmoid(diag_logits).numpy()
        val_auroc = roc_auc_score(ydiag_val, diag_probs)

    if val_auroc > best_auroc:
        best_auroc = val_auroc
        best_state = model.state_dict()  # save the best version so far

    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:2d} | Loss: {total_loss:.2f} | Val AUROC: {val_auroc:.4f}")

print(f"\nBest validation AUROC during training: {best_auroc:.4f}")

# ---- Step 5: Load best model and evaluate on TEST set (final, unbiased score) ----
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
print(classification_report(yrisk_test, risk_preds, target_names=['Low', 'Medium', 'High']))

# ---- Step 6: Save the trained model ----
torch.save(model.state_dict(), 'models/risk/dl_model_v1.pt')
print("\nModel saved to models/risk/dl_model_v1.pt")