"""
train_dl_recurrence.py
Purpose: Train a Deep Learning model (Neural Network) on the Recurrence
dataset to predict whether cancer will recur (Yes/No).
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import os

torch.manual_seed(42)
os.makedirs('models/recurrence', exist_ok=True)

# ---- Step 1: Load preprocessed data (already SMOTE-balanced) ----
data = np.load('data/processed/recurrence_module_data.npz')
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

Xtr = torch.tensor(X_train, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.float32)
Xval = torch.tensor(X_val, dtype=torch.float32)
Xtest = torch.tensor(X_test, dtype=torch.float32)

# ---- Step 2: Define a smaller Neural Network ----
# Smaller because the dataset is small (only ~360 training rows) -> a huge
# network would just memorize the data instead of learning general patterns (overfitting)
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MLP(X_train.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

# ---- Step 3: Training loop with validation-based best model tracking ----
epochs = 80
best_auroc = 0
best_state = None

for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    logits = model(Xtr)
    loss = loss_fn(logits, ytr)
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(Xval)).numpy()
        val_auroc = roc_auc_score(y_val, val_probs)

    if val_auroc > best_auroc:
        best_auroc = val_auroc
        best_state = model.state_dict()

    if epoch % 15 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | Val AUROC: {val_auroc:.4f}")

print(f"\nBest validation AUROC during training: {best_auroc:.4f}")

# ---- Step 4: Load best model, evaluate on TEST set ----
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_probs = torch.sigmoid(model(Xtest)).numpy()
    test_preds = (test_probs > 0.5).astype(int)

test_auroc = roc_auc_score(y_test, test_probs)
test_f1 = f1_score(y_test, test_preds)

print("\n===== DEEP LEARNING MODEL (MLP) - Recurrence Module =====")
print(f"Test AUROC: {test_auroc:.4f}")
print(f"Test F1: {test_f1:.4f}")
print("\nDetailed report:")
print(classification_report(y_test, test_preds, target_names=['No Recurrence', 'Recurred']))

# ---- Step 5: Save the trained model ----
torch.save(model.state_dict(), 'models/recurrence/dl_model.pt')
print("\nModel saved to models/recurrence/dl_model.pt")