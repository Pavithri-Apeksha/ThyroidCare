"""
fairness_analysis_recurrence.py
Purpose: Fairness audit + calibration + additional evaluation metrics for
the Recurrence module.

  1) Fairness analysis by Gender (the only demographic attribute available
     in this dataset - no Ethnicity/Country columns exist here).
  2) Additional evaluation metrics: AUPRC, Brier score,
     sensitivity @ 90% specificity.
  3) Calibration (Isotonic Regression) + reliability diagram.

Uses the trained XGBoost model (models/recurrence/xgboost_model.pkl).

IMPORTANT CAVEAT: The recurrence test set has only 55 patients (this is a
small clinical dataset, 364 rows total after de-duplication). Splitting by
Gender leaves ~25-30 patients per group, which is too small for a
statistically confident fairness conclusion. This script still reports the
breakdown for completeness/transparency, but the sample-size limitation
should be explicitly acknowledged in the report rather than treated as a
strong finding either way.

This script re-runs the exact same preprocessing + split logic as
preprocess_recurrence.py (same random_state=42), keeping the raw Gender
label alongside the split. X_val/X_test are NOT touched by SMOTE (SMOTE is
only applied to the training set in the original pipeline), so they can be
directly traced back to the raw data with no synthetic-row issues.
No models are retrained.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    recall_score, precision_score, confusion_matrix
)

RAW_PATH = 'data/raw/recurrence_dataset.csv'
os.makedirs('reports', exist_ok=True)

# ============================================================
# STEP 1: Rebuild the exact same preprocessing as preprocess_recurrence.py,
#         keeping the raw Gender label alongside X
# ============================================================
df = pd.read_csv(RAW_PATH)
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {before - len(df)} duplicate rows -> {len(df)} rows remain")

# Keep raw Gender BEFORE encoding, for fairness grouping later
raw_gender = df['Gender'].copy()  # 'M' / 'F'

continuous_cols = ['Age']
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
for col in ['Smoking', 'Hx Smoking', 'Hx Radiothreapy']:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

nominal_cols = ['Thyroid Function', 'Physical Examination', 'Adenopathy',
                'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']

df_encoded = pd.get_dummies(df, columns=nominal_cols, prefix=nominal_cols)
onehot_cols = [c for c in df_encoded.columns
               if any(c.startswith(p + '_') for p in nominal_cols)]

target_encoder = LabelEncoder()
df_encoded['Recurred_enc'] = target_encoder.fit_transform(df_encoded['Recurred'])

feature_cols = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy'] + continuous_cols + onehot_cols
X = df_encoded[feature_cols].astype(float)
y = df_encoded['Recurred_enc'].values

# ---- Same two-stage split, same random_state, but also split the raw Gender label ----
X_train, X_temp, y_train, y_temp, gender_train, gender_temp = train_test_split(
    X, y, raw_gender, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test, gender_val, gender_test = train_test_split(
    X_temp, y_temp, gender_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# ---- Scale continuous (fit on train only, matches original pipeline) ----
# NOTE: fit on pre-SMOTE X_train, identical to original pipeline order
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_val[continuous_cols] = scaler.transform(X_val[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

print(f"Reconstructed split -> Val: {X_val.shape[0]}, Test: {X_test.shape[0]} (should match original: 55, 55)")

# ============================================================
# STEP 2: Load trained XGBoost model, get predictions
# ============================================================
model = joblib.load('models/recurrence/xgboost_model.pkl')

val_probs = model.predict_proba(X_val)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs > 0.5).astype(int)

# ============================================================
# STEP 3: FAIRNESS ANALYSIS - Gender only (only demographic column available)
# ============================================================
def fairness_table(raw_group_series, y_true, y_prob, y_pred):
    rows = []
    for group in sorted(raw_group_series.unique()):
        mask = (raw_group_series.values == group)
        n = mask.sum()
        yt, yp, ypred = y_true[mask], y_prob[mask], y_pred[mask]
        try:
            auroc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan
        except Exception:
            auroc = np.nan
        rec = recall_score(yt, ypred, zero_division=0)
        prec = precision_score(yt, ypred, zero_division=0)
        rows.append({'Group': group, 'N': n, 'Recurred_Recall': round(rec, 3),
                      'Recurred_Precision': round(prec, 3),
                      'AUROC': round(auroc, 3) if not np.isnan(auroc) else 'N/A'})
    return pd.DataFrame(rows).sort_values('N', ascending=False)

print("\n===== FAIRNESS: Gender (CAUTION: small n per group, see caveat in docstring) =====")
gender_tbl = fairness_table(gender_test, y_test, test_probs, test_preds)
print(gender_tbl.to_string(index=False))
gender_tbl.to_csv('reports/fairness_recurrence_gender.csv', index=False)
print("Saved: reports/fairness_recurrence_gender.csv")

# ============================================================
# STEP 4: EXTRA METRICS - AUPRC, Brier score, sensitivity @ 90% specificity
# ============================================================
auprc = average_precision_score(y_test, test_probs)
brier = brier_score_loss(y_test, test_probs)

best_sens = 0
for t in np.linspace(0.01, 0.99, 99):
    preds_t = (test_probs > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_t).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    if specificity >= 0.90 and sensitivity > best_sens:
        best_sens = sensitivity

print("\n===== EXTRA METRICS (Recurrence - XGBoost) =====")
print(f"AUPRC: {auprc:.4f}")
print(f"Brier Score: {brier:.4f} (lower is better, 0 = perfect)")
print(f"Sensitivity @ 90% Specificity: {best_sens:.4f}")

with open('reports/extra_metrics_recurrence.json', 'w') as f:
    json.dump({'AUPRC': round(auprc, 4), 'Brier_Score': round(brier, 4),
                'Sensitivity_at_90pct_Specificity': round(best_sens, 4)}, f, indent=2)
print("Saved: reports/extra_metrics_recurrence.json")

# ============================================================
# STEP 5: CALIBRATION - Isotonic Regression (fit on val, apply to test)
# ============================================================
# CAUTION: val set is only 55 rows, so the isotonic fit itself is fairly
# noisy here - report this as a limitation, don't overclaim precision.
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(val_probs, y_val)
test_probs_calibrated = iso.predict(test_probs)

brier_before = brier_score_loss(y_test, test_probs)
brier_after = brier_score_loss(y_test, test_probs_calibrated)
print(f"\nBrier Score BEFORE calibration: {brier_before:.4f}")
print(f"Brier Score AFTER calibration:  {brier_after:.4f}")
print("(Note: calibration fit on only 55 validation samples - treat as exploratory, not definitive.)")

def reliability_curve(y_true, y_prob, n_bins=5):  # fewer bins - small sample
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    mean_pred, mean_true = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            mean_pred.append(y_prob[mask].mean())
            mean_true.append(y_true[mask].mean())
    return mean_pred, mean_true

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
mp, mt = reliability_curve(y_test, test_probs)
plt.plot(mp, mt, 'o-', label=f'Before calibration (Brier={brier_before:.3f})')
mp2, mt2 = reliability_curve(y_test, test_probs_calibrated)
plt.plot(mp2, mt2, 's-', label=f'After calibration (Brier={brier_after:.3f})')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Observed Fraction Positive')
plt.title('Calibration: Reliability Diagram (Recurrence Model)\n(n=55, exploratory)')
plt.legend()
plt.tight_layout()
plt.savefig('reports/calibration_curve_recurrence.png', dpi=150)
plt.close()
print("Saved: reports/calibration_curve_recurrence.png")

joblib.dump(iso, 'models/recurrence/isotonic_calibrator.pkl')
print("Saved: models/recurrence/isotonic_calibrator.pkl")

print("\n===== ANALYSIS COMPLETE =====")