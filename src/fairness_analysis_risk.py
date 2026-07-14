"""
fairness_analysis_risk.py
Purpose: Fairness audit + calibration + additional evaluation metrics for the Risk (Diagnosis) module.
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
    precision_recall_curve, recall_score, precision_score, confusion_matrix
)

RAW_PATH = 'data/raw/thyroid_cancer_risk_data.csv'
os.makedirs('reports', exist_ok=True)

# Rebuild the exact same preprocessing as preprocess_risk.py, but keep raw Gender/Ethnicity/Country alongside X

df = pd.read_csv(RAW_PATH)
df = df.drop(columns=['Patient_ID'])

binary_cols = ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency',
               'Smoking', 'Obesity', 'Diabetes']
continuous_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']

# Keep raw copies BEFORE any encoding, for fairness grouping later
raw_gender = df['Gender'].copy()
raw_country = df['Country'].copy()
raw_ethnicity = df['Ethnicity'].copy()

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

df = pd.get_dummies(df, columns=['Country', 'Ethnicity'], prefix=['Country', 'Ethnicity'])
onehot_cols = [c for c in df.columns if c.startswith('Country_') or c.startswith('Ethnicity_')]

diagnosis_encoder = LabelEncoder()
df['Diagnosis_enc'] = diagnosis_encoder.fit_transform(df['Diagnosis'])
risk_order = ['Low', 'Medium', 'High']
df['Risk_enc'] = df['Thyroid_Cancer_Risk'].map({v: i for i, v in enumerate(risk_order)})

feature_cols = binary_cols + ['Gender'] + continuous_cols + onehot_cols
X = df[feature_cols].astype(float)
y_diag = df['Diagnosis_enc'].values
y_risk = df['Risk_enc'].values

# Same two-stage split, same random_state, but also split the raw labels
(X_train, X_temp, ydiag_train, ydiag_temp, yrisk_train, yrisk_temp,
 gender_train, gender_temp, country_train, country_temp,
 eth_train, eth_temp) = train_test_split(
    X, y_diag, y_risk, raw_gender, raw_country, raw_ethnicity,
    test_size=0.2, stratify=y_diag, random_state=42
)
(X_val, X_test, ydiag_val, ydiag_test, yrisk_val, yrisk_test,
 gender_val, gender_test, country_val, country_test,
 eth_val, eth_test) = train_test_split(
    X_temp, ydiag_temp, yrisk_temp, gender_temp, country_temp, eth_temp,
    test_size=0.5, stratify=ydiag_temp, random_state=42
)

# Scale continuous features (fit on train only, matches original pipeline)
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_val[continuous_cols] = scaler.transform(X_val[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

print(f"Reconstructed split -> Test set size: {X_test.shape[0]} (should match original: 21270)")

# Load trained XGBoost model, get predictions
model = joblib.load('models/risk/xgboost_model.pkl')

val_probs = model.predict_proba(X_val)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs > 0.5).astype(int)

# FAIRNESS ANALYSIS - Gender / Ethnicity / Country
def fairness_table(raw_group_series, y_true, y_prob, y_pred, min_group_size=30):
    rows = []
    for group in sorted(raw_group_series.unique()):
        mask = (raw_group_series.values == group)
        n = mask.sum()
        if n < min_group_size:
            continue  # skip tiny groups, not statistically meaningful
        yt, yp, ypred = y_true[mask], y_prob[mask], y_pred[mask]
        try:
            auroc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan
        except Exception:
            auroc = np.nan
        rec = recall_score(yt, ypred, zero_division=0)
        prec = precision_score(yt, ypred, zero_division=0)
        rows.append({'Group': group, 'N': n, 'Malignant_Recall': round(rec, 3),
                      'Malignant_Precision': round(prec, 3),
                      'AUROC': round(auroc, 3) if not np.isnan(auroc) else 'N/A'})
    return pd.DataFrame(rows).sort_values('N', ascending=False)

print("\n===== FAIRNESS: Gender =====")
gender_tbl = fairness_table(gender_test, ydiag_test, test_probs, test_preds)
print(gender_tbl.to_string(index=False))

print("\n===== FAIRNESS: Ethnicity =====")
eth_tbl = fairness_table(eth_test, ydiag_test, test_probs, test_preds)
print(eth_tbl.to_string(index=False))

print("\n===== FAIRNESS: Country (top 10 by sample size) =====")
country_tbl = fairness_table(country_test, ydiag_test, test_probs, test_preds)
country_tbl = country_tbl.sort_values('N', ascending=False).head(10)
print(country_tbl.to_string(index=False))

# Save all fairness tables to CSV for the report
gender_tbl.to_csv('reports/fairness_gender.csv', index=False)
eth_tbl.to_csv('reports/fairness_ethnicity.csv', index=False)
country_tbl.to_csv('reports/fairness_country.csv', index=False)
print("\nSaved: reports/fairness_gender.csv, fairness_ethnicity.csv, fairness_country.csv")

# Fairness bar chart (Malignant Recall by Ethnicity)
plt.figure(figsize=(8, 5))
plt.bar(eth_tbl['Group'], eth_tbl['Malignant_Recall'], color='steelblue')
overall_recall = recall_score(ydiag_test, test_preds)
plt.axhline(overall_recall, color='red', linestyle='--',
            label=f'Overall Recall ({overall_recall:.2f})')
plt.ylabel('Malignant Recall (Sensitivity)')
plt.title('Model Fairness: Malignant Recall by Ethnicity')
plt.xticks(rotation=30, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('reports/fairness_ethnicity_chart.png', dpi=150)
plt.close()
print("Saved: reports/fairness_ethnicity_chart.png")

# EXTRA METRICS - AUPRC, Brier score, sensitivity @ 90% specificity
auprc = average_precision_score(ydiag_test, test_probs)
brier = brier_score_loss(ydiag_test, test_probs)

# Sensitivity at 90% specificity: scan thresholds, find best sensitivity among thresholds where specificity >= 0.90
best_sens = 0
for t in np.linspace(0.01, 0.99, 99):
    preds_t = (test_probs > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(ydiag_test, preds_t).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    if specificity >= 0.90 and sensitivity > best_sens:
        best_sens = sensitivity

print("\n===== EXTRA METRICS (Risk/Diagnosis - XGBoost) =====")
print(f"AUPRC: {auprc:.4f}")
print(f"Brier Score: {brier:.4f} (lower is better, 0 = perfect)")
print(f"Sensitivity @ 90% Specificity: {best_sens:.4f}")

with open('reports/extra_metrics.json', 'w') as f:
    json.dump({'AUPRC': round(auprc, 4), 'Brier_Score': round(brier, 4),
                'Sensitivity_at_90pct_Specificity': round(best_sens, 4)}, f, indent=2)
print("Saved: reports/extra_metrics.json")

# CALIBRATION - Isotonic Regression (fit on val, apply to test)
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(val_probs, ydiag_val)
test_probs_calibrated = iso.predict(test_probs)

brier_before = brier_score_loss(ydiag_test, test_probs)
brier_after = brier_score_loss(ydiag_test, test_probs_calibrated)
print(f"\nBrier Score BEFORE calibration: {brier_before:.4f}")
print(f"Brier Score AFTER calibration:  {brier_after:.4f}")

# Reliability diagram (before vs after)
def reliability_curve(y_true, y_prob, n_bins=10):
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
mp, mt = reliability_curve(ydiag_test, test_probs)
plt.plot(mp, mt, 'o-', label=f'Before calibration (Brier={brier_before:.3f})')
mp2, mt2 = reliability_curve(ydiag_test, test_probs_calibrated)
plt.plot(mp2, mt2, 's-', label=f'After calibration (Brier={brier_after:.3f})')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Observed Fraction Positive')
plt.title('Calibration: Reliability Diagram (Risk/Diagnosis Model)')
plt.legend()
plt.tight_layout()
plt.savefig('reports/calibration_curve.png', dpi=150)
plt.close()
print("Saved: reports/calibration_curve.png")

# Save the calibrator for use in the web app later
joblib.dump(iso, 'models/risk/isotonic_calibrator.pkl')
print("Saved: models/risk/isotonic_calibrator.pkl")

print("\n ANALYSIS COMPLETE")