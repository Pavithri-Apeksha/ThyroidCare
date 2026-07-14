"""
bootstrap_fairness_ci.py
Purpose: Add statistical rigor to the fairness finding by computing bootstrap 95% confidence intervals for AUROC within each ethnicity group.

"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RAW_PATH = 'data/raw/thyroid_cancer_risk_data.csv'
N_BOOTSTRAP = 1000
CONFIDENCE = 0.95

os.makedirs('reports', exist_ok=True)

# Rebuild the exact same split as fairness_analysis_risk.py
df = pd.read_csv(RAW_PATH)
df = df.drop(columns=['Patient_ID'])

binary_cols = ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency',
               'Smoking', 'Obesity', 'Diabetes']
continuous_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']

raw_ethnicity = df['Ethnicity'].copy()

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df = pd.get_dummies(df, columns=['Country', 'Ethnicity'], prefix=['Country', 'Ethnicity'])
onehot_cols = [c for c in df.columns if c.startswith('Country_') or c.startswith('Ethnicity_')]

diagnosis_encoder = LabelEncoder()
df['Diagnosis_enc'] = diagnosis_encoder.fit_transform(df['Diagnosis'])

feature_cols = binary_cols + ['Gender'] + continuous_cols + onehot_cols
X = df[feature_cols].astype(float)
y_diag = df['Diagnosis_enc'].values

X_train, X_temp, ydiag_train, ydiag_temp, eth_train, eth_temp = train_test_split(
    X, y_diag, raw_ethnicity, test_size=0.2, stratify=y_diag, random_state=42
)
X_val, X_test, ydiag_val, ydiag_test, eth_val, eth_test = train_test_split(
    X_temp, ydiag_temp, eth_temp, test_size=0.5, stratify=ydiag_temp, random_state=42
)

scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

model = joblib.load('models/risk/xgboost_model.pkl')
test_probs = model.predict_proba(X_test)[:, 1]

# Bootstrap AUROC confidence intervals per ethnicity group
def bootstrap_auroc_ci(y_true, y_prob, n_boot=1000, ci=0.95, seed=42):
    """Returns (point_estimate, lower_bound, upper_bound)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    point = roc_auc_score(y_true, y_prob)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue  # skip degenerate resamples
        boot_scores.append(roc_auc_score(yt, yp))
    lower = np.percentile(boot_scores, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_scores, (1 + ci) / 2 * 100)
    return point, lower, upper

print(f"Bootstrap AUROC 95% Confidence Intervals by Ethnicity ({N_BOOTSTRAP} resamples)")
print("=" * 75)

results = []
for group in sorted(eth_test.unique()):
    mask = (eth_test.values == group)
    n = mask.sum()
    if n < 30:
        continue
    yt = ydiag_test[mask]
    yp = test_probs[mask]
    point, lower, upper = bootstrap_auroc_ci(yt, yp, n_boot=N_BOOTSTRAP, ci=CONFIDENCE)
    results.append({'Ethnicity': group, 'N': n, 'AUROC': round(point, 3),
                     'CI_Lower': round(lower, 3), 'CI_Upper': round(upper, 3)})
    print(f"{group:16s} N={n:5d}  AUROC={point:.3f}  95% CI=[{lower:.3f}, {upper:.3f}]")

results_df = pd.DataFrame(results)
results_df.to_csv('reports/fairness_ethnicity_bootstrap_ci.csv', index=False)
print("\nSaved: reports/fairness_ethnicity_bootstrap_ci.csv")

# Check whether the Caucasian and Asian CIs overlap 
print("\n" + "=" * 75)
caucasian = results_df[results_df['Ethnicity'] == 'Caucasian']
asian = results_df[results_df['Ethnicity'] == 'Asian']
if not caucasian.empty and not asian.empty:
    c_upper = caucasian['CI_Upper'].values[0]
    a_lower = asian['CI_Lower'].values[0]
    if c_upper < a_lower:
        print("RESULT: Caucasian and Asian confidence intervals do NOT overlap.")
        print("-> The AUROC gap is statistically significant at the 95% confidence level.")
    else:
        print("RESULT: Confidence intervals overlap.")
        print("-> The gap should be reported as suggestive but not conclusively significant.")

print("\n BOOTSTRAP ANALYSIS COMPLETE=")