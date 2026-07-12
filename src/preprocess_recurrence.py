"""
Step 3b: Preprocessing - Recurrence Module (recurrence_dataset.csv)
Purpose: Clean duplicates, encode categoricals, scale continuous features,
split data, apply SMOTE (small dataset -> oversampling helps here),
save encoders/scalers for backend use.

Target: Recurred (binary: No/Yes)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import json
import os

RAW_PATH = 'data/raw/recurrence_dataset.csv'
OUT_DIR = 'data/processed'
ENCODER_DIR = 'models/recurrence'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

df = pd.read_csv(RAW_PATH)
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {before - len(df)} duplicate rows -> {len(df)} rows remain")

continuous_cols = ['Age']
# Gender: F/M, others: No/Yes
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
for col in ['Smoking', 'Hx Smoking', 'Hx Radiothreapy']:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

nominal_cols = ['Thyroid Function', 'Physical Examination', 'Adenopathy',
                'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']

df_encoded = pd.get_dummies(df, columns=nominal_cols, prefix=nominal_cols)
onehot_cols = [c for c in df_encoded.columns
               if any(c.startswith(p + '_') for p in nominal_cols)]

target_encoder = LabelEncoder()
df_encoded['Recurred_enc'] = target_encoder.fit_transform(df_encoded['Recurred'])  # No=0, Yes=1

feature_cols = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy'] + continuous_cols + onehot_cols
X = df_encoded[feature_cols].astype(float)
y = df_encoded['Recurred_enc'].values

# ---- Stratified split (70/15/15 - small dataset needs bigger val/test proportion) ----
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# ---- Scale continuous ----
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_val[continuous_cols] = scaler.transform(X_val[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

print(f"\nBefore SMOTE - Train balance: {np.bincount(y_train)} (0=No, 1=Yes)")

# ---- SMOTE on training data only (small dataset -> oversampling minority helps deep model learn) ----
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE  - Train balance: {np.bincount(y_train_sm)} (0=No, 1=Yes)")

np.savez(f'{OUT_DIR}/recurrence_module_data.npz',
         X_train=X_train_sm.values if hasattr(X_train_sm, 'values') else X_train_sm,
         X_val=X_val.values, X_test=X_test.values,
         y_train=y_train_sm, y_val=y_val, y_test=y_test)

joblib.dump(scaler, f'{ENCODER_DIR}/scaler.pkl')
with open(f'{ENCODER_DIR}/feature_columns.json', 'w') as f:
    json.dump({
        'feature_cols': feature_cols,
        'continuous_cols': continuous_cols,
        'onehot_cols': onehot_cols,
        'nominal_cols': nominal_cols,
        'target_classes': list(target_encoder.classes_),
        'nominal_options': {col: sorted(df[col].dropna().unique().tolist()) for col in nominal_cols}
    }, f, indent=2)

print("\nRECURRENCE MODULE PREPROCESSING DONE")
print(f"Total features: {len(feature_cols)}")
print(f"Train (post-SMOTE): {X_train_sm.shape}, Val: {X_val.shape}, Test: {X_test.shape}")