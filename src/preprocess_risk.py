"""
Preprocessing - Risk Module (thyroid_cancer_risk_data.csv)
Purpose: Encode categoricals, scale continuous features, split data,save encoders/scalers for later use in the backend API.

Targets (dual):
  - Diagnosis (binary: Benign/Malignant)
  - Thyroid_Cancer_Risk (3-class: Low/Medium/High)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
import os

RAW_PATH = 'data/raw/thyroid_cancer_risk_data.csv'
OUT_DIR = 'data/processed'
ENCODER_DIR = 'models/risk'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

df = pd.read_csv(RAW_PATH)
df = df.drop(columns=['Patient_ID'])  # not predictive, just an ID

# Define column groups
binary_cols = ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency',
               'Smoking', 'Obesity', 'Diabetes']  # Yes/No
nominal_cols = ['Gender', 'Country', 'Ethnicity']  # multi-category, no order
continuous_cols = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']

# Encode binary Yes/No columns as 0/1
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

#  Encode Gender separately (Male/Female)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

#  One-hot encode nominal columns (Country, Ethnicity)
df = pd.get_dummies(df, columns=['Country', 'Ethnicity'], prefix=['Country', 'Ethnicity'])
onehot_cols = [c for c in df.columns if c.startswith('Country_') or c.startswith('Ethnicity_')]

# Encode targets
diagnosis_encoder = LabelEncoder()
df['Diagnosis_enc'] = diagnosis_encoder.fit_transform(df['Diagnosis'])  # Benign=0, Malignant=1

risk_order = ['Low', 'Medium', 'High']
df['Risk_enc'] = df['Thyroid_Cancer_Risk'].map({v: i for i, v in enumerate(risk_order)})

# Feature columns used for training
feature_cols = binary_cols + ['Gender'] + continuous_cols + onehot_cols

X = df[feature_cols].astype(float)
y_diag = df['Diagnosis_enc'].values
y_risk = df['Risk_enc'].values

# Stratified split (80/10/10) using Diagnosis as stratify key 
X_train, X_temp, ydiag_train, ydiag_temp, yrisk_train, yrisk_temp = train_test_split(
    X, y_diag, y_risk, test_size=0.2, stratify=y_diag, random_state=42
)
X_val, X_test, ydiag_val, ydiag_test, yrisk_val, yrisk_test = train_test_split(
    X_temp, ydiag_temp, yrisk_temp, test_size=0.5, stratify=ydiag_temp, random_state=42
)

# Scale continuous features (fit on train only, prevent leakage)
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_val[continuous_cols] = scaler.transform(X_val[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

# Save processed arrays
np.savez(f'{OUT_DIR}/risk_module_data.npz',
         X_train=X_train.values, X_val=X_val.values, X_test=X_test.values,
         ydiag_train=ydiag_train, ydiag_val=ydiag_val, ydiag_test=ydiag_test,
         yrisk_train=yrisk_train, yrisk_val=yrisk_val, yrisk_test=yrisk_test)

#Save encoders/scaler/feature list for backend use
joblib.dump(scaler, f'{ENCODER_DIR}/scaler.pkl')
with open(f'{ENCODER_DIR}/feature_columns.json', 'w') as f:
    json.dump({
        'feature_cols': feature_cols,
        'binary_cols': binary_cols,
        'continuous_cols': continuous_cols,
        'onehot_cols': onehot_cols,
        'diagnosis_classes': list(diagnosis_encoder.classes_),
        'risk_classes': risk_order,
        'countries': sorted(df.filter(like='Country_').columns.str.replace('Country_', '').tolist()),
        'ethnicities': sorted(df.filter(like='Ethnicity_').columns.str.replace('Ethnicity_', '').tolist()),
    }, f, indent=2)

print("RISK MODULE PREPROCESSING DONE")
print(f"Total features: {len(feature_cols)}")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Diagnosis train balance: {np.bincount(ydiag_train)} (0=Benign, 1=Malignant)")
print(f"Risk train balance: {np.bincount(yrisk_train)} (0=Low, 1=Medium, 2=High)")