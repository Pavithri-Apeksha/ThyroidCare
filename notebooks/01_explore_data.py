"""
Data Exploration
Purpose: Understand both datasets before preprocessing - check missing values, data types, class balance, and basic statistics.
"""
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("=" * 70)
print("DATASET 1: Thyroid Cancer Risk Prediction (Risk Module)")
print("=" * 70)
risk_df = pd.read_csv('data/raw/thyroid_cancer_risk_data.csv')
print(f"\nShape: {risk_df.shape}")
print(f"\nColumns & dtypes:\n{risk_df.dtypes}")
print(f"\nMissing values:\n{risk_df.isnull().sum()[risk_df.isnull().sum() > 0]}")
if risk_df.isnull().sum().sum() == 0:
    print("No missing values found.")
print(f"\nDuplicate rows: {risk_df.duplicated().sum()}")
print(f"\nTarget 1 - Thyroid_Cancer_Risk distribution:\n{risk_df['Thyroid_Cancer_Risk'].value_counts()}")
print(f"\nTarget 2 - Diagnosis distribution:\n{risk_df['Diagnosis'].value_counts()}")
print(f"\nSample rows:\n{risk_df.head(3)}")

print("\n\n" + "=" * 70)
print("DATASET 2: Differentiated Thyroid Cancer Recurrence (Recurrence Module)")
print("=" * 70)
rec_df = pd.read_csv('data/raw/recurrence_dataset.csv')
print(f"\nShape: {rec_df.shape}")
print(f"\nColumns & dtypes:\n{rec_df.dtypes}")
print(f"\nMissing values:\n{rec_df.isnull().sum()[rec_df.isnull().sum() > 0]}")
if rec_df.isnull().sum().sum() == 0:
    print("No missing values found.")
print(f"\nDuplicate rows: {rec_df.duplicated().sum()}")
print(f"\nTarget - Recurred distribution:\n{rec_df['Recurred'].value_counts()}")
print(f"\nSample rows:\n{rec_df.head(3)}")