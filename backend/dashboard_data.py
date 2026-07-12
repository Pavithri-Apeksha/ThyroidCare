"""
backend/dashboard_data.py
Purpose: Data-loading logic for the Model Insights dashboard page.
Reads the already-generated fairness/calibration/SHAP artifacts from
reports/ and returns clean data structures. No Streamlit/UI code here -
keeps the dashboard page thin (separation of concerns).
"""
import pandas as pd
import json
import os

REPORTS_DIR = 'reports'

# ---- Known evaluation metrics from training logs (static summary table) ----
# These come directly from the console output when each model was trained/
# evaluated (train_xgboost_risk.py, train_dl_risk_v2.py, etc.) - not
# re-computed here, just organized for display.
MODEL_COMPARISON = {
    "Risk / Diagnosis": [
        {"Model": "Logistic Regression", "AUROC": 0.665, "F1": 0.444},
        {"Model": "XGBoost", "AUROC": 0.700, "F1": 0.551},
        {"Model": "Deep Learning (MLP)", "AUROC": 0.698, "F1": 0.554},
    ],
    "Recurrence": [
        {"Model": "Logistic Regression", "AUROC": 0.968, "F1": 0.938},
        {"Model": "XGBoost", "AUROC": 0.960, "F1": 0.938},
        {"Model": "Deep Learning (MLP)", "AUROC": 0.955, "F1": 0.909},
    ],
}

def get_model_comparison():
    """Returns dict of {module_name: list of {Model, AUROC, F1}}."""
    return MODEL_COMPARISON

def load_extra_metrics():
    """Returns dict with 'risk' and 'recurrence' extra metrics (AUPRC,
    Brier score, sensitivity@90specificity), or None if a file is missing."""
    out = {}
    for key, fname in [('risk', 'extra_metrics.json'), ('recurrence', 'extra_metrics_recurrence.json')]:
        path = os.path.join(REPORTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                out[key] = json.load(f)
        else:
            out[key] = None
    return out

def load_fairness_tables():
    """Returns dict of DataFrames (or None if missing) for each fairness
    breakdown that was generated."""
    files = {
        'risk_gender': 'fairness_gender.csv',
        'risk_ethnicity': 'fairness_ethnicity.csv',
        'risk_country': 'fairness_country.csv',
        'recurrence_gender': 'fairness_recurrence_gender.csv',
    }
    out = {}
    for key, fname in files.items():
        path = os.path.join(REPORTS_DIR, fname)
        out[key] = pd.read_csv(path) if os.path.exists(path) else None
    return out

def get_chart_paths():
    """Returns dict of {label: path} for chart images, only including
    ones that actually exist on disk."""
    candidates = {
        'Ethnicity Fairness (Risk)': 'fairness_ethnicity_chart.png',
        'Calibration Curve (Risk)': 'calibration_curve.png',
        'Calibration Curve (Recurrence)': 'calibration_curve_recurrence.png',
        'SHAP Summary (Risk)': 'shap_summary_risk.png',
        'SHAP Patient Example (Risk)': 'shap_patient_example_risk.png',
        'SHAP Summary (Recurrence)': 'shap_summary_recurrence.png',
        'SHAP Patient Example (Recurrence)': 'shap_patient_example_recurrence.png',
    }
    out = {}
    for label, fname in candidates.items():
        path = os.path.join(REPORTS_DIR, fname)
        if os.path.exists(path):
            out[label] = path
    return out

def load_audit_log(limit=50):
    """Delegates to backend.audit_log so the dashboard has one place to
    pull prediction history from."""
    from backend.audit_log import read_log
    return read_log(limit=limit)