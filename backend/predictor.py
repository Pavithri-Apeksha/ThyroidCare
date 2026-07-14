"""
backend/predictor.py

"""
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import shap
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root


# Model architecture definitions (must match training scripts exactly)
class RiskMultiTaskMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.diag_head = nn.Linear(32, 1)
        self.risk_head = nn.Linear(32, 3)

    def forward(self, x):
        h = self.shared(x)
        return self.diag_head(h).squeeze(-1), self.risk_head(h)


class RecurrenceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Load everything ONCE at import time (fast repeated predictions)
def _load_json(path):
    with open(path) as f:
        return json.load(f)


RISK_META = _load_json(f'{BASE_DIR}/models/risk/feature_columns.json')
RISK_SCALER = joblib.load(f'{BASE_DIR}/models/risk/scaler.pkl')
RISK_MODEL = RiskMultiTaskMLP(len(RISK_META['feature_cols']))
RISK_MODEL.load_state_dict(torch.load(f'{BASE_DIR}/models/risk/dl_model_v2.pt', map_location='cpu'))
RISK_MODEL.eval()
try:
    RISK_CALIBRATOR = joblib.load(f'{BASE_DIR}/models/risk/isotonic_calibrator.pkl')
except FileNotFoundError:
    RISK_CALIBRATOR = None

REC_META = _load_json(f'{BASE_DIR}/models/recurrence/feature_columns.json')
REC_SCALER = joblib.load(f'{BASE_DIR}/models/recurrence/scaler.pkl')
REC_MODEL = RecurrenceMLP(len(REC_META['feature_cols']))
REC_MODEL.load_state_dict(torch.load(f'{BASE_DIR}/models/recurrence/dl_model.pt', map_location='cpu'))
REC_MODEL.eval()
try:
    REC_CALIBRATOR = joblib.load(f'{BASE_DIR}/models/recurrence/isotonic_calibrator.pkl')
except FileNotFoundError:
    REC_CALIBRATOR = None


# RISK MODULE
def build_risk_vector(patient: dict) -> np.ndarray:
    """
    patient dict expected keys (raw, human-readable values):
      Family_History, Radiation_Exposure, Iodine_Deficiency, Smoking,
      Obesity, Diabetes  -> 'Yes' / 'No'
      Gender -> 'Male' / 'Female'
      Age, TSH_Level, T3_Level, T4_Level, Nodule_Size -> numbers
      Country -> one of RISK_META['countries']
      Ethnicity -> one of RISK_META['ethnicities']
    """
    row = {}
    for col in RISK_META['binary_cols']:
        row[col] = 1 if patient[col] == 'Yes' else 0
    row['Gender'] = 1 if patient['Gender'] == 'Male' else 0
    for col in RISK_META['continuous_cols']:
        row[col] = float(patient[col])
    for c in RISK_META['countries']:
        row[f'Country_{c}'] = 1 if patient['Country'] == c else 0
    for e in RISK_META['ethnicities']:
        row[f'Ethnicity_{e}'] = 1 if patient['Ethnicity'] == e else 0

    vec = np.array([[row[col] for col in RISK_META['feature_cols']]], dtype=float)

    # scale continuous columns in-place (same scaler used in training)
    cont_idx = [RISK_META['feature_cols'].index(c) for c in RISK_META['continuous_cols']]
    vec[:, cont_idx] = RISK_SCALER.transform(vec[:, cont_idx])
    return vec


def predict_risk(patient: dict, explain: bool = True) -> dict:
    x = build_risk_vector(patient)
    xb = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        diag_logit, risk_logits = RISK_MODEL(xb)
        diag_prob_raw = torch.sigmoid(diag_logit).numpy()
        risk_probs = torch.softmax(risk_logits, dim=1).numpy()[0]

    diag_prob = diag_prob_raw.copy()
    if RISK_CALIBRATOR is not None:
        diag_prob = RISK_CALIBRATOR.predict(diag_prob_raw)

    result = {
        'diagnosis': RISK_META['diagnosis_classes'][int(diag_prob[0] > 0.5)],
        'malignant_probability': float(diag_prob[0]),
        'risk_level': RISK_META['risk_classes'][int(np.argmax(risk_probs))],
        'risk_level_probabilities': {
            cls: float(p) for cls, p in zip(RISK_META['risk_classes'], risk_probs)
        },
    }

    if explain:
        result['shap_explanation'] = _explain_risk(x)

    return result


def _explain_risk(x_vec: np.ndarray, nsamples: int = 100):
    """SHAP explanation for ONE patient (fast KernelExplainer, small background)."""
    data = np.load(f'{BASE_DIR}/data/processed/risk_module_data.npz')
    X_train = data['X_train']
    background = X_train[np.random.choice(len(X_train), 30, replace=False)]

    def predict_fn(x_numpy):
        with torch.no_grad():
            xb = torch.tensor(x_numpy, dtype=torch.float32)
            logit, _ = RISK_MODEL(xb)
            return torch.sigmoid(logit).numpy()

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(x_vec, nsamples=nsamples)[0]

    contributions = sorted(
        zip(RISK_META['feature_cols'], shap_values),
        key=lambda t: -abs(t[1])
    )
    return [{'feature': f, 'impact': float(v)} for f, v in contributions[:8]]


# RECURRENCE MODULE
def build_recurrence_vector(patient: dict) -> np.ndarray:
    """
    patient dict expected keys:
      Gender -> 'M' / 'F'
      Smoking, Hx Smoking, Hx Radiothreapy -> 'Yes' / 'No'
      Age -> number
      Thyroid Function, Physical Examination, Adenopathy, Pathology,
      Focality, Risk, T, N, M, Stage, Response -> categorical strings
      (must match one of REC_META['nominal_options'][<col>])
    """
    row = {}
    row['Gender'] = 1 if patient['Gender'] == 'M' else 0
    for col in ['Smoking', 'Hx Smoking', 'Hx Radiothreapy']:
        row[col] = 1 if patient[col] == 'Yes' else 0
    row['Age'] = float(patient['Age'])

    for nom_col in REC_META['nominal_cols']:
        for option in REC_META['nominal_options'][nom_col]:
            key = f'{nom_col}_{option}'
            if key in REC_META['feature_cols']:
                row[key] = 1 if patient[nom_col] == option else 0

    vec = np.array([[row[col] for col in REC_META['feature_cols']]], dtype=float)
    cont_idx = [REC_META['feature_cols'].index(c) for c in REC_META['continuous_cols']]
    vec[:, cont_idx] = REC_SCALER.transform(vec[:, cont_idx])
    return vec


def predict_recurrence(patient: dict, explain: bool = True) -> dict:
    x = build_recurrence_vector(patient)
    xb = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logit = REC_MODEL(xb)
        prob_raw = torch.sigmoid(logit).numpy()

    prob = prob_raw.copy()
    if REC_CALIBRATOR is not None:
        prob = REC_CALIBRATOR.predict(prob_raw)

    result = {
        'recurred_prediction': REC_META['target_classes'][int(prob[0] > 0.5)],
        'recurrence_probability': float(prob[0]),
    }

    if explain:
        result['shap_explanation'] = _explain_recurrence(x)

    return result


def _explain_recurrence(x_vec: np.ndarray, nsamples: int = 100):
    data = np.load(f'{BASE_DIR}/data/processed/recurrence_module_data.npz')
    X_train = data['X_train']
    background = X_train[np.random.choice(len(X_train), 30, replace=False)]

    def predict_fn(x_numpy):
        with torch.no_grad():
            xb = torch.tensor(x_numpy, dtype=torch.float32)
            return torch.sigmoid(REC_MODEL(xb)).numpy()

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(x_vec, nsamples=nsamples)[0]

    contributions = sorted(
        zip(REC_META['feature_cols'], shap_values),
        key=lambda t: -abs(t[1])
    )
    return [{'feature': f, 'impact': float(v)} for f, v in contributions[:8]]