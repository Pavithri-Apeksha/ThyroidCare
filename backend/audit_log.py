"""
backend/audit_log.py
Purpose: Every time a prediction is made, log it to a CSV file
(timestamp, module, prediction, probability, input summary).
This satisfies the proposal's "audit logs" requirement for transparency
in a clinical decision support tool.
"""
import csv
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = f'{BASE_DIR}/reports/audit_log.csv'

FIELDNAMES = ['timestamp', 'module', 'prediction', 'probability', 'input_summary']


def log_prediction(module: str, prediction: str, probability: float, patient: dict):
    """
    module: 'risk' or 'recurrence'
    prediction: the predicted label (e.g. 'Malignant', 'Yes')
    probability: the model's probability score
    patient: the raw input dict - we store a short readable summary,
             NOT the full record (keeps the log compact and avoids
             storing anything resembling a full patient profile)
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)

    input_summary = '; '.join(f'{k}={v}' for k, v in list(patient.items())[:5])

    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'module': module,
            'prediction': prediction,
            'probability': round(float(probability), 4),
            'input_summary': input_summary,
        })


def read_log(limit: int = 50):
    """Returns the most recent `limit` log entries as a list of dicts.
    Used by the Model Insights dashboard to show 'Prediction History'."""
    if not os.path.isfile(LOG_PATH):
        return []
    with open(LOG_PATH, newline='') as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:][::-1]  # most recent first