# ThyroidCare 🩺

Deep Learning-Based Prognostic Model for Thyroid Cancer Risk and Recurrence

A research prototype clinical decision support web app that predicts:
- **Thyroid cancer risk** before diagnosis (Diagnosis + Risk Level)
- **Recurrence risk** for post-treatment patients

Built with PyTorch, XGBoost, SHAP explainability, and a Streamlit web interface.

> **Disclaimer:** This is a research/educational prototype only. It is NOT a validated diagnostic tool and must not be used for real clinical decisions.

---

## Features

-  **Risk Prediction** — Diagnosis (Benign/Malignant) + Risk Level (Low/Medium/High), with SHAP explanations
- **Recurrence Prediction** — Post-treatment recurrence probability, with SHAP explanations
- **Model Insights Dashboard** — Performance metrics, calibration curves, global SHAP importance, fairness analysis across gender/ethnicity/country
- **PDF Report Export** — Downloadable prediction report per patient
- **Audit Log** — Every prediction logged for transparency
- **FAQ Chatbot** — Rule-based Q&A about the model and its limitations

## Models

| Module | Baseline | Deep Learning | Test AUROC |
|---|---|---|---|
| Risk (Diagnosis) | Logistic Regression / XGBoost | Multi-Task MLP (PyTorch) | ~0.70 |
| Recurrence | Logistic Regression / XGBoost | MLP (PyTorch) | ~0.96 |

## Tech Stack

- **ML/DL:** PyTorch, XGBoost, scikit-learn, SHAP
- **Web App:** Streamlit
- **PDF Reports:** fpdf2

## Project Structure

```
ThyroidCare/
├── backend/           # Core logic: prediction, PDF export, audit log, FAQ data
├── frontend/          # Streamlit app (Home + multi-page dashboard)
├── src/                # Model training & preprocessing scripts
├── data/               # Raw and processed datasets
├── models/             # Trained models, scalers, encoders
├── reports/             # SHAP plots, fairness/calibration charts, logs
└── .streamlit/          # App theme config
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python src/preprocess_risk.py
python src/preprocess_recurrence.py

# 3. Train models
python src/train_baseline_risk.py
python src/train_baseline_recurrence.py
python src/train_xgboost_risk.py
python src/train_xgboost_recurrence.py
python src/train_dl_risk_v2.py
python src/train_dl_recurrence.py

# 4. Run the web app
streamlit run frontend/app.py
```

## Datasets

- [Thyroid Cancer Risk Prediction Dataset](https://www.kaggle.com/) (Kaggle)
- [Differentiated Thyroid Cancer Recurrence](https://www.kaggle.com/) (Kaggle)

## Author

H.P.P. Apeksha (CS/2020/070)
Faculty of Computing and Technology, University of Kelaniya
Supervisor: Mr. Akash Tharuka
