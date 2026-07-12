"""
frontend/pages/5_About.py
Purpose: Model card / methodology summary page - dataset info, approach,
and known limitations, for transparency (proposal requirement).
"""
import streamlit as st

st.set_page_config(page_title="About - ThyroidCare", page_icon="", layout="wide")
st.title("About ThyroidCare")

st.markdown(
    """
    **ThyroidCare** is a research prototype developed as part of an undergraduate
    research project: *"Deep Learning-Based Prognostic Model for Thyroid Cancer
    Risk and Recurrence"* (Faculty of Computing and Technology, University of Kelaniya).
    """
)

st.divider()
st.subheader("Datasets")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Risk / Diagnosis Module**")
    st.markdown(
        """
        - Source: Thyroid Cancer Risk Prediction Dataset (Kaggle)
        - Size: ~212,691 patient records
        - Targets: Diagnosis (Benign/Malignant), Risk Level (Low/Medium/High)
        - Features: Age, Gender, Country, Ethnicity, Family History,
          Radiation Exposure, Iodine Deficiency, Smoking, Obesity, Diabetes,
          TSH/T3/T4 Levels, Nodule Size
        """
    )
with d2:
    st.markdown("**Recurrence Module**")
    st.markdown(
        """
        - Source: Differentiated Thyroid Cancer Recurrence Dataset (Kaggle)
        - Size: ~364 patient records (post de-duplication)
        - Target: Recurred (Yes/No)
        - Features: Age, Gender, Smoking history, Radiotherapy history,
          Thyroid Function, Physical Examination, Adenopathy, Pathology,
          Focality, Risk Category, TNM Staging, Treatment Response
        """
    )

st.info("Both datasets are fully anonymized with no personally identifiable information.")

st.divider()
st.subheader("Methodology")
st.markdown(
    """
    1. **Preprocessing** — schema validation, categorical encoding (binary +
       one-hot), continuous feature scaling, stratified train/validation/test
       splits, SMOTE oversampling for the (small, imbalanced) Recurrence dataset.
    2. **Baseline models** — Logistic Regression, XGBoost.
    3. **Deep learning models** — Multi-layer perceptron (MLP); a multi-task
       variant was used for the Risk module to jointly predict Diagnosis and
       Risk Level.
    4. **Evaluation** — AUROC, F1, AUPRC, Brier score, sensitivity at 90%
       specificity.
    5. **Explainability** — SHAP (SHapley Additive exPlanations) for both
       global feature importance and per-patient explanations.
    6. **Fairness auditing** — performance breakdown across Gender,
       Ethnicity, and Country subgroups.
    7. **Calibration** — Isotonic Regression to align predicted probabilities
       with observed outcome frequencies.
    8. **Deployment** — this Streamlit web application, with PDF report
       export and an audit log of predictions made.
    """
)

st.divider()
st.subheader("Known Limitations")
st.markdown(
    """
    - **'Medium' risk class is not reliably separable** from 'Low' using the
      available features in the Risk dataset - confirmed across both XGBoost
      and the deep learning model. This is treated as a dataset-level
      limitation rather than a model or training deficiency.
    - **Fairness gap in the Risk model**: AUROC ranges from ~0.58 (Caucasian)
      to ~0.79 (Asian) across ethnicity subgroups. Predictions for lower-AUROC
      groups should be treated with extra caution.
    - **Recurrence module has a small dataset** (~364 patients total, 55 in
      the test set), which limits statistical confidence in subgroup fairness
      comparisons and calibration.
    - Advanced architectures mentioned in the original proposal (TabNet,
      FT-Transformer, DeepSurv/Cox survival models) were **not implemented**
      within the project timeline; the Recurrence dataset also lacks a
      time-to-event column, which would be required for true survival
      modeling (DeepSurv/Cox/DeepHit).
    - This tool is a **research prototype only** - it is not clinically
      validated, not approved by any medical authority, and must not be used
      for real diagnostic or treatment decisions.
    """
)

st.divider()
st.caption("ThyroidCare - H.P.P. Apeksha (CS/2020/070) - Faculty of Computing and Technology, University of Kelaniya")