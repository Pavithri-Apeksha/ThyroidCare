"""
frontend/app.py
Purpose: Home page of the ThyroidCare app. Streamlit auto-detects this as
the entry point, and everything in frontend/pages/ becomes sidebar
navigation automatically.

Run with: streamlit run frontend/app.py   (from the project root folder)
"""
import streamlit as st

st.set_page_config(
    page_title="ThyroidCare",
    page_icon="",
    layout="wide",
)

st.title("ThyroidCare")
st.subheader("Deep Learning-Based Prognostic Tool for Thyroid Cancer Risk and Recurrence")

st.markdown(
    """
    Welcome to **ThyroidCare** — a research prototype clinical decision support tool
    that predicts thyroid cancer risk before diagnosis and recurrence risk after
    treatment, using deep learning models with explainable AI (SHAP).
    """
)

st.warning(
    "**Research & educational prototype only.** This tool is NOT a validated "
    "diagnostic system and must not be used for real clinical decisions. "
    "Always consult a qualified healthcare professional."
)

st.divider()

# ---- Model overview cards ----
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Risk Model AUROC", "0.70", help="Diagnosis prediction (Benign/Malignant)")
with col2:
    st.metric("Recurrence Model AUROC", "0.96", help="Recurrence prediction (Yes/No)")
with col3:
    st.metric("Risk Dataset Size", "212,691", help="Patient records")
with col4:
    st.metric("Recurrence Dataset Size", "364", help="Patient records (post-treatment)")

st.divider()

# ---- Navigation guide ----
st.markdown("### What you can do here")

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        **Risk Prediction**
        Enter patient demographic and clinical data to get an early
        thyroid cancer risk assessment (Diagnosis + Risk Level), with
        a SHAP explanation and a downloadable PDF report.

        **Recurrence Prediction**
        For post-treatment patients: enter clinical/pathology details
        to estimate the probability of cancer recurrence.
        """
    )
with c2:
    st.markdown(
        """
        **Model Insights Dashboard**
        Explore model performance (ROC curves, confusion matrices),
        global SHAP feature importance, calibration, and fairness
        across demographic groups — all at the aggregate level (no
        individual patient data is shown).

        **FAQ Chatbot** & **About**
        Get quick answers to common questions, and read about the
        dataset, methodology, and known limitations of this project.
        """
    )

st.divider()
st.caption("Use the sidebar to navigate between pages.")