"""
frontend/pages/Recurrence_Prediction.py
Purpose: Clinical input form for the Recurrence module (post-treatment patients). Calls backend.predictor to get a recurrence prediction with SHAP explanation, logs it, and offers a downloadable PDF report.
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.predictor import predict_recurrence, REC_META
from backend.audit_log import log_prediction
from backend.pdf_report import generate_prediction_pdf

st.set_page_config(page_title="Recurrence Prediction - ThyroidCare", page_icon="", layout="wide")
st.title("Thyroid Cancer Recurrence Prediction")
st.caption("For post-treatment patients: estimate the probability of cancer recurrence.")

opts = REC_META["nominal_options"]

with st.form("recurrence_form"):
    st.subheader("Patient & Clinical Information")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        gender = st.selectbox("Gender", ["F", "M"])
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        hx_smoking = st.selectbox("History of Smoking", ["No", "Yes"])
        hx_radio = st.selectbox("History of Radiotherapy", ["No", "Yes"])
        thyroid_function = st.selectbox("Thyroid Function", opts["Thyroid Function"])
    with c2:
        physical_exam = st.selectbox("Physical Examination", opts["Physical Examination"])
        adenopathy = st.selectbox("Adenopathy", opts["Adenopathy"])
        pathology = st.selectbox("Pathology", opts["Pathology"])
        focality = st.selectbox("Focality", opts["Focality"])
        risk = st.selectbox("Risk Category", opts["Risk"])
    with c3:
        t_stage = st.selectbox("T (Tumor) Stage", opts["T"])
        n_stage = st.selectbox("N (Node) Stage", opts["N"])
        m_stage = st.selectbox("M (Metastasis) Stage", opts["M"])
        stage = st.selectbox("Overall Stage", opts["Stage"])
        response = st.selectbox("Treatment Response", opts["Response"])

    submitted = st.form_submit_button("Predict Recurrence", use_container_width=True)

if submitted:
    patient = {
        "Gender": gender, "Smoking": smoking, "Hx Smoking": hx_smoking,
        "Hx Radiothreapy": hx_radio, "Age": age,
        "Thyroid Function": thyroid_function, "Physical Examination": physical_exam,
        "Adenopathy": adenopathy, "Pathology": pathology, "Focality": focality,
        "Risk": risk, "T": t_stage, "N": n_stage, "M": m_stage,
        "Stage": stage, "Response": response,
    }

    with st.spinner("Running model + generating explanation..."):
        result = predict_recurrence(patient)
        log_prediction("recurrence", result["recurred_prediction"],
                        result["recurrence_probability"], patient)

    st.divider()
    st.subheader("Prediction Result")

    r1, r2 = st.columns(2)
    with r1:
        if result["recurred_prediction"] == "Yes":
            st.error(f"**Recurrence Prediction: {result['recurred_prediction']}**")
        else:
            st.success(f"**Recurrence Prediction: {result['recurred_prediction']}**")
    with r2:
        st.metric("Recurrence Probability", f"{result['recurrence_probability']:.1%}")
        st.progress(min(result["recurrence_probability"], 1.0))

    st.markdown("##### Top Factors Driving This Prediction (SHAP)")
    st.caption("Positive values increase recurrence risk; negative values decrease it.")
    for item in result["shap_explanation"]:
        direction = "increases risk" if item["impact"] > 0 else "decreases risk"
        st.write(f"**{item['feature']}** — {direction} (impact: {item['impact']:+.3f})")

    # PDF export
    os.makedirs("reports/generated", exist_ok=True)
    pdf_path = f"reports/generated/recurrence_report_{age}_{gender}.pdf"
    generate_prediction_pdf("recurrence", patient, result, pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download PDF Report", f, file_name="ThyroidCare_Recurrence_Report.pdf",
            mime="application/pdf", use_container_width=True
        )