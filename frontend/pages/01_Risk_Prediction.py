"""
frontend/pages/Risk_Prediction.py
Purpose: Patient input form for the Risk module. Calls backend.predictor to get Diagnosis + Risk Level predictions with a SHAP explanation, logs the prediction, and offers a downloadable PDF report.
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.predictor import predict_risk, RISK_META
from backend.audit_log import log_prediction
from backend.pdf_report import generate_prediction_pdf

st.set_page_config(page_title="Risk Prediction - ThyroidCare", page_icon="", layout="wide")
st.title("Thyroid Cancer Risk Prediction")
st.caption("Enter patient demographic and clinical data for an early risk assessment.")

with st.form("risk_form"):
    st.subheader("Patient Information")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        country = st.selectbox("Country", RISK_META["countries"])
        ethnicity = st.selectbox("Ethnicity", RISK_META["ethnicities"])
    with c2:
        tsh = st.number_input("TSH Level (mIU/L)", min_value=0.0, max_value=50.0, value=2.5, step=0.1)
        t3 = st.number_input("T3 Level (ng/dL)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        t4 = st.number_input("T4 Level (µg/dL)", min_value=0.0, max_value=20.0, value=8.0, step=0.1)
        nodule_size = st.number_input("Nodule Size (cm)", min_value=0.0, max_value=15.0, value=1.5, step=0.1)
    with c3:
        family_history = st.selectbox("Family History of Thyroid Cancer", ["No", "Yes"])
        radiation = st.selectbox("Radiation Exposure", ["No", "Yes"])
        iodine = st.selectbox("Iodine Deficiency", ["No", "Yes"])
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        obesity = st.selectbox("Obesity", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])

    submitted = st.form_submit_button("🔬 Predict Risk", use_container_width=True)

if submitted:
    patient = {
        "Family_History": family_history, "Radiation_Exposure": radiation,
        "Iodine_Deficiency": iodine, "Smoking": smoking, "Obesity": obesity,
        "Diabetes": diabetes, "Gender": gender, "Age": age, "TSH_Level": tsh,
        "T3_Level": t3, "T4_Level": t4, "Nodule_Size": nodule_size,
        "Country": country, "Ethnicity": ethnicity,
    }

    with st.spinner("Running model + generating explanation..."):
        result = predict_risk(patient)
        log_prediction("risk", result["diagnosis"], result["malignant_probability"], patient)

    st.divider()
    st.subheader("Prediction Result")

    r1, r2 = st.columns(2)
    with r1:
        if result["diagnosis"] == "Malignant":
            st.error(f"**Diagnosis Prediction: {result['diagnosis']}**")
        else:
            st.success(f"**Diagnosis Prediction: {result['diagnosis']}**")
        st.metric("Malignant Probability", f"{result['malignant_probability']:.1%}")
    with r2:
        st.info(f"**Risk Level: {result['risk_level']}**")
        st.progress(result["risk_level_probabilities"][result["risk_level"]])
        if result["risk_level"] == "Medium":
            st.caption(
                "Note: 'Medium' risk classification has known reliability "
                "limitations in this model - see the FAQ page for details."
            )

    st.markdown("##### Risk Level Probabilities")
    st.bar_chart(result["risk_level_probabilities"])

    st.markdown("##### Top Factors Driving This Prediction (SHAP)")
    st.caption("Positive values increase malignancy risk; negative values decrease it.")
    for item in result["shap_explanation"]:
        direction = "increases risk" if item["impact"] > 0 else "decreases risk"
        st.write(f"**{item['feature']}** — {direction} (impact: {item['impact']:+.3f})")

    #  PDF export
    os.makedirs("reports/generated", exist_ok=True)
    pdf_path = f"reports/generated/risk_report_{age}_{gender}.pdf"
    generate_prediction_pdf("risk", patient, result, pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download PDF Report", f, file_name="ThyroidCare_Risk_Report.pdf",
            mime="application/pdf", use_container_width=True
        )