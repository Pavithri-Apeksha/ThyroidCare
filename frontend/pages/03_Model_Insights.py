"""
frontend/pages/Model_Insights.py
Purpose: Aggregate-level model performance dashboard - fairness breakdown,
calibration, global SHAP feature importance, and prediction history.
No individual patient records from the training data are shown here -
only aggregate statistics and this app's own audit log.
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.dashboard_data import (
    get_model_comparison, load_extra_metrics, load_fairness_tables,
    get_chart_paths, load_audit_log
)

st.set_page_config(page_title="Model Insights - ThyroidCare", page_icon="", layout="wide")
st.title("Model Insights Dashboard")
st.caption("Aggregate model performance, fairness, and explainability - no individual patient records are shown.")

tab1, tab2, tab3 = st.tabs(["Performance & Calibration", "⚖️ Fairness", "🕓 Prediction History"])

charts = get_chart_paths()

# ============================================================
# TAB 1: Performance & Calibration
# ============================================================
with tab1:
    st.subheader("Model Comparison")
    comparison = get_model_comparison()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk / Diagnosis Module**")
        st.dataframe(comparison["Risk / Diagnosis"], use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Recurrence Module**")
        st.dataframe(comparison["Recurrence"], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Additional Evaluation Metrics")
    extra = load_extra_metrics()
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Risk / Diagnosis**")
        if extra['risk']:
            st.metric("AUPRC", extra['risk']['AUPRC'])
            st.metric("Brier Score", extra['risk']['Brier_Score'])
            st.metric("Sensitivity @ 90% Specificity", extra['risk']['Sensitivity_at_90pct_Specificity'])
        else:
            st.info("Not yet generated - run fairness_analysis_risk.py")
    with m2:
        st.markdown("**Recurrence**")
        if extra['recurrence']:
            st.metric("AUPRC", extra['recurrence']['AUPRC'])
            st.metric("Brier Score", extra['recurrence']['Brier_Score'])
            st.metric("Sensitivity @ 90% Specificity", extra['recurrence']['Sensitivity_at_90pct_Specificity'])
        else:
            st.info("Not yet generated - run fairness_analysis_recurrence.py")

    st.divider()
    st.subheader("Calibration Curves")
    st.caption("Compares predicted probabilities against observed outcomes - closer to the diagonal is better calibrated.")
    cc1, cc2 = st.columns(2)
    with cc1:
        if 'Calibration Curve (Risk)' in charts:
            st.image(charts['Calibration Curve (Risk)'], caption="Risk Module - Calibration improved after Isotonic Regression")
        else:
            st.info("Chart not found.")
    with cc2:
        if 'Calibration Curve (Recurrence)' in charts:
            st.image(charts['Calibration Curve (Recurrence)'], caption="Recurrence Module - small validation set (n=55), exploratory")
        else:
            st.info("Chart not found.")

    st.divider()
    st.subheader("Global Feature Importance (SHAP)")
    sc1, sc2 = st.columns(2)
    with sc1:
        if 'SHAP Summary (Risk)' in charts:
            st.image(charts['SHAP Summary (Risk)'], caption="Risk Module - Global SHAP Summary")
    with sc2:
        if 'SHAP Summary (Recurrence)' in charts:
            st.image(charts['SHAP Summary (Recurrence)'], caption="Recurrence Module - Global SHAP Summary")

# ============================================================
# TAB 2: Fairness
# ============================================================
with tab2:
    st.subheader("Fairness Across Demographic Groups")
    st.caption("Aggregate group-level statistics only - no individual patient rows are shown.")
    fairness = load_fairness_tables()

    st.markdown("#### Risk / Diagnosis Module")
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("**By Gender**")
        if fairness['risk_gender'] is not None:
            st.dataframe(fairness['risk_gender'], use_container_width=True, hide_index=True)
    with fc2:
        st.markdown("**By Ethnicity**")
        if fairness['risk_ethnicity'] is not None:
            st.dataframe(fairness['risk_ethnicity'], use_container_width=True, hide_index=True)

    if 'Ethnicity Fairness (Risk)' in charts:
        st.image(charts['Ethnicity Fairness (Risk)'], caption="Malignant Recall by Ethnicity")

    st.warning(
        "**Fairness finding:** The Risk model shows a substantial performance gap "
        "across ethnicity groups (AUROC ranging from ~0.58 for Caucasian patients to "
        "~0.79 for Asian patients). Predictions for lower-AUROC groups should be "
        "interpreted with additional caution. This is discussed as a key limitation "
        "in the accompanying research report."
    )

    st.markdown("**By Country (Top 10)**")
    if fairness['risk_country'] is not None:
        st.dataframe(fairness['risk_country'], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Recurrence Module")
    st.caption("Only Gender is available as a demographic attribute in this dataset. "
               "Test set size is small (n=55), so group-level comparisons here are exploratory.")
    if fairness['recurrence_gender'] is not None:
        st.dataframe(fairness['recurrence_gender'], use_container_width=True, hide_index=True)

# ============================================================
# TAB 3: Prediction History (this app's own audit log)
# ============================================================
with tab3:
    st.subheader("Prediction History (Audit Log)")
    st.caption("Every prediction made through this app is logged here for transparency.")
    log_entries = load_audit_log(limit=50)
    if log_entries:
        st.dataframe(log_entries, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(log_entries)} most recent prediction(s).")
    else:
        st.info("No predictions logged yet. Make a prediction on the Risk or Recurrence page to see it appear here.")