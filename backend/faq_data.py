"""
backend/faq_data.py
Purpose: Simple rule-based FAQ "chatbot" data for the app.
"""

FAQS = [
    {
        "question": "What is TSH and why does it matter?",
        "answer": "TSH (Thyroid-Stimulating Hormone) is produced by the pituitary "
                  "gland and controls how much hormone your thyroid makes. Abnormal "
                  "TSH levels (too high or too low) can be an early signal of thyroid "
                  "dysfunction and are one of the features this model uses for prediction."
    },
    {
        "question": "What does 'Medium' risk level mean?",
        "answer": "In this dataset, the 'Medium' risk category was found to overlap "
                  "heavily with 'Low' risk in terms of the available features. Both our "
                  "deep learning model and a strong baseline (XGBoost) struggled to "
                  "separate these two categories reliably. Treat 'Medium' predictions "
                  "with extra caution - this is a documented limitation of the dataset, "
                  "not a diagnostic guarantee."
    },
    {
        "question": "How reliable is this model?",
        "answer": "The Diagnosis prediction (Benign/Malignant) achieves an AUROC of "
                  "around 0.70 on unseen test data, and the Recurrence model achieves "
                  "around 0.95. These are research-grade results from a student project - "
                  "NOT a clinically validated diagnostic tool. Always consult a qualified "
                  "clinician for medical decisions."
    },
    {
        "question": "What is SHAP and what do the charts mean?",
        "answer": "SHAP (SHapley Additive exPlanations) shows which input features pushed "
                  "a prediction higher or lower. A positive SHAP value means that feature "
                  "increased the predicted risk for this patient; a negative value means it "
                  "decreased it. This helps explain WHY the model made a certain prediction, "
                  "instead of treating it as a black box."
    },
    {
        "question": "Is this tool fair across different patient groups?",
        "answer": "We evaluated model performance separately across gender, ethnicity, and "
                  "country groups (see the Model Insights dashboard). Some performance gaps "
                  "exist between groups, which is disclosed transparently rather than hidden. "
                  "This is an active area for improvement in future versions."
    },
    {
        "question": "Can I use this for an actual diagnosis?",
        "answer": "No. This is a research and educational prototype built for a university "
                  "project. It has not been clinically validated, approved by any medical "
                  "authority, or tested in real clinical settings. Please consult a licensed "
                  "healthcare professional for any real diagnosis or treatment decision."
    },
    {
        "question": "What data was this model trained on?",
        "answer": "The Risk module was trained on a public Kaggle dataset of ~212,000 "
                  "synthetic patient records with demographic and clinical features. The "
                  "Recurrence module was trained on a smaller real clinical dataset (~380 "
                  "differentiated thyroid cancer patients). See the About page for full details."
    },
    {
        "question": "Why do I need to provide my Country and Ethnicity?",
        "answer": "These fields were present as risk factors in the training dataset and "
                  "the model uses them as inputs. No personally identifiable information is "
                  "collected or stored - only aggregated, anonymized values are used for "
                  "the prediction itself, and no raw patient data is retained beyond the "
                  "current session (only a summary is logged for audit purposes)."
    },
]


def get_answer(question: str):
    """Exact or partial match lookup - used by the simple chatbot page."""
    question_lower = question.lower().strip()
    for faq in FAQS:
        if question_lower in faq["question"].lower() or faq["question"].lower() in question_lower:
            return faq["answer"]
    # fallback: keyword overlap scoring
    best_match, best_score = None, 0
    q_words = set(question_lower.split())
    for faq in FAQS:
        f_words = set(faq["question"].lower().split())
        score = len(q_words & f_words)
        if score > best_score:
            best_score, best_match = score, faq
    if best_match and best_score >= 2:
        return best_match["answer"]
    return ("I don't have a predefined answer for that. Try one of the suggested "
            "questions below, or consult a qualified clinician for medical advice.")
