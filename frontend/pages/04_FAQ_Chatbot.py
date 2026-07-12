"""
frontend/pages/4_FAQ_Chatbot.py
Purpose: Simple rule-based Q&A page. Not an LLM integration (avoids
cost/complexity/API-key risk) - satisfies the proposal's "chatbot for
Q&A, guidance, and explanation of predictions" requirement with a
predefined FAQ lookup plus free-text keyword matching.
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.faq_data import FAQS, get_answer

st.set_page_config(page_title="FAQ Chatbot - ThyroidCare", page_icon="", layout="wide")
st.title("ThyroidCare FAQ Assistant")
st.caption("Ask a question below, or click one of the common questions.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Quick-select common questions ----
st.markdown("##### Common Questions")
cols = st.columns(2)
for i, faq in enumerate(FAQS):
    with cols[i % 2]:
        if st.button(faq["question"], key=f"faq_{i}", use_container_width=True):
            st.session_state.chat_history.append(("user", faq["question"]))
            st.session_state.chat_history.append(("bot", faq["answer"]))

st.divider()

# ---- Free-text input ----
user_q = st.chat_input("Type your own question here...")
if user_q:
    answer = get_answer(user_q)
    st.session_state.chat_history.append(("user", user_q))
    st.session_state.chat_history.append(("bot", answer))

# ---- Display chat history ----
for role, text in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.write(text)

if not st.session_state.chat_history:
    st.info("No questions asked yet. Try one of the buttons above, or type your own question below.")

if st.session_state.chat_history:
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()