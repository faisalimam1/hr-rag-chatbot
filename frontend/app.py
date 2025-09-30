# test commit
"""
Streamlit frontend to query the FastAPI backend.

Usage (local):
  streamlit run frontend/app.py
"""
import streamlit as st
import requests
import os

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("RAG HR Chatbot — HR Policy Assistant")

st.write("Welcome! You can explore HR policies here. Type your question and I’ll find the answer from the document")

q = st.text_input("Ask a question about the HR policy", placeholder="e.g. How many earned leaves do I get?")

col1, col2 = st.columns([3, 1])
with col2:
    top_k = st.number_input("Top sources", min_value=1, max_value=10, value=5, step=1)

if st.button("Ask") and q.strip():
    with st.spinner("Contacting backend..."):
        try:
            resp = requests.post(f"{BACKEND}/query", json={"q": q, "top_k": int(top_k)}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(data.get("answer", "No answer returned."))

    st.subheader("Sources (provenance)")
    for s in data.get("sources", []):
        st.markdown(f"**ID:** `{s['id']}` — page {s.get('page')}, score: {s.get('score', 0):.3f}")
        st.write(s.get("text", "")[:800] + ("..." if len(s.get("text", "")) > 800 else ""))

    st.caption(
        f"Latency: {data.get('meta', {}).get('latency_ms', '?')} ms — cached: {data.get('meta', {}).get('cached')}"
    )

# --- Footer ---
st.markdown("<p style='text-align: center; font-size: 14px;'>© Faisal Imam 2025</p>", unsafe_allow_html=True)
