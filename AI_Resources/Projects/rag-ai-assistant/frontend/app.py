
# === frontend/app.py ===
import streamlit as st
import requests

st.set_page_config(page_title="RAG AI Assistant", layout="centered")
st.title("\U0001F4DA RAG PDF Chatbot")
st.markdown("Ask any question from your uploaded documents")

query = st.text_input("\U0001F50D Enter your question",
                      placeholder="e.g., What is a transformer model?")

if st.button("Ask"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("\U0001F9E0 Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query", json={"query": query})
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("\U0001F9E0 Answer")
                    st.markdown(data.get("answer", "_No answer found._"))
                    sources = data.get("sources", [])
                    if sources:
                        st.subheader("\U0001F4C4 Sources")
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**{i}.** {source}")
                    else:
                        st.info(
                            "No source documents were used to generate this answer.")
                else:
                    st.error(f"❌ Backend Error: {response.text}")
            except Exception as e:
                st.error(f"🔌 Failed to connect to backend:\n`{e}`")
