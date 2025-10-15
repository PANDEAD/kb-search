import streamlit as st
import requests

st.set_page_config(page_title="KB Search", layout="centered")

st.title("📚 Knowledge Base Search Engine")

backend_url = "http://127.0.0.1:8000"

st.subheader("Upload PDFs")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files and st.button("Upload and Index"):
    files = [("files", (f.name, f.read(), "application/pdf")) for f in uploaded_files]
    res = requests.post(f"{backend_url}/ingest", files=files)
    st.success(res.json()["message"])

st.subheader("Ask a Question")
query = st.text_input("Enter your question:")
if st.button("Ask"):
    res = requests.post(f"{backend_url}/ask", data={"query": query})
    data = res.json()
    st.markdown("###  Answer")
    st.write(data["answer"])
    st.markdown(f"**Sources:** {', '.join(set(data['sources']))}")
