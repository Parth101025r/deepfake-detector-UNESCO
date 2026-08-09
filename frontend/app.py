from __future__ import annotations

import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")
BACKEND_START_COMMAND = "uvicorn backend.main:app --reload --port 8080"

st.set_page_config(page_title="Multimodal Fake News Detector", page_icon=":newspaper:", layout="wide")

st.title("Multimodal Fake News Detection using RAG")
st.caption("Claim verification using multimodal retrieval-augmented reasoning")

with st.sidebar:
    st.header("Backend")
    st.caption(API_BASE_URL)
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.ok:
            health = health_response.json()
            st.success("API connected")
            st.write(f"Verifier ready: {health.get('retrieval_ready', False)}")
        else:
            st.warning(f"API returned HTTP {health_response.status_code}")
    except requests.exceptions.RequestException:
        st.error("API is not running")
        st.code(BACKEND_START_COMMAND, language="powershell")

default_claim = "Narendra Modi died in a car accident."
claim = st.text_area("Enter a text claim", value=default_claim, height=140)
image_file = st.file_uploader("Upload an associated image (optional)", type=["png", "jpg", "jpeg"])
top_k = st.slider("Number of evidence snippets", min_value=1, max_value=5, value=3)
if image_file is not None:
    st.image(image_file, caption="Uploaded image", use_container_width=True)

if st.button("Verify Multimodal Claim", type="primary", use_container_width=True):
    if not claim.strip():
        st.warning("Please enter a text claim first.")
    else:
        with st.spinner("Analyzing claim..."):
            
            files = {}
            if image_file is not None:
                files["image"] = (image_file.name, image_file.getvalue(), image_file.type)
            
            data = {
                "claim": claim.strip(), 
                "top_k": top_k
            }
            
            try:
                response = requests.post(
                    f"{API_BASE_URL}/verify",
                    data=data,
                    files=files if files else None,
                    timeout=180,
                )
            except requests.exceptions.RequestException as exc:
                st.error("Could not connect to the backend API.")
                st.info(
                    "Start the FastAPI backend in another terminal, then try again. "
                    "Use API_BASE_URL if your backend runs on a different host or port."
                )
                st.code(BACKEND_START_COMMAND, language="powershell")
                with st.expander("Connection details"):
                    st.code(str(exc))
                st.stop()

        if response.ok:
            data = response.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Verdict", data["predicted_label"])
            col2.metric("Confidence", f"{data['confidence']:.2f}")
            col3.metric("Image Status", data["image_status"])

            st.subheader("Explanation")
            st.write(data["explanation"])

            with st.expander("Model Details"):
                st.write("The verifier combines the claim, optional image, and retrieved context before producing a label.")
                st.write(f"Internal model signal: {data['classifier_predicted_label']} ({data['classifier_confidence']:.2f})")

        else:
            st.error(f"Backend error: {response.status_code}")
            try:
                st.json(response.json())
            except Exception:
                st.text(response.text)
