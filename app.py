from __future__ import annotations
import os
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")

st.set_page_config(page_title="Parallel Multimodal Fake News Detector", page_icon="📰", layout="wide")

st.title("Parallel Multimodal Fake News Detection System")
st.caption("Live Web Search Evidence Pipeline + CLIP Multimodal Classifier fused by Final Decision Engine")

default_claim = "Scientists discover evidence of past liquid water on Mars."
claim = st.text_area("Enter a text claim to verify", value=default_claim, height=120)
image_file = st.file_uploader("Upload an associated claim image (Optional)", type=["png", "jpg", "jpeg"])
top_k = st.slider("Number of Live Web Search Headlines to Rank", min_value=1, max_value=5, value=3)

if st.button("Verify Claim (Parallel Pipeline)", type="primary", use_container_width=True):
    if not claim.strip():
        st.warning("Please enter a text claim first.")
    else:
        with st.spinner("Running Web Search Branch & CLIP Multimodal Classifier Branch..."):
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
            except Exception as e:
                st.error(f"Could not connect to backend server at {API_BASE_URL}. Error: {e}")
                st.stop()

        if response.ok:
            res = response.json()

            # Banner: Final Decision Engine Result
            st.divider()
            st.header("🎯 Final Decision Engine Output")
            
            label = res.get("predicted_label", "Uncertain")
            if label == "Fake":
                verdict_color = "red"
            elif label == "Real":
                verdict_color = "green"
            else:
                verdict_color = "orange"
            
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"### Final Verdict: :{verdict_color}[**{label}**]")
            m2.metric("Combined Confidence", f"{res.get('confidence', 0.0):.2%}")
            m3.metric("Pipeline Architecture", "SentenceTransformers + CLIP")

            st.info(f"**Explanation:** {res.get('explanation', 'N/A')}")
            st.divider()

            # Two Column Parallel Branches View
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("🌐 Branch A: Two-Stage Web Evidence Verification")
                ev_summary = res.get("evidence_summary", {})
                gen_q = ev_summary.get("generated_query", "")
                if gen_q:
                    st.caption(f"🔍 Extracted Keyword Query: `{gen_q}`")
                
                # Render Stance Metrics Summary
                s_count = ev_summary.get("supports_count", 0)
                c_count = ev_summary.get("contradicts_count", 0)
                u_count = ev_summary.get("unrelated_count", 0)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("🟢 Supporting", s_count)
                c2.metric("🔴 Contradicting", c_count)
                c3.metric("⚪ Unrelated", u_count)
                
                evidence_list = res.get("evidence", [])
                if not evidence_list:
                    st.warning("No trusted live web search evidence retrieved.")
                else:
                    for item in evidence_list:
                        label = item.get("evidence_label", "UNRELATED")
                        if label == "SUPPORTS":
                            badge = "🟢 **SUPPORTS**"
                        elif label == "CONTRADICTS":
                            badge = "🔴 **CONTRADICTS**"
                        else:
                            badge = "⚪ **UNRELATED**"

                        with st.container(border=True):
                            st.markdown(f"{badge} | **Rank #{item.get('rank', 1)}: [{item.get('title', 'Untitled')}]({item.get('url', '#')})**")
                            st.caption(
                                f"NLI Confidence: {item.get('stance_score', 0.0):.2%} | "
                                f"Topical Similarity: {item.get('score', 0.0):.4f} | "
                                f"Source: {item.get('url', 'Unknown')}"
                            )
                            st.markdown(f"**Extracted Evidence Sentence:**\n> {item.get('evidence_sentence', 'N/A')}")
                            st.write(f"**Snippet:** {item.get('snippet', 'No snippet available.')}")
                            if item.get("stance_reason"):
                                st.caption(f"💡 *Verification Note: {item.get('stance_reason')}*")

            with col_right:
                st.subheader("🖼️ Branch B: CLIP Multimodal Classifier")
                mm_pred = res.get("multimodal_prediction", {})
                
                with st.container(border=True):
                    st.markdown(f"**Predicted Label**: `{mm_pred.get('label', 'N/A')}`")
                    st.markdown(f"**Visual-Textual Confidence**: `{mm_pred.get('confidence', 0.0):.2%}`")
                    if image_file:
                        st.image(image_file, caption="Input Claim Image", use_column_width=True)
                    else:
                        st.caption("No image uploaded (used fallback image tensor representation).")
        else:
            st.error(f"Backend error: {response.status_code}")
            try:
                st.json(response.json())
            except Exception:
                st.text(response.text)

