import streamlit as st
import requests

st.set_page_config(page_title="Traffic Inference UI", layout="centered")
st.title("Traffic Inference UI")

api_base = st.text_input("API Base URL", "http://api:8000")
task = st.selectbox("Task", ["app", "att"])
up = st.file_uploader("Upload val.csv", type=["csv"])

if st.button("Predict"):
    if not up:
        st.error("Please upload a CSV first.")
    else:
        files = {"file": (up.name, up.getvalue(), "text/csv")}
        with st.spinner("Running inference..."):
            r = requests.post(f"{api_base}/predict/{task}", files=files, timeout=300)
        if r.status_code != 200:
            st.error(r.text)
        else:
            out_name = f"val_with_prediction_{task}.csv"
            st.success("Done.")
            st.download_button("Download result", data=r.content, file_name=out_name, mime="text/csv")
