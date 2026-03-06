import streamlit as st
import requests

st.set_page_config(page_title="Health Insurance Premium Predictor", page_icon="🏥")

st.title("🏥 Health Insurance Premium Predictor")
st.markdown("---")

API_URL = st.text_input("API URL", value="http://51.20.135.84:8000")

st.subheader("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=28)
    bmi = st.number_input("BMI", min_value=1.0, value=22.5, step=0.1)
    diabetic = st.selectbox("Diabetic", ["No", "Yes"])
    smoker = st.selectbox("Smoker", ["No", "Yes"])

with col2:
    gender = st.selectbox("Gender", ["male", "female"])
    bloodpressure = st.number_input("Blood Pressure", min_value=0, value=80)
    children = st.number_input("Children", min_value=0, value=0)
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

st.markdown("---")

if st.button("🔮 Predict Premium", use_container_width=True):
    payload = {
        "age": float(age),
        "gender": gender,
        "bmi": float(bmi),
        "bloodpressure": int(bloodpressure),
        "diabetic": diabetic,
        "children": int(children),
        "smoker": smoker,
        "region": region
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            claim = response.json()["claim"]
            st.success(f"### 💰 Estimated Premium: **${claim:,.2f}**")
        else:
            st.error(f"API Error: {response.status_code} — {response.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")