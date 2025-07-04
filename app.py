import streamlit as st
import joblib
import numpy as np

# Load model and columns
model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Risk Predictor")

user_input = {}

# ----- NUMERIC INPUTS -----
st.subheader("📊 Health Metrics")
user_input["BMI"] = st.number_input("BMI", min_value=0.0, step=0.1)

# Dropdown for PhysicalHealth
physical_map = {
    "✅ Great (0–3 bad days)": 2,
    "⚖️ Average (4–10 bad days)": 7,
    "❌ Poor (11–30 bad days)": 20
}
user_input["PhysicalHealth"] = physical_map[st.selectbox("Physical Health (past 30 days)", list(physical_map.keys()))]

# Dropdown for MentalHealth
mental_map = {
    "✅ Great (0–3 bad days)": 2,
    "⚖️ Average (4–10 bad days)": 7,
    "❌ Poor (11–30 bad days)": 20
}
user_input["MentalHealth"] = mental_map[st.selectbox("Mental Health (past 30 days)", list(mental_map.keys()))]

# Dropdown for SleepTime
sleep_map = {
    "💤 Too little (<4 hrs)": 3,
    "😴 Not enough (4–6 hrs)": 5,
    "✅ Healthy (7–9 hrs)": 8,
    "🛌 Too much (10+ hrs)": 10
}
user_input["SleepTime"] = sleep_map[st.selectbox("Sleep Time (average hrs/night)", list(sleep_map.keys()))]

# ----- CHECKBOX INPUTS -----
st.subheader("✅ Lifestyle / Medical History")
checkbox_fields = [
    "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
    "Diabetic", "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer"
]
cols = st.columns(3)
for i, field in enumerate(checkbox_fields):
    user_input[field] = 1 if cols[i % 3].checkbox(field) else 0

# ----- DROPDOWN ENCODINGS -----
st.subheader("🧬 Demographics")
# Encoded dropdowns (you may adjust encoding if needed)
user_input["Sex"] = 1 if st.radio("Sex", ["Female", "Male"]) == "Male" else 0
user_input["AgeCategory"] = st.number_input("Age Category (as encoded number)", min_value=0, max_value=13)
user_input["Race"] = st.number_input("Race (as encoded number)", min_value=0, max_value=5)
user_input["GenHealth"] = st.number_input("General Health (0=Excellent → 4=Poor)", min_value=0, max_value=4)

# ----- PREDICTION -----
if st.button("Predict"):
    input_array = np.array([list(user_input[col] for col in columns)])
    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error("🚨 The patient is AT RISK of heart disease.")
    else:
        st.success("✅ The patient is NOT at risk of heart disease.")

    st.info(f"Confidence Score: {confidence:.2f}")
