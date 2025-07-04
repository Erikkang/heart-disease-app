import streamlit as st
import joblib
import numpy as np

# Load the model and feature columns
model = joblib.load('heart_model.pkl')
columns = joblib.load('columns.pkl')

st.set_page_config(page_title="Heart Disease Risk Predictor")
st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter patient health information below to check risk for heart disease.")

# Dynamically build the input form
user_input = []
for col in columns:
    val = st.number_input(f"{col}", format="%.2f")
    user_input.append(val)

# Show button for prediction
if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    confidence = model.predict_proba([user_input])[0][1]

    st.subheader("🩺 Prediction Result")
    if prediction == 1:
        st.error("⚠️ Patient is **at risk** of heart disease.")
    else:
        st.success("✅ Patient is **not at risk**.")

    st.info(f"Confidence score: **{confidence:.2f}**")
