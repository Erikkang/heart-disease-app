import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Risk Predictor")

# Load the model and columns
try:
    model = joblib.load("heart_model.pkl")
    columns = joblib.load("columns.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model or columns: {e}")
    st.stop()

st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter patient details below to predict heart disease risk.")

# Input form
user_input = []
for col in columns:
    val = st.number_input(f"{col}", format="%.2f")
    user_input.append(val)

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict([user_input])[0]
        confidence = model.predict_proba([user_input])[0][1]

        st.subheader("🩺 Prediction Result")
        if prediction == 1:
            st.error("⚠️ Patient is **at risk** of heart disease.")
        else:
            st.success("✅ Patient is **not at risk**.")
        st.info(f"Confidence score: **{confidence:.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
