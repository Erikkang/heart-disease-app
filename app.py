import streamlit as st
import joblib
import numpy as np

# Load model and columns
model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Risk Predictor")

# Create input dictionary
user_input = {}

# Define numeric and checkbox columns
numeric_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'AgeCategory']
checkbox_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalActivity', 
                 'DiffWalking', 'Diabetic', 'Asthma', 'KidneyDisease', 'SkinCancer']

# Collect numeric inputs first
st.subheader("📊 Health Metrics")
for col in numeric_cols:
    user_input[col] = st.number_input(f"{col}", min_value=0.0, step=1.0)

# Then collect checkbox inputs
st.subheader("✅ Health Conditions")
for col in checkbox_cols:
    user_input[col] = 1 if st.checkbox(f"{col.replace('_', ' ')}") else 0

# Remaining columns (like encoded categories)
for col in columns:
    if col not in numeric_cols + checkbox_cols:
        user_input[col] = st.number_input(f"{col}", min_value=0, step=1)

# Prediction
if st.button("Predict"):
    input_array = np.array([list(user_input.values())])
    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error("🚨 Patient is AT RISK of heart disease.")
        st.info(f"Confidence score: {confidence:.2f}")
    else:
        st.success("✅ Patient is NOT at risk of heart disease.")
        st.info(f"Confidence score: {(1 - confidence):.2f}")
