import streamlit as st
import joblib
import numpy as np

# Load model and columns
model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Risk Predictor")

# Create a dictionary to store inputs
user_input = {}

# Loop through columns and assign inputs
for col in columns:
    # For numeric inputs (customize as needed)
    if col in ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'AgeCategory']:
        user_input[col] = st.number_input(f"{col}", min_value=0.0, step=1.0)
    
    # For binary yes/no features — use checkbox
    elif col in ['Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalActivity', 'DiffWalking', 
                 'Diabetic', 'Asthma', 'KidneyDisease', 'SkinCancer']:
        user_input[col] = 1 if st.checkbox(f"{col.replace('_', ' ')}?") else 0

    # For others that still need numbers (like encoded categorical variables)
    else:
        user_input[col] = st.number_input(f"{col}", min_value=0, step=1)

# When button is clicked
if st.button("Predict"):
    input_array = np.array([list(user_input.values())])
    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error("🚨 Patient is AT RISK of heart disease.")
        st.info(f"Confidence score: {confidence:.2f}")
    else:
        st.success("✅ Patient is NOT at risk of heart disease.")
        st.info(f"Confidence score: {1 - confidence:.2f}")
