import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Diabetes Prediction App")

st.write("Enter Patient Details")

# User Input
Pregnancies = st.number_input("Pregnancies", 0, 20)

Glucose = st.number_input("Glucose", 0)

BloodPressure = st.number_input("Blood Pressure", 0)

SkinThickness = st.number_input("Skin Thickness", 0)

Insulin = st.number_input("Insulin", 0)

BMI = st.number_input("BMI", 0.0)

DiabetesPedigreeFunction = st.number_input("DPF", 0.0)

Age = st.number_input("Age", 1, 120)

# Convert input into array
input_data = np.array([[Pregnancies,
                        Glucose,
                        BloodPressure,
                        SkinThickness,
                        Insulin,
                        BMI,
                        DiabetesPedigreeFunction,
                        Age]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"High Diabetes Risk (Probability: {probability:.2f})")

    else:
        st.success(f"Low Diabetes Risk (Probability: {probability:.2f})")