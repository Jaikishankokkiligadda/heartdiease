import os
import pickle
import streamlit as st
import numpy as np

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict risk of heart disease.")

# Load model
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "Heart_disease_model_pipeline.pkl")

if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()  # Stop execution if model is missing

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0,1])
oldpeak = st.number_input("ST Depression", value=1.0)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (0-3)", [0,1,2,3])

# Prediction
if st.button("Predict"):
    input_data = np.array([[float(age), float(sex), float(cp), float(trestbps), float(chol), 
                            float(fbs), float(restecg), float(thalach), float(exang), 
                            float(oldpeak), float(slope), float(ca), float(thal)]])
    
    prediction = model.predict(input_data)
    
    # Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1]  # Probability of heart disease
        st.write(f"🔹 Risk Probability: **{proba*100:.2f}%**")
    
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
