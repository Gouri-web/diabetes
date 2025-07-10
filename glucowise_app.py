import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------- UI Setup ----------
st.set_page_config(page_title="Glucowise", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #121212;
            color: white;
        }
        img {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox("Navigation", ["Welcome", "Check Diabetes"])

if page == "Welcome":
    st.title("Glucowise")
    st.markdown("A Smart Diabetes Prediction and Monitoring Tool")
    st.image("glucowise_icon.png", width=150)

elif page == "Check Diabetes":
    st.title("Diabetes Check")

    name = st.text_input("Your Name")
    email = st.text_input("Your Email")

    st.subheader("Input Health Metrics")

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 140, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 500, 80)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 100, 30)

    if st.button("Predict"):
        inputs = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        inputs_scaled = scaler.transform(inputs)
        prediction = model.predict(inputs_scaled)[0]

        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        st.success(f"Prediction: {result}")
