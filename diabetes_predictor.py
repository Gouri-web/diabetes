import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to check diabetes prediction using Random Forest")

# Sidebar - User input
st.sidebar.header("Patient Information")

def user_input_features():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 15, 1)
    glucose = st.sidebar.slider("Glucose", 40, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 40, 130, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0.0, 800.0, 79.0)
    bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
    diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 18, 100, 33)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

user_input = user_input_features()

# Split data and train model
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

st.subheader("Prediction:")
result = "🟢 No Diabetes" if prediction[0] == 0 else "🔴 Positive for Diabetes"
st.write(result)

st.subheader("Prediction Probability:")
st.write(f"No Diabetes: {prediction_proba[0][0]:.2f} | Diabetes: {prediction_proba[0][1]:.2f}")

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy on Test Set: **{acc:.2f}**")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)