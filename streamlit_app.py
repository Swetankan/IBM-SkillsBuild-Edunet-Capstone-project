import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Heart Disease Prediction App")
st.markdown("This app predicts whether a person has heart disease using various ML models.")

# Input fields
def user_input_features():
    age = st.slider('Age', 18, 100, 45)
    sex = st.selectbox('Sex', [0, 1])
    cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    trestbps = st.slider('Resting Blood Pressure (trestbps)', 80, 200, 120)
    chol = st.slider('Cholesterol (chol)', 100, 400, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.selectbox('Resting ECG (restecg)', [0, 1, 2])
    thalach = st.slider('Max Heart Rate Achieved (thalach)', 70, 210, 150)
    exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.slider('ST depression (oldpeak)', 0.0, 6.0, 1.0)
    slope = st.selectbox('Slope of ST segment (slope)', [0, 1, 2])
    ca = st.selectbox('Number of major vessels (ca)', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Load trained model and scaler
model = joblib.load("model.pkl")  # Ensure 'model.pkl' is saved in the same folder
scaler = joblib.load("scaler.pkl")  # Ensure 'scaler.pkl' is saved too

# Preprocess and predict
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)

# Output
st.subheader('Prediction')
st.write('Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease')
