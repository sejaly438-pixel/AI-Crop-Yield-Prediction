import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("crop_model.pkl")

st.title("🌾 AI Crop Yield Predictor")

st.write("Enter environmental and soil parameters to predict crop yield.")

# Input fields with explanation
st.info("This AI model predicts crop yield based on environmental conditions and soil nutrients.")

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=120.0,
    help="Total rainfall in millimeters. Example: 100 - 300 mm"
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=28.0,
    help="Average temperature in Celsius. Example: 20 - 35 °C"
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    help="Air humidity percentage. Example: 40 - 80%"
)

nitrogen = st.number_input(
    "Nitrogen in Soil (kg/ha)",
    min_value=0.0,
    max_value=200.0,
    value=80.0,
    help="Nitrogen content in soil. Example: 50 - 150"
)

phosphorus = st.number_input(
    "Phosphorus in Soil (kg/ha)",
    min_value=0.0,
    max_value=200.0,
    value=40.0,
    help="Phosphorus content in soil. Example: 20 - 100"
)

potassium = st.number_input(
    "Potassium in Soil (kg/ha)",
    min_value=0.0,
    max_value=200.0,
    value=50.0,
    help="Potassium content in soil. Example: 30 - 120"
)

# Prediction button
if st.button("Predict Yield"):
    
    features = np.array([[rainfall, temperature, humidity, nitrogen, phosphorus, potassium]])
    
    prediction = model.predict(features)

    st.success(f"Predicted Crop Yield: {prediction[0]:.2f} tons per hectare")