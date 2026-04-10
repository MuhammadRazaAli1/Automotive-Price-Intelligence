"""
Automotive Price Intelligence - Web Interface
Author: Muhammad Raza Ali
Description: A Streamlit web application that serves the trained Random Forest model
for real-time used car price predictions.
"""

import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

# Configure page settings
st.set_page_config(page_title="UniCarPrice Predictor", page_icon="🚗", layout="centered")

MODEL_PATH = os.path.join("models", "model.joblib")

# CSS for a cleaner look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px; background-color: #007bff; color: white;}
    .stButton>button:hover {background-color: #0056b3; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("🚗 UniCarPrice Predictor")
st.markdown("Enter the specifications of your vehicle below to get a real-time, AI-driven price estimate.")

# Load Model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

if not model:
    st.error("⚠️ Model not found! Please train the model first by running `python src/train.py`.")
    st.stop()

# UI Layout configuration
st.subheader("Vehicle Specifications")

col1, col2 = st.columns(2)

with col1:
    make = st.text_input("Brand / Make", value="Toyota", placeholder="e.g. Honda")
    model_name = st.text_input("Model", value="Corolla", placeholder="e.g. Civic")
    year = st.number_input("Manufacture Year", min_value=1990, max_value=datetime.now().year, value=2018)
    odometer = st.number_input("Odometer (km)", min_value=0, value=60000, step=1000)
    body_type = st.selectbox("Body Type", ["sedan", "hatchback", "suv", "truck", "van", "other"], index=0)

with col2:
    fuel = st.selectbox("Fuel Type", ["petrol", "diesel", "hybrid", "electric", "other"], index=0)
    transmission = st.selectbox("Transmission", ["manual", "automatic", "cvt", "other"], index=1)
    engine_cc = st.number_input("Engine Size (cc)", min_value=600, max_value=8000, value=1300, step=100)
    power_hp = st.number_input("Horsepower (HP)", min_value=30, max_value=1200, value=95, step=10)
    condition = st.selectbox("Condition", ["new", "like new", "good", "fair", "salvage"], index=2)

st.subheader("Listing Details")
with st.expander("Add Optional Details (Improves Prediction)"):
    seats = st.slider("Number of Seats", min_value=2, max_value=10, value=5)
    state = st.text_input("State / Region", value="Punjab")
    seller_type = st.selectbox("Seller Type", ["dealer", "individual"], index=1)
    title = st.text_input("Ad Title", value="Toyota Corolla 2018 Low Mileage")
    description = st.text_area("Ad Description", value="Well maintained, single owner, service history available.")

# Prediction Logic
if st.button("Calculate Estimated Price"):
    with st.spinner("Analyzing market data..."):
        # Match the exact column names expected by the derive_features function in train.py
        input_data = pd.DataFrame([{
            "make": make, "model": model_name, "year": year, 
            "odometer": odometer, "body_type": body_type,
            "fuel": fuel, "transmission": transmission, 
            "engine_cc": engine_cc, "power_hp": power_hp, 
            "condition": condition, "seats": seats, 
            "state": state, "seller_type": seller_type,
            "title": title, "description": description
        }])
        
        # Note: derive_features handles 'car_age' and 'combined_text' logic automatically
        # when the pipeline transforms the data if it was set up inside the pipeline. 
        # Since derive_features was outside the pipeline in train.py, we apply it here too:
        
        current_year = datetime.now().year
        input_data["car_age"] = max(current_year - year, 0)
        input_data["combined_text"] = (title + " " + description).strip()

        try:
            prediction = model.predict(input_data)[0]
            st.success(f"💰 Estimated Market Value: **Rs. {prediction:,.0f}**")
            st.balloons()
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
