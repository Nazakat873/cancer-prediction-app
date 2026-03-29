import streamlit as st
import numpy as np
import joblib

st.title("Personalized Cancer Prediction System")

# Load model safely
model = joblib.load("model.pkl")
pca = joblib.load("pca.pkl")

st.write("Enter patient gene expression values")

inputs = []
for i in range(10):
    val = st.number_input(f"Gene {i+1}", value=0.0)
    inputs.append(val)

full_input = np.zeros(20531)
full_input[:10] = inputs

if st.button("Predict"):
    transformed = pca.transform([full_input])
    prediction = model.predict(transformed)
    prob = model.predict_proba(transformed)

    st.success(f"Predicted Class: {prediction[0]}")
    st.write("Confidence:", prob)
