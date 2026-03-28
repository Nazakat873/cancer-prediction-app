import streamlit as st
import numpy as np
import pickle

st.title("Personalized Cancer Prediction System")

st.write("Enter patient gene expression values")

# Load model + PCA
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

# For demo: take only 10 inputs (instead of 20k)
inputs = []
for i in range(10):
    val = st.number_input(f"Gene {i+1}", value=0.0)
    inputs.append(val)

# Pad remaining features with zeros
full_input = np.zeros(20531)
full_input[:10] = inputs

if st.button("Predict"):
    transformed = pca.transform([full_input])
    prediction = model.predict(transformed)
    prob = model.predict_proba(transformed)

    st.success(f"Predicted Class: {prediction[0]}")
    st.write("Prediction Confidence:", prob)
