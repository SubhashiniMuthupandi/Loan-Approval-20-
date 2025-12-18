import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model
try:
    with open('Model7/model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'Model7/model.pkl' exists.")

st.title("Loan Approval Prediction App")
st.write("Enter the details below to check the loan approval status.")

# Input fields
income = st.number_input("Annual Income", min_value=0, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
loan_amount = st.number_input("Loan Amount Requested", min_value=0, value=20000)

# Predict button
if st.button("Predict Status"):
    # Create a dataframe for the input to match feature names
    features = pd.DataFrame([[income, credit_score, loan_amount]], 
                            columns=['Income', 'Credit Score', 'Loan Amount'])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Display result
    result = prediction[0]
    if result == 'Approved':
        st.success(f"Result: {result}")
    else:
        st.error(f"Result: {result}")

st.info("Note: This model has an accuracy of approximately 54.4%.")
