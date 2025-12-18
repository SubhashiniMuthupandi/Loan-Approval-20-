import streamlit as st
import pandas as pd
import pickle
import os

# Get the absolute path to the model file
# This helps prevent 'File Not Found' errors
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'Model7', 'model.pkl')

st.title("Loan Approval Prediction App")

# Check if model exists before loading
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    st.write("Enter the details below to check the loan approval status.")

    # Input fields
    income = st.number_input("Annual Income", min_value=0, value=67669)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    loan_amount = st.number_input("Loan Amount Requested", min_value=0, value=45000)

    if st.button("Predict Status"):
        # Features must match the training column names exactly
        features = pd.DataFrame([[income, credit_score, loan_amount]], 
                                columns=['Income', 'Credit Score', 'Loan Amount'])
        
        prediction = model.predict(features)
        
        if prediction[0] == 'Approved':
            st.success(f"Result: {prediction[0]}")
        else:
            st.error(f"Result: {prediction[0]}")
else:
    st.error(f"Error: Could not find 'model.pkl' at {model_path}")
    st.info("Please make sure you have a folder named 'Model7' with 'model.pkl' inside it.")
