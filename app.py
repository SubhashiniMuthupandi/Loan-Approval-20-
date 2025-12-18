import streamlit as st
import pandas as pd
import pickle
import os

# --- CONFIGURATION ---
# We use the filename you provided: 'loan_approval_model.pkl'
MODEL_FOLDER = "Model7"
MODEL_FILE = "loan_approval_model.pkl"

# This logic finds the folder regardless of where you run the script from
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, MODEL_FOLDER, MODEL_FILE)

st.set_page_config(page_title="Loan Approval Predictor")
st.title("🏦 Loan Approval Prediction App")

# --- LOAD MODEL ---
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    st.success("✅ Model loaded successfully!")
    
    st.write("### Enter Applicant Details")
    
    # Input fields based on your dataset
    income = st.number_input("Annual Income", min_value=0, value=67000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    loan_amount = st.number_input("Loan Amount Requested", min_value=0, value=20000)

    if st.button("Predict Approval Status"):
        # The dataframe columns must match the training features exactly
        features = pd.DataFrame([[income, credit_score, loan_amount]], 
                                columns=['Income', 'Credit Score', 'Loan Amount'])
        
        prediction = model.predict(features)
        
        if prediction[0] == 'Approved':
            st.balloons()
            st.success(f"Prediction: **{prediction[0]}**")
        else:
            st.error(f"Prediction: **{prediction[0]}**")

else:
    # --- TROUBLESHOOTING UI ---
    st.error(f"❌ Could not find the file: {MODEL_FILE}")
    st.info(f"The app is looking in this location: `{model_path}`")
    
    st.write("---")
    st.write("### 🛠️ Troubleshooting Checklist:")
    st.write(f"1. Is there a folder named `{MODEL_FOLDER}` in your project directory?")
    st.write(f"2. Is the file inside that folder named exactly `{MODEL_FILE}`?")
    
    # Show what files actually exist to help the user
    st.write("#### Files found in your project folder:")
    st.code(os.listdir(base_path))
    
    if os.path.exists(os.path.join(base_path, MODEL_FOLDER)):
        st.write(f"#### Files found inside {MODEL_FOLDER}:")
        st.code(os.listdir(os.path.join(base_path, MODEL_FOLDER)))
