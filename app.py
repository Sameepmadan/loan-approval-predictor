import streamlit as st
import numpy as np
import pandas as pd
import joblib


model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")

st.title("🏦 Loan Approval Predictor")
st.write("Write your details below to check if your loan will be approved or not.")


col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=130)
    loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 60, 300, 240, 84, 36, 12])
    credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])


if st.button("Predict Loan Approval"):

 
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dependents_val = 3 if dependents == "3+" else int(dependents)
    education_val = 0 if education == "Graduate" else 1
    self_employed_val = 1 if self_employed == "Yes" else 0
    property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_val = property_map[property_area]
    credit_val = 1 if credit_history == "Good (1)" else 0

    total_income = applicant_income + coapplicant_income

    # Sample DataFrame banao - training ke exact same column order mein
    input_data = pd.DataFrame({
        'Gender': [gender_val],
        'Married': [married_val],
        'Dependents': [dependents_val],
        'Education': [education_val],
        'Self_Employed': [self_employed_val],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_val],
        'Property_Area': [property_val],
        'Total_Income': [total_income],
        'Total_Income_log': [np.log(total_income) if total_income > 0 else 0],
        'LoanAmount_log': [np.log(loan_amount)]
    })


    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.divider()

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved! (Confidence: {probability[0][1]*100:.1f}%)")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {probability[0][0]*100:.1f}%)")
