import streamlit as st

from src.predict import predict_loan

st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="wide"
)

st.title(
    "🏦 Loan Approval Prediction"
)

st.write(
    """
    This application uses a Stacking Classifier.

    Base Models:
    - Random Forest
    - Logistic Regression
    - XGBoost

    Meta Learner:
    - Logistic Regression
    """
)

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

with col2:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=0
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

    property_area = st.selectbox(
        "Property Area",
        [
            "Urban",
            "Semiurban",
            "Rural"
        ]
    )

if st.button(
    "Predict Loan Status"
):

    features = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    prediction, probability = predict_loan(
        features
    )

    if prediction == 1:

        st.success(
            f"Loan Approved ✅\n\nConfidence: {probability:.2%}"
        )

    else:

        st.error(
            f"Loan Rejected ❌\n\nConfidence: {(1-probability):.2%}"
        )