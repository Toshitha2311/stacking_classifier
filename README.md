# Loan Approval Prediction using Stacking Classifier

## Overview

Predict loan approval status using a Stacking Classifier.

## Base Models

- Random Forest Classifier
- Logistic Regression
- XGBoost Classifier

## Meta Model

- Logistic Regression

## Dataset

Loan Prediction Dataset

## Features

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

## Target

Loan_Status

Y = Approved

N = Rejected

## Installation

pip install -r requirements.txt

## Train

cd src

python train.py

## Evaluate

python evaluate.py

## Run Application

cd ..

streamlit run app.py