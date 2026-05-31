import os
import joblib

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier
)

from sklearn.linear_model import (
    LogisticRegression
)

from xgboost import XGBClassifier

from data_loader import load_data

from preprocessing import build_preprocessor


df = load_data()

df.drop(
    columns=["Loan_ID"],
    inplace=True
)

df["Loan_Status"] = (
    df["Loan_Status"]
    .map(
        {
            "Y": 1,
            "N": 0
        }
    )
)

X = df.drop(
    "Loan_Status",
    axis=1
)

y = df["Loan_Status"]

numerical_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

preprocessor = build_preprocessor(
    numerical_features,
    categorical_features
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=1
)

lr = LogisticRegression(
    max_iter=1000
)

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

estimators = [
    ("rf", rf),
    ("lr", lr),
    ("xgb", xgb)
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=3,
    n_jobs=1
)

model = Pipeline(
    [
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            stack
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(
    X_train,
    y_train
)

os.makedirs(
    "../models",
    exist_ok=True
)

joblib.dump(
    model,
    "../models/stack_classifier.pkl"
)

print("Training Complete")
print("Model Saved")