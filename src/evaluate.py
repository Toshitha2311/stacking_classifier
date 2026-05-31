import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

from data_loader import load_data


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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load(
    "../models/stack_classifier.pkl"
)

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        predictions
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        predictions
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        predictions
    )
)

print(
    "F1 Score:",
    f1_score(
        y_test,
        predictions
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        probabilities
    )
)

print(
    "\nConfusion Matrix\n"
)

print(
    confusion_matrix(
        y_test,
        predictions
    )
)