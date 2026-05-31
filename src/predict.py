import joblib
import pandas as pd

model = joblib.load(
    "models/stack_classifier.pkl"
)


def predict_loan(features):

    df = pd.DataFrame(
        [features]
    )

    prediction = model.predict(
        df
    )[0]

    probability = model.predict_proba(
        df
    )[0][1]

    return prediction, probability