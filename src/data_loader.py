import pandas as pd


def load_data():

    df = pd.read_csv(
        "../data/loan_data.csv"
    )

    return df