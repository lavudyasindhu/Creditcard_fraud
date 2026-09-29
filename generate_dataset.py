"""
Creates a SYNTHETIC dataset with the same columns as the Kaggle
"Credit Card Fraud Detection" dataset (Time, V1..V28, Amount, Class).

Use this only if you don't have the real file. To use the real data,
download creditcard.csv from Kaggle and put it in the dataset/ folder
(the app will then use it automatically).
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset", "creditcard.csv")


def generate(n_rows=30000, fraud_rate=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    # Legitimate transactions: standard normal features
    legit = rng.normal(0, 1, size=(n_legit, 28))

    # Fraud transactions: shifted on some features, with more spread.
    # The shift is moderate on purpose, so the two classes overlap and the
    # threshold visibly changes the results.
    fraud = rng.normal(0, 1.3, size=(n_fraud, 28))
    shift = np.zeros(28)
    shift[[0, 2, 3, 6, 9, 10, 13, 16]] = [-1.4, -1.2, 1.2, -1.0, -1.1, 1.0, -1.3, -0.9]
    fraud += shift

    X = np.vstack([legit, fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)

    time = rng.uniform(0, 172792, size=len(y))
    amount = np.where(
        y == 1,
        rng.lognormal(4.2, 1.4, size=len(y)),
        rng.lognormal(3.6, 1.2, size=len(y)),
    ).round(2)

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time.round(0))
    df["Amount"] = amount
    df["Class"] = y
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def ensure_dataset():
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        print("creditcard.csv not found -> generating a synthetic dataset...")
        generate().to_csv(CSV_PATH, index=False)
        print("Saved:", CSV_PATH)
    return CSV_PATH


if __name__ == "__main__":
    ensure_dataset()
