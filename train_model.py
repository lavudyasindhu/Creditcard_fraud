"""
Trains the fraud-detection model and saves it to model/trained_model.pkl
Run:  python train_model.py
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from generate_dataset import ensure_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_model.pkl")


def load_split():
    """Load the CSV and return the SAME train/test split every time."""
    df = pd.read_csv(ensure_dataset())
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train():
    X_train, X_test, y_train, y_test = load_split()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Test accuracy at threshold 0.50: {model.score(X_test, y_test):.4f}")
    return model


if __name__ == "__main__":
    train()
