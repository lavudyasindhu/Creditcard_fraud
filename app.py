"""
Credit Card Fraud Detection - Flask backend
Run:  python app.py     then open http://127.0.0.1:5000
"""
import os
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from train_model import MODEL_PATH, load_split, train

app = Flask(__name__)

# ---------- Load model + test data once, at startup ----------
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train()
_, X_test, _, y_test = load_split()
X_test = X_test.reset_index(drop=True)
y_true = y_test.reset_index(drop=True).to_numpy()

# Fraud probability for every test transaction (the model's real output)
fraud_prob = model.predict_proba(X_test)[:, 1]


# ---------- Helpers ----------
def compute_metrics(threshold):
    """Apply the threshold to the real probabilities and compute metrics."""
    y_pred = (fraud_prob >= threshold).astype(int)   # >= threshold -> Fraud

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def safe_div(a, b):
        return a / b if b else 0.0

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, accuracy=accuracy,
                recall=recall, precision=precision, f1=f1)


def clamp_threshold(value):
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def describe_transaction(idx, threshold):
    p = float(fraud_prob[idx])
    row = X_test.iloc[idx]
    return dict(
        id=int(idx),
        probability=p,
        prediction="Fraudulent" if p >= threshold else "Legitimate",
        actual="Fraudulent" if y_true[idx] == 1 else "Legitimate",
        amount=float(row["Amount"]),
        time=float(row["Time"]),
        features={k: round(float(row[k]), 3) for k in ["V1", "V2", "V3", "V4", "V14"]},
    )


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/evaluate")
def evaluate():
    """Metrics for a threshold (+ optional prediction for one transaction)."""
    threshold = clamp_threshold(request.args.get("threshold", 0.5))
    result = dict(threshold=threshold, metrics=compute_metrics(threshold),
                  total=int(len(y_true)), total_fraud=int(y_true.sum()))
    sample_id = request.args.get("sample_id", type=int)
    if sample_id is not None and 0 <= sample_id < len(y_true):
        result["transaction"] = describe_transaction(sample_id, threshold)
    return jsonify(result)


@app.route("/api/sample")
def sample():
    """Pick a random test transaction: kind = any | fraud | legit."""
    kind = request.args.get("kind", "any")
    threshold = clamp_threshold(request.args.get("threshold", 0.5))
    if kind == "fraud":
        pool = np.where(y_true == 1)[0]
    elif kind == "legit":
        pool = np.where(y_true == 0)[0]
    else:
        pool = np.arange(len(y_true))
    idx = int(np.random.choice(pool))
    return jsonify(describe_transaction(idx, threshold))


@app.route("/api/curve")
def curve():
    """Accuracy / recall / precision for thresholds 0.00 ... 1.00."""
    points = []
    for t in np.round(np.arange(0, 1.0001, 0.01), 2):
        m = compute_metrics(float(t))
        points.append(dict(threshold=float(t), accuracy=m["accuracy"],
                           recall=m["recall"], precision=m["precision"]))
    return jsonify(points)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
