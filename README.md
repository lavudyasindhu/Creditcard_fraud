# Credit Card Fraud Detection (threshold demo)

## Run
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

On first run the app creates a synthetic `dataset/creditcard.csv` (if missing) and trains
`model/trained_model.pkl`. To use the REAL Kaggle dataset, download `creditcard.csv`
into `dataset/`, delete `model/trained_model.pkl`, then run `python train_model.py`.

## How the threshold works
`app.py` computes the model's fraud probability for every test transaction once. For each
slider value: `probability >= threshold -> Fraud, else Legitimate`, then TP/TN/FP/FN,
accuracy, recall, precision and F1 are recomputed from those predictions.
