import joblib
import pandas as pd

MODEL_PATH = "models/final_model.pkl"

# Ek baar load hoga, jab server start hoga
_package = joblib.load(MODEL_PATH)
MODEL = _package["model"]
SCALER = _package["scaler"]
THRESHOLD = _package["threshold"]
FEATURES = _package["features"]


def predict_fraud(transaction: dict):
    missing = [f for f in FEATURES if f not in transaction]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    df = pd.DataFrame([transaction], columns=FEATURES)

    scaled = SCALER.transform(df)
    fraud_probability = float(MODEL.predict_proba(scaled)[0][1])

    prediction = int(fraud_probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "result": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": fraud_probability,
        "threshold": float(THRESHOLD),
    }