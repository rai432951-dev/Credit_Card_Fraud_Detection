import pandas as pd
import joblib


# ==========================================
# FILE PATH
# ==========================================

MODEL_PATH = "models/final_model.pkl"


# ==========================================
# LOAD FINAL MODEL
# ==========================================

def load_model():

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    scaler = package["scaler"]
    threshold = package["threshold"]
    features = package["features"]

    return model, scaler, threshold, features


# ==========================================
# PREDICT FRAUD
# ==========================================

def predict_fraud(transaction):

    model, scaler, threshold, features = load_model()

    # Convert transaction into DataFrame
    transaction_df = pd.DataFrame(
        [transaction],
        columns=features
    )

    # Scale input
    transaction_scaled = scaler.transform(
        transaction_df
    )

    # Get fraud probability
    fraud_probability = model.predict_proba(
        transaction_scaled
    )[0][1]

    # Apply final threshold
    prediction = int(
        fraud_probability >= threshold
    )

    # Convert prediction to readable result
    if prediction == 1:
        result = "Fraud"
    else:
        result = "Legitimate"

    return {
        "prediction": prediction,
        "result": result,
        "fraud_probability": float(fraud_probability),
        "threshold": threshold
    }


# ==========================================
# TEST PREDICTION
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("FRAUD PREDICTION TEST")
    print("=" * 60)

    # Example transaction
    transaction = {
        "Time": 406,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62
    }

    result = predict_fraud(transaction)

    print("\nPrediction Result:")
    print(result)