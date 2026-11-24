import pandas as pd
import joblib
import json


# ==========================================
# FILE PATHS
# ==========================================

DATA_PATH = "data/processed/creditcard_cleaned.csv"

RF_MODEL_PATH = "models/random_forest.pkl"
RF_SCALER_PATH = "models/random_forest_scaler.pkl"

FINAL_MODEL_PATH = "models/final_model.pkl"
FINAL_CONFIG_PATH = "models/final_config.json"


# ==========================================
# FINAL THRESHOLD
# ==========================================

FINAL_THRESHOLD = 0.50


# ==========================================
# LOAD DATA
# ==========================================

def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


# ==========================================
# CREATE FINAL MODEL PACKAGE
# ==========================================

def create_final_model():

    print("=" * 60)
    print("FINAL MODEL PACKAGING")
    print("=" * 60)

    # Load trained Random Forest
    model = joblib.load(RF_MODEL_PATH)

    # Load scaler used with Random Forest
    scaler = joblib.load(RF_SCALER_PATH)

    # Load dataset to get exact feature order
    df = load_data()

    feature_names = [
        column for column in df.columns
        if column != "Class"
    ]

    # Create model package
    final_model = {
        "model": model,
        "scaler": scaler,
        "threshold": FINAL_THRESHOLD,
        "features": feature_names
    }

    # Save complete model package
    joblib.dump(
        final_model,
        FINAL_MODEL_PATH
    )

    # Save configuration separately
    config = {
        "model_name": "Random Forest",
        "threshold": FINAL_THRESHOLD,
        "number_of_features": len(feature_names),
        "features": feature_names
    }

    with open(
        FINAL_CONFIG_PATH,
        "w"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    print("\nFinal model created successfully!")

    print("\nModel:")
    print("Random Forest")

    print("\nThreshold:")
    print(FINAL_THRESHOLD)

    print("\nNumber of features:")
    print(len(feature_names))

    print("\nFeatures:")
    print(feature_names)

    print("\nFinal model saved to:")
    print(FINAL_MODEL_PATH)

    print("\nFinal configuration saved to:")
    print(FINAL_CONFIG_PATH)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    create_final_model()