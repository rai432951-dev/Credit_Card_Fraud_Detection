import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE


# ==========================================
# FILE PATHS
# ==========================================

DATA_PATH = "data/processed/creditcard_cleaned.csv"

RF_MODEL_PATH = "models/random_forest.pkl"
RF_SCALER_PATH = "models/random_forest_scaler.pkl"

XGB_MODEL_PATH = "models/xgboost.pkl"
XGB_SCALER_PATH = "models/xgboost_scaler.pkl"


# ==========================================
# LOAD DATA
# ==========================================

def load_data():
    return pd.read_csv(DATA_PATH)


# ==========================================
# PREPARE TEST DATA
# ==========================================

def prepare_test_data(df):

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Use scaler fitted during training
    scaler = StandardScaler()

    scaler.fit(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, y_test


# ==========================================
# THRESHOLD ANALYSIS
# ==========================================

def analyze_thresholds(model, X_test, y_test, model_name):

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]

    results = []

    for threshold in thresholds:

        y_pred = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        results.append({
            "Model": model_name,
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    return pd.DataFrame(results)


# ==========================================
# PLOT RESULTS
# ==========================================

def plot_results(results, model_name):

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["Threshold"],
        results["Precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        results["Threshold"],
        results["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        results["Threshold"],
        results["F1-Score"],
        marker="o",
        label="F1-Score"
    )

    plt.title(
        f"{model_name} - Threshold Analysis"
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")

    plt.xticks(
        results["Threshold"]
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
    f"data/processed/{model_name.lower().replace(' ', '_')}_threshold_analysis.png"
)

plt.close()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("FRAUD DETECTION THRESHOLD ANALYSIS")
    print("=" * 60)

    # Load dataset
    df = load_data()

    print("\nDataset loaded successfully!")

    # Prepare test data
    X_test, y_test = prepare_test_data(df)

    # --------------------------------------
    # Random Forest
    # --------------------------------------

    print("\nLoading Random Forest...")

    rf_model = joblib.load(
        RF_MODEL_PATH
    )

    rf_scaler = joblib.load(
        RF_SCALER_PATH
    )

    # IMPORTANT:
    # Transform test data using saved RF scaler
    X = df.drop("Class", axis=1)
    y = df["Class"]

    _, X_test_raw, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_test_rf = rf_scaler.transform(
        X_test_raw
    )

    rf_results = analyze_thresholds(
        rf_model,
        X_test_rf,
        y_test,
        "Random Forest"
    )

    print("\n" + "=" * 60)
    print("RANDOM FOREST THRESHOLD RESULTS")
    print("=" * 60)

    print(rf_results.to_string(index=False))

    plot_results(
        rf_results,
        "Random Forest"
    )

    # --------------------------------------
    # XGBoost
    # --------------------------------------

    print("\nLoading XGBoost...")

    xgb_model = joblib.load(
        XGB_MODEL_PATH
    )

    xgb_scaler = joblib.load(
        XGB_SCALER_PATH
    )

    X_test_xgb = xgb_scaler.transform(
        X_test_raw
    )

    xgb_results = analyze_thresholds(
        xgb_model,
        X_test_xgb,
        y_test,
        "XGBoost"
    )

    print("\n" + "=" * 60)
    print("XGBOOST THRESHOLD RESULTS")
    print("=" * 60)

    print(xgb_results.to_string(index=False))

    plot_results(
        xgb_results,
        "XGBoost"
    )

    # --------------------------------------
    # Combined Results
    # --------------------------------------

    all_results = pd.concat(
        [
            rf_results,
            xgb_results
        ],
        ignore_index=True
    )

    all_results.to_csv(
        "data/processed/threshold_results.csv",
        index=False
    )

    print("\nThreshold analysis saved to:")

    print(
        "data/processed/threshold_results.csv"
    )