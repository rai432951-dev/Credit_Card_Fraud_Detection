import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from imblearn.over_sampling import SMOTE


DATA_PATH = "data/processed/creditcard_cleaned.csv"
MODEL_PATH = "models/logistic_regression.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def prepare_data(df):

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE ONLY to training data
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled,
        y_train
    )

    return (
        X_train_resampled,
        X_test_scaled,
        y_train_resampled,
        y_test,
        scaler
    )


def train_model(X_train, y_train):

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Model training completed!")

    return model


def evaluate_model(model, X_test, y_test):

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)

    # Prediction probabilities
    y_probability = model.predict_proba(X_test)[:, 1]

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\nROC-AUC Score:")
    print(roc_auc)

    # PR-AUC
    pr_auc = average_precision_score(y_test, y_probability)

    print("\nPR-AUC Score:")
    print(pr_auc)


def save_model(model, scaler):

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")


if __name__ == "__main__":

    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION")
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    # Load data
    df = load_data()

    print("\nDataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")

    # Prepare data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = prepare_data(df)

    print("\nTraining data after SMOTE:")
    print(X_train.shape)

    print("\nTesting data:")
    print(X_test.shape)

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(
        model,
        X_test,
        y_test
    )

    # Save
    save_model(model, scaler)