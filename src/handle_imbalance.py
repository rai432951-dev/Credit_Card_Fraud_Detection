import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/processed/creditcard_cleaned.csv"


def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def prepare_data(df):

    print("=" * 60)
    print("CLASS IMBALANCE HANDLING")
    print("=" * 60)

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

    print("\nBefore SMOTE:")
    print("Training class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    # Scale features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE ONLY to training data
    smote = SMOTE(
        random_state=42
    )

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled,
        y_train
    )

    print("\nAfter SMOTE:")
    print("Training class distribution:")
    print(pd.Series(y_train_resampled).value_counts())

    print("\nTraining data shape after SMOTE:")
    print(X_train_resampled.shape)

    print("\nTesting data shape:")
    print(X_test_scaled.shape)

    return (
        X_train_resampled,
        X_test_scaled,
        y_train_resampled,
        y_test,
        scaler
    )


if __name__ == "__main__":

    df = load_data()

    prepare_data(df)