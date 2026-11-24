import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/creditcard_cleaned.csv"


def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def preprocess_data(df):

    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    print("\n1. Features shape:")
    print(X.shape)

    print("\n2. Target shape:")
    print(y.shape)

    print("\n3. Target distribution:")
    print(y.value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n4. Training data shape:")
    print(X_train.shape)

    print("\n5. Testing data shape:")
    print(X_test.shape)

    print("\n6. Training target distribution:")
    print(y_train.value_counts())

    print("\n7. Testing target distribution:")
    print(y_test.value_counts())

    # Feature scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n8. Scaling completed.")

    print("\nScaled training data shape:")
    print(X_train_scaled.shape)

    print("\nScaled testing data shape:")
    print(X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":

    df = load_data()

    preprocess_data(df)