import pandas as pd


DATA_PATH = "data/raw/creditcard.csv"


def load_data():
    """Load the raw credit card fraud dataset."""
    return pd.read_csv(DATA_PATH)


def inspect_data(df):
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    # 1. Shape
    print("\n1. Dataset Shape:")
    print(df.shape)

    # 2. Columns
    print("\n2. Columns:")
    print(df.columns.tolist())

    # 3. Data types
    print("\n3. Data Types:")
    print(df.dtypes)

    # 4. Missing values
    print("\n4. Missing Values:")
    print(df.isnull().sum())

    # 5. Total missing values
    print("\n5. Total Missing Values:")
    print(df.isnull().sum().sum())

    # 6. Duplicate rows
    print("\n6. Duplicate Rows:")
    print(df.duplicated().sum())

    # 7. Class distribution
    print("\n7. Class Distribution:")
    print(df["Class"].value_counts())

    # 8. Class percentage
    print("\n8. Class Percentage:")
    print(df["Class"].value_counts(normalize=True) * 100)

    # 9. Statistical summary
    print("\n9. Statistical Summary:")
    print(df.describe())


if __name__ == "__main__":
    df = load_data()

    inspect_data(df)