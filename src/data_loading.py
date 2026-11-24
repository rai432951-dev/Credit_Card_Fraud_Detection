import pandas as pd


DATA_PATH = "data/raw/creditcard.csv"


def load_data():
    """Load the credit card fraud dataset."""
    df = pd.read_csv(DATA_PATH)
    return df


if __name__ == "__main__":

    df = load_data()

    print("========================================")
    print("CREDIT CARD FRAUD DETECTION PROJECT")
    print("========================================")

    print("\nDataset loaded successfully!")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)