import pandas as pd


RAW_DATA_PATH = "data/raw/creditcard.csv"
CLEANED_DATA_PATH = "data/processed/creditcard_cleaned.csv"


def load_data():
    """Load the raw dataset."""
    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df):
    """Clean the credit card fraud dataset."""

    print("=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    # Original shape
    print("\nOriginal dataset shape:")
    print(df.shape)

    # Check missing values
    missing_values = df.isnull().sum().sum()
    print("\nTotal missing values:")
    print(missing_values)

    # Check duplicate rows
    duplicate_count = df.duplicated().sum()
    print("\nDuplicate rows before cleaning:")
    print(duplicate_count)

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Check duplicates after cleaning
    print("\nDuplicate rows after cleaning:")
    print(df.duplicated().sum())

    # Final shape
    print("\nCleaned dataset shape:")
    print(df.shape)

    return df


def save_cleaned_data(df):
    """Save cleaned dataset."""
    df.to_csv(CLEANED_DATA_PATH, index=False)

    print("\nCleaned dataset saved successfully!")
    print(f"File location: {CLEANED_DATA_PATH}")


if __name__ == "__main__":

    df = load_data()

    cleaned_df = clean_data(df)

    save_cleaned_data(cleaned_df)