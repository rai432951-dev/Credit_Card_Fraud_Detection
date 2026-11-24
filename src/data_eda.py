import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/creditcard_cleaned.csv"


def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def perform_eda(df):
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Basic information
    print("\n1. Dataset Shape:")
    print(df.shape)

    # Class distribution
    print("\n2. Class Distribution:")
    print(df["Class"].value_counts())

    print("\n3. Class Percentage:")
    print(df["Class"].value_counts(normalize=True) * 100)

    # Transaction amount
    print("\n4. Transaction Amount Statistics:")
    print(df["Amount"].describe())

    # Fraud transaction amount
    print("\n5. Fraud Transaction Amount Statistics:")
    print(df[df["Class"] == 1]["Amount"].describe())

    # Legitimate transaction amount
    print("\n6. Legitimate Transaction Amount Statistics:")
    print(df[df["Class"] == 0]["Amount"].describe())


def visualize_data(df):

    # 1. Class Distribution
    plt.figure(figsize=(8, 5))

    sns.countplot(data=df, x="Class")

    plt.title("Fraud vs Legitimate Transactions")
    plt.xlabel("Transaction Class")
    plt.ylabel("Number of Transactions")
    plt.show()

    # 2. Transaction Amount Distribution
    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="Amount",
        bins=50
    )

    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.show()

    # 3. Fraud vs Legitimate Amount
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Class",
        y="Amount"
    )

    plt.title("Transaction Amount by Class")
    plt.xlabel("Transaction Class")
    plt.ylabel("Amount")
    plt.show()

    # 4. Transaction Time Distribution
    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="Time",
        bins=50
    )

    plt.title("Transaction Time Distribution")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":

    df = load_data()

    perform_eda(df)

    visualize_data(df)