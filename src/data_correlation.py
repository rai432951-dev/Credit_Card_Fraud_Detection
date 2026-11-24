import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/creditcard_cleaned.csv"


def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv(DATA_PATH)


def correlation_analysis(df):

    print("=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Correlation with target variable
    class_correlation = correlation_matrix["Class"].sort_values(
        ascending=False
    )

    print("\nCorrelation with Class:")
    print(class_correlation)

    # Plot heatmap
    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    df = load_data()

    correlation_analysis(df)