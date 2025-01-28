import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(data):
    """
    Perform exploratory data analysis (EDA) on the dataset.
    - Visualizes fraud vs. non-fraud distribution.
    - Analyzes the transaction amount distribution.

    Args:
        data (pd.DataFrame): The dataset for analysis.
    """
    try:
        # Fraud vs. Non-Fraud Distribution
        plt.figure(figsize=(8, 6))
        counts = data['Class'].value_counts()
        counts.plot(kind='bar', rot=0)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Distribution of Fraud vs Non-Fraud Transactions')
        plt.show()

        # Transaction Amount Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data['normalizedAmount'], bins=50, kde=True, color='blue', alpha=0.7)
        plt.xlabel('Normalized Transaction Amount')
        plt.ylabel('Frequency')
        plt.title('Transaction Amount Distribution')
        plt.show()

    except KeyError as e:
        print(f"Missing key column: {e}")
    except Exception as e:
        print(f"Error during EDA: {e}")


def calculate_statistics(data):
    """
    Calculate and display basic statistics of the dataset.
    - Mean transaction amount for fraud and non-fraud transactions.
    - Fraud vs. non-fraud transaction counts.

    Args:
        data (pd.DataFrame): The dataset for analysis.
    """
    try:
        # Calculate mean transaction amount for fraud and non-fraud
        mean_amount_fraud = data[data['Class'] == 1]['normalizedAmount'].mean()
        mean_amount_non_fraud = data[data['Class'] == 0]['normalizedAmount'].mean()

        print(f"Mean transaction amount for fraud transactions: {mean_amount_fraud:.2f}")
        print(f"Mean transaction amount for non-fraud transactions: {mean_amount_non_fraud:.2f}")

    except KeyError as e:
        print(f"Missing key column: {e}")
    except Exception as e:
        print(f"Error during statistical calculations: {e}")


def plot_correlation_matrix(data):
    """
    Generate a correlation matrix to analyze feature relationships.

    Args:
        data (pd.DataFrame): The dataset for analysis.
    """
    try:
        correlation_matrix = data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
        plt.title('Feature Correlation Matrix')
        plt.show()
    except Exception as e:
        print(f"Error during correlation matrix generation: {e}")


if __name__ == "__main__":
    import pandas as pd

    # Example usage
    dataset_path = "data/creditcard.csv"
    df = pd.read_csv(dataset_path)

    perform_eda(df)
    calculate_statistics(df)
    plot_correlation_matrix(df)
