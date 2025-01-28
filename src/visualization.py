import matplotlib.pyplot as plt
import seaborn as sns


def plot_fraud_distribution(data):
    """
    Plot the distribution of fraud vs. non-fraud transactions.

    Args:
        data (pd.DataFrame): The dataset containing the 'Class' column.
    """
    try:
        counts = data['Class'].value_counts()
        plt.figure(figsize=(8, 6))
        counts.plot(kind='bar', rot=0, color=['green', 'red'])
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Distribution of Fraud vs Non-Fraud Transactions')
        plt.xticks([0, 1], ['Non-Fraud', 'Fraud'], rotation=0)
        plt.show()
    except KeyError:
        print("Error: 'Class' column not found in the dataset.")
    except Exception as e:
        print(f"Error during fraud distribution plot: {e}")


def plot_correlation_matrix(data):
    """
    Plot the correlation matrix of the dataset.

    Args:
        data (pd.DataFrame): The dataset for which the correlation matrix is computed.
    """
    try:
        correlation_matrix = data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
        plt.title('Feature Correlation Matrix')
        plt.show()
    except Exception as e:
        print(f"Error during correlation matrix plot: {e}")


def plot_transaction_amount(data):
    """
    Plot the distribution of transaction amounts (normalized).

    Args:
        data (pd.DataFrame): The dataset containing the 'normalizedAmount' column.
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data['normalizedAmount'], bins=50, kde=True, color='blue', alpha=0.7)
        plt.xlabel('Normalized Transaction Amount')
        plt.ylabel('Frequency')
        plt.title('Transaction Amount Distribution')
        plt.show()
    except KeyError:
        print("Error: 'normalizedAmount' column not found in the dataset.")
    except Exception as e:
        print(f"Error during transaction amount plot: {e}")


def plot_pca_scatter(pca_data, labels, title="PCA Anomaly Detection"):
    """
    Plot a PCA scatter plot with color-coded labels.

    Args:
        pca_data (pd.DataFrame): Data containing PCA-transformed features with 'PC1' and 'PC2'.
        labels (pd.Series): Labels for coloring the data points.
        title (str): Title of the scatter plot.
    """
    try:
        plt.figure(figsize=(8, 6))
        colors = ['red' if label == -1 else 'green' for label in labels]
        plt.scatter(pca_data['PC1'], pca_data['PC2'], c=colors, alpha=0.7)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(title)
        plt.show()
    except KeyError as e:
        print(f"Error: Missing required column for PCA scatter plot: {e}")
    except Exception as e:
        print(f"Error during PCA scatter plot: {e}")


def plot_fraud_pie_chart(data):
    """
    Plot a pie chart showing the ratio of fraud to non-fraud transactions.

    Args:
        data (pd.DataFrame): The dataset containing the 'Class' column.
    """
    try:
        counts = data['Class'].value_counts()
        labels = ['Non-Fraud', 'Fraud']
        colors = ['green', 'red']
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title('Fraud vs Non-Fraud Transaction Ratio')
        plt.show()
    except KeyError:
        print("Error: 'Class' column not found in the dataset.")
    except Exception as e:
        print(f"Error during pie chart plot: {e}")


if __name__ == "__main__":
    import pandas as pd

    # Example usage
    dataset_path = "data/creditcard.csv"
    df = pd.read_csv(dataset_path)

    # Visualization examples
    plot_fraud_distribution(df)
    plot_correlation_matrix(df)
    plot_transaction_amount(df)
    # Example PCA data
    # df_pca = pd.DataFrame({'PC1': [1, 2, 3], 'PC2': [4, 5, 6]})
    # labels = [1, -1, 1]
    # plot_pca_scatter(df_pca, labels)
    plot_fraud_pie_chart(df)
