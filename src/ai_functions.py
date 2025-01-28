import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def detect_outliers(data, contamination=0.01):
    """
    Detect outliers using the Isolation Forest algorithm.

    Args:
        data (pd.DataFrame): The dataset for analysis (features only, without labels).
        contamination (float): The expected proportion of outliers in the data.

    Returns:
        np.ndarray: Array of predictions (-1 for outliers, 1 for inliers).
    """
    try:
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = isolation_forest.fit_predict(data)
        return predictions
    except Exception as e:
        print(f"Error during outlier detection: {e}")
        return None


def perform_pca(data, n_components=2):
    """
    Perform Principal Component Analysis (PCA) for dimensionality reduction.

    Args:
        data (pd.DataFrame): The dataset for dimensionality reduction.
        n_components (int): Number of principal components to compute.

    Returns:
        np.ndarray: Transformed data in reduced dimensions.
    """
    try:
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        return reduced_data
    except Exception as e:
        print(f"Error during PCA: {e}")
        return None


def classify_transactions(data, labels):
    """
    Train and evaluate a logistic regression model for fraud classification.

    Args:
        data (pd.DataFrame): Feature dataset for classification.
        labels (pd.Series): Target labels (e.g., 'Class' column).

    Returns:
        None
    """
    try:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Train a logistic regression model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions and evaluate
        predictions = model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    except Exception as e:
        print(f"Error during classification: {e}")


def detect_anomalies_with_lof(data, contamination=0.01):
    """
    Detect anomalies using the Local Outlier Factor (LOF) algorithm.

    Args:
        data (pd.DataFrame): The dataset for anomaly detection (features only).
        contamination (float): The expected proportion of anomalies in the data.

    Returns:
        np.ndarray: Array of predictions (-1 for anomalies, 1 for normal points).
    """
    try:
        lof = LocalOutlierFactor(contamination=contamination)
        predictions = lof.fit_predict(data)
        return predictions
    except Exception as e:
        print(f"Error during anomaly detection with LOF: {e}")
        return None


if __name__ == "__main__":
    import pandas as pd

    # Example usage
    dataset_path = "data/creditcard.csv"
    df = pd.read_csv(dataset_path)

    # Preprocess data
    features = df.drop(columns=["Class", "Time"])  # Drop non-feature columns
    labels = df["Class"]

    # Outlier detection
    outliers = detect_outliers(features)
    print(f"Outliers detected: {np.sum(outliers == -1)}")

    # Dimensionality reduction
    pca_result = perform_pca(features)
    print(f"PCA result shape: {pca_result.shape}")

    # Classification
    classify_transactions(features, labels)

    # Anomaly detection with LOF
    anomalies = detect_anomalies_with_lof(features)
    print(f"Anomalies detected: {np.sum(anomalies == -1)}")
