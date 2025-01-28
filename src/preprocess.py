import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        
    Returns:
        pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def preprocess_data(data):
    """
    Preprocess the dataset for analysis and modeling.
    - Standardizes the 'Amount' column.
    - Drops unnecessary columns (e.g., 'Amount').
    - Ensures data is ready for further processing.

    Args:
        data (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    try:
        # Standardizing the 'Amount' column
        scaler = StandardScaler()
        data['normalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
        
        # Dropping the original 'Amount' column
        data = data.drop(['Amount'], axis=1)
        
        print("Preprocessing completed successfully.")
        return data
    except KeyError as e:
        print(f"Column missing in dataset: {e}")
        return None
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    dataset_path = "data/creditcard.csv"
    df = load_data(dataset_path)
    if df is not None:
        processed_data = preprocess_data(df)
        print(processed_data.head())
