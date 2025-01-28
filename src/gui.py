import tkinter as tk
from tkinter import messagebox
from preprocess import load_data, preprocess_data
from analytics import perform_eda, calculate_statistics, plot_correlation_matrix
from ai_functions import detect_outliers, perform_pca, classify_transactions, detect_anomalies_with_lof
from visualization import plot_fraud_distribution, plot_transaction_amount, plot_fraud_pie_chart

# Initialize global variables
dataset = None
preprocessed_data = None


def load_and_preprocess():
    """Load and preprocess the dataset."""
    global dataset, preprocessed_data
    dataset_path = "data/creditcard.csv"  # Adjust path if necessary
    dataset = load_data(dataset_path)
    if dataset is not None:
        preprocessed_data = preprocess_data(dataset)
        messagebox.showinfo("Success", "Dataset loaded and preprocessed successfully!")


def perform_data_analytics():
    """Perform data analytics tasks."""
    if preprocessed_data is not None:
        perform_eda(preprocessed_data)
        calculate_statistics(preprocessed_data)
        plot_correlation_matrix(preprocessed_data)
    else:
        messagebox.showwarning("Warning", "Please load and preprocess the dataset first!")


def perform_ai_tasks():
    """Perform AI tasks such as outlier detection and classification."""
    if preprocessed_data is not None:
        # Outlier Detection
        outliers = detect_outliers(preprocessed_data.drop(columns=["Class"]))
        preprocessed_data["Outliers"] = outliers

        # PCA
        pca_data = perform_pca(preprocessed_data.drop(columns=["Class"]))
        plot_fraud_distribution(preprocessed_data)

        # Classification
        classify_transactions(preprocessed_data.drop(columns=["Class", "Outliers"]), preprocessed_data["Class"])

        # Anomaly Detection
        anomalies = detect_anomalies_with_lof(preprocessed_data.drop(columns=["Class", "Outliers"]))
        preprocessed_data["Anomalies"] = anomalies

        plot_fraud_pie_chart(preprocessed_data)
    else:
        messagebox.showwarning("Warning", "Please load and preprocess the dataset first!")


# Create the main GUI window
window = tk.Tk()
window.title("Fraud Detection System")
window.geometry("600x400")

# Add GUI elements
title_label = tk.Label(window, text="Fraud Detection System", font=("Arial", 18, "bold"))
title_label.pack(pady=20)

load_button = tk.Button(window, text="Load and Preprocess Data", command=load_and_preprocess, width=25, height=2)
load_button.pack(pady=10)

analytics_button = tk.Button(window, text="Perform Data Analytics", command=perform_data_analytics, width=25, height=2)
analytics_button.pack(pady=10)

ai_button = tk.Button(window, text="Perform AI Tasks", command=perform_ai_tasks, width=25, height=2)
ai_button.pack(pady=10)

exit_button = tk.Button(window, text="Exit", command=window.quit, width=25, height=2)
exit_button.pack(pady=10)

# Run the GUI event loop
window.mainloop()
