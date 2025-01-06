import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

# Load the dataset
dataset_path = r"C:\Users\harih\Downloads\creditcard_1683007639882.csv"
df = pd.read_csv(dataset_path)

# Preprocessing
scaler = StandardScaler()
df['normalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Amount'], axis=1)

# Function for Data Analytics (DA) Implementation
def da_implementation():
    # DA Task 1: Data Visualization - Plot the distribution of fraud vs non-fraud transactions
    plt.figure(figsize=(8, 6))
    counts = df['Class'].value_counts()
    counts.plot(kind='bar', rot=0)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Distribution of Fraud vs Non-Fraud Transactions')
    plt.show()

    # DA Task 2: Feature Analysis - Compute and display correlation matrix
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation='vertical')
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Correlation Matrix')
    plt.show()

    # DA Task 3: Transaction Amount Analysis - Compare the mean transaction amount for fraud and non-fraud transactions
    mean_amount_fraud = df[df['Class'] == 1]['normalizedAmount'].mean()
    mean_amount_non_fraud = df[df['Class'] == 0]['normalizedAmount'].mean()
    print(f"Mean transaction amount for fraud transactions: {mean_amount_fraud}")
    print(f"Mean transaction amount for non-fraud transactions: {mean_amount_non_fraud}")

    # DA Task 4: Time Analysis - Analyze the distribution of transactions over time
    plt.figure(figsize=(10, 6))
    plt.hist(df['Time'], bins=50, color='blue', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('Distribution of Transactions over Time')
    plt.show()

# Function for Artificial Intelligence (AI) Implementation
def ai_implementation():
    # Initialize 'Outlier' and 'Anomaly' columns with default values
    # AI Task 1: Outlier Detection - Use Isolation Forest to detect outliers
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        features = df.drop(['Class'], axis=1)  # Exclude non-feature columns
        clf = IsolationForest(contamination=0.01)
        clf.fit(features)
        y_pred = clf.predict(features)
        df['Outlier'] = y_pred
        outlier_count = len(df[df['Outlier'] == -1])
        print(f"Number of detected outliers: {outlier_count}")

    # AI Task 2: Dimensionality Reduction - Apply PCA for feature dimensionality reduction
    print("PRINCIPAL COMPONENT ANALYSIS")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['Outlier'] = y_pred
    plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Outlier'], cmap='coolwarm')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA-Outlier Detection')
    plt.show()

    # AI Task 3: Classification - Train and evaluate a classification model
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression

    y = df['Class']
    X = df.drop(['Class', 'Outlier'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # AI Task 4: Anomaly Detection - Use Local Outlier Factor (LOF) for anomaly detection
    lof = LocalOutlierFactor(contamination=0.01)
    y_pred_lof = lof.fit_predict(X)
    df['Anomaly'] = y_pred_lof
    anomaly_count = len(df[df['Anomaly'] == -1])
    print(f"Number of detected anomalies: {anomaly_count}")

    # Compute PCA on feature matrix X
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['Anomaly'] = y_pred_lof

    # Visualize the anomalies
    print("PCA ANOMALY")
    plt.figure(figsize=(8, 6))
    colors = np.where(df['Anomaly'] == -1, 'red', 'green')
    plt.scatter(df_pca['PC1'], df_pca['PC2'], c=colors)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Anomaly Detection using LOF')
    plt.show()

    fraud = len(df[df['Class'] == 1])
    notfraud = len(df[df['Class'] == 0])

    # Data to plot
    labels = ['Fraud', 'Not Fraud']
    sizes = [fraud, notfraud]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=0)
    plt.title('Ratio of Fraud Vs Non-Fraud\n', fontsize=20)
    sns.set_context("paper", font_scale=2)

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb_model = gnb.fit(X_train, y_train)

    # Predict on test set
    gnb_pred = gnb_model.predict(X_test)
    print(accuracy_score(y_test, gnb_pred))

    from sklearn.ensemble import RandomForestClassifier
    randf = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

    # Predict on test set
    randf_pred = randf.predict(X_test)
    print(accuracy_score(y_test, randf_pred))

    print('Random Forest classification_report')
    print('....' * 10)
    print(classification_report(y_test, randf_pred))

# Create GUI
window = tk.Tk()
window.title("Fraud Transaction Detection")
window.geometry("500x300")

# Create buttons for DA implementation
da_button = tk.Button(window, text="Data Analytics (DA) Implementation", command=da_implementation)
da_button.pack(pady=10)

# Create buttons for AI implementation
ai_button = tk.Button(window, text="Artificial Intelligence (AI) Implementation", command=ai_implementation)
ai_button.pack(pady=10)

# Start the GUI event loop
window.mainloop()
