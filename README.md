# Fraudulent Transaction Detection

This project implements a system for detecting fraudulent transactions using a combination of **Data Analytics (DA)** and **Artificial Intelligence (AI)** techniques. The system processes transactional data to identify potential frauds, leveraging advanced algorithms for anomaly detection, classification, and visualization.

---

## Features

### Data Analytics (DA)
1. **Data Visualization**: Provides insights into fraud and non-fraud transaction distributions.
2. **Feature Analysis**: Computes correlation matrices to analyze feature relationships.
3. **Transaction Analysis**:
   - Mean transaction amount comparison for fraud and non-fraud transactions.
   - Temporal distribution of transactions.

### Artificial Intelligence (AI)
1. **Outlier Detection**: Uses Isolation Forest to identify potential outliers.
2. **Dimensionality Reduction**: Applies Principal Component Analysis (PCA) for efficient data visualization.
3. **Classification**:
   - Trains models like Logistic Regression, Naive Bayes, and Random Forest.
   - Evaluates performance using precision, recall, F1-score, and accuracy.
4. **Anomaly Detection**: Utilizes Local Outlier Factor (LOF) to detect anomalies.

### Graphical User Interface (GUI)
A simple, user-friendly interface built with **Tkinter**, enabling:
- Execution of DA tasks.
- AI model training, evaluation, and anomaly detection.

---

## Technologies Used

### Programming Languages and Libraries
- **Python**: Core programming language.
- **Pandas**: Data manipulation.
- **NumPy**: Numerical computations.
- **Matplotlib** and **Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning algorithms.
- **Tkinter**: GUI development.

### Algorithms
- **Isolation Forest**: For outlier detection.
- **Principal Component Analysis (PCA)**: For dimensionality reduction.
- **Logistic Regression**: For classification.
- **Local Outlier Factor (LOF)**: For anomaly detection.
- **Random Forest** and **Naive Bayes**: For enhanced classification performance.

---

## Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Attributes**:
  - **Features**: 30+ attributes including `Time`, `Amount`, and anonymized variables.
  - **Class**: Binary (1 - Fraudulent, 0 - Non-Fraudulent).
  - **Instances**: 284,807 transactions.

---

## How to Use

### Prerequisites
1. Python 3.x installed on your system.
2. Required libraries: Install using `pip install -r requirements.txt`.

### Steps to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/anna123venkat/FraudTransactionDetection/blob/main/README.md
   cd FraudTransactionDetection
   ```

2. Run the Python script:
   ```bash
   python fraud_detection.py
   ```

3. Use the GUI to interact with the system.
   - Click **"Data Analytics Implementation"** for DA tasks.
   - Click **"Artificial Intelligence Implementation"** for AI tasks.

---

## Results
1. **Data Visualizations**:
   - Distribution of fraud vs non-fraud transactions.
   - Correlation matrix and temporal analysis.
2. **Model Performance**:
   - Logistic Regression achieved high accuracy for fraud detection.
   - Random Forest provided the best F1-score and precision.
3. **Anomaly Detection**:
   - LOF successfully identified anomalies, visualized with PCA.

---

## File Structure
```
FraudTransactionDetection/
├── fraud_detection.py       # Main implementation script.
├── README.md                # Project documentation.
├── requirements.txt         # Dependencies for the project.
├── dataset/                 # Folder for datasets.
└── results/                 # Output visualizations and logs.
```

---

## Future Enhancements
1. Implement real-time fraud detection.
2. Integrate more sophisticated AI models, such as neural networks.
3. Extend the system to handle multi-source datasets.

---

## Authors
- **Hariharanvignesh K** (9517202109021)
- **Prasanna Venkatesh S** (9517202109040)
- **Sakthi Jeganathan R** (9517202109045)

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.
