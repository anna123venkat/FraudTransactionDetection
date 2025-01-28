import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from ai_functions import detect_outliers, perform_pca, classify_transactions, detect_anomalies_with_lof


class TestAIFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up data for testing."""
        # Generate synthetic data
        cls.data, cls.labels = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
            weights=[0.9, 0.1]
        )
        cls.df = pd.DataFrame(cls.data, columns=[f"Feature_{i}" for i in range(10)])
        cls.df['Class'] = cls.labels

    def test_detect_outliers(self):
        """Test the detect_outliers function."""
        features = self.df.drop(columns=['Class'])
        outliers = detect_outliers(features, contamination=0.01)
        self.assertEqual(len(outliers), len(features))
        self.assertIn(-1, outliers)
        self.assertIn(1, outliers)

    def test_perform_pca(self):
        """Test the perform_pca function."""
        features = self.df.drop(columns=['Class'])
        pca_result = perform_pca(features, n_components=2)
        self.assertEqual(pca_result.shape, (len(features), 2))
        self.assertIsInstance(pca_result, np.ndarray)

    def test_classify_transactions(self):
        """Test the classify_transactions function."""
        features = self.df.drop(columns=['Class'])
        labels = self.df['Class']
        try:
            classify_transactions(features, labels)
        except Exception as e:
            self.fail(f"classify_transactions raised an exception: {e}")

    def test_detect_anomalies_with_lof(self):
        """Test the detect_anomalies_with_lof function."""
        features = self.df.drop(columns=['Class'])
        anomalies = detect_anomalies_with_lof(features, contamination=0.01)
        self.assertEqual(len(anomalies), len(features))
        self.assertIn(-1, anomalies)
        self
