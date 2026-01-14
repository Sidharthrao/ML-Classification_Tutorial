"""
Unit tests for preprocessing pipeline.
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocessor import FraudDetectionPreprocessor


class TestPreprocessor(unittest.TestCase):
    """Test cases for FraudDetectionPreprocessor."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'step': [1, 2, 3],
            'type': ['TRANSFER', 'PAYMENT', 'CASH_OUT'],
            'amount': [100.0, 200.0, 300.0],
            'nameOrig': ['C1', 'C2', 'C3'],
            'oldbalanceOrg': [1000.0, 2000.0, 3000.0],
            'newbalanceOrig': [900.0, 1800.0, 2700.0],
            'nameDest': ['M1', 'M2', 'C1'],
            'oldbalanceDest': [0.0, 0.0, 100.0],
            'newbalanceDest': [100.0, 200.0, 400.0],
        })
    
    def test_preprocessor_fit_transform(self):
        """Test preprocessor fit and transform."""
        preprocessor = FraudDetectionPreprocessor()
        X_transformed = preprocessor.fit_transform(self.sample_data)
        
        self.assertIsNotNone(X_transformed)
        self.assertGreater(len(X_transformed.columns), len(self.sample_data.columns))
        self.assertEqual(len(X_transformed), len(self.sample_data))
    
    def test_preprocessor_save_load(self):
        """Test preprocessor save and load."""
        from config.config import MODEL_PATHS
        
        preprocessor = FraudDetectionPreprocessor()
        preprocessor.fit(self.sample_data)
        
        # Save
        preprocessor.save(MODEL_PATHS["preprocessor"])
        
        # Load
        loaded_preprocessor = FraudDetectionPreprocessor.load(MODEL_PATHS["preprocessor"])
        
        self.assertIsNotNone(loaded_preprocessor)
        self.assertTrue(loaded_preprocessor.is_fitted_)


if __name__ == '__main__':
    unittest.main()

