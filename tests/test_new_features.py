"""
Test script for new features (SVM, AdaBoost, MLP, Feature Selection, Drift Detection).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_trainer import FraudDetectionModelTrainer
from src.preprocessing.feature_selection import FeatureSelector
from src.monitoring.drift_detector import DriftDetector
from src.preprocessing.preprocessor import FraudDetectionPreprocessor

def test_feature_selection():
    print("\n--- Testing Feature Selection ---")
    # Synthetic data
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    selector = FeatureSelector(method="select_k_best", n_features_to_select=5)
    X_selected = selector.fit_transform(X, y)
    
    print(f"Original shape: {X.shape}")
    print(f"Selected shape: {X_selected.shape}")
    assert X_selected.shape[1] == 5
    print("Feature Selection Test Passed!")

def test_new_models():
    print("\n--- Testing New Models ---")
    X = pd.DataFrame(np.random.rand(50, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 50))
    
    for model_type in ["svm", "adaboost", "mlp"]:
        print(f"Testing {model_type}...")
        trainer = FraudDetectionModelTrainer(model_type=model_type, use_smote=False)
        trainer.fit(X, y)
        preds = trainer.predict(X)
        assert len(preds) == 50
        print(f"{model_type} trained and predicted successfully")
    print("All New Models Test Passed!")

def test_drift_detection():
    print("\n--- Testing Drift Detection ---")
    X_ref = pd.DataFrame({'f1': np.random.normal(0, 1, 100)})
    X_curr = pd.DataFrame({'f1': np.random.normal(5, 1, 100)}) # Drifted
    
    detector = DriftDetector(p_value_threshold=0.05)
    detector.fit(X_ref)
    report = detector.detect_drift(X_curr)
    
    print(f"Drift detected: {report['n_drift_detected']}")
    assert report['drift_detected_features'] == ['f1']
    print("Drift Detection Test Passed!")

if __name__ == "__main__":
    try:
        test_feature_selection()
        test_new_models()
        test_drift_detection()
        print("\nAll Tests Passed Successfully!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        sys.exit(1)
