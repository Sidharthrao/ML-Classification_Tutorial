"""
Drift detection module.
Implements statistical tests to detect distribution drift between training and serving data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import ks_2samp
from src.utils.logger import setup_logger
from config.config import DRIFT_DETECTION_CONFIG

logger = setup_logger("drift_detector")

class DriftDetector:
    """Detects data drift between reference (training) and current (serving) data."""
    
    def __init__(self, p_value_threshold: float = None):
        """
        Initialize drift detector.
        
        Args:
            p_value_threshold: Threshold for p-value to indicate drift.
                             If p-value < threshold, we reject null hypothesis (that distributions are same).
        """
        self.p_value_threshold = p_value_threshold or DRIFT_DETECTION_CONFIG.get("p_value_threshold", 0.05)
        self.reference_data = None
        self.drift_report = {}
        
    def fit(self, X_reference: pd.DataFrame) -> 'DriftDetector':
        """
        Set reference data (usually training data).
        
        Args:
            X_reference: Reference data DataFrame
        """
        self.reference_data = X_reference.copy()
        logger.info(f"Drift detector fitted with {len(X_reference)} samples and {X_reference.shape[1]} features")
        return self
    
    def detect_drift(self, X_current: pd.DataFrame) -> Dict:
        """
        Detect drift in current data compared to reference data.
        
        Args:
            X_current: Current data DataFrame
            
        Returns:
            Drift report dictionary
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call fit() first.")
        
        logger.info(f"Running drift detection on {len(X_current)} samples")
        
        drift_results = {}
        drift_detected_features = []
        
        # Check drift for each feature
        for column in self.reference_data.columns:
            if column not in X_current.columns:
                logger.warning(f"Feature {column} missing in current data")
                continue
            
            # Use Kolmogorov-Smirnov test for numerical features
            # (Assuming most features are numerical or treated as such after preprocessing)
            if np.issubdtype(self.reference_data[column].dtype, np.number) and \
               np.issubdtype(X_current[column].dtype, np.number):
                
                statistic, p_value = ks_2samp(
                    self.reference_data[column], 
                    X_current[column]
                )
                
                is_drift = p_value < self.p_value_threshold
                
                drift_results[column] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift_detected": bool(is_drift)
                }
                
                if is_drift:
                    drift_detected_features.append(column)
        
        self.drift_report = {
            "n_features_checked": len(drift_results),
            "n_drift_detected": len(drift_detected_features),
            "drift_detected_features": drift_detected_features,
            "feature_details": drift_results
        }
        
        logger.info(f"Drift detected in {len(drift_detected_features)}/{len(drift_results)} features")
        
        return self.drift_report

if __name__ == "__main__":
    # Test drift detection
    logger.info("Testing drift detection")
    
    # Generate synthetic data
    np.random.seed(42)
    X_ref = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000)
    })
    
    # Generate drift (shift mean)
    X_curr = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1, 1000), # Drifted
        'feature2': np.random.normal(0, 1, 1000)    # Same
    })
    
    detector = DriftDetector()
    detector.fit(X_ref)
    report = detector.detect_drift(X_curr)
    
    print(f"Drift Report: {report['n_drift_detected']} features drifted")
    print(f"Drifted features: {report['drift_detected_features']}")
