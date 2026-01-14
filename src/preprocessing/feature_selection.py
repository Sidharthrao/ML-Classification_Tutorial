"""
Feature selection module.
Implements Recursive Feature Elimination (RFE) and SelectKBest.
"""
import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from config.config import FEATURE_SELECTION_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger("feature_selector")

class FeatureSelector:
    """Feature selector for selecting most important features."""
    
    def __init__(
        self,
        method: str = "rfe",
        n_features_to_select: int = 20,
        step: int = 1,
        random_state: int = 42
    ):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('rfe' or 'select_k_best')
            n_features_to_select: Number of features to select
            step: Step size for RFE
            random_state: Random state
        """
        self.method = method
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.random_state = random_state
        self.selected_features_ = None
        self.selector = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """Fit feature selector."""
        logger.info(f"Fitting feature selector with method={self.method}")
        
        if self.method == "rfe":
            # Use Random Forest as estimator for RFE
            estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.selector = RFE(
                estimator=estimator,
                n_features_to_select=self.n_features_to_select,
                step=self.step,
                verbose=1
            )
            self.selector.fit(X, y)
            self.selected_features_ = X.columns[self.selector.support_].tolist()
            
        elif self.method == "select_k_best":
            self.selector = SelectKBest(score_func=f_classif, k=self.n_features_to_select)
            self.selector.fit(X, y)
            mask = self.selector.get_support()
            self.selected_features_ = X.columns[mask].tolist()
            
        else:
            logger.warning(f"Unknown feature selection method: {self.method}. keeping all features.")
            self.selected_features_ = X.columns.tolist()
            
        logger.info(f"Selected {len(self.selected_features_)} features: {self.selected_features_}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to keep only selected features."""
        if self.selected_features_ is None:
            raise ValueError("Feature selector not fitted yet.")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(X, y).transform(X)

if __name__ == "__main__":
    # Test feature selection
    from src.data.data_loader import load_and_prepare_data
    from src.data.data_splitter import split_features_target
    from src.preprocessing.preprocessor import FraudDetectionPreprocessor
    
    df = load_and_prepare_data(max_rows=1000)
    X, y = split_features_target(df)
    
    preprocessor = FraudDetectionPreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    selector = FeatureSelector(method="select_k_best", n_features_to_select=10)
    X_selected = selector.fit_transform(X_transformed, y)
    
    print(f"Original shape: {X_transformed.shape}")
    print(f"Selected shape: {X_selected.shape}")
