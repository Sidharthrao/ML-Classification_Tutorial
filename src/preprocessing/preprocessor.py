"""
Preprocessing pipeline module.
Sklearn-compatible transformer for consistent preprocessing.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from typing import Optional, List
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.feature_engineering import create_all_features
from config.config import FEATURE_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger("preprocessor")


class FraudDetectionPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible preprocessor for fraud detection.
    Handles feature engineering, encoding, scaling, and imputation.
    """
    
    def __init__(
        self,
        feature_config: Optional[dict] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        use_one_hot: bool = True,
        use_scaling: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            feature_config: Configuration for feature engineering
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            use_one_hot: Whether to use one-hot encoding for categoricals
            use_scaling: Whether to scale numerical features
        """
        self.feature_config = feature_config or FEATURE_CONFIG
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.use_one_hot = use_one_hot
        self.use_scaling = use_scaling
        
        # Transformers
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # Feature names after transformation
        self.feature_names_ = None
        self.is_fitted_ = False
        
    def _identify_columns(self, X: pd.DataFrame) -> None:
        """Identify categorical and numerical columns if not provided."""
        if not self.categorical_columns and not self.numerical_columns:
            # Identify columns based on dtype
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    if col not in self.categorical_columns:
                        self.categorical_columns.append(col)
                elif X[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    if col not in self.numerical_columns:
                        self.numerical_columns.append(col)
        
        logger.debug(f"Categorical columns: {self.categorical_columns}")
        logger.debug(f"Numerical columns: {self.numerical_columns}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FraudDetectionPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional, not used)
        
        Returns:
            Self
        """
        logger.info("Fitting preprocessor")
        
        # Create features
        X_features = create_all_features(X, self.feature_config)
        
        # Identify columns if not already done
        self._identify_columns(X_features)
        
        # Handle missing values
        X_imputed = X_features.copy()
        if self.numerical_columns:
            numerical_data = X_imputed[self.numerical_columns].select_dtypes(include=[np.number])
            self.imputer.fit(numerical_data)
        
        # Fit encoders
        if self.categorical_columns:
            categorical_data = X_imputed[self.categorical_columns]
            
            if self.use_one_hot:
                # Fit one-hot encoder
                self.one_hot_encoder.fit(categorical_data)
            else:
                # Fit label encoder (for single column)
                if len(self.categorical_columns) == 1:
                    self.label_encoder.fit(categorical_data[self.categorical_columns[0]])
        
        # Fit scaler
        if self.use_scaling and self.numerical_columns:
            numerical_data = X_imputed[self.numerical_columns].select_dtypes(include=[np.number])
            if len(numerical_data.columns) > 0:
                self.scaler.fit(self.imputer.transform(numerical_data))
        
        # Store feature names
        self.feature_names_ = self._get_feature_names(X_features)
        self.is_fitted_ = True
        
        logger.info(f"Preprocessor fitted. Output features: {len(self.feature_names_)}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted_:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.debug("Transforming data")
        
        # Create features
        X_features = create_all_features(X, self.feature_config)
        
        # Handle missing values
        X_processed = X_features.copy()
        
        # Impute numerical columns
        if self.numerical_columns:
            numerical_data = X_processed[self.numerical_columns].select_dtypes(include=[np.number])
            if len(numerical_data.columns) > 0:
                numerical_imputed = self.imputer.transform(numerical_data)
                
                # Scale if enabled
                if self.use_scaling:
                    numerical_scaled = self.scaler.transform(numerical_imputed)
                else:
                    numerical_scaled = numerical_imputed
                
                # Update DataFrame
                for i, col in enumerate(numerical_data.columns):
                    X_processed[col] = numerical_scaled[:, i]
        
        # Encode categorical columns
        if self.categorical_columns:
            categorical_data = X_processed[self.categorical_columns]
            
            if self.use_one_hot:
                # One-hot encode
                categorical_encoded = self.one_hot_encoder.transform(categorical_data)
                categorical_encoded_df = pd.DataFrame(
                    categorical_encoded,
                    columns=self.one_hot_encoder.get_feature_names_out(self.categorical_columns),
                    index=X_processed.index
                )
                # Drop original categorical columns and add encoded ones
                X_processed = X_processed.drop(columns=self.categorical_columns)
                X_processed = pd.concat([X_processed, categorical_encoded_df], axis=1)
            else:
                # Label encode
                if len(self.categorical_columns) == 1:
                    X_processed[self.categorical_columns[0]] = self.label_encoder.transform(
                        categorical_data[self.categorical_columns[0]]
                    )
        
        # Select features in correct order
        if self.feature_names_ is not None:
            # Get available features
            available_features = [f for f in self.feature_names_ if f in X_processed.columns]
            X_processed = X_processed[available_features]
        
        logger.debug(f"Transformed shape: {X_processed.shape}")
        return X_processed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get feature names after transformation."""
        feature_names = []
        
        # Numerical features
        if self.numerical_columns:
            numerical_cols = [col for col in self.numerical_columns if col in X.columns]
            feature_names.extend(numerical_cols)
        
        # Categorical features
        if self.categorical_columns:
            if self.use_one_hot:
                # Get one-hot feature names
                categorical_cols = [col for col in self.categorical_columns if col in X.columns]
                if categorical_cols:
                    try:
                        # Fit one-hot encoder temporarily to get feature names
                        temp_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                        temp_encoder.fit(X[categorical_cols])
                        feature_names.extend(temp_encoder.get_feature_names_out(categorical_cols))
                    except:
                        pass
            else:
                # Label encoded features
                categorical_cols = [col for col in self.categorical_columns if col in X.columns]
                feature_names.extend(categorical_cols)
        
        # Other features (not in numerical or categorical)
        other_cols = [
            col for col in X.columns
            if col not in self.categorical_columns + self.numerical_columns
            and col not in ['step', 'nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
        ]
        feature_names.extend(other_cols)
        
        return feature_names
    
    def save(self, filepath: Path) -> None:
        """Save preprocessor to file."""
        logger.info(f"Saving preprocessor to {filepath}")
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: Path) -> 'FraudDetectionPreprocessor':
        """Load preprocessor from file."""
        logger.info(f"Loading preprocessor from {filepath}")
        return joblib.load(filepath)


if __name__ == "__main__":
    # Test preprocessor
    from src.data.data_loader import load_and_prepare_data
    from src.data.data_splitter import split_features_target
    
    logger.info("Testing preprocessor")
    df = load_and_prepare_data(max_rows=1000)
    X, y = split_features_target(df)
    
    preprocessor = FraudDetectionPreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Feature names: {preprocessor.feature_names_}")

