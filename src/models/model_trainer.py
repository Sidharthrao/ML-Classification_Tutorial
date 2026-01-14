"""
Model training module.
Handles model training with hyperparameter tuning and class imbalance handling.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import joblib
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier # Multi-layer Perceptron (Neural Net)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

from config.config import (
    MODEL_CONFIG, XGBOOST_PARAMS, LIGHTGBM_PARAMS, SMOTE_CONFIG,
    SVM_PARAMS, ADABOOST_PARAMS, MLP_PARAMS,
    MODEL_PATHS, DIRECTORIES
)
from src.utils.logger import setup_logger

logger = setup_logger("model_trainer")

# Try importing Optuna for hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Using default hyperparameters.")


class FraudDetectionModelTrainer:
    """Model trainer for fraud detection with class imbalance handling."""
    
    def __init__(
        self,
        model_type: str = "xgboost",
        use_smote: bool = True,
        smote_config: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'logistic', 'svm', 'adaboost', 'mlp')
            use_smote: Whether to use SMOTE for class imbalance
            smote_config: Configuration for SMOTE
            random_state: Random seed
        """
        self.model_type = model_type.lower()
        self.use_smote = use_smote
        self.smote_config = smote_config or SMOTE_CONFIG
        self.random_state = random_state
        
        self.model = None
        self.smote = None
        self.best_params_ = None
        
        logger.info(f"Initialized trainer with model_type={model_type}, use_smote={use_smote}")
    
    def _create_model(self, params: Optional[Dict] = None) -> Any:
        """Create model instance based on model_type."""
        if params is None:
            params = {}
        
        if self.model_type == "xgboost":
            default_params = XGBOOST_PARAMS.copy()
            default_params.update(params)
            default_params['random_state'] = self.random_state
            return XGBClassifier(**default_params)
        
        elif self.model_type == "lightgbm":
            default_params = LIGHTGBM_PARAMS.copy()
            default_params.update(params)
            default_params['random_state'] = self.random_state
            return LGBMClassifier(**default_params)
        
        elif self.model_type == "random_forest":
            rf_params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': self.random_state,
                **params
            }
            return RandomForestClassifier(**rf_params)
        
        elif self.model_type == "logistic":
            lr_params = {
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': self.random_state,
                **params
            }
            return LogisticRegression(**lr_params)
        
        elif self.model_type == "svm":
            svm_params = SVM_PARAMS.copy()
            svm_params.update(params)
            svm_params['random_state'] = self.random_state
            return SVC(**svm_params)
        
        elif self.model_type == "adaboost":
            ada_params = ADABOOST_PARAMS.copy()
            ada_params.update(params)
            ada_params['random_state'] = self.random_state
            return AdaBoostClassifier(**ada_params)
        
        elif self.model_type == "mlp":
            mlp_params = MLP_PARAMS.copy()
            mlp_params.update(params)
            mlp_params['random_state'] = self.random_state
            return MLPClassifier(**mlp_params)
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _create_smote(self) -> Any:
        """Create SMOTE instance."""
        k_neighbors = self.smote_config.get('k_neighbors', 5)
        random_state = self.smote_config.get('random_state', self.random_state)
        return SMOTE(
            k_neighbors=k_neighbors,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparameters: bool = False,
        n_trials: int = 20
    ) -> 'FraudDetectionModelTrainer':
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            tune_hyperparameters: Whether to tune hyperparameters
            n_trials: Number of trials for hyperparameter tuning
        
        Returns:
            Self
        """
        logger.info(f"Training {self.model_type} model")
        logger.info(f"Training data shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
        
        # Handle class imbalance with SMOTE
        if self.use_smote:
            logger.info("Applying SMOTE for class imbalance")
            self.smote = self._create_smote()
            
            # Check if we have enough samples for SMOTE
            min_class_count = y.value_counts().min()
            k_neighbors = self.smote_config.get('k_neighbors', 5)
            
            if min_class_count <= k_neighbors:
                logger.warning(
                    f"Not enough samples for SMOTE (min_class_count={min_class_count}, "
                    f"k_neighbors={k_neighbors}). Using class_weight instead."
                )
                self.use_smote = False
            else:
                try:
                    X_resampled, y_resampled = self.smote.fit_resample(X, y)
                    logger.info(f"After SMOTE: {X_resampled.shape}, Target distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
                    X, y = X_resampled, y_resampled
                except Exception as e:
                    logger.warning(f"SMOTE failed: {str(e)}. Using class_weight instead.")
                    self.use_smote = False
        
        # Hyperparameter tuning
        if tune_hyperparameters and OPTUNA_AVAILABLE:
            logger.info("Starting hyperparameter tuning with Optuna")
            best_params = self._tune_hyperparameters(X, y, n_trials=n_trials)
            self.best_params_ = best_params
            self.model = self._create_model(best_params)
        else:
            self.model = self._create_model()
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X, y)
        logger.info("Model training completed")
        
        return self
    
    def _tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 20
    ) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna."""
        logger.info(f"Hyperparameter tuning with {n_trials} trials")
        
        def objective(trial):
            if self.model_type == "xgboost":
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 50, 200),
                }
            elif self.model_type == "lightgbm":
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 50, 200),
                }
            else:
                # For other models, use simpler tuning
                params = {}
            
            model = self._create_model(params)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(
                model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1
            )
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name=f'{self.model_type}_tuning')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")
        
        return study.best_params
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, filepath: Path) -> None:
        """Save model to file."""
        logger.info(f"Saving model to {filepath}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load(cls, filepath: Path) -> Any:
        """Load model from file."""
        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)


if __name__ == "__main__":
    # Test model training
    from src.data.data_loader import load_and_prepare_data
    from src.data.data_splitter import split_features_target
    from src.preprocessing.preprocessor import FraudDetectionPreprocessor
    
    logger.info("Testing model training")
    df = load_and_prepare_data(max_rows=5000)
    X, y = split_features_target(df)
    
    # Preprocess
    preprocessor = FraudDetectionPreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    # Train model
    trainer = FraudDetectionModelTrainer(model_type="xgboost", use_smote=True)
    trainer.fit(X_transformed, y, tune_hyperparameters=False)
    
    # Predictions
    y_pred = trainer.predict(X_transformed[:100])
    y_proba = trainer.predict_proba(X_transformed[:100])
    
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Probabilities shape: {y_proba.shape}")

