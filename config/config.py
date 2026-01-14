"""
Configuration module for Fraud Detection ML Pipeline.
Contains all configuration parameters, paths, and hyperparameters.
"""
import os
from pathlib import Path
from typing import Dict, Any

# Project root directory - absolute path to 1.Fraud_Detection
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Database path - relative to project root (goes up to Capstone_Projects level)
DATABASE_PATH = PROJECT_ROOT.parent.parent.parent / "Database.db"

# Virtual environment path at repository root
REPO_ROOT = PROJECT_ROOT.parent.parent.parent.parent.parent
VENV_PATH = REPO_ROOT / "venv"

# Data split configuration
TRAIN_SIZE = 4_000_000  # First 4M records for training
EVAL_SIZE = 1_000_000  # Next 1M records for evaluation
# Remaining records for production/testing

# Directory paths (relative to PROJECT_ROOT)
DIRECTORIES = {
    "models": PROJECT_ROOT / "models",
    "logs": PROJECT_ROOT / "logs",
    "reports": PROJECT_ROOT / "reports",
    "notebooks": PROJECT_ROOT / "notebooks",
    "data": PROJECT_ROOT / "data",
}

# Model file paths
MODEL_PATHS = {
    "preprocessor": DIRECTORIES["models"] / "preprocessor.pkl",
    "model": DIRECTORIES["models"] / "model.pkl",
    "feature_names": DIRECTORIES["models"] / "feature_names.pkl",
    "label_encoder": DIRECTORIES["models"] / "label_encoder.pkl",
}

# Logging configuration
LOG_CONFIG = {
    "log_file": DIRECTORIES["logs"] / "fraud_detection.log",
    "log_level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", #Fomat -> timestamp - name - level - message
}

# Model hyperparameters
MODEL_CONFIG = {
    "primary_model": "xgboost",  # xgboost, lightgbm, random_forest
    "use_smote": True,
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
    "scoring_metric": "roc_auc",
}

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "scale_pos_weight": 100,  # For class imbalance
    "random_state": 42,
}

# LightGBM hyperparameters (alternative)
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "scale_pos_weight": 100,
    "random_state": 42,
}

# SVM hyperparameters
SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "probability": True,  # Required for ROC-AUC
    "class_weight": "balanced",
    "random_state": 42,
}

# AdaBoost hyperparameters
ADABOOST_PARAMS = {
    "n_estimators": 50,
    "learning_rate": 1.0,
    "random_state": 42,
}

# MLP hyperparameters
MLP_PARAMS = {
    "hidden_layer_sizes": (100, 50),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": "auto",
    "learning_rate": "constant",
    "max_iter": 200,
    "random_state": 42,
}

# SMOTE configuration
SMOTE_CONFIG = {
    "k_neighbors": 5,
    "random_state": 42,
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "use_account_frequency": False,  # Set to True if computationally feasible
    "use_time_features": True,
    "use_balance_features": True,
    "use_transaction_features": True,
}

# Feature selection configuration
FEATURE_SELECTION_CONFIG = {
    "method": "rfe",  # rfe, select_k_best, or None
    "n_features_to_select": 20,
    "step": 1,
}

# Drift detection configuration
DRIFT_DETECTION_CONFIG = {
    "drift_method": "ks_test",
    "p_value_threshold": 0.05,
}

# Evaluation thresholds
EVALUATION_CONFIG = {
    "optimize_threshold": True,
    "target_recall": 0.90,  # Target recall for fraud detection
    "precision_weight": 0.3,
    "recall_weight": 0.7,
}

# Comprehensive classification metrics configuration
# Educational: All metrics used in classification problems
CLASSIFICATION_METRICS = {
    # Basic Metrics
    "accuracy": True,
    "precision": True,
    "recall": True,  # Also called Sensitivity or TPR
    "f1_score": True,
    "f2_score": True,  # Emphasizes recall more than F1
    "fbeta_score": True,  # Configurable beta
    
    # Advanced Metrics
    "specificity": True,  # True Negative Rate (TNR)
    "sensitivity": True,  # Same as Recall/TPR
    "false_positive_rate": True,  # FPR
    "false_negative_rate": True,  # FNR
    "true_positive_rate": True,  # TPR (same as Recall)
    "true_negative_rate": True,  # TNR (same as Specificity)
    
    # AUC Metrics
    "roc_auc": True,  # Area Under ROC Curve
    "pr_auc": True,  # Area Under Precision-Recall Curve (Average Precision)
    
    # Correlation Metrics
    "matthews_corrcoef": True,  # MCC - balanced metric for imbalanced data
    "cohen_kappa": True,  # Agreement between predicted and actual
    
    # Confusion Matrix Components
    "confusion_matrix": True,  # Full confusion matrix
    "true_positives": True,
    "true_negatives": True,
    "false_positives": True,
    "false_negatives": True,
    
    # Classification Report
    "classification_report": True,  # Detailed per-class metrics
}

# Database table name
DB_TABLE_NAME = "Fraud_detection"

# Column mappings for type conversion
COLUMN_TYPES = {
    "step": "int64",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int64",
    "isFlaggedFraud": "int64",
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "threaded": True,
}

def ensure_directories() -> None:
    """Create all necessary directories if they don't exist."""
    for dir_path in DIRECTORIES.values():
        dir_path.mkdir(parents=True, exist_ok=True)


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        "project_root": str(PROJECT_ROOT),
        "database_path": str(DATABASE_PATH),
        "venv_path": str(VENV_PATH),
        "directories": {k: str(v) for k, v in DIRECTORIES.items()},
        "model_paths": {k: str(v) for k, v in MODEL_PATHS.items()},
        "log_config": LOG_CONFIG,
        "model_config": MODEL_CONFIG,
        "db_table_name": DB_TABLE_NAME,
    }


if __name__ == "__main__":
    # Ensure directories exist
    ensure_directories()
    print("Configuration loaded successfully!")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Database Path: {DATABASE_PATH}")
    print(f"Database exists: {DATABASE_PATH.exists()}")

