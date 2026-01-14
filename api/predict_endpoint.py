"""
Prediction endpoint logic for Flask API.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.predict import load_model_artifacts
from src.utils.logger import setup_logger

logger = setup_logger("predict_endpoint")

# Global variables for model artifacts (loaded once at startup)
_model = None
_preprocessor = None
_feature_names = None


def initialize_model():
    """Initialize model artifacts (call once at API startup)."""
    global _model, _preprocessor, _feature_names
    
    if _model is None:
        logger.info("Initializing model artifacts...")
        _model, _preprocessor, _feature_names = load_model_artifacts()
        logger.info("Model artifacts initialized successfully")


def predict_single(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict fraud for a single transaction.
    
    Args:
        transaction_data: Dictionary with transaction features
    
    Returns:
        Dictionary with prediction results
    """
    global _model, _preprocessor
    
    if _model is None:
        initialize_model()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Preprocess
        X_transformed = _preprocessor.transform(df)
        
        # Predict
        prediction = _model.predict(X_transformed)[0]
        probabilities = _model.predict_proba(X_transformed)[0]
        fraud_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        return {
            "transaction_id": transaction_data.get('step', 'unknown'),
            "is_fraud": bool(prediction),
            "fraud_probability": float(fraud_probability),
            "confidence": "high" if fraud_probability > 0.8 or fraud_probability < 0.2 else "medium"
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise ValueError(f"Prediction failed: {str(e)}")


def predict_batch(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict fraud for multiple transactions.
    
    Args:
        transactions: List of transaction dictionaries
    
    Returns:
        List of prediction results
    """
    global _model, _preprocessor
    
    if _model is None:
        initialize_model()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Preprocess
        X_transformed = _preprocessor.transform(df)
        
        # Predict
        predictions = _model.predict(X_transformed)
        probabilities = _model.predict_proba(X_transformed)
        fraud_probabilities = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        # Format results
        results = []
        for i, transaction in enumerate(transactions):
            results.append({
                "transaction_id": transaction.get('step', i),
                "is_fraud": bool(predictions[i]),
                "fraud_probability": float(fraud_probabilities[i]),
                "confidence": "high" if fraud_probabilities[i] > 0.8 or fraud_probabilities[i] < 0.2 else "medium"
            })
        
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise ValueError(f"Batch prediction failed: {str(e)}")


def validate_transaction(transaction_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate transaction data.
    
    Args:
        transaction_data: Dictionary with transaction features
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = [
        'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
        'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'
    ]
    
    for field in required_fields:
        if field not in transaction_data:
            return False, f"Missing required field: {field}"
        
        if transaction_data[field] is None:
            return False, f"Field {field} cannot be None"
    
    # Validate types
    try:
        float(transaction_data['amount'])
        float(transaction_data['oldbalanceOrg'])
        float(transaction_data['newbalanceOrig'])
        float(transaction_data['oldbalanceDest'])
        float(transaction_data['newbalanceDest'])
        int(transaction_data['step'])
    except (ValueError, TypeError):
        return False, "Invalid data types in transaction"
    
    return True, None


def get_model_info() -> Dict[str, Any]:
    """Get model information."""
    global _model, _preprocessor, _feature_names
    
    if _model is None:
        initialize_model()
    
    model_type = type(_model).__name__
    n_features = len(_feature_names) if _feature_names else None
    
    info = {
        "model_type": model_type,
        "num_features": n_features,
        "model_loaded": _model is not None
    }
    
    if hasattr(_model, 'feature_importances_'):
        info["feature_importance_available"] = True
    
    return info

