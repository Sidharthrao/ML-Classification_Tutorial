"""
Batch prediction script for Fraud Detection Model.
Loads saved model and preprocessor for predictions on new data.
"""
import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
from config.config import MODEL_PATHS, DIRECTORIES
from src.preprocessing.preprocessor import FraudDetectionPreprocessor
from src.models.model_trainer import FraudDetectionModelTrainer
from src.utils.logger import setup_logger

logger = setup_logger("predict")


def load_model_artifacts():
    """Load saved model, preprocessor, and feature names."""
    logger.info("Loading model artifacts...")
    
    # Check if model files exist
    if not MODEL_PATHS["model"].exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATHS['model']}")
    if not MODEL_PATHS["preprocessor"].exists():
        raise FileNotFoundError(f"Preprocessor file not found: {MODEL_PATHS['preprocessor']}")
    
    # Load preprocessor
    preprocessor = FraudDetectionPreprocessor.load(MODEL_PATHS["preprocessor"])
    
    # Load model
    model = FraudDetectionModelTrainer.load(MODEL_PATHS["model"])
    
    # Load feature names
    feature_names = joblib.load(MODEL_PATHS["feature_names"])
    
    logger.info("Model artifacts loaded successfully")
    return model, preprocessor, feature_names


def predict_from_csv(csv_path: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Make predictions from CSV file.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Load model artifacts
    model, preprocessor, feature_names = load_model_artifacts()
    
    # Preprocess
    logger.info("Preprocessing data...")
    X_transformed = preprocessor.transform(df)
    
    # Predict
    logger.info("Making predictions...")
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)
    
    # Create results dataframe
    results = df.copy()
    results['predicted_fraud'] = predictions
    results['fraud_probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
    results['is_fraud'] = (results['fraud_probability'] >= 0.5).astype(int)
    
    # Save results
    if output_path is None:
        output_path = DIRECTORIES["reports"] / f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Log summary
    fraud_count = results['is_fraud'].sum()
    logger.info(f"Predictions complete: {len(results)} transactions, {fraud_count} flagged as fraud")
    
    return results


def predict_from_db(query: str, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Make predictions from database query.
    
    Args:
        query: SQL query string
        output_path: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    from src.data.data_loader import load_data_from_db
    from config.config import DATABASE_PATH
    
    logger.info("Loading data from database...")
    # Note: This is a simplified version. In production, you'd want a proper query interface
    df = pd.read_sql_query(query, sqlite3.connect(str(DATABASE_PATH)))
    
    # Use same prediction logic as CSV
    return predict_from_dataframe(df, output_path)


def predict_from_dataframe(df: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Make predictions from DataFrame.
    
    Args:
        df: Input DataFrame
        output_path: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    # Load model artifacts
    model, preprocessor, feature_names = load_model_artifacts()
    
    # Preprocess
    logger.info("Preprocessing data...")
    X_transformed = preprocessor.transform(df)
    
    # Predict
    logger.info("Making predictions...")
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)
    
    # Create results dataframe
    results = df.copy()
    results['predicted_fraud'] = predictions
    results['fraud_probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
    results['is_fraud'] = (results['fraud_probability'] >= 0.5).astype(int)
    
    # Save results if path provided
    if output_path:
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    
    # Log summary
    fraud_count = results['is_fraud'].sum()
    logger.info(f"Predictions complete: {len(results)} transactions, {fraud_count} flagged as fraud")
    
    return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Fraud Detection Batch Prediction")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (optional)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else None
    
    try:
        results = predict_from_csv(input_path, output_path)
        logger.info("Prediction completed successfully")
        return results
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import sqlite3
    from typing import Optional
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Prediction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        sys.exit(1)

