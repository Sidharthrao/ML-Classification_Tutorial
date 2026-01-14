"""
Main training script for Fraud Detection Model.
Orchestrates data loading, preprocessing, training, and evaluation.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import joblib
from config.config import (
    ensure_directories, MODEL_PATHS, DIRECTORIES, MODEL_CONFIG,
    TRAIN_SIZE, EVAL_SIZE, DATABASE_PATH
)
from src.data.data_loader import load_and_prepare_data
from src.data.data_splitter import split_data, split_features_target
from src.preprocessing.preprocessor import FraudDetectionPreprocessor
from src.models.model_trainer import FraudDetectionModelTrainer
from src.evaluation.model_evaluator import ModelEvaluator
from src.utils.logger import setup_logger

logger = setup_logger("train_model")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Fraud Detection Model Training Pipeline")
    logger.info("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Step 1: Load data
    logger.info("\n[Step 1/6] Loading data from database...")
    try:
        df = load_and_prepare_data(
            db_path=DATABASE_PATH,
            chunk_size=100000,
            max_rows=None  # Load all data
        )
        logger.info(f"Loaded {len(df):,} records")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise
    
    # Step 2: Split data
    logger.info("\n[Step 2/6] Splitting data into train/eval/test sets...")
    train_df, eval_df, test_df = split_data(
        df,
        train_size=TRAIN_SIZE,
        eval_size=EVAL_SIZE,
        preserve_order=True  # Preserve temporal order
    )
    
    # Step 3: Split features and target
    logger.info("\n[Step 3/6] Preparing features and targets...")
    X_train, y_train = split_features_target(train_df)
    X_eval, y_eval = split_features_target(eval_df)
    
    logger.info(f"Train set: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Eval set: X={X_eval.shape}, y={y_eval.shape}")
    
    # Step 4: Preprocessing
    logger.info("\n[Step 4/6] Fitting preprocessing pipeline...")
    preprocessor = FraudDetectionPreprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_eval_transformed = preprocessor.transform(X_eval)
    
    logger.info(f"Train features after preprocessing: {X_train_transformed.shape}")
    logger.info(f"Feature names: {len(preprocessor.feature_names_)} features")
    
    # Save preprocessor and feature names
    logger.info("Saving preprocessor...")
    preprocessor.save(MODEL_PATHS["preprocessor"])
    joblib.dump(preprocessor.feature_names_, MODEL_PATHS["feature_names"])
    
    # Step 5: Model training
    logger.info("\n[Step 5/6] Training model...")
    model_type = MODEL_CONFIG.get("primary_model", "xgboost")
    use_smote = MODEL_CONFIG.get("use_smote", True)
    
    trainer = FraudDetectionModelTrainer(
        model_type=model_type,
        use_smote=use_smote,
        random_state=MODEL_CONFIG.get("random_state", 42)
    )
    
    # Train with or without hyperparameter tuning
    tune_hp = False  # Set to True for hyperparameter tuning (takes longer)
    trainer.fit(
        X_train_transformed,
        y_train,
        tune_hyperparameters=tune_hp,
        n_trials=20
    )
    
    # Save model
    logger.info("Saving trained model...")
    trainer.save(MODEL_PATHS["model"])
    
    # Step 6: Evaluation
    logger.info("\n[Step 6/6] Evaluating model...")
    evaluator = ModelEvaluator(
        model=trainer.model,
        preprocessor=preprocessor,
        optimize_threshold=True,
        target_recall=0.90
    )
    
    # Evaluate on evaluation set
    metrics = evaluator.evaluate(X_eval_transformed, y_eval, save_plots=True)
    
    # Generate SHAP plots (optional, computationally expensive)
    logger.info("Generating SHAP plots (this may take time)...")
    try:
        evaluator.generate_shap_plots(X_eval_transformed.sample(min(1000, len(X_eval_transformed))))
    except Exception as e:
        logger.warning(f"SHAP plot generation failed: {str(e)}")
    
    # Generate and save report
    logger.info("Generating evaluation report...")
    model_info = {
        "Model Type": model_type,
        "Use SMOTE": use_smote,
        "Training Samples": len(X_train),
        "Evaluation Samples": len(X_eval),
        "Number of Features": X_train_transformed.shape[1],
    }
    
    if trainer.best_params_:
        model_info["Best Hyperparameters"] = trainer.best_params_
    
    report = evaluator.generate_report(metrics, model_info)
    
    report_path = DIRECTORIES["reports"] / "model_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Pipeline Completed Successfully!")
    logger.info("=" * 60)
    logger.info(f"\nModel saved to: {MODEL_PATHS['model']}")
    logger.info(f"Preprocessor saved to: {MODEL_PATHS['preprocessor']}")
    logger.info(f"Evaluation report saved to: {report_path}")
    logger.info(f"\n{'='*60}")
    logger.info("COMPREHENSIVE EVALUATION METRICS")
    logger.info(f"{'='*60}")
    
    logger.info("\n📊 Basic Metrics:")
    logger.info(f"  - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    logger.info(f"  - Precision: {metrics.get('precision', 'N/A'):.4f}")
    logger.info(f"  - Recall (Sensitivity): {metrics.get('recall', 'N/A'):.4f}")
    logger.info(f"  - F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")
    logger.info(f"  - F2-Score: {metrics.get('f2_score', 'N/A'):.4f}")
    logger.info(f"  - F0.5-Score: {metrics.get('f0.5_score', 'N/A'):.4f}")
    
    logger.info("\n📈 Advanced Metrics:")
    logger.info(f"  - Specificity (TNR): {metrics.get('specificity', 'N/A'):.4f}")
    logger.info(f"  - False Positive Rate: {metrics.get('false_positive_rate', 'N/A'):.4f}")
    logger.info(f"  - False Negative Rate: {metrics.get('false_negative_rate', 'N/A'):.4f}")
    
    logger.info("\n🎯 AUC Metrics:")
    logger.info(f"  - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
    logger.info(f"  - PR-AUC (Average Precision): {metrics.get('pr_auc', 'N/A'):.4f}")
    
    logger.info("\n🔗 Correlation Metrics:")
    logger.info(f"  - Matthews Correlation Coefficient: {metrics.get('matthews_corrcoef', 'N/A'):.4f}")
    logger.info(f"  - Cohen's Kappa: {metrics.get('cohen_kappa', 'N/A'):.4f}")
    
    logger.info("\n📋 Confusion Matrix Components:")
    logger.info(f"  - True Positives: {metrics.get('true_positives', 'N/A'):,}")
    logger.info(f"  - True Negatives: {metrics.get('true_negatives', 'N/A'):,}")
    logger.info(f"  - False Positives: {metrics.get('false_positives', 'N/A'):,}")
    logger.info(f"  - False Negatives: {metrics.get('false_negatives', 'N/A'):,}")
    
    if 'optimal_threshold' in metrics:
        optimal_metrics = metrics.get('metrics_at_optimal_threshold', {})
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMAL THRESHOLD METRICS (Threshold: {metrics['optimal_threshold']:.4f})")
        logger.info(f"{'='*60}")
        logger.info(f"  - Accuracy: {optimal_metrics.get('accuracy', 'N/A'):.4f}")
        logger.info(f"  - Precision: {optimal_metrics.get('precision', 'N/A'):.4f}")
        logger.info(f"  - Recall: {optimal_metrics.get('recall', 'N/A'):.4f}")
        logger.info(f"  - F1-Score: {optimal_metrics.get('f1_score', 'N/A'):.4f}")
        logger.info(f"  - F2-Score: {optimal_metrics.get('f2_score', 'N/A'):.4f}")
        logger.info(f"  - Specificity: {optimal_metrics.get('specificity', 'N/A'):.4f}")
        logger.info(f"  - ROC-AUC: {optimal_metrics.get('roc_auc', 'N/A'):.4f}")
        logger.info(f"  - PR-AUC: {optimal_metrics.get('pr_auc', 'N/A'):.4f}")
        logger.info(f"  - MCC: {optimal_metrics.get('matthews_corrcoef', 'N/A'):.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

