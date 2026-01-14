"""
Model evaluation module.
Comprehensive evaluation with metrics, SHAP values, and report generation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, fbeta_score, accuracy_score,
    precision_recall_curve, roc_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)

from config.config import DIRECTORIES, EVALUATION_CONFIG, CLASSIFICATION_METRICS
from src.utils.logger import setup_logger

logger = setup_logger("model_evaluator")

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Feature importance plots will be limited.")


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""
    
    def __init__(
        self,
        model: Any,
        preprocessor: Optional[Any] = None,
        optimize_threshold: bool = True,
        target_recall: float = 0.90
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            preprocessor: Optional preprocessor for feature names
            optimize_threshold: Whether to optimize classification threshold
            target_recall: Target recall for threshold optimization
        """
        self.model = model
        self.preprocessor = preprocessor
        self.optimize_threshold = optimize_threshold
        self.target_recall = target_recall
        self.optimal_threshold_ = None
        self.metrics_ = {}
        
        # Set style for plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            X: Feature DataFrame
            y_true: True labels
            save_plots: Whether to save plots
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting model evaluation")
        
        # Get predictions and probabilities
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        # If binary classification, get probabilities for positive class
        if y_proba.shape[1] == 2:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba[:, -1]
        
        # Calculate metrics with default threshold (0.5)
        metrics = self._calculate_metrics(y_true, y_pred, y_proba_positive)
        self.metrics_ = metrics
        
        # Optimize threshold if requested
        if self.optimize_threshold:
            optimal_threshold = self._optimize_threshold(y_true, y_proba_positive)
            self.optimal_threshold_ = optimal_threshold
            
            # Recalculate metrics with optimal threshold
            y_pred_optimal = (y_proba_positive >= optimal_threshold).astype(int)
            metrics_optimal = self._calculate_metrics(y_true, y_pred_optimal, y_proba_positive)
            metrics['optimal_threshold'] = optimal_threshold
            metrics['metrics_at_optimal_threshold'] = metrics_optimal
        
        # Generate visualizations
        if save_plots:
            self._generate_plots(X, y_true, y_pred, y_proba_positive, metrics)
        
        logger.info("Model evaluation completed")
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        Educational: Includes all common classification metrics for comparison.
        """
        metrics = {}
        
        # Get confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Total samples
        total = len(y_true)
        
        # Basic Metrics
        if CLASSIFICATION_METRICS.get("accuracy", True):
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if CLASSIFICATION_METRICS.get("precision", True):
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        
        if CLASSIFICATION_METRICS.get("recall", True):
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        
        if CLASSIFICATION_METRICS.get("f1_score", True):
            metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        if CLASSIFICATION_METRICS.get("f2_score", True):
            metrics['f2_score'] = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        
        if CLASSIFICATION_METRICS.get("fbeta_score", True):
            # Using beta=0.5 to emphasize precision
            metrics['f0.5_score'] = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        
        # Advanced Metrics from Confusion Matrix
        if CLASSIFICATION_METRICS.get("specificity", True):
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        if CLASSIFICATION_METRICS.get("sensitivity", True):
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if CLASSIFICATION_METRICS.get("false_positive_rate", True):
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        if CLASSIFICATION_METRICS.get("false_negative_rate", True):
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        if CLASSIFICATION_METRICS.get("true_positive_rate", True):
            metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if CLASSIFICATION_METRICS.get("true_negative_rate", True):
            metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # AUC Metrics
        if CLASSIFICATION_METRICS.get("roc_auc", True):
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        if CLASSIFICATION_METRICS.get("pr_auc", True):
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        
        # Correlation Metrics
        if CLASSIFICATION_METRICS.get("matthews_corrcoef", True):
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        if CLASSIFICATION_METRICS.get("cohen_kappa", True):
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion Matrix Components
        if CLASSIFICATION_METRICS.get("confusion_matrix", True):
            metrics['confusion_matrix'] = cm.tolist()
        
        if CLASSIFICATION_METRICS.get("true_positives", True):
            metrics['true_positives'] = int(tp)
        
        if CLASSIFICATION_METRICS.get("true_negatives", True):
            metrics['true_negatives'] = int(tn)
        
        if CLASSIFICATION_METRICS.get("false_positives", True):
            metrics['false_positives'] = int(fp)
        
        if CLASSIFICATION_METRICS.get("false_negatives", True):
            metrics['false_negatives'] = int(fn)
        
        # Additional useful metrics
        metrics['positive_predictive_value'] = metrics.get('precision', 0.0)  # Same as precision
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Prevalence (base rate of positive class)
        metrics['prevalence'] = (tp + fn) / total if total > 0 else 0.0
        
        # Classification Report
        if CLASSIFICATION_METRICS.get("classification_report", True):
            metrics['classification_report'] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        
        logger.info(f"Calculated {len(metrics)} metrics")
        return metrics
    
    def _optimize_threshold(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray
    ) -> float:
        """Optimize classification threshold based on target recall."""
        logger.info(f"Optimizing threshold for target recall: {self.target_recall}")
        
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Find threshold that achieves target recall
        target_idx = np.argmax(recall_vals >= self.target_recall)
        
        if target_idx > 0:
            optimal_threshold = thresholds[target_idx - 1]
        else:
            # If target recall not achievable, use default
            optimal_threshold = 0.5
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        return optimal_threshold
    
    def _generate_plots(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        metrics: Dict[str, float]
    ) -> None:
        """Generate evaluation plots."""
        logger.info("Generating evaluation plots")
        
        reports_dir = DIRECTORIES["reports"]
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1])
        axes[1].set_title('Normalized Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(reports_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(reports_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {metrics["pr_auc"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(reports_dir / "pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Feature Importance (if model supports it)
        if hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance(X)
        
        # 5. Metrics Comparison Visualization
        self._plot_metrics_comparison(metrics)
        
        logger.info("Evaluation plots saved to reports/ directory")
    
    def _plot_feature_importance(self, X: pd.DataFrame, top_n: int = 20) -> None:
        """Plot feature importance."""
        if not hasattr(self.model, 'feature_importances_'):
            return
        
        reports_dir = DIRECTORIES["reports"]
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, y='feature', x='importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.savefig(reports_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, metrics: Dict[str, Any]) -> None:
        """Create comprehensive metrics comparison visualization."""
        reports_dir = DIRECTORIES["reports"]
        
        # Extract key metrics for comparison
        key_metrics = {
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'F2-Score': metrics.get('f2_score', 0),
            'Specificity': metrics.get('specificity', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
            'PR-AUC': metrics.get('pr_auc', 0),
            'MCC': metrics.get('matthews_corrcoef', 0),
            'Cohen Kappa': metrics.get('cohen_kappa', 0),
        }
        
        # If optimal threshold metrics exist, create comparison
        if 'metrics_at_optimal_threshold' in metrics:
            optimal_metrics = metrics['metrics_at_optimal_threshold']
            key_metrics_optimal = {
                'Accuracy': optimal_metrics.get('accuracy', 0),
                'Precision': optimal_metrics.get('precision', 0),
                'Recall': optimal_metrics.get('recall', 0),
                'F1-Score': optimal_metrics.get('f1_score', 0),
                'F2-Score': optimal_metrics.get('f2_score', 0),
                'Specificity': optimal_metrics.get('specificity', 0),
                'ROC-AUC': optimal_metrics.get('roc_auc', 0),
                'PR-AUC': optimal_metrics.get('pr_auc', 0),
                'MCC': optimal_metrics.get('matthews_corrcoef', 0),
                'Cohen Kappa': optimal_metrics.get('cohen_kappa', 0),
            }
            
            # Create comparison plot
            fig, ax = plt.subplots(figsize=(14, 8))
            x = np.arange(len(key_metrics))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, list(key_metrics.values()), width, 
                          label=f'Threshold 0.5', alpha=0.8)
            bars2 = ax.bar(x + width/2, list(key_metrics_optimal.values()), width,
                          label=f'Optimal Threshold ({metrics.get("optimal_threshold", 0):.3f})', alpha=0.8)
            
            ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title('Comprehensive Metrics Comparison: Default vs Optimal Threshold', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(list(key_metrics.keys()), rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.1])
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(reports_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Single threshold plot
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(key_metrics))
            bars = ax.bar(x, list(key_metrics.values()), alpha=0.8, color='steelblue')
            
            ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title('Comprehensive Classification Metrics', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(list(key_metrics.keys()), rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.1])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(reports_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Metrics comparison plot saved")
    
    def generate_shap_plots(
        self,
        X: pd.DataFrame,
        sample_size: int = 1000
    ) -> None:
        """Generate SHAP plots for model interpretability."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP plots.")
            return
        
        logger.info("Generating SHAP plots")
        reports_dir = DIRECTORIES["reports"]
        
        # Sample data for SHAP (computationally expensive)
        X_sample = X.sample(min(sample_size, len(X)), random_state=42)
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.savefig(reports_dir / "shap_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("SHAP plots generated successfully")
            
        except Exception as e:
            logger.warning(f"SHAP plot generation failed: {str(e)}")
    
    def generate_report(
        self,
        metrics: Dict[str, Any],
        model_info: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive markdown evaluation report with all classification metrics.
        Educational: Includes all metrics for learning and comparison.
        
        Args:
            metrics: Evaluation metrics
            model_info: Additional model information
        
        Returns:
            Markdown report string
        """
        logger.info("Generating comprehensive evaluation report")
        
        report = f"""# Fraud Detection Model - Comprehensive Evaluation Report

## Model Information
"""
        
        if model_info:
            for key, value in model_info.items():
                report += f"- **{key}**: {value}\n"
        
        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            report += f"""
## Confusion Matrix

| | Predicted: Legitimate | Predicted: Fraud |
|---|---|---|
| **Actual: Legitimate** | {cm[0][0]:,} (TN) | {cm[0][1]:,} (FP) |
| **Actual: Fraud** | {cm[1][0]:,} (FN) | {cm[1][1]:,} (TP) |

**Legend**: TN = True Negative, FP = False Positive, FN = False Negative, TP = True Positive

### Confusion Matrix Components
- **True Positives (TP)**: {metrics.get('true_positives', 'N/A'):,}
- **True Negatives (TN)**: {metrics.get('true_negatives', 'N/A'):,}
- **False Positives (FP)**: {metrics.get('false_positives', 'N/A'):,}
- **False Negatives (FN)**: {metrics.get('false_negatives', 'N/A'):,}

"""
        
        # Comprehensive Metrics Table
        report += """
## Comprehensive Classification Metrics

### Standard Threshold (0.5)

| Metric Category | Metric Name | Value | Description |
|---|---|---|---|
"""
        
        # Basic Metrics
        report += f"""| **Basic Metrics** | Accuracy | {metrics.get('accuracy', 'N/A'):.4f} | Overall correctness |
| | Precision | {metrics.get('precision', 'N/A'):.4f} | True positives / (True positives + False positives) |
| | Recall (Sensitivity) | {metrics.get('recall', 'N/A'):.4f} | True positives / (True positives + False negatives) |
| | F1-Score | {metrics.get('f1_score', 'N/A'):.4f} | Harmonic mean of precision and recall |
| | F2-Score | {metrics.get('f2_score', 'N/A'):.4f} | Emphasizes recall more than F1 |
| | F0.5-Score | {metrics.get('f0.5_score', 'N/A'):.4f} | Emphasizes precision more than F1 |
"""
        
        # Advanced Metrics
        report += f"""| **Advanced Metrics** | Specificity (TNR) | {metrics.get('specificity', 'N/A'):.4f} | True negatives / (True negatives + False positives) |
| | Sensitivity (TPR) | {metrics.get('sensitivity', 'N/A'):.4f} | Same as Recall |
| | False Positive Rate (FPR) | {metrics.get('false_positive_rate', 'N/A'):.4f} | False positives / (False positives + True negatives) |
| | False Negative Rate (FNR) | {metrics.get('false_negative_rate', 'N/A'):.4f} | False negatives / (False negatives + True positives) |
| | True Positive Rate (TPR) | {metrics.get('true_positive_rate', 'N/A'):.4f} | Same as Recall/Sensitivity |
| | True Negative Rate (TNR) | {metrics.get('true_negative_rate', 'N/A'):.4f} | Same as Specificity |
"""
        
        # AUC Metrics
        report += f"""| **AUC Metrics** | ROC-AUC | {metrics.get('roc_auc', 'N/A'):.4f} | Area Under ROC Curve (ranking quality) |
| | PR-AUC (Average Precision) | {metrics.get('pr_auc', 'N/A'):.4f} | Area Under Precision-Recall Curve (better for imbalanced data) |
"""
        
        # Correlation Metrics
        report += f"""| **Correlation Metrics** | Matthews Correlation Coefficient (MCC) | {metrics.get('matthews_corrcoef', 'N/A'):.4f} | Balanced metric for imbalanced data (-1 to +1) |
| | Cohen's Kappa | {metrics.get('cohen_kappa', 'N/A'):.4f} | Agreement between predicted and actual (-1 to +1) |
"""
        
        # Additional Metrics
        report += f"""| **Additional Metrics** | Positive Predictive Value (PPV) | {metrics.get('positive_predictive_value', 'N/A'):.4f} | Same as Precision |
| | Negative Predictive Value (NPV) | {metrics.get('negative_predictive_value', 'N/A'):.4f} | True negatives / (True negatives + False negatives) |
| | Prevalence | {metrics.get('prevalence', 'N/A'):.4f} | Base rate of positive class in dataset |
"""
        
        report += "\n"
        
        # Optimal Threshold Metrics
        if 'optimal_threshold' in metrics:
            optimal_metrics = metrics.get('metrics_at_optimal_threshold', {})
            optimal_threshold = metrics.get('optimal_threshold', 0.5)
            
            report += f"""### Optimal Threshold ({optimal_threshold:.4f})

| Metric Category | Metric Name | Value | Change from Default |
|---|---|---|---|
"""
            
            # Compare key metrics
            comparison_metrics = [
                ('Accuracy', 'accuracy'),
                ('Precision', 'precision'),
                ('Recall', 'recall'),
                ('F1-Score', 'f1_score'),
                ('F2-Score', 'f2_score'),
                ('Specificity', 'specificity'),
                ('ROC-AUC', 'roc_auc'),
                ('PR-AUC', 'pr_auc'),
                ('MCC', 'matthews_corrcoef'),
                ('Cohen Kappa', 'cohen_kappa'),
            ]
            
            for metric_name, metric_key in comparison_metrics:
                default_val = metrics.get(metric_key, 0)
                optimal_val = optimal_metrics.get(metric_key, 0)
                change = optimal_val - default_val
                change_str = f"{change:+.4f}" if change != 0 else "0.0000"
                
                report += f"| | {metric_name} | {optimal_val:.4f} | {change_str} |\n"
            
            report += "\n"
        
        # Classification Report
        if 'classification_report' in metrics:
            report += "## Detailed Classification Report\n\n"
            report += "```\n"
            cls_report = metrics['classification_report']
            if isinstance(cls_report, dict):
                # Format the classification report dictionary
                report += f"              precision    recall  f1-score   support\n\n"
                for class_name, class_metrics in cls_report.items():
                    if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                        report += f"{class_name:15s} {class_metrics['precision']:8.4f} {class_metrics['recall']:8.4f} "
                        report += f"{class_metrics['f1-score']:8.4f} {class_metrics['support']:8.0f}\n"
                if 'accuracy' in cls_report:
                    report += f"\naccuracy                            {cls_report['accuracy']:8.4f}\n"
                if 'macro avg' in cls_report:
                    macro = cls_report['macro avg']
                    report += f"macro avg    {macro['precision']:8.4f} {macro['recall']:8.4f} "
                    report += f"{macro['f1-score']:8.4f} {macro['support']:8.0f}\n"
                if 'weighted avg' in cls_report:
                    weighted = cls_report['weighted avg']
                    report += f"weighted avg {weighted['precision']:8.4f} {weighted['recall']:8.4f} "
                    report += f"{weighted['f1-score']:8.4f} {weighted['support']:8.0f}\n"
            report += "```\n\n"
        
        report += """
## Visualizations

The following plots have been generated:
- **Confusion Matrix** (`confusion_matrix.png`) - Shows TP, TN, FP, FN
- **ROC Curve** (`roc_curve.png`) - True Positive Rate vs False Positive Rate
- **Precision-Recall Curve** (`pr_curve.png`) - Precision vs Recall (better for imbalanced data)
- **Feature Importance** (`feature_importance.png`) - Top features contributing to predictions
- **Metrics Comparison** (`metrics_comparison.png`) - Side-by-side comparison of all metrics
- **SHAP Summary Plot** (`shap_summary.png`) - Model interpretability (if SHAP is available)

## Educational Notes on Metrics

### When to Use Which Metric?

1. **Accuracy**: Good for balanced datasets, misleading for imbalanced data
2. **Precision**: Important when false positives are costly (e.g., blocking legitimate transactions)
3. **Recall**: Critical for fraud detection - we want to catch as many fraud cases as possible
4. **F1-Score**: Balanced view when precision and recall are equally important
5. **F2-Score**: Emphasizes recall - use when missing fraud is worse than false alarms
6. **ROC-AUC**: Good for ranking quality, less sensitive to class imbalance
7. **PR-AUC**: Better than ROC-AUC for imbalanced data (like fraud detection)
8. **MCC**: Best single metric for imbalanced binary classification
9. **Specificity**: Important for customer experience (correctly identifying legitimate transactions)

### For Fraud Detection:
- **Primary Focus**: Recall (catch fraud) and PR-AUC (overall performance on imbalanced data)
- **Secondary Focus**: Precision (reduce false alarms) and Specificity (customer experience)
- **Overall Quality**: MCC and F2-Score provide balanced views

## Recommendations

Based on the evaluation metrics:
"""
        
        # Smart recommendations
        recall = metrics.get('recall', 0)
        precision = metrics.get('precision', 0)
        roc_auc = metrics.get('roc_auc', 0)
        pr_auc = metrics.get('pr_auc', 0)
        mcc = metrics.get('matthews_corrcoef', 0)
        
        if recall < 0.80:
            report += "- **⚠️ Low Recall**: Consider adjusting threshold or using SMOTE to improve fraud detection.\n"
        
        if precision < 0.50:
            report += "- **⚠️ Low Precision**: High false positive rate. Consider threshold optimization.\n"
        
        if roc_auc > 0.90:
            report += "- **✅ Excellent ROC-AUC**: Model has strong ranking quality.\n"
        
        if pr_auc > 0.80:
            report += "- **✅ Excellent PR-AUC**: Model performs well on imbalanced data.\n"
        
        if mcc > 0.50:
            report += "- **✅ Good MCC**: Model shows strong correlation between predictions and actual values.\n"
        
        if recall > 0.85 and precision > 0.60:
            report += "- **✅ Strong Performance**: Model balances fraud detection (recall) and accuracy (precision) well.\n"
        
        return report


if __name__ == "__main__":
    # Test evaluator
    from src.data.data_loader import load_and_prepare_data
    from src.data.data_splitter import split_features_target
    from src.preprocessing.preprocessor import FraudDetectionPreprocessor
    from src.models.model_trainer import FraudDetectionModelTrainer
    
    logger.info("Testing model evaluator")
    df = load_and_prepare_data(max_rows=2000)
    X, y = split_features_target(df)
    
    # Preprocess and train
    preprocessor = FraudDetectionPreprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    trainer = FraudDetectionModelTrainer(model_type="xgboost", use_smote=True)
    trainer.fit(X_transformed, y)
    
    # Evaluate
    evaluator = ModelEvaluator(trainer.model, preprocessor)
    metrics = evaluator.evaluate(X_transformed, y)
    
    print("Evaluation metrics:", metrics)

