"""
Visualization utility functions.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import DIRECTORIES
from src.utils.logger import setup_logger

logger = setup_logger("visualizations")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_class_distribution(y: pd.Series, save_path: Optional[Path] = None) -> None:
    """Plot class distribution."""
    plt.figure(figsize=(8, 6))
    y.value_counts().plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, columns: list, save_path: Optional[Path] = None) -> None:
    """Plot distributions of selected features."""
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(columns[:len(axes)]):
        if col in df.columns:
            df[col].hist(bins=50, ax=axes[i])
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide extra subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature distribution plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """Plot correlation matrix."""
    plt.figure(figsize=(12, 10))
    correlation = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()

