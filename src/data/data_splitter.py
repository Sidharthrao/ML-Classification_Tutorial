"""
Data splitting module.
Handles splitting of dataset into train, evaluation, and test sets.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import TRAIN_SIZE, EVAL_SIZE
from src.utils.logger import setup_logger

logger = setup_logger("data_splitter")


def split_data(
    df: pd.DataFrame,
    train_size: int = TRAIN_SIZE,
    eval_size: int = EVAL_SIZE,
    random_state: int = 42,
    preserve_order: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, evaluation, and test sets.
    
    Args:
        df: Input DataFrame
        train_size: Number of records for training (default: 4M)
        eval_size: Number of records for evaluation (default: 1M)
        random_state: Random seed for reproducibility
        preserve_order: If True, maintain temporal order (use step column)
    
    Returns:
        Tuple of (train_df, eval_df, test_df)
    """
    logger.info(f"Splitting data: train={train_size:,}, eval={eval_size:,}")
    
    total_rows = len(df)
    logger.info(f"Total rows: {total_rows:,}")
    
    if preserve_order and 'step' in df.columns:
        # Sort by step to preserve temporal order
        df_sorted = df.sort_values('step').reset_index(drop=True)
        logger.info("Preserving temporal order based on 'step' column")
    else:
        df_sorted = df.copy()
        if not preserve_order:
            # Shuffle for random split
            df_sorted = df_sorted.sample(frac=1, random_state=random_state).reset_index(drop=True)
            logger.info("Randomly shuffling data")
    
    # Calculate split indices
    train_end = min(train_size, total_rows)
    eval_end = min(train_size + eval_size, total_rows)
    
    # Split data
    train_df = df_sorted.iloc[:train_end].copy()
    eval_df = df_sorted.iloc[train_end:eval_end].copy()
    test_df = df_sorted.iloc[eval_end:].copy()
    
    logger.info(f"Train set: {len(train_df):,} rows")
    logger.info(f"Eval set: {len(eval_df):,} rows")
    logger.info(f"Test set: {len(test_df):,} rows")
    
    # Log class distribution for each split
    if 'isFraud' in train_df.columns:
        logger.info(f"Train fraud distribution:\n{train_df['isFraud'].value_counts()}")
        logger.info(f"Eval fraud distribution:\n{eval_df['isFraud'].value_counts()}")
        logger.info(f"Test fraud distribution:\n{test_df['isFraud'].value_counts()}")
    
    return train_df, eval_df, test_df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "isFraud"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
    
    Returns:
        Tuple of (features_df, target_series)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Drop target and other non-feature columns
    columns_to_drop = [
        target_column,
        'isFlaggedFraud',  # Business flag, not a feature
    ]
    
    # Keep only columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    X = df.drop(columns=columns_to_drop).copy()
    y = df[target_column].copy()
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Feature columns: {list(X.columns)}")
    
    return X, y


def get_stratified_split_info(df: pd.DataFrame, target_column: str = "isFraud") -> dict:
    """
    Get information about class distribution for stratified splitting.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
    
    Returns:
        Dictionary with class distribution information
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    class_counts = df[target_column].value_counts().sort_index()
    class_proportions = df[target_column].value_counts(normalize=True).sort_index()
    
    info = {
        "class_counts": class_counts.to_dict(),
        "class_proportions": class_proportions.to_dict(),
        "total_samples": len(df),
        "n_classes": len(class_counts),
        "imbalance_ratio": class_counts.min() / class_counts.max() if len(class_counts) > 1 else 1.0
    }
    
    logger.info(f"Class distribution: {info['class_counts']}")
    logger.info(f"Imbalance ratio: {info['imbalance_ratio']:.6f}")
    
    return info


if __name__ == "__main__":
    # Test splitting (requires data to be loaded first)
    from src.data.data_loader import load_and_prepare_data
    
    logger.info("Testing data splitting")
    df = load_and_prepare_data(max_rows=10000)
    train_df, eval_df, test_df = split_data(df, train_size=5000, eval_size=3000)
    
    print(f"Train: {len(train_df)}, Eval: {len(eval_df)}, Test: {len(test_df)}")
    
    # Test feature/target split
    X_train, y_train = split_features_target(train_df)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

