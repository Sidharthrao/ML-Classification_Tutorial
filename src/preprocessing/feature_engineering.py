"""
Feature engineering module.
Creates derived features from raw transaction data.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import FEATURE_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger("feature_engineering")


def create_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create balance-related features.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with balance features added
    """
    logger.debug("Creating balance features")
    df_features = df.copy()
    
    # Balance differences
    df_features['balance_diff_orig'] = (
        df_features['oldbalanceOrg'] - df_features['newbalanceOrig']
    )
    df_features['balance_diff_dest'] = (
        df_features['newbalanceDest'] - df_features['oldbalanceDest']
    )
    
    # Zero balance flags
    df_features['balance_orig_zero'] = (df_features['oldbalanceOrg'] == 0).astype(int)
    df_features['balance_dest_zero'] = (df_features['oldbalanceDest'] == 0).astype(int)
    
    # Zero balance after transaction
    df_features['zero_balance_after_transaction'] = (
        df_features['newbalanceOrig'] == 0
    ).astype(int)
    
    # Balance ratios
    df_features['balance_orig_ratio'] = np.where(
        df_features['oldbalanceOrg'] > 0,
        df_features['amount'] / df_features['oldbalanceOrg'],
        0
    )
    
    df_features['balance_dest_ratio'] = np.where(
        df_features['oldbalanceDest'] > 0,
        df_features['amount'] / (df_features['oldbalanceDest'] + 1),
        0
    )
    
    return df_features


def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction-related features.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with transaction features added
    """
    logger.debug("Creating transaction features")
    df_features = df.copy()
    
    # Log amount (handle zero and negative)
    df_features['amount_log'] = np.log1p(df_features['amount'].clip(lower=0))
    
    # Amount per original balance
    df_features['amount_per_balance_orig'] = (
        df_features['amount'] / (df_features['oldbalanceOrg'] + 1)
    )
    
    # Transaction type is already present, will be encoded separately
    
    # Check if transaction empties origin account
    df_features['empties_origin'] = (
        (df_features['oldbalanceOrg'] > 0) & 
        (df_features['newbalanceOrig'] == 0)
    ).astype(int)
    
    # Check if transaction creates new destination balance
    df_features['creates_dest_balance'] = (
        (df_features['oldbalanceDest'] == 0) & 
        (df_features['newbalanceDest'] > 0)
    ).astype(int)
    
    # Amount categories (buckets)
    df_features['amount_category'] = pd.cut(
        df_features['amount'],
        bins=[0, 100, 1000, 10000, 100000, float('inf')],
        labels=['very_small', 'small', 'medium', 'large', 'very_large'],
        include_lowest=True
    )
    
    return df_features


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-related features from step column.
    
    Args:
        df: Input DataFrame with 'step' column
    
    Returns:
        DataFrame with time features added
    """
    logger.debug("Creating time features")
    df_features = df.copy()
    
    if 'step' not in df_features.columns:
        logger.warning("'step' column not found, skipping time features")
        return df_features
    
    # Hour of day (step represents hours)
    df_features['hour_of_day'] = df_features['step'] % 24
    
    # Day of week (assuming step 0 is start of week)
    df_features['day_of_week'] = (df_features['step'] // 24) % 7
    
    # Is weekend
    df_features['is_weekend'] = (
        (df_features['day_of_week'] == 5) | (df_features['day_of_week'] == 6)
    ).astype(int)
    
    # Is business hours (9-17)
    df_features['is_business_hours'] = (
        (df_features['hour_of_day'] >= 9) & (df_features['hour_of_day'] < 17)
    ).astype(int)
    
    # Is night (22-6)
    df_features['is_night'] = (
        (df_features['hour_of_day'] >= 22) | (df_features['hour_of_day'] < 6)
    ).astype(int)
    
    return df_features


def create_account_features(df: pd.DataFrame, use_frequency: bool = False) -> pd.DataFrame:
    """
    Create account-related features.
    
    Args:
        df: Input DataFrame
        use_frequency: Whether to compute account frequency (computationally expensive)
    
    Returns:
        DataFrame with account features added
    """
    logger.debug("Creating account features")
    df_features = df.copy()
    
    # Same account transfer flag
    df_features['same_account_transfer'] = (
        df_features['nameOrig'] == df_features['nameDest']
    ).astype(int)
    
    # Account name prefixes (C = customer, M = merchant)
    df_features['orig_is_customer'] = (
        df_features['nameOrig'].str.startswith('C', na=False)
    ).astype(int)
    df_features['dest_is_customer'] = (
        df_features['nameDest'].str.startswith('C', na=False)
    ).astype(int)
    
    if use_frequency:
        logger.info("Computing account frequency features (this may take time)")
        # Frequency of origin account
        orig_counts = df_features['nameOrig'].value_counts()
        df_features['orig_account_frequency'] = df_features['nameOrig'].map(orig_counts)
        
        # Frequency of destination account
        dest_counts = df_features['nameDest'].value_counts()
        df_features['dest_account_frequency'] = df_features['nameDest'].map(dest_counts)
        
        logger.debug("Account frequency features computed")
    else:
        logger.debug("Skipping account frequency features (computationally expensive)")
    
    return df_features


def create_all_features(
    df: pd.DataFrame,
    feature_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Create all engineered features.
    
    Args:
        df: Input DataFrame
        feature_config: Configuration dictionary for feature creation
    
    Returns:
        DataFrame with all features added
    """
    if feature_config is None:
        feature_config = FEATURE_CONFIG
    
    logger.info("Starting feature engineering pipeline")
    df_features = df.copy()
    
    # Balance features
    if feature_config.get("use_balance_features", True):
        df_features = create_balance_features(df_features)
    
    # Transaction features
    if feature_config.get("use_transaction_features", True):
        df_features = create_transaction_features(df_features)
    
    # Time features
    if feature_config.get("use_time_features", True):
        df_features = create_time_features(df_features)
    
    # Account features
    df_features = create_account_features(
        df_features,
        use_frequency=feature_config.get("use_account_frequency", False)
    )
    
    logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
    logger.debug(f"New feature columns: {[col for col in df_features.columns if col not in df.columns]}")
    
    return df_features


if __name__ == "__main__":
    # Test feature engineering
    from src.data.data_loader import load_and_prepare_data
    
    logger.info("Testing feature engineering")
    df = load_and_prepare_data(max_rows=1000)
    df_features = create_all_features(df)
    
    print(f"Original columns: {len(df.columns)}")
    print(f"Features columns: {len(df_features.columns)}")
    print(f"New features: {[col for col in df_features.columns if col not in df.columns]}")

