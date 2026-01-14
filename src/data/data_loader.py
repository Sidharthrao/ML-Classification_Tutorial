"""
Data loading module.
Handles database connection and data extraction from SQLite database.
"""
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import DATABASE_PATH, DB_TABLE_NAME, COLUMN_TYPES
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")


def load_data_from_db(
    db_path: Optional[Path] = None,
    table_name: Optional[str] = None,
    chunk_size: int = 100000,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load data from SQLite database with chunking for large datasets.
    
    Args:
        db_path: Path to database file (defaults to config DATABASE_PATH)
        table_name: Name of table to load (defaults to config DB_TABLE_NAME)
        chunk_size: Number of rows to load per chunk
        max_rows: Maximum number of rows to load (None for all)
    
    Returns:
        DataFrame containing the loaded data
    """
    if db_path is None:
        db_path = DATABASE_PATH
    if table_name is None:
        table_name = DB_TABLE_NAME
    
    logger.info(f"Loading data from {db_path} table {table_name}")
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        
        # Get total row count
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        logger.info(f"Total rows in table: {total_rows}")
        
        # Determine how many rows to load
        rows_to_load = min(total_rows, max_rows) if max_rows else total_rows
        
        # Load data in chunks
        chunks = []
        offset = 0
        
        while offset < rows_to_load:
            current_chunk_size = min(chunk_size, rows_to_load - offset)
            query = f"SELECT * FROM {table_name} LIMIT {current_chunk_size} OFFSET {offset}"
            
            chunk = pd.read_sql_query(query, conn)
            chunks.append(chunk)
            
            offset += current_chunk_size
            logger.debug(f"Loaded {offset:,}/{rows_to_load:,} rows")
        
        conn.close()
        
        # Concatenate all chunks
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Successfully loaded {len(df):,} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def convert_column_types(df: pd.DataFrame, column_types: Optional[dict] = None) -> pd.DataFrame:
    """
    Convert column types according to schema.
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to types (defaults to config COLUMN_TYPES)
    
    Returns:
        DataFrame with converted types
    """
    if column_types is None:
        column_types = COLUMN_TYPES
    
    logger.info("Converting column types")
    df_converted = df.copy()
    
    for column, dtype in column_types.items():
        if column in df_converted.columns:
            try:
                if dtype == "float64":
                    df_converted[column] = pd.to_numeric(df_converted[column], errors="coerce")
                elif dtype == "int64":
                    df_converted[column] = pd.to_numeric(df_converted[column], errors="coerce").astype("Int64")
                elif dtype == "category":
                    df_converted[column] = df_converted[column].astype("category")
                elif dtype == "string":
                    df_converted[column] = df_converted[column].astype("string")
                else:
                    df_converted[column] = df_converted[column].astype(dtype)
                
                logger.debug(f"Converted {column} to {dtype}")
                
            except Exception as e:
                logger.warning(f"Failed to convert {column} to {dtype}: {str(e)}")
    
    return df_converted


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data validation checks.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Validated DataFrame
    
    Raises:
        ValueError: If critical validation checks fail
    """
    logger.info("Validating data")
    
    # Check required columns
    required_columns = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", 
                        "newbalanceOrig", "nameDest", "oldbalanceDest", 
                        "newbalanceDest", "isFraud", "isFlaggedFraud"]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Log basic statistics
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Missing values per column:\n{df.isnull().sum()}")
    logger.info(f"Fraud class distribution:\n{df['isFraud'].value_counts()}")
    
    # Check for negative amounts (should not exist)
    negative_amounts = (df['amount'] < 0).sum()
    if negative_amounts > 0:
        logger.warning(f"Found {negative_amounts} transactions with negative amounts")
    
    return df


def load_and_prepare_data(
    db_path: Optional[Path] = None,
    table_name: Optional[str] = None,
    chunk_size: int = 100000,
    max_rows: Optional[int] = None,
    convert_types: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Complete data loading pipeline: load, convert types, and validate.
    
    Args:
        db_path: Path to database file
        table_name: Name of table to load
        chunk_size: Number of rows to load per chunk
        max_rows: Maximum number of rows to load
        convert_types: Whether to convert column types
        validate: Whether to validate data
    
    Returns:
        Prepared DataFrame
    """
    logger.info("Starting data loading pipeline")
    
    # Load data
    df = load_data_from_db(db_path, table_name, chunk_size, max_rows)
    
    # Convert types
    if convert_types:
        df = convert_column_types(df)
    
    # Validate
    if validate:
        df = validate_data(df)
    
    logger.info("Data loading pipeline completed successfully")
    return df


if __name__ == "__main__":
    # Test data loading
    df = load_and_prepare_data(max_rows=1000)
    print(f"Loaded {len(df)} rows")
    print(df.head())
    print(df.info())

