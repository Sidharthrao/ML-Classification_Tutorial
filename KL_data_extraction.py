"""
KL_data_extraction.py

Comprehensive Data Extraction Script for Fraud Detection Project
==================================================================

Purpose:
--------
Standalone script to extract fraud detection data from SQLite database with:
- Schema inspection and documentation
- Data quality reporting
- Multiple export formats (CSV, Parquet, Pickle)
- Memory-efficient chunked processing
- Progress tracking and detailed logging
- Sample data generation for EDA

Author: Kilo Code
Date: 2025-11-12
Educational Focus: Production-grade data extraction patterns

Learning Objectives:
-------------------
1. Database connection management and error handling
2. Efficient data extraction for large datasets (6M+ records)
3. Data quality assessment and reporting
4. Memory-efficient chunked processing
5. Multiple export format handling
6. Progress tracking for long-running operations
7. Comprehensive logging practices
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import json
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("⚠️  PyArrow not available. Parquet export will be disabled.")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Database configuration
DATABASE_PATH = Path("../Database.db")  # Relative to project root
TABLE_NAME = "Fraud_detection"

# Output directories
OUTPUT_DIR = Path("data")
REPORTS_DIR = Path("reports")
LOGS_DIR = Path("logs")

# Data extraction configuration
DEFAULT_CHUNK_SIZE = 100_000  # Process 100k rows at a time
SAMPLE_SIZE = 10_000  # Sample size for quick EDA

# Column schema
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

# Expected columns
REQUIRED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", 
    "newbalanceDest", "isFraud", "isFlaggedFraud"
]

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: Path = LOGS_DIR) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Educational Note: Proper logging is crucial for production systems
    to track operations, debug issues, and monitor performance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"data_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger
    logger = logging.getLogger("KL_DataExtraction")
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - user-friendly messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Initialize logger
logger = setup_logging()

# ============================================================================
# DATABASE CONNECTION & INSPECTION
# ============================================================================

class DatabaseInspector:
    """
    Inspect SQLite database schema and structure.
    
    Educational Note: Always inspect your data source before extraction
    to understand structure, size, and potential issues.
    """
    
    def __init__(self, db_path: Path, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        self.logger = logging.getLogger("KL_DataExtraction.Inspector")
    
    def connect(self) -> sqlite3.Connection:
        """Establish database connection with error handling."""
        try:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")
            
            conn = sqlite3.connect(str(self.db_path))
            self.logger.info(f"✅ Connected to database: {self.db_path}")
            return conn
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def get_table_info(self) -> Dict[str, Any]:
        """
        Extract comprehensive table information.
        
        Returns:
            Dictionary with table metadata including:
            - row_count: Total number of rows
            - column_count: Number of columns
            - columns: List of column names and types
            - size_mb: Approximate size in MB
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            row_count = cursor.fetchone()[0]
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            columns_info = cursor.fetchall()
            
            columns = [
                {
                    "name": col[1],
                    "type": col[2],
                    "nullable": not bool(col[3]),
                    "default": col[4],
                    "primary_key": bool(col[5])
                }
                for col in columns_info
            ]
            
            # Estimate size (rough approximation)
            cursor.execute(f"SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0] / (1024 * 1024)  # Convert to MB
            
            info = {
                "table_name": self.table_name,
                "row_count": row_count,
                "column_count": len(columns),
                "columns": columns,
                "database_size_mb": round(db_size, 2),
                "inspection_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"📊 Table Info: {row_count:,} rows, {len(columns)} columns")
            return info
            
        finally:
            conn.close()
    
    def get_sample_data(self, n_rows: int = 5) -> pd.DataFrame:
        """Get sample rows from table."""
        conn = self.connect()
        try:
            query = f"SELECT * FROM {self.table_name} LIMIT {n_rows}"
            df = pd.read_sql_query(query, conn)
            self.logger.info(f"📋 Retrieved {len(df)} sample rows")
            return df
        finally:
            conn.close()
    
    def get_column_statistics(self) -> pd.DataFrame:
        """
        Get basic statistics for each column.
        
        Educational Note: Understanding column distributions helps
        identify data quality issues and inform preprocessing strategies.
        """
        conn = self.connect()
        try:
            # Get distinct counts for each column
            stats = []
            for col in REQUIRED_COLUMNS:
                query = f"SELECT COUNT(DISTINCT {col}) as distinct_count FROM {self.table_name}"
                result = pd.read_sql_query(query, conn)
                stats.append({
                    "column": col,
                    "distinct_count": result['distinct_count'][0],
                    "expected_type": COLUMN_TYPES.get(col, "unknown")
                })
            
            stats_df = pd.DataFrame(stats)
            self.logger.info("📊 Column statistics computed")
            return stats_df
            
        finally:
            conn.close()

    def generate_inspection_report(self, output_dir: Path) -> Path:
        """Generate comprehensive inspection report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"database_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Gather all information
        table_info = self.get_table_info()
        sample_data = self.get_sample_data()
        column_stats = self.get_column_statistics()
        
        # Create report
        report = {
            "inspection_metadata": {
                "database_path": str(self.db_path),
                "inspection_time": datetime.now().isoformat(),
                "script_version": "KL_1.0"
            },
            "table_information": table_info,
            "column_statistics": column_stats.to_dict(orient='records'),
            "sample_data": sample_data.to_dict(orient='records')
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📄 Inspection report saved: {report_path}")
        return report_path

# ============================================================================
# DATA EXTRACTOR
# ============================================================================

class FraudDataExtractor:
    """
    Main data extraction class with multiple export formats.
    
    Educational Note: This class demonstrates production patterns for
    extracting large datasets efficiently while maintaining data quality.
    """
    
    def __init__(
        self,
        db_path: Path = DATABASE_PATH,
        table_name: str = TABLE_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.chunk_size = chunk_size
        self.logger = logging.getLogger("KL_DataExtraction.Extractor")
        self.inspector = DatabaseInspector(db_path, table_name)
    
    def extract_full_dataset(
        self,
        output_format: str = 'csv',
        output_dir: Path = OUTPUT_DIR,
        stratified: bool = False
    ) -> Path:
        """
        Extract full dataset with progress tracking.
        
        Args:
            output_format: Export format ('csv', 'parquet', 'pickle')
            output_dir: Output directory path
            stratified: Whether to maintain fraud/legitimate ratio
        
        Returns:
            Path to exported file
        
        Educational Note: Chunked processing prevents memory issues
        with large datasets by processing data in manageable batches.
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get total row count
        table_info = self.inspector.get_table_info()
        total_rows = table_info['row_count']
        
        self.logger.info(f"🔄 Extracting {total_rows:,} rows in chunks of {self.chunk_size:,}")
        
        # Connect to database
        conn = sqlite3.connect(str(self.db_path))
        
        try:
            # Extract data in chunks with progress bar
            chunks = []
            offset = 0
            
            with tqdm(total=total_rows, desc="Extracting data", unit="rows") as pbar:
                while offset < total_rows:
                    # Calculate chunk size
                    current_chunk_size = min(self.chunk_size, total_rows - offset)
                    
                    # Query chunk
                    query = f"SELECT * FROM {self.table_name} LIMIT {current_chunk_size} OFFSET {offset}"
                    chunk = pd.read_sql_query(query, conn)
                    chunks.append(chunk)
                    
                    # Update progress
                    offset += current_chunk_size
                    pbar.update(current_chunk_size)
            
            # Combine all chunks
            self.logger.info("🔗 Combining chunks...")
            df = pd.concat(chunks, ignore_index=True)
            
            # Convert column types
            df = self._convert_column_types(df)
            
            # Validate data
            self._validate_data(df)
            
            # Export to specified format
            output_path = self._export_data(df, output_format, output_dir)
            
            self.logger.info(f"✅ Full dataset extracted: {len(df):,} rows")
            return output_path
            
        finally:
            conn.close()
    
    def extract_sample(
        self,
        sample_size: int = SAMPLE_SIZE,
        stratified: bool = True,
        output_dir: Path = OUTPUT_DIR
    ) -> Tuple[Path, Path, Path]:
        """
        Extract stratified sample for quick EDA.
        
        Args:
            sample_size: Number of rows to sample
            stratified: Maintain fraud/legitimate ratio
            output_dir: Output directory
        
        Returns:
            Tuple of (csv_path, pickle_path, summary_path)
        
        Educational Note: Stratified sampling ensures representative
        samples from imbalanced datasets, crucial for fraud detection.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        
        try:
            if stratified:
                # Get fraud and legitimate samples proportionally
                self.logger.info(f"📊 Extracting stratified sample of {sample_size:,} rows")
                
                # Get fraud count
                fraud_query = f"SELECT COUNT(*) FROM {self.table_name} WHERE isFraud = 1"
                total_query = f"SELECT COUNT(*) FROM {self.table_name}"
                
                fraud_count = pd.read_sql_query(fraud_query, conn).iloc[0, 0]
                total_count = pd.read_sql_query(total_query, conn).iloc[0, 0]
                
                fraud_ratio = fraud_count / total_count
                fraud_sample_size = int(sample_size * fraud_ratio)
                legit_sample_size = sample_size - fraud_sample_size
                
                self.logger.info(f"  Fraud samples: {fraud_sample_size:,}")
                self.logger.info(f"  Legitimate samples: {legit_sample_size:,}")
                
                # Extract stratified samples
                fraud_query = f"""
                SELECT * FROM {self.table_name} 
                WHERE isFraud = 1 
                ORDER BY RANDOM() 
                LIMIT {fraud_sample_size}
                """
                
                legit_query = f"""
                SELECT * FROM {self.table_name} 
                WHERE isFraud = 0 
                ORDER BY RANDOM() 
                LIMIT {legit_sample_size}
                """
                
                df_fraud = pd.read_sql_query(fraud_query, conn)
                df_legit = pd.read_sql_query(legit_query, conn)
                
                # Combine and shuffle
                df = pd.concat([df_fraud, df_legit], ignore_index=True)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                
            else:
                # Random sample
                self.logger.info(f"📊 Extracting random sample of {sample_size:,} rows")
                query = f"""
                SELECT * FROM {self.table_name} 
                ORDER BY RANDOM() 
                LIMIT {sample_size}
                """
                df = pd.read_sql_query(query, conn)
            
            # Convert types and validate
            df = self._convert_column_types(df)
            self._validate_data(df)
            
            # Export in multiple formats for convenience
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            csv_path = output_dir / f"fraud_sample_{sample_size}_{timestamp}.csv"
            pickle_path = output_dir / f"fraud_sample_{sample_size}_{timestamp}.pkl"
            
            df.to_csv(csv_path, index=False)
            df.to_pickle(pickle_path)
            
            # Generate summary
            summary_path = self._generate_sample_summary(df, output_dir, timestamp)
            
            self.logger.info(f"✅ Sample extracted: {len(df):,} rows")
            self.logger.info(f"  CSV: {csv_path}")
            self.logger.info(f"  Pickle: {pickle_path}")
            self.logger.info(f"  Summary: {summary_path}")
            
            return csv_path, pickle_path, summary_path
            
        finally:
            conn.close()
    
    def _convert_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame columns to appropriate types.
        
        Educational Note: Proper data types reduce memory usage and
        enable correct operations (e.g., categorical for efficiency).
        """
        self.logger.info("🔄 Converting column types...")
        df_converted = df.copy()
        
        for column, dtype in COLUMN_TYPES.items():
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
                    
                    self.logger.debug(f"  ✓ {column} -> {dtype}")
                except Exception as e:
                    self.logger.warning(f"  ⚠️  {column} conversion failed: {e}")
        
        return df_converted
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate extracted data quality.
        
        Educational Note: Always validate data after extraction to catch
        corruption, missing values, or unexpected patterns early.
        """
        self.logger.info("🔍 Validating data quality...")
        
        # Check required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for empty dataframe
        if df.empty:
            raise ValueError("Extracted dataframe is empty")
        
        # Log data quality metrics
        self.logger.info(f"  Shape: {df.shape}")
        self.logger.info(f"  Missing values: {df.isnull().sum().sum():,}")
        self.logger.info(f"  Duplicate rows: {df.duplicated().sum():,}")
        self.logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Fraud distribution
        if 'isFraud' in df.columns:
            fraud_dist = df['isFraud'].value_counts()
            fraud_pct = (fraud_dist[1] / len(df) * 100) if len(fraud_dist) > 1 else 0
            self.logger.info(f"  Fraud rate: {fraud_pct:.4f}%")
            self.logger.info(f"  Fraud count: {fraud_dist.get(1, 0):,}")
            self.logger.info(f"  Legitimate count: {fraud_dist.get(0, 0):,}")
        
        self.logger.info("✅ Data validation passed")
    
    def _export_data(
        self,
        df: pd.DataFrame,
        format: str,
        output_dir: Path
    ) -> Path:
        """
        Export dataframe to specified format.
        
        Educational Note: Different formats have different trade-offs:
        - CSV: Human-readable, universal, but slower and larger
        - Parquet: Fast, compressed, efficient, but requires libraries
        - Pickle: Python-specific, preserves types, convenient for ML
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'csv':
            output_path = output_dir / f"fraud_detection_full_{timestamp}.csv"
            self.logger.info(f"💾 Exporting to CSV: {output_path}")
            df.to_csv(output_path, index=False)
            
        elif format.lower() == 'parquet':
            if not PARQUET_AVAILABLE:
                self.logger.warning("Parquet not available, falling back to CSV")
                return self._export_data(df, 'csv', output_dir)
            
            output_path = output_dir / f"fraud_detection_full_{timestamp}.parquet"
            self.logger.info(f"💾 Exporting to Parquet: {output_path}")
            df.to_parquet(output_path, index=False, compression='snappy')
            
        elif format.lower() == 'pickle':
            output_path = output_dir / f"fraud_detection_full_{timestamp}.pkl"
            self.logger.info(f"💾 Exporting to Pickle: {output_path}")
            df.to_pickle(output_path)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"✅ Export complete: {file_size_mb:.2f} MB")
        
        return output_path
    
    def _generate_sample_summary(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        timestamp: str
    ) -> Path:
        """Generate comprehensive summary of sample data."""
        summary_path = output_dir / f"sample_summary_{timestamp}.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FRAUD DETECTION SAMPLE DATA SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sample Size: {len(df):,} rows\n")
            f.write(f"Features: {len(df.columns)} columns\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
            
            f.write("FRAUD DISTRIBUTION:\n")
            f.write("-" * 70 + "\n")
            fraud_counts = df['isFraud'].value_counts()
            for label, count in fraud_counts.items():
                label_text = "Fraud" if label == 1 else "Legitimate"
                percentage = (count / len(df)) * 100
                f.write(f"{label_text}: {count:,} ({percentage:.2f}%)\n")
            f.write("\n")
            
            f.write("DATA QUALITY:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Missing Values: {df.isnull().sum().sum():,}\n")
            f.write(f"Duplicate Rows: {df.duplicated().sum():,}\n\n")
            
            f.write("NUMERICAL FEATURES SUMMARY:\n")
            f.write("-" * 70 + "\n")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            f.write(df[numeric_cols].describe().to_string())
            f.write("\n\n")
            
            f.write("CATEGORICAL FEATURES:\n")
            f.write("-" * 70 + "\n")
            if 'type' in df.columns:
                f.write("Transaction Types:\n")
                type_counts = df['type'].value_counts()
                for trans_type, count in type_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  {trans_type}: {count:,} ({percentage:.2f}%)\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        return summary_path

# ============================================================================
# DATA QUALITY REPORTER
# ============================================================================

class DataQualityReporter:
    """
    Generate comprehensive data quality reports.
    
    Educational Note: Data quality reporting is essential for
    understanding data reliability and informing preprocessing decisions.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.logger = logging.getLogger("KL_DataExtraction.QualityReporter")
    
    def generate_comprehensive_report(self, output_dir: Path = REPORTS_DIR) -> Path:
        """Generate detailed data quality report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            self._write_header(f)
            self._write_overview(f)
            self._write_missing_values_analysis(f)
            self._write_outlier_analysis(f)
            self._write_distribution_analysis(f)
            self._write_recommendations(f)
        
        self.logger.info(f"📄 Quality report generated: {report_path}")
        return report_path
    
    def _write_header(self, f):
        f.write("# Data Quality Report - Fraud Detection Dataset\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Dataset Size:** {len(self.df):,} rows × {len(self.df.columns)} columns\n\n")
        f.write("---\n\n")
    
    def _write_overview(self, f):
        f.write("## 1. Dataset Overview\n\n")
        f.write(f"- **Total Records:** {len(self.df):,}\n")
        f.write(f"- **Total Features:** {len(self.df.columns)}\n")
        f.write(f"- **Memory Usage:** {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        f.write(f"- **Duplicate Rows:** {self.df.duplicated().sum():,}\n\n")
        
        # Fraud distribution
        if 'isFraud' in self.df.columns:
            fraud_dist = self.df['isFraud'].value_counts()
            fraud_pct = (fraud_dist[1] / len(self.df) * 100) if len(fraud_dist) > 1 else 0
            f.write(f"### Fraud Distribution\n")
            f.write(f"- **Fraud Rate:** {fraud_pct:.4f}%\n")
            f.write(f"- **Fraud Cases:** {fraud_dist.get(1, 0):,}\n")
            f.write(f"- **Legitimate Cases:** {fraud_dist.get(0, 0):,}\n")
            f.write(f"- **Class Imbalance Ratio:** {fraud_dist.get(0, 0) / fraud_dist.get(1, 1):.2f}:1\n\n")
    
    def _write_missing_values_analysis(self, f):
        f.write("## 2. Missing Values Analysis\n\n")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        if missing.sum() == 0:
            f.write("✅ **No missing values detected!**\n\n")
        else:
            f.write("| Column | Missing Count | Percentage |\n")
            f.write("|--------|---------------|------------|\n")
            for col in missing[missing > 0].index:
                f.write(f"| {col} | {missing[col]:,} | {missing_pct[col]:.2f}% |\n")
            f.write("\n")
    
    def _write_outlier_analysis(self, f):
        f.write("## 3. Outlier Analysis (IQR Method)\n\n")
        f.write("| Column | Outliers | Percentage |\n")
        f.write("|--------|----------|------------|\n")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['isFraud', 'isFlaggedFraud', 'step']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_pct = (outliers / len(self.df)) * 100
                f.write(f"| {col} | {outliers:,} | {outlier_pct:.2f}% |\n")
        f.write("\n")
    
    def _write_distribution_analysis(self, f):
        f.write("## 4. Distribution Analysis\n\n")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        f.write("### Skewness\n\n")
        f.write("| Column | Skewness | Interpretation |\n")
        f.write("|--------|----------|----------------|\n")
        
        for col in numeric_cols:
            if col not in ['isFraud', 'isFlaggedFraud']:
                skew = self.df[col].skew()
                interpretation = "Highly skewed" if abs(skew) > 1 else "Moderately skewed" if abs(skew) > 0.5 else "Approximately symmetric"
                f.write(f"| {col} | {skew:.2f} | {interpretation} |\n")
        f.write("\n")
    
    def _write_recommendations(self, f):
        f.write("## 5. Recommendations\n\n")
        
        # Based on fraud rate
        if 'isFraud' in self.df.columns:
            fraud_rate = (self.df['isFraud'].sum() / len(self.df)) * 100
            if fraud_rate < 1:
                f.write("- ⚠️ **Severe class imbalance detected**. Use SMOTE, ADASYN, or class weights.\n")
        
        # Based on skewness
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        skewed_cols = [col for col in numeric_cols if abs(self.df[col].skew()) > 1]
        if skewed_cols:
            f.write(f"- 📊 **Log transformation recommended** for: {', '.join(skewed_cols)}\n")
        
        # Based on outliers
        has_outliers = False
        for col in numeric_cols:
            if col not in ['isFraud', 'isFlaggedFraud', 'step']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers / len(self.df) > 0.05:
                    has_outliers = True
                    break
        
        if has_outliers:
            f.write("- 🎯 **Use RobustScaler** instead of StandardScaler for outlier resistance.\n")
        
        f.write("\n---\n\n*Report generated by KL_data_extraction.py*\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function with user-friendly CLI.
    
    Educational Note: Well-structured main() functions make scripts
    easy to use and maintain. Always provide clear options and feedback.
    """
    print("=" * 70)
    print("FRAUD DETECTION DATA EXTRACTION TOOL")
    print("=" * 70)
    print()
    
    # Step 1: Database Inspection
    print("📊 Step 1: Database Inspection")
    print("-" * 70)
    inspector = DatabaseInspector(DATABASE_PATH, TABLE_NAME)
    
    try:
        # Get and display table info
        table_info = inspector.get_table_info()
        print(f"✅ Database: {DATABASE_PATH.name}")
        print(f"✅ Table: {table_info['table_name']}")
        print(f"✅ Total Rows: {table_info['row_count']:,}")
        print(f"✅ Columns: {table_info['column_count']}")
        print(f"✅ Database Size: {table_info['database_size_mb']} MB")
        print()
        
        # Generate inspection report
        report_path = inspector.generate_inspection_report(REPORTS_DIR)
        print(f"📄 Inspection report saved: {report_path}")
        print()
        
    except Exception as e:
        logger.error(f"❌ Inspection failed: {e}")
        return
    
    # Step 2: Extract Sample Data
    print("📊 Step 2: Sample Data Extraction")
    print("-" * 70)
    extractor = FraudDataExtractor()
    
    try:
        csv_path, pickle_path, summary_path = extractor.extract_sample(
            sample_size=SAMPLE_SIZE,
            stratified=True
        )
        print(f"✅ Sample data extracted successfully!")
        print()
        
        # Load sample for quality report
        df_sample = pd.read_pickle(pickle_path)
        
    except Exception as e:
        logger.error(f"❌ Sample extraction failed: {e}")
        return
    
    # Step 3: Generate Data Quality Report
    print("📊 Step 3: Data Quality Report")
    print("-" * 70)
    
    try:
        reporter = DataQualityReporter(df_sample)
        quality_report_path = reporter.generate_comprehensive_report()
        print(f"✅ Quality report generated: {quality_report_path}")
        print()
        
    except Exception as e:
        logger.error(f"❌ Quality report generation failed: {e}")
    
    # Step 4: Optional Full Dataset Extraction
    print("📊 Step 4: Full Dataset Extraction (Optional)")
    print("-" * 70)
    print("This will extract the full 6M+ dataset.")
    print("⚠️  This may take several minutes and require significant disk space.")
    print()
    
    user_input = input("Extract full dataset? (y/N): ").strip().lower()
    
    if user_input == 'y':
        try:
            output_path = extractor.extract_full_dataset(
                output_format='pickle',  # Fastest for ML workflows
                stratified=False
            )
            print(f"✅ Full dataset extracted: {output_path}")
            print()
        except Exception as e:
            logger.error(f"❌ Full extraction failed: {e}")
    else:
        print("⏭️  Full extraction skipped.")
        print()
    
    # Summary
    print("=" * 70)
    print("✅ DATA EXTRACTION COMPLETE!")
    print("=" * 70)
    print()
    print("📁 Output Locations:")
    print(f"  - Data samples: {OUTPUT_DIR}/")
    print(f"  - Reports: {REPORTS_DIR}/")
    print(f"  - Logs: {LOGS_DIR}/")
    print()
    print("📖 Next Steps:")
    print("  1. Review the data quality report")
    print("  2. Explore sample data in KL_eda.ipynb")
    print("  3. Use extracted data for model training")
    print()
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Extraction interrupted by user")
        logger.info("Extraction interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)