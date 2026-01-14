# Fraud Detection ML Pipeline

An industry-ready machine learning pipeline for detecting fraudulent transactions in financial data.

## Project Overview

This project implements a complete ML pipeline for fraud detection using:
- **Dataset**: 6M+ financial transactions from SQLite database
- **Task**: Binary classification (fraud vs. legitimate)
- **Challenge**: Extreme class imbalance (0.13% fraud rate)
- **Deployment**: Both batch script and Flask API

## Project Structure

```
1.Fraud_Detection/
├── config/              # Configuration parameters
├── src/                 # Source code
│   ├── data/           # Data loading and splitting
│   ├── preprocessing/  # Feature engineering and preprocessing
│   ├── models/         # Model training
│   ├── evaluation/     # Model evaluation
│   └── utils/           # Utility functions
├── scripts/            # Training and prediction scripts
├── api/                # Flask API endpoints
├── models/             # Saved model artifacts
├── notebooks/          # Jupyter notebooks for EDA
├── reports/            # Evaluation reports and plots
├── logs/               # Application logs
└── tests/              # Unit tests
```

## Setup Instructions

### 1. Virtual Environment Setup

The project uses a shared virtual environment at the repository root:

```bash
# Navigate to repository root
cd /path/to/Project-Rogue

# Create virtual environment (if not exists)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate      # On Windows
```

### 2. Install Dependencies

```bash
# Navigate to project directory
cd "Inttrvu/Capstone_Projects/Capstone_Project - Classification/1.Fraud_Detection"

# Install requirements
pip install -r requirements.txt
```

### 3. Database Setup

The database file is located at:
```
Inttrvu/Capstone_Projects/Database.db
```

The project automatically uses this database. Ensure the `Fraud_detection` table exists.

## Usage

### Training the Model

Train the model using the provided script:

```bash
# Activate virtual environment first
source ../../../../venv/bin/activate

# Run training script
python scripts/train_model.py
```

The training pipeline will:
1. Load data from the database (6M+ records)
2. Split into train (4M), eval (1M), and test sets
3. Engineer features and preprocess data
4. Train XGBoost model with SMOTE for class imbalance
5. Evaluate on evaluation set
6. Save model artifacts to `models/` directory
7. Generate evaluation report in `reports/model_report.md`

**Training Time**: Approximately 30-60 minutes depending on hardware.

### Batch Prediction

Make predictions on new data from a CSV file:

```bash
python scripts/predict.py --input data/new_transactions.csv --output predictions.csv
```

The script will:
- Load saved model and preprocessor
- Apply consistent preprocessing
- Generate predictions with fraud probabilities
- Save results to CSV file

### Flask API

Start the Flask API for real-time predictions:

```bash
python api/app.py
```

The API will start on `http://0.0.0.0:5000` by default.

#### API Endpoints

**Health Check**
```bash
GET /health
```

**Model Information**
```bash
GET /model_info
```

**Single Prediction**
```bash
POST /predict
Content-Type: application/json

{
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36,
    "nameDest": "M1979787155",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}
```

**Batch Prediction**
```bash
POST /predict_batch
Content-Type: application/json

[
    {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        ...
    },
    {
        "step": 2,
        "type": "PAYMENT",
        ...
    }
]
```

#### Example API Usage

```python
import requests

# Single prediction
response = requests.post('http://localhost:5000/predict', json={
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36,
    "nameDest": "M1979787155",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
})

result = response.json()
print(f"Fraud: {result['is_fraud']}, Probability: {result['fraud_probability']:.4f}")
```

## Model Details

### Selected Model
- **Primary Model**: XGBoost Classifier
- **Alternative Models**: LightGBM, Random Forest, Logistic Regression

### Features

The model uses engineered features from raw transaction data:

**Balance Features**:
- Balance differences (origin/destination)
- Zero balance flags
- Balance ratios

**Transaction Features**:
- Log-transformed amount
- Amount per balance ratio
- Transaction type encoding
- Account emptying flags

**Time Features**:
- Hour of day
- Day of week
- Business hours flag
- Night time flag

**Account Features**:
- Same account transfer flag
- Account type flags (customer/merchant)

### Class Imbalance Handling

- **SMOTE**: Synthetic Minority Oversampling Technique
- **Class Weights**: Scale positive class weight in models
- **Threshold Optimization**: Optimize classification threshold for target recall

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve (important for imbalanced data)

## Model Artifacts

After training, the following files are saved in `models/`:
- `preprocessor.pkl`: Preprocessing pipeline
- `model.pkl`: Trained model
- `feature_names.pkl`: Feature names for consistency

## Reports

Evaluation reports and visualizations are saved in `reports/`:
- `model_report.md`: Comprehensive evaluation report
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve
- `pr_curve.png`: Precision-Recall curve
- `feature_importance.png`: Feature importance plot
- `shap_summary.png`: SHAP summary plot (if available)

## Configuration

Model and pipeline settings can be modified in `config/config.py`:
- Data split ratios
- Model hyperparameters
- Feature engineering options
- Evaluation thresholds

## Logging

Application logs are saved to `logs/fraud_detection.log`. Log levels can be configured in the logger setup.

## Troubleshooting

### Database Connection Issues
- Verify database path in `config/config.py`
- Ensure database file exists and is accessible

### Model Not Found
- Ensure training script has been run successfully
- Check that model files exist in `models/` directory

### Memory Issues
- Reduce `chunk_size` in data loader for systems with limited RAM
- Consider using smaller sample sizes for evaluation

### SHAP Plot Generation Fails
- SHAP is optional and requires sufficient memory
- Reduce `sample_size` in `generate_shap_plots()` if needed

## Performance Considerations

- **Training**: Expect 30-60 minutes for full dataset
- **Prediction**: ~1000 transactions/second on modern hardware
- **Memory**: Requires ~8GB RAM for full dataset processing

## Future Enhancements

- Real-time streaming predictions
- Model versioning and A/B testing
- Online learning for model updates
- Enhanced feature engineering with account patterns
- Ensemble methods with multiple models

## License

See LICENSE file in repository root.

## Contact

For questions or issues, please contact the project maintainer.

