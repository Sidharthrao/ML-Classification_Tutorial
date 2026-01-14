"""
Flask API application for Fraud Detection Model.
Provides RESTful API endpoints for real-time fraud detection.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.predict_endpoint import (
    predict_single, predict_batch, validate_transaction, get_model_info, initialize_model
)
from config.config import API_CONFIG
from src.utils.logger import setup_logger

logger = setup_logger("flask_api")

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize model at startup
def startup():
    """Initialize model when API starts."""
    logger.info("Initializing Flask API...")
    try:
        initialize_model()
        logger.info("Flask API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

# Initialize on module import
startup()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "fraud_detection_api"
    }), 200


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information."""
    try:
        info = get_model_info()
        return jsonify(info), 200
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Predict fraud for a single transaction."""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate transaction
        is_valid, error_msg = validate_transaction(data)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        # Make prediction
        result = predict_single(data)
        
        return jsonify(result), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch_endpoint():
    """Predict fraud for multiple transactions."""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if not isinstance(data, list):
            return jsonify({"error": "Data must be a list of transactions"}), 400
        
        if len(data) == 0:
            return jsonify({"error": "Empty transaction list"}), 400
        
        # Validate all transactions
        for i, transaction in enumerate(data):
            is_valid, error_msg = validate_transaction(transaction)
            if not is_valid:
                return jsonify({"error": f"Transaction {i}: {error_msg}"}), 400
        
        # Make predictions
        results = predict_batch(data)
        
        return jsonify({
            "count": len(results),
            "results": results
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Initialize model
    try:
        initialize_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        sys.exit(1)
    
    # Run Flask app
    logger.info(f"Starting Flask API on {API_CONFIG['host']}:{API_CONFIG['port']}")
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug'],
        threaded=API_CONFIG['threaded']
    )

