"""
REST API for ETHUSDT prediction system.
"""
from flask import Flask, request, jsonify
import threading
import time
import logging

from utils.data_collector import BinanceDataCollector
from prediction.prediction_engine import PredictionEngine
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Initialize Flask app
app = Flask(__name__)

# Initialize components
data_collector = BinanceDataCollector()
prediction_engine = PredictionEngine()

# Global state
latest_data = {}
update_thread = None
thread_running = False

def update_data_continuously():
    """Update data continuously in a background thread"""
    global latest_data, thread_running
    
    logger.info("Starting continuous data update thread")
    
    while thread_running:
        try:
            # Update data for all timeframes
            latest_data = data_collector.update_data()
            
            logger.info(f"Data updated successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Sleep for the update interval
            time.sleep(config.UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            time.sleep(60)  # Sleep longer on error
    
    logger.info("Data update thread stopped")

def start_update_thread():
    """Start the continuous update thread"""
    global update_thread, thread_running
    
    if update_thread is None or not update_thread.is_alive():
        thread_running = True
        update_thread = threading.Thread(target=update_data_continuously)
        update_thread.daemon = True  # Thread will close when main program exits
        update_thread.start()
        logger.info("Background data update thread started")

def initialize_system():
    """Initialize the prediction system"""
    try:
        # Load models
        models = prediction_engine.load_models()
        
        if not models:
            logger.warning("No models loaded. Predictions may not be available.")
        else:
            logger.info(f"Loaded {len(models)} prediction models")
        
        # Start background data updates
        start_update_thread()
        
        logger.info("System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return False

@app.route('/predict', methods=['GET'])
def predict():
    """
    Prediction endpoint.
    
    Query parameters:
    - symbol: Trading symbol (default: ETHUSDT)
    - interval: Candle interval (default: 5m)
    
    Returns:
    - JSON with prediction data
    """
    try:
        # Get query parameters
        symbol = request.args.get('symbol', config.SYMBOL)
        interval = request.args.get('interval', config.TIMEFRAMES["primary"])
        
        # Validate parameters
        if symbol != config.SYMBOL:
            return jsonify({
                "error": f"Symbol not supported: {symbol}. Currently only {config.SYMBOL} is available."
            }), 400
            
        if interval != config.TIMEFRAMES["primary"] and interval not in config.TIMEFRAMES["secondary"]:
            return jsonify({
                "error": f"Interval not supported: {interval}. Supported intervals: {config.TIMEFRAMES['primary']} (primary), {', '.join(config.TIMEFRAMES['secondary'])} (secondary)"
            }), 400
        
        # Check if we have data
        if not latest_data or interval not in latest_data or latest_data[interval] is None:
            return jsonify({
                "error": "No data available. Please try again later."
            }), 503
            
        # Generate prediction
        prediction = prediction_engine.predict(latest_data[interval])
        
        if prediction is None:
            return jsonify({
                "error": "Failed to generate prediction. Models may not be trained."
            }), 500
            
        # Return prediction
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """
    Status endpoint.
    
    Returns:
    - JSON with system status
    """
    try:
        # Check if data is available
        data_available = bool(latest_data)
        
        # Check if models are loaded
        models_loaded = bool(prediction_engine.models)
        
        # Get timeframes with data
        available_timeframes = list(latest_data.keys()) if data_available else []
        
        # Return status
        return jsonify({
            "status": "running",
            "data_available": data_available,
            "models_loaded": models_loaded,
            "available_timeframes": available_timeframes,
            "symbol": config.SYMBOL,
            "primary_timeframe": config.TIMEFRAMES["primary"],
            "update_interval": config.UPDATE_INTERVAL,
            "thread_running": thread_running
        })
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Initialize the system
    logger.info("Initializing prediction system...")
    success = initialize_system()
    
    if success:
        logger.info("Starting API server...")
        # Run the Flask app
        app.run(host=config.API_HOST, port=config.API_PORT, debug=False)
    else:
        logger.error("Failed to initialize prediction system. Exiting.")
