"""
Main Streamlit application for ETHUSDT prediction dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import os
import json
from datetime import datetime, timedelta
import pytz

from utils.data_collector import BinanceDataCollector
from utils.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from prediction.prediction_engine import PredictionEngine
import config
from dashboard.charts import (
    plot_candlestick_chart, 
    plot_prediction_history, 
    plot_technical_indicators,
    plot_confidence_distribution,
    plot_model_accuracy
)
from dashboard.metrics import (
    display_current_prediction,
    display_model_performance,
    display_data_stats,
    display_system_status
)

# Set page config
st.set_page_config(
    page_title="ETHUSDT Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.data_collector = None
    st.session_state.prediction_engine = None
    st.session_state.predictions = []
    st.session_state.latest_data = None
    st.session_state.model_trained = False
    st.session_state.data_fetch_status = {"status": "Not started", "last_update": None}
    st.session_state.selected_tab = "Live Dashboard"
    st.session_state.training_metrics = None
    st.session_state.update_thread = None
    st.session_state.thread_running = False

def initialize_system():
    """Initialize the prediction system"""
    if st.session_state.initialized:
        return

    with st.spinner("Initializing ETHUSDT Prediction System..."):
        try:
            # Initialize data collector
            st.session_state.data_collector = BinanceDataCollector()
            
            # Initialize prediction engine
            st.session_state.prediction_engine = PredictionEngine()
            
            # Load models if available
            models = st.session_state.prediction_engine.load_models()
            if models:
                st.session_state.model_trained = True

            st.session_state.initialized = True
            
            # Update status
            st.session_state.data_fetch_status = {
                "status": "Initialized", 
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            st.error(f"Error initializing system: {e}")

def fetch_data():
    """Fetch the latest data from Binance"""
    if not st.session_state.initialized:
        st.warning("System not initialized yet")
        return None
        
    try:
        # Update data for all timeframes
        st.session_state.data_fetch_status = {
            "status": "Fetching data...",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        data = st.session_state.data_collector.update_data()
        
        st.session_state.latest_data = data.get(config.TIMEFRAMES["primary"])
        
        st.session_state.data_fetch_status = {
            "status": "Data fetched successfully",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return data
    except Exception as e:
        st.session_state.data_fetch_status = {
            "status": f"Error: {e}",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return None

def get_prediction():
    """Get the latest prediction"""
    if not st.session_state.initialized or not st.session_state.latest_data is not None:
        st.warning("System not initialized or no data available")
        return None
        
    if not st.session_state.model_trained:
        st.warning("Models not trained or loaded")
        return None
        
    try:
        # Get prediction
        prediction = st.session_state.prediction_engine.predict(
            st.session_state.latest_data,
            use_cache=True
        )
        
        # Add to predictions history
        if prediction and 'cached' not in prediction:
            st.session_state.predictions.append(prediction)
            # Keep only the last 100 predictions
            if len(st.session_state.predictions) > 100:
                st.session_state.predictions = st.session_state.predictions[-100:]
        
        return prediction
    except Exception as e:
        st.error(f"Error getting prediction: {e}")
        return None

def train_models():
    """Train all prediction models"""
    if not st.session_state.initialized or st.session_state.latest_data is None:
        st.warning("System not initialized or no data available")
        return
        
    try:
        # Process data for training
        with st.spinner("Processing data for training..."):
            dp = DataProcessor()
            processed_data = dp.process_data(st.session_state.latest_data)
            
            # Prepare sequence data and image data
            sequence_data = dp.prepare_sequence_data(processed_data)
            image_data = dp.prepare_cnn_data(processed_data)
            
            if sequence_data is None or image_data is None:
                st.error("Failed to prepare data for training")
                return
        
        # Train models
        with st.spinner("Training models... This may take a while"):
            trainer = ModelTrainer()
            models = trainer.train_all_models(sequence_data, image_data)
            
            if models:
                st.session_state.model_trained = True
                st.session_state.prediction_engine.models = models
                
                # Store training metrics
                st.session_state.training_metrics = {
                    'models': list(models.keys()),
                    'evaluations': trainer.evaluation_results,
                    'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.success("Models trained successfully!")
            else:
                st.error("Model training failed")
    except Exception as e:
        st.error(f"Error training models: {e}")

def update_data_continuously():
    """Update data continuously in a separate thread"""
    while st.session_state.thread_running:
        try:
            fetch_data()
            get_prediction()
            # Sleep for the update interval
            time.sleep(config.UPDATE_INTERVAL)
        except Exception as e:
            print(f"Error in update thread: {e}")
            time.sleep(60)  # Sleep longer on error

def start_update_thread():
    """Start the continuous update thread"""
    if not st.session_state.thread_running:
        st.session_state.thread_running = True
        thread = threading.Thread(target=update_data_continuously)
        thread.daemon = True  # Thread will close when main program exits
        thread.start()
        st.session_state.update_thread = thread
        st.success("Background data updates started")

def stop_update_thread():
    """Stop the continuous update thread"""
    if st.session_state.thread_running:
        st.session_state.thread_running = False
        st.session_state.update_thread = None
        st.info("Background data updates stopped")

# Sidebar
with st.sidebar:
    st.title("ETHUSDT Prediction System")
    st.write("AI-driven trading signal generator")
    
    # Initialize button
    if not st.session_state.initialized:
        if st.button("Initialize System"):
            initialize_system()
    else:
        st.success("System initialized")
    
    # Navigation
    st.subheader("Navigation")
    tabs = ["Live Dashboard", "Model Training", "System Status", "API Guide"]
    selected_tab = st.radio("Select View", tabs, index=tabs.index(st.session_state.selected_tab))
    st.session_state.selected_tab = selected_tab
    
    # Data controls
    if st.session_state.initialized:
        st.subheader("Data Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Fetch Data"):
                fetch_data()
                
        with col2:
            if not st.session_state.thread_running:
                if st.button("Auto Update"):
                    start_update_thread()
            else:
                if st.button("Stop Updates"):
                    stop_update_thread()
        
        # Show last update time
        if st.session_state.data_fetch_status["last_update"]:
            st.caption(f"Last update: {st.session_state.data_fetch_status['last_update']}")

# Main content
if st.session_state.selected_tab == "Live Dashboard":
    st.title("Live Trading Dashboard")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system first")
    else:
        # Initialize system if not done yet
        if st.session_state.latest_data is None:
            fetch_data()
        
        # Get latest prediction
        prediction = get_prediction()
        
        # Display prediction and chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Candlestick chart
            if st.session_state.latest_data is not None:
                st.subheader("ETHUSDT Price Chart (5m)")
                chart = plot_candlestick_chart(st.session_state.latest_data.iloc[-100:])
                st.plotly_chart(chart, use_container_width=True)
                
                # Technical indicators
                st.subheader("Technical Indicators")
                indicators_chart = plot_technical_indicators(st.session_state.latest_data.iloc[-100:])
                st.plotly_chart(indicators_chart, use_container_width=True)
        
        with col2:
            # Current prediction
            st.subheader("Current Prediction")
            if prediction:
                display_current_prediction(prediction)
            else:
                st.info("No prediction available")
            
            # Confidence distribution
            if st.session_state.predictions:
                st.subheader("Prediction Confidence")
                confidence_chart = plot_confidence_distribution(st.session_state.predictions[-20:])
                st.plotly_chart(confidence_chart, use_container_width=True)
        
        # Prediction history
        st.subheader("Prediction History")
        if st.session_state.predictions:
            history_chart = plot_prediction_history(st.session_state.predictions)
            st.plotly_chart(history_chart, use_container_width=True)
            
            # Show most recent predictions in a table
            with st.expander("Recent Predictions"):
                recent_preds = pd.DataFrame(st.session_state.predictions[-10:])
                recent_preds['timestamp'] = pd.to_datetime(recent_preds['timestamp'])
                recent_preds = recent_preds.sort_values('timestamp', ascending=False)
                st.dataframe(recent_preds, use_container_width=True)
        else:
            st.info("No prediction history available")

elif st.session_state.selected_tab == "Model Training":
    st.title("Model Training & Performance")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system first")
    else:
        # Data statistics
        if st.session_state.latest_data is not None:
            display_data_stats(st.session_state.latest_data)
        
        # Training controls
        st.subheader("Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Train All Models"):
                train_models()
        
        with col2:
            if st.session_state.model_trained:
                st.success("Models are trained and ready")
            else:
                st.warning("Models not trained yet")
        
        # Model performance metrics
        if st.session_state.model_trained and st.session_state.training_metrics:
            display_model_performance(st.session_state.training_metrics)
            
            # Model accuracy comparison
            st.subheader("Model Accuracy Comparison")
            accuracy_chart = plot_model_accuracy(st.session_state.training_metrics['evaluations'])
            st.plotly_chart(accuracy_chart, use_container_width=True)

elif st.session_state.selected_tab == "System Status":
    st.title("System Status")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system first")
    else:
        # Display system status
        display_system_status(
            data_status=st.session_state.data_fetch_status,
            model_status=st.session_state.model_trained,
            thread_status=st.session_state.thread_running,
            prediction_count=len(st.session_state.predictions)
        )
        
        # Data preview
        if st.session_state.latest_data is not None:
            with st.expander("Preview Latest Data"):
                st.dataframe(st.session_state.latest_data.tail(10), use_container_width=True)
        
        # Configuration parameters
        with st.expander("System Configuration"):
            # Display key configuration parameters
            st.json({
                "Symbol": config.SYMBOL,
                "Primary Timeframe": config.TIMEFRAMES["primary"],
                "Secondary Timeframes": config.TIMEFRAMES["secondary"],
                "Update Interval": f"{config.UPDATE_INTERVAL} seconds",
                "Prediction Window": f"{config.PREDICTION_WINDOW} candles",
                "Sequence Length": config.SEQUENCE_LENGTH,
                "Price Movement Threshold": f"{config.PRICE_MOVEMENT_THRESHOLD * 100}%",
                "Target PNL Threshold": f"${config.TARGET_PNL_THRESHOLD}",
                "Confidence Threshold": config.CONFIDENCE_THRESHOLD,
                "Prediction Validity": f"{config.VALIDITY_MINUTES} minutes"
            })

elif st.session_state.selected_tab == "API Guide":
    st.title("REST API Documentation")
    
    st.write("""
    ## API Endpoints
    
    The prediction system provides a REST API for integrating with trading bots or other applications.
    
    ### Main Endpoint
    
    **Prediction Endpoint:** `/predict`
    
    This endpoint returns the latest prediction for the specified symbol and interval.
    
    **Parameters:**
    - `symbol` (optional): Trading symbol (default: ETHUSDT)
    - `interval` (optional): Candle interval (default: 5m)
    
    **Example Request:**
    ```
    GET /predict?symbol=ETHUSDT&interval=5m
    ```
    
    **Example Response:**
    ```json
    {
      "trend": "long",
      "confidence": 0.87,
      "price": 3415.72,
      "valid_for_minutes": 15,
      "reason": "LSTM+Meta agree; RSI=32; Bollinger Lower touched; historical match score: 0.93",
      "timestamp": "2023-05-15 14:30:45"
    }
    ```
    
    ### Response Fields
    
    - `trend`: Predicted trend direction ("long", "short", or "neutral")
    - `confidence`: Confidence score between 0 and 1
    - `price`: Predicted future price target
    - `valid_for_minutes`: How long the prediction is valid for
    - `reason`: Explanation of the prediction
    - `timestamp`: When the prediction was generated
    
    ### Server Information
    
    The API server runs on port 8000 by default.
    
    ### Usage with curl
    
    ```bash
    curl "http://localhost:8000/predict?symbol=ETHUSDT&interval=5m"
    ```
    """)
    
    st.info("The API server must be started separately by running `python api.py`")

# Initialize on startup
if not st.session_state.initialized:
    initialize_system()
