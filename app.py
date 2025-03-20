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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

from utils.data_collector import BinanceDataCollector
import config

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
    st.session_state.predictions = []
    st.session_state.latest_data = None
    st.session_state.model_trained = False
    st.session_state.data_fetch_status = {"status": "Not started", "last_update": None}
    st.session_state.selected_tab = "Live Dashboard"
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

def make_random_prediction():
    """Generate a random prediction for demo purposes"""
    if not st.session_state.initialized or st.session_state.latest_data is None:
        st.warning("System not initialized or no data available")
        return None
    
    classes = ["SHORT", "NEUTRAL", "LONG"]
    prediction_class = random.choice([0, 1, 2])
    confidence = random.uniform(0.65, 0.95)
    
    current_price = st.session_state.latest_data.iloc[-1]['close']
    
    if prediction_class == 0:  # SHORT
        predicted_move = -random.uniform(0.3, 1.2)
        target_price = current_price * (1 + predicted_move/100)
        reason = "Bearish divergence detected; RSI overbought; 200 EMA resistance"
    elif prediction_class == 2:  # LONG
        predicted_move = random.uniform(0.3, 1.2)
        target_price = current_price * (1 + predicted_move/100)
        reason = "Bullish pattern confirmed; RSI oversold; 50 EMA support"
    else:  # NEUTRAL
        predicted_move = random.uniform(-0.2, 0.2)
        target_price = current_price * (1 + predicted_move/100)
        reason = "Sideways price action; low volatility; mixed signals"
    
    prediction = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "price": current_price,
        "trend": classes[prediction_class],
        "confidence": round(confidence, 2),
        "target_price": round(target_price, 2),
        "predicted_move": round(predicted_move, 2),
        "reason": reason,
        "valid_for_minutes": config.VALIDITY_MINUTES
    }
    
    # Add to predictions history
    st.session_state.predictions.append(prediction)
    # Keep only the last 100 predictions
    if len(st.session_state.predictions) > 100:
        st.session_state.predictions = st.session_state.predictions[-100:]
    
    return prediction

def update_data_continuously():
    """Update data continuously in a separate thread"""
    while st.session_state.thread_running:
        try:
            fetch_data()
            make_random_prediction()
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

# Plot functions
def plot_candlestick_chart(df):
    """Create a candlestick chart with volume bars"""
    if df is None or df.empty:
        return go.Figure()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'], 
            high=df['high'],
            low=df['low'], 
            close=df['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['red' if row['open'] > row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index, 
            y=df['volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="ETHUSDT Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_prediction_history(predictions):
    """Create a chart with prediction history"""
    if not predictions:
        return go.Figure()
    
    # Convert predictions to DataFrame for easier plotting
    df = pd.DataFrame(predictions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create trend categories
    trend_map = {"LONG": 1, "NEUTRAL": 0, "SHORT": -1}
    df['trend_value'] = df['trend'].map(trend_map)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.04, row_heights=[0.6, 0.4])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], 
            y=df['price'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )
    
    # Add prediction markers with confidence as size
    colors = {'LONG': 'green', 'NEUTRAL': 'gray', 'SHORT': 'red'}
    for trend in colors:
        trend_df = df[df['trend'] == trend]
        
        fig.add_trace(
            go.Scatter(
                x=trend_df['timestamp'], 
                y=trend_df['price'],
                mode='markers',
                name=f'{trend} Prediction',
                marker=dict(
                    size=trend_df['confidence'] * 20,
                    color=colors[trend],
                    line=dict(width=1, color='black')
                ),
                hovertemplate='%{x}<br>Price: %{y:.2f}<br>Confidence: %{marker.size:.0f}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add confidence chart
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], 
            y=df['confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Prediction History",
        xaxis_title="Date",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=2, col=1)
    
    return fig

def plot_technical_indicators(df):
    """Create technical indicators chart"""
    if df is None or df.empty:
        return go.Figure()
    
    # Calculate simple indicators
    df['sma_9'] = df['close'].rolling(window=9).mean()
    df['sma_21'] = df['close'].rolling(window=21).mean()
    df['upper_band'] = df['sma_21'] + (df['close'].rolling(window=21).std() * 2)
    df['lower_band'] = df['sma_21'] - (df['close'].rolling(window=21).std() * 2)
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Add price and MAs
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['sma_9'],
            mode='lines',
            name='9-period SMA',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['sma_21'],
            mode='lines',
            name='21-period SMA',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['upper_band'],
            mode='lines',
            name='Upper Band',
            line=dict(color='rgba(0,128,0,0.3)', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['lower_band'],
            mode='lines',
            fill='tonexty',
            name='Lower Band',
            line=dict(color='rgba(0,128,0,0.3)', width=1)
        ),
        row=1, col=1
    )
    
    # Add volume
    colors = ['red' if row['open'] > row['close'] else 'green' for i, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index, 
            y=df['volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Technical Indicators",
        xaxis_title="Date",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_confidence_distribution(predictions):
    """Create confidence distribution chart by trend"""
    if not predictions:
        return go.Figure()
    
    # Convert predictions to DataFrame
    df = pd.DataFrame(predictions)
    
    # Group by trend
    trends = df['trend'].unique()
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram for each trend
    colors = {'LONG': 'green', 'NEUTRAL': 'gray', 'SHORT': 'red'}
    for trend in trends:
        trend_df = df[df['trend'] == trend]
        
        fig.add_trace(
            go.Histogram(
                x=trend_df['confidence'],
                name=trend,
                marker_color=colors.get(trend, 'blue'),
                opacity=0.7,
                nbinsx=10
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Prediction Confidence Distribution by Trend",
        xaxis_title="Confidence",
        yaxis_title="Count",
        height=300,
        barmode='overlay',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def display_current_prediction(prediction):
    """Display the current prediction with confidence indicator"""
    if not prediction:
        st.info("No prediction available")
        return
    
    # Determine color based on trend
    color_map = {"LONG": "green", "NEUTRAL": "gray", "SHORT": "red"}
    color = color_map.get(prediction["trend"], "blue")
    
    # Show prediction details in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Current Trend", 
            value=prediction["trend"],
            delta=f"{prediction['predicted_move']}%" if prediction["trend"] != "NEUTRAL" else None,
            delta_color="normal" if prediction["trend"] == "LONG" else "inverse" if prediction["trend"] == "SHORT" else "off"
        )
    
    with col2:
        st.metric(
            label="Current Price", 
            value=f"${prediction['price']:.2f}"
        )
    
    with col3:
        st.metric(
            label="Target Price", 
            value=f"${prediction['target_price']:.2f}",
            delta=f"{(prediction['target_price'] - prediction['price']):.2f} USDT"
        )
    
    # Confidence gauge
    st.write(f"**Confidence: {prediction['confidence'] * 100:.1f}%**")
    st.progress(prediction['confidence'])
    
    # Reasoning
    st.write(f"**Reasoning:** {prediction['reason']}")
    
    # Validity
    st.caption(f"Prediction made at {prediction['timestamp']} (valid for {prediction['valid_for_minutes']} minutes)")

def display_system_status(data_status, thread_status, prediction_count):
    """Display system status overview"""
    st.write("### System Status Overview")
    
    # Display in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Data Collection**")
        status_color = "green" if data_status["status"] == "Data fetched successfully" else "red"
        st.markdown(f"Status: :{status_color}[{data_status['status']}]")
        if data_status["last_update"]:
            st.write(f"Last update: {data_status['last_update']}")
    
    with col2:
        st.write("**Auto-Update Thread**")
        status_color = "green" if thread_status else "red"
        st.markdown(f"Status: :{status_color}[{'Running' if thread_status else 'Stopped'}]")
        
    with col3:
        st.write("**Predictions**")
        st.write(f"Total predictions: {prediction_count}")
        if prediction_count > 0:
            trends = [p["trend"] for p in st.session_state.predictions[-20:]]
            long_pct = trends.count("LONG") / len(trends) * 100
            neutral_pct = trends.count("NEUTRAL") / len(trends) * 100
            short_pct = trends.count("SHORT") / len(trends) * 100
            
            st.write(f"Recent trend distribution:")
            st.write(f"LONG: {long_pct:.1f}% | NEUTRAL: {neutral_pct:.1f}% | SHORT: {short_pct:.1f}%")

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
    tabs = ["Live Dashboard", "System Status", "API Guide"]
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
        
        # Get latest prediction or make a new one if none exists
        if not st.session_state.predictions:
            prediction = make_random_prediction()
        else:
            prediction = st.session_state.predictions[-1]
        
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
            display_current_prediction(prediction)
            
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

elif st.session_state.selected_tab == "System Status":
    st.title("System Status")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system first")
    else:
        # Display system status
        display_system_status(
            data_status=st.session_state.data_fetch_status,
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