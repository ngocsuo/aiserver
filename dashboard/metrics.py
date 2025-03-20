"""
Metric display functions for the Streamlit dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import config

def display_current_prediction(prediction):
    """
    Display the current prediction with confidence indicator.
    
    Args:
        prediction (dict): Prediction dictionary
    """
    # Get prediction details
    trend = prediction.get('trend', 'neutral').upper()
    confidence = prediction.get('confidence', 0.0)
    price = prediction.get('price', 0.0)
    valid_minutes = prediction.get('valid_for_minutes', 0)
    reason = prediction.get('reason', 'No reason provided')
    timestamp = prediction.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Determine color based on trend and confidence
    if trend == 'LONG':
        color = 'green'
        emoji = '🔼'
    elif trend == 'SHORT':
        color = 'red'
        emoji = '🔽'
    else:
        color = 'gray'
        emoji = '◀▶'
    
    # Create colored box for trend with large font
    st.markdown(
        f"""
        <div style="
            background-color: {color}; 
            padding: 10px; 
            border-radius: 5px; 
            text-align: center;
            color: white;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 10px;
        ">
            {emoji} {trend} {emoji}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Show confidence with progress bar
    st.metric("Confidence", f"{confidence:.2f}")
    st.progress(float(confidence))
    
    # Show predicted price and validity
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Price", f"${price:.2f}")
    
    with col2:
        st.metric("Valid For", f"{valid_minutes} min")
    
    # Show reasoning
    st.markdown("**Reasoning:**")
    st.info(reason)
    
    # Show timestamp
    st.caption(f"Generated at: {timestamp}")

def display_model_performance(training_metrics):
    """
    Display model performance metrics.
    
    Args:
        training_metrics (dict): Dictionary with training metrics
    """
    st.subheader("Model Performance")
    
    # Show models trained
    st.write(f"**Models in Ensemble:** {', '.join(training_metrics['models']).upper()}")
    
    # Display training time
    st.write(f"**Last Training:** {training_metrics['training_time']}")
    
    # Create a table with model accuracies
    accuracies = {}
    for model, results in training_metrics['evaluations'].items():
        accuracies[model] = results.get('accuracy', 0)
    
    # Sort models by accuracy
    sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    
    # Create columns for model metrics
    cols = st.columns(len(sorted_models))
    
    for i, (model, accuracy) in enumerate(sorted_models):
        with cols[i]:
            # Show model name and accuracy
            st.metric(
                f"{model.upper()}",
                f"{accuracy:.2f}",
                delta=None
            )
            
            # Add color indicator
            color = 'green' if accuracy >= 0.7 else 'orange' if accuracy >= 0.5 else 'red'
            st.markdown(
                f"""
                <div style="
                    background-color: {color}; 
                    height: 5px; 
                    border-radius: 2px; 
                    width: 100%;
                "></div>
                """, 
                unsafe_allow_html=True
            )

def display_data_stats(data):
    """
    Display statistics about the training data.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
    """
    st.subheader("Data Statistics")
    
    # Calculate basic stats
    data_points = len(data)
    date_range = (data.index.min(), data.index.max())
    time_period = (date_range[1] - date_range[0]).total_seconds() / 3600  # in hours
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Points", f"{data_points:,}")
    
    with col2:
        st.metric("Time Range", f"{time_period:.1f} hours")
    
    with col3:
        latest_price = data['close'].iloc[-1]
        price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
        price_change_pct = price_change / data['close'].iloc[-2] * 100
        st.metric("Latest Price", f"${latest_price:.2f}", f"{price_change_pct:.2f}%")
    
    # Calculate price statistics
    price_stats = {
        "Min Price": data['low'].min(),
        "Max Price": data['high'].max(),
        "Mean Price": data['close'].mean(),
        "Price Volatility": data['close'].pct_change().std() * 100  # in percentage
    }
    
    # Display price statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Min Price", f"${price_stats['Min Price']:.2f}")
    
    with col2:
        st.metric("Max Price", f"${price_stats['Max Price']:.2f}")
    
    with col3:
        st.metric("Mean Price", f"${price_stats['Mean Price']:.2f}")
    
    with col4:
        st.metric("Volatility", f"{price_stats['Price Volatility']:.2f}%")

def display_system_status(data_status, model_status, thread_status, prediction_count):
    """
    Display system status overview.
    
    Args:
        data_status (dict): Data fetch status
        model_status (bool): Whether models are trained
        thread_status (bool): Whether update thread is running
        prediction_count (int): Number of predictions made
    """
    st.subheader("System Overview")
    
    # System Status Indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_indicator = "🟢" if data_status.get('status') != "Error" else "🔴"
        st.markdown(f"### {data_indicator} Data Collection")
        st.write(f"Status: {data_status.get('status', 'Unknown')}")
        st.write(f"Last Update: {data_status.get('last_update', 'Never')}")
    
    with col2:
        model_indicator = "🟢" if model_status else "🔴"
        st.markdown(f"### {model_indicator} Model Status")
        st.write(f"Models Loaded: {'Yes' if model_status else 'No'}")
    
    with col3:
        thread_indicator = "🟢" if thread_status else "🔴"
        st.markdown(f"### {thread_indicator} Auto-Updates")
        st.write(f"Thread Running: {'Yes' if thread_status else 'No'}")
        st.write(f"Predictions Made: {prediction_count}")
    
    # System Resources and Configuration
    st.subheader("System Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Memory Usage:**")
        # This is a placeholder as actual memory usage monitoring would require additional libraries
        st.progress(0.5)  # Simulated 50% memory usage
        st.caption("Memory usage statistics would be shown here in a real deployment")
    
    with col2:
        st.markdown("**Processing Information:**")
        st.write(f"Symbol: {config.SYMBOL}")
        st.write(f"Primary Timeframe: {config.TIMEFRAMES['primary']}")
        st.write(f"Update Interval: {config.UPDATE_INTERVAL} seconds")
        st.write(f"Prediction Validity: {config.VALIDITY_MINUTES} minutes")
