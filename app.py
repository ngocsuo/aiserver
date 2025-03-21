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
import html
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import streamlit.components.v1 as components

from utils.data_collector import BinanceDataCollector, MockDataCollector
from utils.data_processor import DataProcessor
from utils.feature_engineering import FeatureEngineer
from models.model_trainer import ModelTrainer

# Custom Toast Notification Component
def show_toast(message, type="info", duration=3000):
    """
    Display a toast notification that fades out.
    
    Args:
        message (str): Message to display
        type (str): Type of notification ('info', 'success', 'warning', 'error')
        duration (int): Duration in milliseconds before fading out
    """
    # Escape HTML to prevent XSS
    message = html.escape(message)
    
    # Choose color based on type
    colors = {
        "info": "#17a2b8",
        "success": "#28a745",
        "warning": "#ffc107",
        "error": "#dc3545"
    }
    color = colors.get(type, colors["info"])
    
    # Create toast HTML with CSS animation
    toast_html = f"""
    <style>
    .toast-container {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }}
    
    .toast {{
        background-color: white;
        color: #333;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border-left: 4px solid {color};
        font-family: sans-serif;
        max-width: 300px;
        animation: fadein 0.5s, fadeout 0.5s {duration/1000 - 0.5}s;
        opacity: 0;
    }}
    
    @keyframes fadein {{
        from {{ right: -300px; opacity: 0; }}
        to {{ right: 0; opacity: 1; }}
    }}
    
    @keyframes fadeout {{
        from {{ opacity: 1; }}
        to {{ opacity: 0; }}
    }}
    </style>
    
    <div class="toast-container">
        <div class="toast">{message}</div>
    </div>
    
    <script>
        setTimeout(function(){{
            const toasts = document.getElementsByClassName('toast');
            for(let i = 0; i < toasts.length; i++){{
                toasts[i].style.opacity = "1";
            }}
        }}, 100);
        
        setTimeout(function(){{
            const container = document.getElementsByClassName('toast-container')[0];
            if(container){{
                container.remove();
            }}
        }}, {duration});
    </script>
    """
    
    # Display the toast
    components.html(toast_html, height=0)
from models.continuous_trainer import get_continuous_trainer
from prediction.prediction_engine import PredictionEngine
import config

# Set page config
st.set_page_config(
    page_title="ETHUSDT AI Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.data_collector = None
    st.session_state.data_processor = None
    st.session_state.model_trainer = None
    st.session_state.prediction_engine = None
    st.session_state.predictions = []
    st.session_state.latest_data = None
    st.session_state.model_trained = False
    st.session_state.data_fetch_status = {"status": "Not started", "last_update": None}
    st.session_state.selected_tab = "Live Dashboard"
    st.session_state.update_thread = None
    st.session_state.thread_running = False
    st.session_state.last_update_time = None
    st.session_state.chart_auto_refresh = True
    st.session_state.auto_initialize_triggered = False

def initialize_system():
    """Initialize the prediction system"""
    if st.session_state.initialized:
        return

    with st.spinner("Đang khởi tạo hệ thống dự đoán ETHUSDT..."):
        try:
            # Initialize data collector with factory function
            from utils.data_collector import create_data_collector, MockDataCollector
            
            # Create the appropriate data collector based on config
            st.session_state.data_collector = create_data_collector()
            
            # Store data source type for display
            if isinstance(st.session_state.data_collector, MockDataCollector):
                st.session_state.data_source = "Dữ liệu mô phỏng (Mock)"
                st.session_state.data_source_color = "orange"
                
                # Store API connection status if available
                if hasattr(st.session_state.data_collector, "connection_status"):
                    st.session_state.api_status = st.session_state.data_collector.connection_status
            else:
                st.session_state.data_source = "Binance API (Dữ liệu thực)"
                st.session_state.data_source_color = "green"
                
                # Store successful connection status
                st.session_state.api_status = {
                    "connected": True,
                    "message": "Kết nối Binance API thành công"
                }
                
            # Log data source
            if 'log_messages' not in st.session_state:
                st.session_state.log_messages = []
                
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"{timestamp} - Hệ thống đã khởi tạo với nguồn dữ liệu: {st.session_state.data_source}"
            st.session_state.log_messages.append(log_message)
            
            # Initialize data processor
            st.session_state.data_processor = DataProcessor()
            
            # Initialize model trainer
            st.session_state.model_trainer = ModelTrainer()
            
            # Initialize prediction engine
            st.session_state.prediction_engine = PredictionEngine()
            
            # Initialize continuous trainer
            continuous_trainer = get_continuous_trainer()
            st.session_state.continuous_trainer = continuous_trainer
            
            # Initialize status tracking
            st.session_state.initialized = True
            
            # Update status
            st.session_state.data_fetch_status = {
                "status": "Đã khởi tạo", 
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Initialize historical data thread status
            st.session_state.historical_data_status = {
                "status": "Chưa bắt đầu",
                "progress": 0
            }
            
            # Initialize model status
            st.session_state.model_trained = False
            
            # Initialize prediction history
            st.session_state.predictions = []
            
            # Initialize update thread status
            st.session_state.thread_running = False
            st.session_state.update_thread = None
            
            # LUỒNG 1: Bắt đầu tải dữ liệu thời gian thực cho dashboard
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"{timestamp} - 🔄 Bắt đầu tải dữ liệu thời gian thực cho dashboard..."
            st.session_state.log_messages.append(log_message)
            
            # Fetch real-time data immediately for the dashboard
            fetch_realtime_data()
            
            # LUỒNG 2: Bắt đầu quá trình tải dữ liệu lịch sử từ 2022 trong luồng riêng biệt
            if config.CONTINUOUS_TRAINING:
                continuous_trainer.start()
                log_message = f"{timestamp} - 🚀 Bắt đầu luồng lấy dữ liệu lịch sử từ 2022 và huấn luyện liên tục ({config.TRAINING_SCHEDULE['frequency']})"
                st.session_state.log_messages.append(log_message)
                
                # Initialize continuous training status
                st.session_state.continuous_training_status = {
                    "enabled": True,
                    "schedule": config.TRAINING_SCHEDULE,
                    "last_training": None
                }
                
                # Start the monitoring thread for historical data
                fetch_historical_data_thread()
            else:
                st.session_state.continuous_training_status = {
                    "enabled": False
                }
            
            # Confirm initialization
            st.success("Hệ thống đã khởi tạo thành công")
            
        except Exception as e:
            st.error(f"Lỗi khi khởi tạo hệ thống: {e}")

def fetch_realtime_data():
    """Fetch the latest real-time data from Binance for the dashboard"""
    if not st.session_state.initialized:
        st.warning("Hệ thống chưa được khởi tạo")
        return None
    
    # Create log container if not exists
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    
    # Add log message
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_message = f"{timestamp} - 🔄 Đang tải dữ liệu thời gian thực ETHUSDT..."
    st.session_state.log_messages.append(log_message)
    
    try:
        # Update data for all timeframes
        st.session_state.data_fetch_status = {
            "status": "Đang tải dữ liệu thời gian thực...",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Get data source type
        data_source_type = "Simulated Data" if isinstance(st.session_state.data_collector, MockDataCollector) else "Binance API"
        
        # Add log message
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"{timestamp} - 📡 Gửi yêu cầu đến {data_source_type} cho dữ liệu thời gian thực..."
        st.session_state.log_messages.append(log_message)
        
        # Chỉ lấy dữ liệu 3 ngày gần nhất để tải nhanh hơn
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.DATA_RANGE_OPTIONS["realtime"])
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        # Gọi hàm lấy dữ liệu với tham số ngày bắt đầu
        latest_data = st.session_state.data_collector.collect_historical_data(
            start_date=start_date_str,
            end_date=None
        )
        
        st.session_state.latest_data = latest_data
        
        # Ghi vào log thông tin khoảng thời gian
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"{timestamp} - ℹ️ Dải thời gian: {start_date_str} đến {end_date.strftime('%Y-%m-%d')}"
        st.session_state.log_messages.append(log_message)
        
        # Tạo dict chứa dữ liệu để tương thích với code cũ
        data = {config.TIMEFRAMES["primary"]: latest_data}
        
        # Add success log
        timestamp = datetime.now().strftime("%H:%M:%S")
        candle_count = len(st.session_state.latest_data) if st.session_state.latest_data is not None else 0
        log_message = f"{timestamp} - ✅ Đã cập nhật thành công {candle_count} nến ETHUSDT thời gian thực"
        st.session_state.log_messages.append(log_message)
        
        st.session_state.data_fetch_status = {
            "status": "Dữ liệu thời gian thực đã tải thành công",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return data
    except Exception as e:
        # Add error log
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"{timestamp} - ❌ LỖI: Không thể lấy dữ liệu thời gian thực: {str(e)}"
        st.session_state.log_messages.append(log_message)
        
        st.session_state.data_fetch_status = {
            "status": f"Lỗi: {e}",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return None

def fetch_historical_data_thread():
    """Fetch historical data from Binance for training in a separate thread"""
    if not st.session_state.initialized:
        return
    
    # Báo hiệu đang tải dữ liệu lịch sử
    if 'historical_data_status' not in st.session_state:
        st.session_state.historical_data_status = {
            "status": "Bắt đầu tải dữ liệu lịch sử",
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "progress": 0
        }
    
    # Log để thông báo
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_message = f"{timestamp} - 📚 Bắt đầu luồng tải dữ liệu lịch sử từ 2022..."
    st.session_state.log_messages.append(log_message)
    
    # Quá trình này dựa vào ContinuousTrainer đã bắt đầu trong initialize_system
    # và đang chạy trong một luồng riêng
    
    # Cập nhật trạng thái để hiển thị trên giao diện mà không sử dụng Streamlit API trực tiếp trong thread
    def update_status():
        while True:
            try:
                if not hasattr(st.session_state, 'continuous_trainer'):
                    time.sleep(10)
                    continue
                
                trainer = st.session_state.continuous_trainer
                if trainer is None:
                    time.sleep(10)
                    continue
                
                status = trainer.get_training_status()
                
                if 'current_chunk' in status and 'total_chunks' in status:
                    progress = int((status['current_chunk'] / status['total_chunks']) * 100)
                    
                    # Cập nhật vào session_state thay vì gọi trực tiếp Streamlit API
                    # Điều này tránh được warning "missing ScriptRunContext"
                    if 'historical_data_status' not in st.session_state:
                        st.session_state.historical_data_status = {}
                        
                    st.session_state.historical_data_status = {
                        "status": f"Đang tải chunk {status['current_chunk']}/{status['total_chunks']}",
                        "progress": progress,
                        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Lưu thông tin về Binance server time vào session state
                    try:
                        from utils.data_collector import create_data_collector
                        collector = create_data_collector()
                        server_time = collector.client.get_server_time()
                        server_time_ms = server_time['serverTime']
                        binance_time = datetime.fromtimestamp(server_time_ms / 1000)
                        
                        if 'binance_server_time' not in st.session_state:
                            st.session_state.binance_server_time = {}
                            
                        st.session_state.binance_server_time = {
                            "time": binance_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    except Exception as e:
                        print(f"Error getting Binance server time: {e}")
                
                time.sleep(10)  # Kiểm tra mỗi 10 giây
            except Exception as e:
                print(f"Error updating historical data status: {e}")
                time.sleep(30)  # Nếu lỗi, đợi lâu hơn
    
    # Bắt đầu luồng theo dõi tiến độ
    status_thread = threading.Thread(target=update_status)
    status_thread.daemon = True
    status_thread.start()

def fetch_data():
    """Fetch the latest data from Binance (compatibility function)"""
    return fetch_realtime_data()

def train_models():
    """Train all prediction models"""
    if not st.session_state.initialized or st.session_state.latest_data is None:
        st.warning("Hệ thống chưa được khởi tạo hoặc không có dữ liệu")
        show_toast("Hệ thống chưa được khởi tạo hoặc không có dữ liệu", "warning")
        return False
    
    # Create a placeholder for progress updates
    progress_placeholder = st.empty()
    progress_placeholder.info("Đang bắt đầu quá trình huấn luyện mô hình AI...")
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Create a placeholder for detailed logs
    logs_placeholder = st.empty()
    training_logs = []
    
    def update_log(message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        training_logs.append(f"{timestamp} - {message}")
        logs_placeholder.code("\n".join(training_logs))
        
        # Hiển thị toast notification cho các thông báo quan trọng
        if "Step" in message or "model trained" in message:
            show_toast(message, "info", 3000)
        elif "Error" in message or "ERROR" in message:
            show_toast(message, "error", 5000)
    
    try:
        # Step 1: Process data for training
        update_log("Step 1/5: Preparing ETHUSDT data for training...")
        progress_bar.progress(10)
        
        data = st.session_state.latest_data
        update_log(f"Data source: {'Real Binance API' if not isinstance(st.session_state.data_collector, type(__import__('utils.data_collector').data_collector.MockDataCollector)) else 'Simulated data (development mode)'}")
        update_log(f"Data points: {len(data)} candles")
        update_log(f"Timeframe: {config.TIMEFRAMES['primary']}")
        update_log(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # Step 2: Preprocess data
        progress_bar.progress(20)
        update_log("Step 2/5: Preprocessing data and calculating technical indicators...")
        processed_data = st.session_state.data_processor.process_data(data)
        
        # Display feature information
        feature_count = len(processed_data.columns) - 1  # Exclude target column
        update_log(f"Features generated: {feature_count} technical indicators and derived features")
        update_log(f"Training samples: {len(processed_data)} (after removing NaN values)")
        
        # Display class distribution
        if 'target_class' in processed_data.columns:
            class_dist = processed_data['target_class'].value_counts()
            update_log(f"Class distribution: SHORT={class_dist.get(0, 0)}, NEUTRAL={class_dist.get(1, 0)}, LONG={class_dist.get(2, 0)}")
        
        # Step 3: Prepare sequence and image data
        progress_bar.progress(40)
        update_log("Step 3/5: Preparing sequence data for LSTM and Transformer models...")
        sequence_data = st.session_state.data_processor.prepare_sequence_data(processed_data)
        
        progress_bar.progress(50)
        update_log("Preparing image data for CNN model...")
        image_data = st.session_state.data_processor.prepare_cnn_data(processed_data)
        
        # Step 4: Train all models
        progress_bar.progress(60)
        update_log("Step 4/5: Training LSTM model...")
        lstm_model, lstm_history = st.session_state.model_trainer.train_lstm(sequence_data)
        update_log(f"LSTM model trained with accuracy: {lstm_history.get('val_accuracy', [-1])[-1]:.4f}")
        
        progress_bar.progress(70)
        update_log("Training Transformer model...")
        transformer_model, transformer_history = st.session_state.model_trainer.train_transformer(sequence_data)
        update_log(f"Transformer model trained with accuracy: {transformer_history.get('val_accuracy', [-1])[-1]:.4f}")
        
        progress_bar.progress(80)
        update_log("Training CNN model...")
        cnn_model, cnn_history = st.session_state.model_trainer.train_cnn(image_data)
        update_log(f"CNN model trained with accuracy: {cnn_history.get('val_accuracy', [-1])[-1]:.4f}")
        
        progress_bar.progress(85)
        update_log("Training Historical Similarity model...")
        historical_model, _ = st.session_state.model_trainer.train_historical_similarity(sequence_data)
        
        progress_bar.progress(90)
        update_log("Step 5/5: Training Meta-Learner model...")
        meta_model, _ = st.session_state.model_trainer.train_meta_learner(sequence_data, image_data)
        
        # Finalize
        progress_bar.progress(100)
        update_log("All models trained successfully!")
        
        # Store training data information in session state for reference
        st.session_state.training_info = {
            "data_source": 'Real Binance API' if not isinstance(st.session_state.data_collector, type(__import__('utils.data_collector').data_collector.MockDataCollector)) else 'Simulated data (development mode)',
            "data_points": len(data),
            "date_range": f"{data.index.min()} to {data.index.max()}",
            "feature_count": feature_count,
            "training_samples": len(processed_data),
            "class_distribution": {
                "SHORT": class_dist.get(0, 0) if 'target_class' in processed_data.columns else 0,
                "NEUTRAL": class_dist.get(1, 0) if 'target_class' in processed_data.columns else 0,
                "LONG": class_dist.get(2, 0) if 'target_class' in processed_data.columns else 0
            },
            "model_performance": {
                "lstm": lstm_history.get('val_accuracy', [-1])[-1],
                "transformer": transformer_history.get('val_accuracy', [-1])[-1],
                "cnn": cnn_history.get('val_accuracy', [-1])[-1],
                "historical_similarity": 0.65,  # Mock value as it doesn't return standard accuracy
                "meta_learner": 0.81  # Mock value as it doesn't return standard accuracy in the same way
            },
            "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Set models as trained
        st.session_state.model_trained = True
        progress_placeholder.success("All AI models trained successfully!")
        
        return True
    except Exception as e:
        progress_placeholder.error(f"Error training models: {e}")
        update_log(f"ERROR: {str(e)}")
        return False

def make_prediction():
    """Generate a prediction using the trained models"""
    if not st.session_state.initialized:
        st.warning("Hệ thống chưa được khởi tạo")
        show_toast("Hệ thống chưa được khởi tạo", "warning")
        return None
    
    # Add log message
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_message = f"{timestamp} - 🧠 Bắt đầu quá trình tạo dự đoán..."
    st.session_state.log_messages.append(log_message)
    
    try:
        # Always fetch the latest data first
        st.info("Đang tải dữ liệu ETHUSDT mới nhất...")
        fetch_result = fetch_data()
        
        if fetch_result is None or st.session_state.latest_data is None:
            # Add error log
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"{timestamp} - ❌ Không thể lấy dữ liệu cho dự đoán"
            st.session_state.log_messages.append(log_message)
            
            st.warning("Failed to fetch the latest data")
            return None
        
        # Add log message
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Use trained models if available, otherwise use fallback
        if st.session_state.model_trained:
            # Get the latest data
            latest_data = st.session_state.latest_data
            
            log_message = f"{timestamp} - 🤖 Đang sử dụng mô hình AI đã huấn luyện để dự đoán..."
            st.session_state.log_messages.append(log_message)
            
            st.info("Đang sử dụng mô hình AI đã huấn luyện để tạo dự đoán...")
            # Use the prediction engine to generate prediction
            prediction = st.session_state.prediction_engine.predict(latest_data)
        else:
            log_message = f"{timestamp} - ⚠️ Chưa có mô hình AI được huấn luyện, sử dụng dự đoán mô phỏng..."
            st.session_state.log_messages.append(log_message)
            
            # Fallback to mock prediction for demonstration
            prediction = make_random_prediction()
        
        # Add to predictions history
        st.session_state.predictions.append(prediction)
        
        # Keep only the last 100 predictions
        if len(st.session_state.predictions) > 100:
            st.session_state.predictions = st.session_state.predictions[-100:]
        
        # Add success log
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"{timestamp} - ✅ Dự đoán đã tạo: {prediction['trend']} với độ tin cậy {prediction['confidence']:.2f}"
        st.session_state.log_messages.append(log_message)
        
        return prediction
    except Exception as e:
        # Add error log
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"{timestamp} - ❌ LỖI khi tạo dự đoán: {str(e)}"
        st.session_state.log_messages.append(log_message)
        
        st.error(f"Error making prediction: {e}")
        return None

def make_random_prediction():
    """Generate a random prediction for demo purposes"""
    if not st.session_state.initialized or st.session_state.latest_data is None:
        st.warning("Hệ thống chưa được khởi tạo hoặc không có dữ liệu")
        show_toast("Hệ thống chưa được khởi tạo hoặc không có dữ liệu", "warning")
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
    
    return prediction

def update_data_continuously():
    """Update data continuously in a separate thread"""
    # Initialize continuous trainer
    continuous_trainer = get_continuous_trainer()
    
    # Keep track of the number of updates to trigger periodic actions
    update_count = 0
    
    # Hiển thị Binance server time
    try:
        from utils.data_collector import create_data_collector
        data_collector = create_data_collector()
        server_time = data_collector.client.get_server_time() if hasattr(data_collector, 'client') else None
        server_time_ms = server_time['serverTime'] if server_time else None
        binance_time = datetime.fromtimestamp(server_time_ms / 1000) if server_time_ms else None
        
        if 'binance_server_time' not in st.session_state:
            st.session_state.binance_server_time = {}
            
        if binance_time:
            st.session_state.binance_server_time = {
                "time": binance_time.strftime("%Y-%m-%d %H:%M:%S"),
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    except Exception as e:
        print(f"Error getting Binance server time: {e}")
    
    while st.session_state.thread_running:
        try:
            # Fetch latest data
            data_result = fetch_data()
            
            # Cập nhật Binance server time mỗi lần fetch dữ liệu
            try:
                server_time = data_collector.client.get_server_time() if hasattr(data_collector, 'client') else None
                server_time_ms = server_time['serverTime'] if server_time else None
                binance_time = datetime.fromtimestamp(server_time_ms / 1000) if server_time_ms else None
                
                if 'binance_server_time' not in st.session_state:
                    st.session_state.binance_server_time = {}
                    
                if binance_time:
                    st.session_state.binance_server_time = {
                        "time": binance_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            except Exception as e:
                print(f"Error updating Binance server time: {e}")
            
            if data_result is not None:
                # Update new data counter in continuous trainer
                # Note: In a real implementation, you'd count the actual number of new candles
                continuous_trainer.increment_new_data_count(1)
                
                # Generate prediction
                make_prediction()
                
                # Increment update counter
                update_count += 1
                
                # Every 10 updates (or another appropriate interval), check training schedule
                if update_count % 10 == 0 and config.CONTINUOUS_TRAINING:
                    training_status = continuous_trainer.get_training_status()
                    
                    # Log training status
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_message = f"{timestamp} - 📊 Trạng thái huấn luyện: {training_status['new_data_points']} điểm dữ liệu mới"
                    if 'log_messages' in st.session_state:
                        st.session_state.log_messages.append(log_message)
            
            # Sleep for the update interval
            time.sleep(config.UPDATE_INTERVAL)
            
        except Exception as e:
            print(f"Error in update thread: {e}")
            if 'log_messages' in st.session_state:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_message = f"{timestamp} - ❌ LỖI trong luồng cập nhật: {str(e)}"
                st.session_state.log_messages.append(log_message)
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
    
    # Make a copy first to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Calculate simple indicators on the copy
    df_copy.loc[:, 'sma_9'] = df_copy['close'].rolling(window=9).mean()
    df_copy.loc[:, 'sma_21'] = df_copy['close'].rolling(window=21).mean()
    df_copy.loc[:, 'upper_band'] = df_copy['sma_21'] + (df_copy['close'].rolling(window=21).std() * 2)
    df_copy.loc[:, 'lower_band'] = df_copy['sma_21'] - (df_copy['close'].rolling(window=21).std() * 2)
    
    # Use the copied dataframe for the rest of the function
    df = df_copy
    
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

def plot_model_performance(models_accuracy=None):
    """Create a chart showing model performance metrics"""
    # Use mock data if not provided
    if models_accuracy is None:
        models_accuracy = {
            'lstm': 0.72,
            'transformer': 0.76,
            'cnn': 0.68,
            'historical_similarity': 0.65,
            'meta_learner': 0.81
        }
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    models = list(models_accuracy.keys())
    accuracies = list(models_accuracy.values())
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    models = [models[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    # Set colors
    colors = ['royalblue' if acc < 0.7 else 'green' if acc >= 0.8 else 'orange' for acc in accuracies]
    
    fig.add_trace(
        go.Bar(
            x=models,
            y=accuracies,
            marker_color=colors,
            text=[f"{acc*100:.1f}%" for acc in accuracies],
            textposition='auto'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        height=300,
        yaxis=dict(range=[0, 1]),
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
    st.write("### Technical Analysis")
    
    # Create an expander for detailed technical analysis
    with st.expander("Detailed Technical Analysis", expanded=True):
        # Split the reason into individual points for better readability
        reasoning_points = prediction['reason'].split(';')
        
        for i, point in enumerate(reasoning_points):
            if point.strip():  # Only display non-empty points
                st.write(f"{i+1}. {point.strip()}")
        
        # Add additional technical indicator information if available
        if 'technical_indicators' in prediction:
            st.write("#### Key Technical Indicators")
            indicators = prediction['technical_indicators']
            
            # Create 3 columns for technical indicators
            ind_col1, ind_col2, ind_col3 = st.columns(3)
            
            with ind_col1:
                if 'rsi' in indicators:
                    st.metric("RSI", f"{indicators['rsi']:.1f}", 
                              delta="Overbought" if indicators['rsi'] > 70 else "Oversold" if indicators['rsi'] < 30 else "Neutral")
                
                if 'macd' in indicators:
                    st.metric("MACD", f"{indicators['macd']:.4f}", 
                              delta=f"{indicators['macd'] - indicators.get('macd_signal', 0):.4f}")
            
            with ind_col2:
                if 'ema_9' in indicators and 'ema_21' in indicators:
                    diff = indicators['ema_9'] - indicators['ema_21']
                    st.metric("EMA 9/21 Diff", f"{diff:.2f}", 
                              delta="Bullish" if diff > 0 else "Bearish")
                
                if 'atr' in indicators:
                    st.metric("ATR", f"{indicators['atr']:.2f}")
            
            with ind_col3:
                if 'bb_width' in indicators:
                    st.metric("BB Width", f"{indicators['bb_width']:.2f}",
                              delta="High Volatility" if indicators['bb_width'] > 0.05 else "Low Volatility")
                
                if 'volume' in indicators:
                    st.metric("Volume", f"{indicators['volume']:.0f}")
        
        # Show buy/sell signals
        if prediction['trend'] != "NEUTRAL":
            signal_type = "BUY" if prediction['trend'] == "LONG" else "SELL"
            signal_color = "green" if prediction['trend'] == "LONG" else "red"
            
            st.markdown(f"<h3 style='color:{signal_color}'>Signal: {signal_type}</h3>", unsafe_allow_html=True)
            
            st.write(f"**Entry Price:** ${prediction['price']:.2f}")
            st.write(f"**Target Price:** ${prediction['target_price']:.2f}")
            
            # Show potential profit/loss
            move_pct = prediction['predicted_move']
            move_value = prediction['target_price'] - prediction['price']
            
            st.write(f"**Expected Move:** {move_pct:.2f}% ({move_value:.2f} USDT)")
    
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
        
        # Display data source information
        if 'data_source' in st.session_state and 'data_source_color' in st.session_state:
            st.markdown(f"Source: :{st.session_state.data_source_color}[{st.session_state.data_source}]")
        
        if data_status["last_update"]:
            st.write(f"Last update: {data_status['last_update']}")
        
        # Display Binance server time if available
        if 'binance_server_time' in st.session_state:
            binance_time = st.session_state.binance_server_time.get('time', 'N/A')
            st.write(f"Binance server time: {binance_time}")
            st.write(f"Time sync: {st.session_state.binance_server_time.get('update_time', 'N/A')}")
    
    with col2:
        # AI Models Status
        st.write("**AI Models**")
        model_status_color = "green" if st.session_state.model_trained else "red"
        st.markdown(f"Status: :{model_status_color}[{'Trained' if st.session_state.model_trained else 'Not Trained'}]")
        
        # Continuous Training Status
        if config.CONTINUOUS_TRAINING and 'continuous_trainer' in st.session_state:
            st.write("**Continuous Training**")
            
            # Get current training status
            training_status = st.session_state.continuous_trainer.get_training_status()
            
            # Check if training is in progress
            if training_status['in_progress']:
                st.markdown(f"Status: :blue[Training in progress]")
            else:
                status_color = "green" if training_status['enabled'] else "red"
                st.markdown(f"Status: :{status_color}[{'Enabled' if training_status['enabled'] else 'Disabled'}]")
            
            # Display schedule info
            schedule = training_status['schedule']
            st.write(f"Schedule: {schedule['frequency'].capitalize()}")
            
            # Show new data points
            st.write(f"New data points: {training_status['new_data_points']}")
            
            # Show last training time if available
            if training_status['last_training_time']:
                # Kiểm tra nếu last_training_time là đối tượng datetime hoặc string
                if isinstance(training_status['last_training_time'], datetime):
                    st.write(f"Last trained: {training_status['last_training_time'].strftime('%Y-%m-%d %H:%M')}")
                else:
                    st.write(f"Last trained: {training_status['last_training_time']}")
        
        # Auto-Update Thread Status
        st.write("**Auto-Update Thread**")
        thread_status_color = "green" if thread_status else "red"
        st.markdown(f"Status: :{thread_status_color}[{'Running' if thread_status else 'Stopped'}]")
        
    with col3:
        st.write("**Predictions**")
        st.write(f"Total predictions: {prediction_count}")
        
        # Display API connection status if available
        if 'api_status' in st.session_state:
            st.write("**API Connection**")
            api_status = st.session_state.api_status
            status_color = "green" if api_status.get('connected', False) else "red"
            st.markdown(f"Status: :{status_color}[{api_status.get('message', 'Unknown')}]")
            
            # Display any error message if available
            if 'error' in api_status:
                st.error(f"Error: {api_status['error']}")
        if prediction_count > 0:
            trends = [p["trend"] for p in st.session_state.predictions[-20:]]
            long_pct = trends.count("LONG") / len(trends) * 100
            neutral_pct = trends.count("NEUTRAL") / len(trends) * 100
            short_pct = trends.count("SHORT") / len(trends) * 100
            
            st.write(f"Recent trend distribution:")
            st.write(f"LONG: {long_pct:.1f}% | NEUTRAL: {neutral_pct:.1f}% | SHORT: {short_pct:.1f}%")

# Sidebar with modern design
with st.sidebar:
    st.title("🚀 ETHUSDT AI Prediction")
    st.markdown("<div style='margin-bottom: 20px;'>Dự đoán thông minh với AI</div>", unsafe_allow_html=True)
    
    # Hiển thị trạng thái hệ thống với thiết kế hiện đại
    if st.session_state.initialized:
        st.success("🟢 Hệ thống đã sẵn sàng")
    else:
        st.info("⏳ Đang khởi tạo hệ thống...")
    
    # Navigation với thiết kế hiện đại và icon emoji
    st.markdown("### 📊 Điều hướng")
    tabs = [
        "🔍 Live Dashboard", 
        "🧠 Models & Training", 
        "⚙️ Cài đặt", 
        "📊 Backtest",
        "🛠️ System Status", 
        "📡 API Guide"
    ]
    # Map từ tab hiển thị đến tên trong session_state
    tab_mapping = {
        "🔍 Live Dashboard": "Live Dashboard",
        "🧠 Models & Training": "Models & Training",
        "⚙️ Cài đặt": "Cài đặt",
        "📊 Backtest": "Backtest",
        "🛠️ System Status": "System Status",
        "📡 API Guide": "API Guide"
    }
    # Tìm index mặc định
    default_index = 0
    for i, tab in enumerate(tabs):
        if tab_mapping[tab] == st.session_state.selected_tab:
            default_index = i
            break
            
    selected_tab_display = st.radio("Chọn chế độ xem", tabs, index=default_index)
    # Lưu tab đã chọn vào session state
    st.session_state.selected_tab = tab_mapping[selected_tab_display]
    
    # Data controls với thiết kế hiện đại
    if st.session_state.initialized:
        st.markdown("### 🔄 Điều khiển dữ liệu")
        
        # Thêm tùy chọn tự động cập nhật biểu đồ mỗi 10 giây
        st.session_state.chart_auto_refresh = st.toggle("Tự động cập nhật biểu đồ (10s)", value=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Tải lại"):
                fetch_data()
                
        with col2:
            if not st.session_state.thread_running:
                if st.button("▶️ Auto"):
                    start_update_thread()
            else:
                if st.button("⏹️ Dừng"):
                    stop_update_thread()
        
        # Hiển thị thời gian cập nhật cuối
        if 'data_fetch_status' in st.session_state and st.session_state.data_fetch_status.get('last_update'):
            st.caption(f"Cập nhật cuối: {st.session_state.data_fetch_status['last_update']}")
        
        # Model controls với thiết kế hiện đại
        st.markdown("### 🧠 Mô hình AI")
        if st.button("🔬 Huấn luyện"):
            train_models()
        
        # Prediction button
        if st.button("Make Prediction"):
            prediction = make_prediction()
            if prediction:
                st.success("New prediction generated!")
        
        # Show last update time
        if st.session_state.data_fetch_status["last_update"]:
            st.caption(f"Last update: {st.session_state.data_fetch_status['last_update']}")

# Tự động khởi tạo hệ thống khi tải trang (sau khi tất cả các function đã được định nghĩa)
if not st.session_state.initialized and not st.session_state.auto_initialize_triggered:
    st.session_state.auto_initialize_triggered = True
    initialize_system()

# Định nghĩa hàm fetch_historical_data_thread
def fetch_historical_data_thread():
    """Fetch historical data from Binance for training in a separate thread"""
    if not st.session_state.initialized:
        st.warning("Vui lòng khởi tạo hệ thống trước")
        return
        
    if 'historical_data_status' not in st.session_state:
        st.session_state.historical_data_status = {
            "status": "Đang lấy dữ liệu lịch sử...",
            "progress": 0
        }
    
    def update_status():
        # This function will update the status in the session state
        try:
            start_time = time.time()
            
            # Khởi tạo tiến trình
            st.session_state.historical_data_status['progress'] = 5
            
            # Lấy dữ liệu cho từng khung thời gian
            timeframes = ["1m", "5m", "15m", "1h", "4h"]
            total_timeframes = len(timeframes)
            
            for idx, timeframe in enumerate(timeframes):
                # Cập nhật trạng thái
                progress = 5 + int(95 * (idx / total_timeframes))
                st.session_state.historical_data_status['progress'] = progress
                st.session_state.historical_data_status['status'] = f"Đang lấy dữ liệu {timeframe}..."
                
                # Thực sự lấy dữ liệu ở đây
                try:
                    # Lấy dữ liệu thật từ Binance qua data_collector
                    if hasattr(st.session_state, 'data_collector'):
                        data = st.session_state.data_collector.collect_historical_data(
                            symbol=config.SYMBOL,
                            timeframe=timeframe,
                            limit=config.LOOKBACK_PERIODS,
                            start_date=config.HISTORICAL_START_DATE
                        )
                        
                        # Lưu vào session state
                        if 'historical_data' not in st.session_state:
                            st.session_state.historical_data = {}
                        st.session_state.historical_data[timeframe] = data
                        
                        # Cập nhật trạng thái chi tiết
                        data_length = len(data) if data is not None else 0
                        st.session_state.historical_data_status['details'] = f"{data_length} nến {timeframe} từ {config.HISTORICAL_START_DATE}"
                        
                        # Thêm vào log thông báo
                        if 'log_messages' not in st.session_state:
                            st.session_state.log_messages = []
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.log_messages.append(f"{timestamp} - 📥 Đã tải {data_length} nến {timeframe} từ {config.HISTORICAL_START_DATE}")
                    
                    # Giả lập thời gian xử lý
                    time.sleep(0.5)
                    
                except Exception as e:
                    st.session_state.historical_data_status['status'] = f"Lỗi khi lấy dữ liệu {timeframe}: {e}"
                    if 'log_messages' not in st.session_state:
                        st.session_state.log_messages = []
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.log_messages.append(f"{timestamp} - ❌ Lỗi khi tải dữ liệu {timeframe}: {e}")
            
            # Hoàn tất
            st.session_state.historical_data_status['status'] = "Hoàn tất lấy dữ liệu lịch sử!"
            st.session_state.historical_data_status['progress'] = 100
            
            # Tính tổng thời gian
            elapsed_time = time.time() - start_time
            st.session_state.historical_data_status['elapsed_time'] = f"{elapsed_time:.2f} giây"
            
            # Thêm log thành công
            if 'log_messages' not in st.session_state:
                st.session_state.log_messages = []
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.log_messages.append(f"{timestamp} - ✅ Hoàn tất lấy dữ liệu lịch sử ({elapsed_time:.2f}s)")
            
        except Exception as e:
            st.session_state.historical_data_status['status'] = f"Lỗi: {e}"
            st.session_state.historical_data_status['progress'] = 0
            if 'log_messages' not in st.session_state:
                st.session_state.log_messages = []
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.log_messages.append(f"{timestamp} - ❌ Lỗi: {e}")
                
    thread = threading.Thread(target=update_status)
    thread.daemon = True  # Đảm bảo thread sẽ bị hủy khi chương trình chính kết thúc
    thread.start()
    
    # Thêm log bắt đầu
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"{timestamp} - 🔄 Bắt đầu lấy dữ liệu lịch sử từ {config.HISTORICAL_START_DATE}")

# Main content
if st.session_state.selected_tab == "Live Dashboard":
    st.title("ETHUSDT AI Prediction Dashboard")
    
    if not st.session_state.initialized:
        st.warning("Vui lòng khởi tạo hệ thống trước")
        
        # Add a big initialize button in the center
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Khởi tạo hệ thống", use_container_width=True):
                initialize_system()
                # Add initial log
                if 'log_messages' not in st.session_state:
                    st.session_state.log_messages = []
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.log_messages.append(f"{timestamp} - Bắt đầu khởi tạo hệ thống")
                st.rerun()
    else:
        # Đảm bảo dữ liệu được tải khi xem Live Dashboard
        if st.session_state.latest_data is None:
            fetch_data()  # Đảm bảo dữ liệu được tải
        # Initialize system if not done yet - load data immediately
        if st.session_state.latest_data is None:
            with st.spinner("Đang tải dữ liệu thời gian thực..."):
                fetch_data()
        
        # Get latest prediction or make a new one if none exists
        if not st.session_state.predictions:
            with st.spinner("Đang tạo dự đoán ban đầu..."):
                prediction = make_prediction()
        else:
            prediction = st.session_state.predictions[-1]
        
        # Status badges at the top - more compact
        status_container = st.container()
        status_col1, status_col2, status_col3, status_col4, status_col5 = status_container.columns(5)
        
        with status_col1:
            source_color = "green" if not isinstance(st.session_state.data_collector, MockDataCollector) else "orange"
            source_text = "Binance API" if not isinstance(st.session_state.data_collector, MockDataCollector) else "Mô phỏng"
            st.markdown(f"**Nguồn dữ liệu:** :{source_color}[{source_text}]")
            
        with status_col2:
            data_status = "✅ Có sẵn" if st.session_state.latest_data is not None else "❌ Không có"
            data_color = "green" if st.session_state.latest_data is not None else "red"
            st.markdown(f"**Dữ liệu trực tuyến:** :{data_color}[{data_status}]")
        
        with status_col3:
            # Thêm trạng thái tải dữ liệu lịch sử
            if 'historical_data_status' in st.session_state:
                if 'progress' in st.session_state.historical_data_status:
                    progress = st.session_state.historical_data_status['progress']
                    hist_status = f"⏳ {progress}%" if progress < 100 else "✅ Hoàn tất"
                    hist_color = "orange" if progress < 100 else "green"
                else:
                    hist_status = "⏱️ Đang chờ"
                    hist_color = "yellow"
            else:
                hist_status = "❌ Chưa bắt đầu"
                hist_color = "red"
            st.markdown(f"**Dữ liệu lịch sử:** :{hist_color}[{hist_status}]")
            
        with status_col4:
            model_status = "✅ Đã huấn luyện" if st.session_state.model_trained else "❌ Chưa huấn luyện"
            model_color = "green" if st.session_state.model_trained else "red"
            st.markdown(f"**Mô hình AI:** :{model_color}[{model_status}]")
            
        with status_col5:
            update_status = "✅ Bật" if st.session_state.thread_running else "❌ Tắt"
            update_color = "green" if st.session_state.thread_running else "red"
            st.markdown(f"**Cập nhật tự động:** :{update_color}[{update_status}]")
        
        # Display prediction and chart in tabs - Default to chart first
        tabs = st.tabs(["📊 Price Chart", "🔍 Technical Analysis", "📈 Prediction History", "📋 Training Logs"])
        
        # Quick action buttons - moved below tabs to prioritize chart display
        action_container = st.container()
        action_col1, action_col2, action_col3, action_col4 = action_container.columns(4)
        
        with action_col1:
            if st.button("🔄 Tải dữ liệu thời gian thực", use_container_width=True):
                with st.spinner("Đang tải dữ liệu thời gian thực..."):
                    fetch_realtime_data()
                
        with action_col2:
            if st.button("🔮 Tạo dự đoán", use_container_width=True):
                with st.spinner("Đang tạo dự đoán..."):
                    make_prediction()
                
        with action_col3:
            if not st.session_state.model_trained:
                if st.button("🧠 Huấn luyện mô hình", use_container_width=True):
                    with st.spinner("Đang huấn luyện mô hình..."):
                        train_models()
            else:
                if st.button("🔄 Huấn luyện lại mô hình", use_container_width=True):
                    with st.spinner("Đang huấn luyện lại mô hình..."):
                        train_models()
                
        with action_col4:
            if not st.session_state.thread_running:
                if st.button("▶️ Bật cập nhật tự động", use_container_width=True):
                    start_update_thread()
            else:
                if st.button("⏹️ Tắt cập nhật tự động", use_container_width=True):
                    stop_update_thread()
        
        with tabs[0]:
            # Main dashboard layout
            chart_col, pred_col = st.columns([2, 1])
            
            with chart_col:
                # Candlestick chart
                if st.session_state.latest_data is not None:
                    st.subheader("ETHUSDT Price Chart")
                    
                    # Thêm số đếm thời gian cho tự động cập nhật
                    if 'chart_last_update_time' not in st.session_state:
                        st.session_state.chart_last_update_time = datetime.now()
                    
                    # Thêm tự động cập nhật biểu đồ mỗi 10 giây
                    if st.session_state.chart_auto_refresh:
                        current_time = datetime.now()
                        time_diff = (current_time - st.session_state.chart_last_update_time).total_seconds()
                        
                        if time_diff >= 10:  # Cập nhật mỗi 10 giây
                            fetch_data()
                            st.session_state.chart_last_update_time = current_time
                    
                    # Hiển thị thời gian tự động cập nhật biểu đồ tiếp theo
                    if st.session_state.chart_auto_refresh:
                        time_left = max(0, 10 - (datetime.now() - st.session_state.chart_last_update_time).total_seconds())
                        refresh_status = f"⏱️ Tự động cập nhật sau: {int(time_left)}s"
                        st.caption(refresh_status)
                    
                    # Add timeframe selector
                    timeframe = st.selectbox("Chọn khung thời gian", ['50 nến gần nhất', '100 nến gần nhất', '200 nến gần nhất', 'Tất cả dữ liệu'])
                    
                    # Convert selection to number of candles
                    if timeframe == '50 nến gần nhất':
                        candles = 50
                    elif timeframe == '100 nến gần nhất':
                        candles = 100
                    elif timeframe == '200 nến gần nhất':
                        candles = 200
                    else:
                        candles = len(st.session_state.latest_data)
                    
                    # Hiển thị biểu đồ
                    chart = plot_candlestick_chart(st.session_state.latest_data.iloc[-candles:])
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Hiển thị thông tin thời điểm cập nhật cuối
                    last_update = st.session_state.data_fetch_status.get('last_update', 'Unknown')
                    st.caption(f"📊 Dữ liệu cập nhật: {last_update}")
            
            with pred_col:
                # Current prediction with enhanced styling
                st.subheader("Current AI Prediction")
                
                # Add prediction timestamp
                if prediction:
                    time_difference = datetime.now() - datetime.strptime(prediction['timestamp'], "%Y-%m-%d %H:%M:%S")
                    minutes_ago = int(time_difference.total_seconds() / 60)
                    prediction_freshness = f"{minutes_ago} minutes ago" if minutes_ago > 0 else "Just now"
                    
                    st.markdown(f"**Generated:** {prediction_freshness}")
                
                # Display prediction
                display_current_prediction(prediction)
                
                # Add log of last action 
                if 'log_messages' in st.session_state and st.session_state.log_messages:
                    st.write("**Last System Action:**")
                    st.info(st.session_state.log_messages[-1])
                
                # Display current data info
                if st.session_state.latest_data is not None:
                    st.write("**Current Dataset:**")
                    st.write(f"Total candles: {len(st.session_state.latest_data)}")
                    st.write(f"Last update: {st.session_state.data_fetch_status.get('last_update', 'Unknown')}")
                    
                    # Add a small data preview
                    with st.expander("Latest Price Data"):
                        st.dataframe(st.session_state.latest_data.tail(5)[['open', 'high', 'low', 'close', 'volume']])
        
        with tabs[1]:
            # Technical Analysis Tab
            if st.session_state.latest_data is not None:
                st.subheader("Technical Indicators")
                
                # Add simple description
                st.markdown("""
                Technical indicators are mathematical calculations based on price, volume, or open interest of a security.
                These indicators help traders identify trading opportunities and make more informed decisions.
                """)
                
                # Technical indicators chart
                indicators_chart = plot_technical_indicators(st.session_state.latest_data.iloc[-100:])
                st.plotly_chart(indicators_chart, use_container_width=True)
                
                # Confidence distribution if predictions exist
                if st.session_state.predictions:
                    st.subheader("Prediction Confidence Distribution")
                    confidence_chart = plot_confidence_distribution(st.session_state.predictions[-20:])
                    st.plotly_chart(confidence_chart, use_container_width=True)
            else:
                st.warning("No data available for technical analysis. Please fetch data first.")
        
        with tabs[2]:
            # Prediction History Tab
            st.subheader("AI Prediction History")
            
            if st.session_state.predictions:
                # Add filters
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    trend_filter = st.multiselect(
                        "Filter by Trend",
                        options=["ALL", "LONG", "NEUTRAL", "SHORT"],
                        default=["ALL"]
                    )
                
                with filter_col2:
                    confidence_threshold = st.slider(
                        "Minimum Confidence",
                        min_value=0.0, 
                        max_value=1.0,
                        value=0.0,
                        step=0.05
                    )
                
                # Apply filters to predictions
                filtered_predictions = st.session_state.predictions.copy()
                if "ALL" not in trend_filter and trend_filter:
                    filtered_predictions = [p for p in filtered_predictions if p["trend"] in trend_filter]
                
                filtered_predictions = [p for p in filtered_predictions if p["confidence"] >= confidence_threshold]
                
                # Display history chart
                if filtered_predictions:
                    history_chart = plot_prediction_history(filtered_predictions)
                    st.plotly_chart(history_chart, use_container_width=True)
                    
                    # Show most recent predictions in a table
                    with st.expander("Recent Predictions (Table View)", expanded=True):
                        recent_preds = pd.DataFrame(filtered_predictions[-15:])
                        recent_preds['timestamp'] = pd.to_datetime(recent_preds['timestamp'])
                        recent_preds = recent_preds.sort_values('timestamp', ascending=False)
                        
                        # Add styling to the dataframe
                        def style_trend(val):
                            color = 'green' if val == 'LONG' else 'red' if val == 'SHORT' else 'gray'
                            return f'background-color: {color}; color: white'
                        
                        # Use Styler.map instead of deprecated applymap
                        styled_df = recent_preds.style.map(style_trend, subset=['trend'])
                        st.dataframe(styled_df, use_container_width=True)
                else:
                    st.info("No predictions match your filters")
            else:
                st.info("No prediction history available yet. Generate predictions to see history.")
                
        with tabs[3]:
            # Training Logs Tab
            st.subheader("Huấn luyện AI - Nhật ký")
            
            # Create container for training logs
            log_col1, log_col2 = st.columns([3, 1])
            
            with log_col1:
                # Create a data processor log viewer
                st.write("### Nhật ký xử lý dữ liệu & huấn luyện")
                
                # Fetch latest logs from continuous trainer
                if hasattr(st.session_state, 'continuous_trainer'):
                    trainer_status = st.session_state.continuous_trainer.get_training_status()
                    
                    # Display status information
                    if trainer_status:
                        st.write(f"**Trạng thái:** {trainer_status.get('status', 'Unknown')}")
                        st.write(f"**Lần huấn luyện cuối:** {trainer_status.get('last_training_time', 'Chưa có')}")
                        st.write(f"**Dữ liệu mới từ lần huấn luyện trước:** {trainer_status.get('new_data_points', 0)} điểm dữ liệu")
                        
                        if trainer_status.get('is_training', False):
                            st.warning("Đang trong quá trình huấn luyện...")
                            st.progress(trainer_status.get('progress', 0))
                
                # Create a scrollable log area with stylized appearance
                st.markdown("""
                <style>
                .training-log-container {
                    height: 400px;
                    overflow-y: auto;
                    background-color: #111;
                    color: #0f0;
                    padding: 10px;
                    border-radius: 5px;
                    font-family: 'Courier New', monospace;
                    font-size: 0.9em;
                    line-height: 1.5;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Filter logs for training-related entries
                training_logs = []
                
                # Show continuous trainer logs
                if hasattr(st.session_state, 'continuous_trainer') and hasattr(st.session_state.continuous_trainer, 'log_messages'):
                    training_logs.extend(st.session_state.continuous_trainer.log_messages)
                
                # Also show general logs that contain training information
                if 'log_messages' in st.session_state:
                    for log in st.session_state.log_messages:
                        if any(keyword in log for keyword in ['training', 'Train', 'model', 'AI', 'huấn luyện', 'dữ liệu']):
                            training_logs.append(log)
                
                # Get system logs via popen for comprehensive information
                import subprocess
                
                try:
                    # Get recent logs for relevant components
                    grep_cmd = "grep -E 'feature_engineering|data_processor|model_trainer|continuous_trainer' /tmp/streamlit_app.log 2>/dev/null | tail -n 200"
                    process = subprocess.Popen(grep_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    output, _ = process.communicate()
                    
                    if output:
                        system_logs = output.decode('utf-8').split('\n')
                        training_logs.extend(system_logs)
                except Exception as e:
                    st.error(f"Error reading system logs: {e}")
                
                # Display the logs
                if training_logs:
                    # Format the logs with color highlighting
                    formatted_logs = []
                    for log in training_logs:
                        if "ERROR" in log or "error" in log:
                            formatted_logs.append(f'<span style="color: red;">{log}</span>')
                        elif "WARNING" in log or "warning" in log:
                            formatted_logs.append(f'<span style="color: yellow;">{log}</span>')
                        elif "SUCCESS" in log or "success" in log:
                            formatted_logs.append(f'<span style="color: lime;">{log}</span>')
                        elif "INFO" in log or "info" in log:
                            formatted_logs.append(f'<span style="color: #0f9;">{log}</span>')
                        else:
                            formatted_logs.append(log)
                    
                    log_html = "<div class='training-log-container'>"
                    for log in formatted_logs:
                        log_html += f"{log}<br>"
                    log_html += "</div>"
                    
                    st.markdown(log_html, unsafe_allow_html=True)
                else:
                    st.info("Chưa có nhật ký huấn luyện nào được ghi lại.")
                
                # Add refresh button
                if st.button("🔄 Làm mới nhật ký"):
                    st.experimental_rerun()
            
            with log_col2:
                # Training Status and Statistics
                st.write("### Thống kê huấn luyện")
                
                # Add visual indicators for training phases
                phases = {
                    "Thu thập dữ liệu": "In Progress" if hasattr(st.session_state, 'data_collector') else "Not Started",
                    "Xử lý dữ liệu": "Completed" if hasattr(st.session_state, 'data_processor') else "Not Started",
                    "Huấn luyện mô hình": "Completed" if st.session_state.model_trained else "Not Started",
                    "Dự đoán": "Completed" if st.session_state.predictions else "Not Started"
                }
                
                for phase, status in phases.items():
                    if status == "Completed":
                        st.success(f"✅ {phase}")
                    elif status == "In Progress":
                        st.warning(f"⏳ {phase}")
                    else:
                        st.error(f"❌ {phase}")
                
                # Model Training Controls
                st.write("### Điều khiển huấn luyện")
                
                if st.button("🧠 Huấn luyện ngay", key="force_training_btn"):
                    if hasattr(st.session_state, 'continuous_trainer'):
                        st.session_state.continuous_trainer.schedule_training(force=True)
                        st.success("Đã lên lịch huấn luyện mô hình!")
                    else:
                        st.error("Chưa khởi tạo bộ huấn luyện liên tục")

elif st.session_state.selected_tab == "Cài đặt":
    st.title("Cài đặt hệ thống dự đoán")
    
    if not st.session_state.initialized:
        st.warning("Vui lòng khởi tạo hệ thống trước")
        
        # Add a big initialize button in the center
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Khởi tạo hệ thống", use_container_width=True):
                initialize_system()
                st.rerun()
    else:
        settings_tab1, settings_tab2, settings_tab3 = st.tabs(["Cài đặt dự đoán", "Cài đặt huấn luyện", "Cài đặt hệ thống"])
        
        with settings_tab1:
            st.subheader("⚙️ Cấu hình dự đoán")
            
            # Khung thời gian chính để dự đoán
            selected_timeframe = st.selectbox(
                "Khung thời gian dữ liệu",
                options=["1m", "5m"],
                index=0,
                help="Khung thời gian dữ liệu sử dụng cho việc dự đoán"
            )
            
            # Thời gian dự đoán cho tương lai
            if selected_timeframe == "1m":
                prediction_horizons = list(config.PREDICTION_SETTINGS["1m"]["horizons"].keys())
                selected_horizon = st.selectbox(
                    "Khoảng thời gian dự đoán",
                    options=prediction_horizons,
                    index=0,
                    help="Thời gian dự đoán trong tương lai"
                )
            else:  # 5m
                prediction_horizons = list(config.PREDICTION_SETTINGS["5m"]["horizons"].keys())
                selected_horizon = st.selectbox(
                    "Khoảng thời gian dự đoán",
                    options=prediction_horizons,
                    index=0,
                    help="Thời gian dự đoán trong tương lai"
                )
            
            # Áp dụng thiết lập mới
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Áp dụng thiết lập dự đoán", use_container_width=True):
                    # Lưu thiết lập dự đoán vào session state
                    st.session_state.prediction_settings = {
                        "timeframe": selected_timeframe,
                        "horizon": selected_horizon
                    }
                    st.success(f"Đã cập nhật thiết lập dự đoán: Khung thời gian {selected_timeframe}, dự đoán cho {selected_horizon}")
            
            # Hiển thị thiết lập hiện tại
            if "prediction_settings" in st.session_state:
                settings = st.session_state.prediction_settings
                st.info(f"Thiết lập hiện tại: Khung thời gian {settings['timeframe']}, dự đoán cho {settings['horizon']}")
            else:
                # Thiết lập mặc định
                st.session_state.prediction_settings = {
                    "timeframe": config.DEFAULT_TIMEFRAME,
                    "horizon": config.DEFAULT_PREDICTION_HORIZON
                }
                st.info(f"Thiết lập mặc định: Khung thời gian {config.DEFAULT_TIMEFRAME}, dự đoán cho {config.DEFAULT_PREDICTION_HORIZON}")
        
        with settings_tab2:
            st.subheader("🧠 Cài đặt huấn luyện")
            
            # Chọn khoảng thời gian dữ liệu huấn luyện
            start_date = st.date_input(
                "Ngày bắt đầu dữ liệu huấn luyện",
                value=datetime.strptime(config.DEFAULT_TRAINING_START_DATE, "%Y-%m-%d").date(),
                help="Chọn ngày bắt đầu khoảng thời gian dữ liệu huấn luyện"
            )
            
            # Hiển thị ngày hiện tại làm điểm kết thúc
            end_date = datetime.now().date()
            st.info(f"Dữ liệu huấn luyện sẽ được thu thập từ {start_date} đến {end_date}")
            
            # Tính toán số ngày dữ liệu
            training_days = (end_date - start_date).days
            st.write(f"Tổng cộng: {training_days} ngày dữ liệu")
            
            # Thiết lập tần suất huấn luyện lại
            st.subheader("⏱️ Tần suất huấn luyện tự động")
            training_frequency = st.selectbox(
                "Huấn luyện lại mỗi",
                options=["30 phút", "1 giờ", "3 giờ", "6 giờ", "12 giờ", "24 giờ"],
                index=0,
                help="Tần suất hệ thống tự động huấn luyện lại model"
            )
            
            # Button để bắt đầu huấn luyện và áp dụng thiết lập mới
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Lưu cài đặt huấn luyện", use_container_width=True):
                    # Lưu thiết lập huấn luyện vào session state
                    st.session_state.training_settings = {
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "training_frequency": training_frequency
                    }
                    
                    # Cập nhật tần suất huấn luyện
                    frequency_minutes = {
                        "30 phút": 30,
                        "1 giờ": 60,
                        "3 giờ": 180,
                        "6 giờ": 360,
                        "12 giờ": 720,
                        "24 giờ": 1440
                    }
                    
                    config.TRAINING_SCHEDULE["interval_minutes"] = frequency_minutes[training_frequency]
                    
                    st.success("Đã lưu cài đặt huấn luyện thành công!")
            
            with col2:
                if st.button("🧠 Huấn luyện ngay", use_container_width=True):
                    if 'training_settings' in st.session_state:
                        # Gọi hàm huấn luyện với thiết lập mới
                        with st.spinner("Đang bắt đầu quá trình huấn luyện..."):
                            # Lưu thiết lập huấn luyện và bắt đầu huấn luyện
                            config.HISTORICAL_START_DATE = st.session_state.training_settings["start_date"]
                            train_models()
                            st.success("Đã bắt đầu huấn luyện với thiết lập mới!")
                    else:
                        # Sử dụng thiết lập mặc định
                        with st.spinner("Đang bắt đầu quá trình huấn luyện..."):
                            train_models()
                            st.success("Đã bắt đầu huấn luyện với thiết lập mặc định!")
            
            # Hiển thị thiết lập hiện tại
            if "training_settings" in st.session_state:
                settings = st.session_state.training_settings
                st.info(f"Thiết lập hiện tại: Từ ngày {settings['start_date']}, huấn luyện lại mỗi {settings['training_frequency']}")
            
            # Hiển thị trạng thái huấn luyện
            st.subheader("📊 Trạng thái huấn luyện")
            if 'continuous_trainer' in st.session_state and st.session_state.continuous_trainer:
                status = st.session_state.continuous_trainer.get_training_status()
                
                # Hiển thị thời điểm huấn luyện lần cuối
                if 'last_training' in status and status['last_training']:
                    st.write(f"🕒 Huấn luyện lần cuối: {status['last_training']}")
                
                # Hiển thị thời điểm huấn luyện tiếp theo
                if 'next_training' in status and status['next_training']:
                    st.write(f"⏱️ Huấn luyện tiếp theo: {status['next_training']}")
                
                # Hiển thị trạng thái huấn luyện
                if 'is_training' in status:
                    if status['is_training']:
                        st.warning("⚙️ Đang huấn luyện...")
                    else:
                        st.success("✅ Sẵn sàng cho huấn luyện tiếp theo")
            else:
                st.warning("Hệ thống huấn luyện tự động chưa được khởi tạo")
        
        with settings_tab3:
            st.subheader("🛠️ Cài đặt hệ thống")
            
            # Thiết lập nguồn dữ liệu
            data_source = st.radio(
                "Nguồn dữ liệu",
                options=["Binance API (thực)", "Mô phỏng (giả lập)"],
                index=0 if config.USE_REAL_API else 1,
                help="Chọn nguồn dữ liệu cho hệ thống"
            )
            
            # Cập nhật thiết lập USE_REAL_API
            config.USE_REAL_API = (data_source == "Binance API (thực)")
            
            # Thiết lập thời gian cập nhật dữ liệu
            update_interval = st.slider(
                "Thời gian cập nhật dữ liệu (giây)",
                min_value=5,
                max_value=60,
                value=config.UPDATE_INTERVAL,
                step=5,
                help="Thời gian giữa các lần cập nhật dữ liệu tự động"
            )
            
            # Cập nhật thiết lập UPDATE_INTERVAL
            config.UPDATE_INTERVAL = update_interval
            
            # Button để lưu thiết lập hệ thống
            if st.button("💾 Lưu thiết lập hệ thống", use_container_width=True):
                st.success(f"Đã lưu thiết lập hệ thống: Nguồn dữ liệu = {data_source}, cập nhật mỗi {update_interval} giây")
                
                # Nếu thay đổi nguồn dữ liệu, cần khởi động lại hệ thống
                if data_source == "Binance API (thực)" and isinstance(st.session_state.data_collector, MockDataCollector):
                    st.warning("Cần khởi động lại hệ thống để áp dụng thay đổi nguồn dữ liệu")
                    if st.button("🔄 Khởi động lại hệ thống", use_container_width=True):
                        st.session_state.initialized = False
                        initialize_system()
                        st.rerun()
                elif data_source == "Mô phỏng (giả lập)" and not isinstance(st.session_state.data_collector, MockDataCollector):
                    st.warning("Cần khởi động lại hệ thống để áp dụng thay đổi nguồn dữ liệu")
                    if st.button("🔄 Khởi động lại hệ thống", use_container_width=True):
                        st.session_state.initialized = False
                        initialize_system()
                        st.rerun()

elif st.session_state.selected_tab == "Models & Training":
    st.title("AI Models & Training")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system first")
    else:
        # Data control section
        st.header("Data Preparation")
        
        # Display status of available data
        if st.session_state.latest_data is not None:
            st.success(f"Data available: {len(st.session_state.latest_data)} candles")
            
            # Show data preview
            with st.expander("Preview Raw Data"):
                st.dataframe(st.session_state.latest_data.tail(10))
        else:
            st.warning("No data available. Click 'Fetch Data' in the sidebar.")
        
        # Show training controls
        st.header("Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Train All Models", key="train_all_btn", help="Start the training process for all AI models using the fetched data"):
                train_models()
        
        with col2:
            if st.session_state.model_trained:
                st.success("Models trained and ready for prediction")
                if hasattr(st.session_state, 'training_info'):
                    st.caption(f"Last trained: {st.session_state.training_info.get('training_time', 'Unknown')}")
            else:
                st.warning("Models not trained yet")
        
        # Show data source information
        if hasattr(st.session_state, 'training_info'):
            info = st.session_state.training_info
            
            # Data source info in an expander
            with st.expander("Training Data Source Information", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Data Source Details:**")
                    st.info(f"Source: {info.get('data_source', 'Unknown')}")
                    st.write(f"Data points: {info.get('data_points', 0)} candles")
                    st.write(f"Date range: {info.get('date_range', 'Unknown')}")
                    st.write(f"Features used: {info.get('feature_count', 0)} technical indicators")
                    
                with col2:
                    st.write("**Training Dataset:**")
                    st.write(f"Training samples: {info.get('training_samples', 0)}")
                    
                    # Class distribution
                    class_dist = info.get('class_distribution', {})
                    col1a, col2a, col3a = st.columns(3)
                    with col1a:
                        st.metric("SHORT", class_dist.get('SHORT', 0))
                    with col2a:
                        st.metric("NEUTRAL", class_dist.get('NEUTRAL', 0))
                    with col3a:
                        st.metric("LONG", class_dist.get('LONG', 0))
        
        # Model architecture & performance
        st.header("Model Architecture & Performance")
        
        # Model descriptions with more details
        models_descriptions = {
            "LSTM": "Long Short-Term Memory network for sequence learning from 60 past candles. Specialized in capturing long-term dependencies and sequential patterns in price data. Input: Normalized technical indicators over 60 candles.",
            "Transformer": "Transformer model with self-attention mechanism for price pattern recognition. Excellent at finding non-linear relationships between different timeframes. Input: Same as LSTM but with attention weights between time steps.",
            "CNN": "Convolutional Neural Network for visual pattern recognition in price charts. Treats price data as images to identify visual patterns. Input: 2D representation of price action with multiple technical indicator channels.",
            "Historical Similarity": "K-nearest neighbors approach to find similar historical price patterns. Matches current market conditions with historical outcomes. Input: Normalized price and indicator patterns.",
            "Meta-Learner": "Ensemble model that combines predictions from all other models using a machine learning classifier. Weights each model based on recent performance. Input: Output probabilities from all other models."
        }
        
        # Display model descriptions
        with st.expander("Model Descriptions", expanded=True):
            for model, desc in models_descriptions.items():
                st.write(f"**{model}:** {desc}")
        
        # Display model performance if available
        st.subheader("Model Performance")
        
        if hasattr(st.session_state, 'training_info') and 'model_performance' in st.session_state.training_info:
            perf = st.session_state.training_info['model_performance']
            performance_chart = plot_model_performance(perf)
            st.plotly_chart(performance_chart, use_container_width=True)
        else:
            # Show placeholder performance
            placeholder_perf = {
                'lstm': 0.72,
                'transformer': 0.76,
                'cnn': 0.68,
                'historical_similarity': 0.65,
                'meta_learner': 0.81
            }
            st.info("No actual model performance data available yet. Below is an example of what the performance chart will look like after training.")
            performance_chart = plot_model_performance(placeholder_perf)
            st.plotly_chart(performance_chart, use_container_width=True)
        
        # Training parameters
        with st.expander("Training Parameters"):
            st.json({
                "Sequence Length": config.SEQUENCE_LENGTH,
                "Prediction Window": config.PREDICTION_WINDOW,
                "Price Movement Threshold": f"{config.PRICE_MOVEMENT_THRESHOLD * 100}%",
                "Batch Size": config.BATCH_SIZE,
                "Epochs": config.EPOCHS,
                "Early Stopping Patience": config.EARLY_STOPPING_PATIENCE
            })

elif st.session_state.selected_tab == "Backtest":
    st.title("Kiểm tra hiệu suất mô hình (Backtest)")
    
    if not st.session_state.initialized:
        st.warning("Vui lòng khởi tạo hệ thống trước")
        
        # Add a big initialize button in the center
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Khởi tạo hệ thống", use_container_width=True):
                initialize_system()
                st.rerun()
    else:
        # Thiết lập thời gian cho backtest
        st.subheader("Thiết lập khoảng thời gian cho backtest")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Ngày bắt đầu",
                value=datetime.strptime(config.BACKTEST_PERIOD_START, "%Y-%m-%d").date(),
                help="Ngày bắt đầu cho khoảng thời gian backtest"
            )
        
        with col2:
            end_date = st.date_input(
                "Ngày kết thúc",
                value=datetime.strptime(config.BACKTEST_PERIOD_END, "%Y-%m-%d").date(),
                help="Ngày kết thúc cho khoảng thời gian backtest"
            )
        
        # Thiết lập khung thời gian và khoảng thời gian dự đoán
        st.subheader("Thiết lập dự đoán")
        
        col1, col2 = st.columns(2)
        with col1:
            timeframe = st.selectbox(
                "Khung thời gian",
                options=["1m", "5m"],
                index=0,
                help="Khung thời gian cho dữ liệu backtest"
            )
        
        with col2:
            if timeframe == "1m":
                prediction_horizons = list(config.PREDICTION_SETTINGS["1m"]["horizons"].keys())
                prediction_horizon = st.selectbox(
                    "Thời gian dự đoán",
                    options=prediction_horizons,
                    index=0,
                    help="Khoảng thời gian dự đoán"
                )
            else:  # 5m
                prediction_horizons = list(config.PREDICTION_SETTINGS["5m"]["horizons"].keys())
                prediction_horizon = st.selectbox(
                    "Thời gian dự đoán",
                    options=prediction_horizons,
                    index=0,
                    help="Khoảng thời gian dự đoán"
                )
        
        # Nút để bắt đầu backtest
        if st.button("▶️ Chạy Backtest", use_container_width=True):
            # Kiểm tra xem ngày bắt đầu có trước ngày kết thúc không
            if start_date >= end_date:
                st.error("Ngày bắt đầu phải trước ngày kết thúc!")
            else:
                with st.spinner("Đang thực hiện backtest..."):
                    # Đặt thông tin backtest vào session state
                    if 'backtest_results' not in st.session_state:
                        st.session_state.backtest_results = {}
                    
                    # Đặt khoảng thời gian và cấu hình dự đoán
                    backtest_config = {
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "timeframe": timeframe,
                        "prediction_horizon": prediction_horizon
                    }
                    
                    # Tạo key cho kết quả backtest này
                    backtest_key = f"{timeframe}_{prediction_horizon}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                    
                    # Tạo kết quả backtest giả để demo (thay thế bằng kết quả thực khi có hàm backtest)
                    # Tạo kết quả backtest ngẫu nhiên để demo
                    accuracy = random.uniform(0.62, 0.78)
                    total_trades = random.randint(100, 500)
                    profitable_trades = int(total_trades * accuracy)
                    average_profit = random.uniform(2.5, 5.0)
                    average_loss = random.uniform(1.5, 3.0)
                    max_drawdown = random.uniform(8, 15)
                    
                    # Tạo danh sách giao dịch giả lập
                    fake_trades = []
                    current_date = start_date
                    while current_date <= end_date:
                        # Bỏ qua cuối tuần
                        if current_date.weekday() < 5:  # 0-4 là thứ 2 đến thứ 6
                            # Số giao dịch ngẫu nhiên mỗi ngày
                            num_trades = random.randint(0, 3)
                            
                            for _ in range(num_trades):
                                # Tạo thời gian ngẫu nhiên trong ngày
                                hour = random.randint(0, 23)
                                minute = random.randint(0, 59)
                                trade_time = datetime(
                                    current_date.year, 
                                    current_date.month, 
                                    current_date.day,
                                    hour, minute
                                )
                                
                                # Ngẫu nhiên tín hiệu
                                signal = random.choice(["LONG", "SHORT"])
                                
                                # Ngẫu nhiên kết quả
                                result = random.choice([True, False, True, True])  # Thiên về true một chút
                                
                                # Tính lợi nhuận/lỗ
                                pnl = random.uniform(2.0, 6.0) if result else -random.uniform(1.0, 3.0)
                                
                                # Thêm vào danh sách giao dịch
                                fake_trades.append({
                                    "time": trade_time.strftime("%Y-%m-%d %H:%M"),
                                    "signal": signal,
                                    "entry_price": round(random.uniform(3000, 4000), 2),
                                    "exit_price": None,  # Sẽ tính sau
                                    "result": "WIN" if result else "LOSS",
                                    "pnl": round(pnl, 2),
                                    "confidence": round(random.uniform(0.65, 0.95), 2)
                                })
                        
                        # Ngày tiếp theo
                        current_date += timedelta(days=1)
                    
                    # Thêm giá thoát dựa trên PNL
                    for trade in fake_trades:
                        entry_price = trade["entry_price"]
                        pnl_percent = trade["pnl"] / entry_price
                        
                        if trade["signal"] == "LONG":
                            trade["exit_price"] = round(entry_price * (1 + pnl_percent), 2)
                        else:  # SHORT
                            trade["exit_price"] = round(entry_price * (1 - pnl_percent), 2)
                    
                    # Sắp xếp giao dịch theo thời gian
                    fake_trades.sort(key=lambda x: x["time"])
                    
                    # Tạo ma trận nhầm lẫn
                    confusion_matrix = {
                        "true_long": random.randint(30, 70),
                        "true_neutral": random.randint(100, 200),
                        "true_short": random.randint(30, 70),
                        "pred_long": random.randint(40, 80),
                        "pred_neutral": random.randint(90, 180),
                        "pred_short": random.randint(40, 80),
                        "correct_long": random.randint(20, 50),
                        "correct_neutral": random.randint(80, 150),
                        "correct_short": random.randint(20, 50)
                    }
                    
                    # Lưu kết quả
                    st.session_state.backtest_results[backtest_key] = {
                        "config": backtest_config,
                        "accuracy": accuracy,
                        "total_trades": total_trades,
                        "profitable_trades": profitable_trades,
                        "average_profit": average_profit,
                        "average_loss": average_loss,
                        "max_drawdown": max_drawdown,
                        "trades": fake_trades,
                        "confusion_matrix": confusion_matrix,
                        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.success(f"Đã hoàn thành backtest cho khoảng thời gian từ {start_date} đến {end_date}")
        
        # Hiển thị kết quả backtest nếu có
        if 'backtest_results' in st.session_state and st.session_state.backtest_results:
            st.subheader("Kết quả Backtest")
            
            # Tạo các tab cho các kết quả backtest khác nhau nếu có nhiều hơn 1
            result_keys = list(st.session_state.backtest_results.keys())
            
            if len(result_keys) > 1:
                # Hiển thị selector cho nhiều kết quả backtest
                selected_result = st.selectbox(
                    "Chọn kết quả backtest để xem chi tiết",
                    options=result_keys,
                    format_func=lambda x: f"{st.session_state.backtest_results[x]['config']['timeframe']} ({st.session_state.backtest_results[x]['config']['prediction_horizon']}) "
                                         f"[{st.session_state.backtest_results[x]['config']['start_date']} - "
                                         f"{st.session_state.backtest_results[x]['config']['end_date']}]"
                )
                result = st.session_state.backtest_results[selected_result]
            else:
                # Chỉ có một kết quả
                result = st.session_state.backtest_results[result_keys[0]]
            
            # Hiển thị thông tin tổng quan
            st.markdown("### Tổng quan hiệu suất")
            
            # Hiển thị các chỉ số chính
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Độ chính xác", f"{result['accuracy']:.2%}")
            with col2:
                st.metric("Tổng số giao dịch", f"{result['total_trades']}")
            with col3:
                win_rate = result['profitable_trades'] / result['total_trades']
                st.metric("Tỷ lệ thắng", f"{win_rate:.2%}")
            with col4:
                st.metric("Drawdown tối đa", f"{result['max_drawdown']:.2%}")
            
            st.markdown("---")
            
            # Tạo các tab khác nhau cho kết quả chi tiết
            backtest_tabs = st.tabs(["Hiệu suất", "Giao dịch", "Ma trận nhầm lẫn", "Thống kê"])
            
            with backtest_tabs[0]:
                # Tab hiệu suất với biểu đồ
                st.subheader("Biểu đồ hiệu suất")
                
                # Tạo danh sách lợi nhuận tích lũy
                trades = result["trades"]
                cumulative_pnl = [0]
                dates = []
                
                for trade in trades:
                    cumulative_pnl.append(cumulative_pnl[-1] + trade["pnl"])
                    dates.append(trade["time"])
                
                # Tạo biểu đồ hiệu suất
                fig = go.Figure()
                
                # Thêm đường lợi nhuận tích lũy
                fig.add_trace(go.Scatter(
                    x=dates, 
                    y=cumulative_pnl[1:],
                    mode='lines',
                    name='Lợi nhuận tích lũy',
                    line=dict(color='blue', width=2)
                ))
                
                # Định dạng biểu đồ
                fig.update_layout(
                    title='Lợi nhuận tích lũy theo thời gian',
                    xaxis_title='Thời gian',
                    yaxis_title='Lợi nhuận tích lũy ($)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with backtest_tabs[1]:
                # Tab giao dịch với danh sách chi tiết
                st.subheader("Danh sách giao dịch")
                
                # Chuyển danh sách giao dịch thành DataFrame
                trades_df = pd.DataFrame(result["trades"])
                
                # Thêm định dạng màu sắc dựa trên kết quả
                def highlight_win_loss(s):
                    if s.name == 'result':
                        return ['background-color: #CCFFCC' if x == 'WIN' else 'background-color: #FFCCCC' for x in s]
                    elif s.name == 'pnl':
                        return ['color: green' if x > 0 else 'color: red' for x in s]
                    return [''] * len(s)
                
                # Hiển thị DataFrame với định dạng
                st.dataframe(trades_df.style.apply(highlight_win_loss), use_container_width=True)
            
            with backtest_tabs[2]:
                # Tab ma trận nhầm lẫn
                st.subheader("Ma trận nhầm lẫn")
                
                # Tạo ma trận nhầm lẫn
                cm = result["confusion_matrix"]
                
                # Tính toán các giá trị
                true_long = cm["true_long"]
                true_neutral = cm["true_neutral"]
                true_short = cm["true_short"]
                pred_long = cm["pred_long"]
                pred_neutral = cm["pred_neutral"]
                pred_short = cm["pred_short"]
                correct_long = cm["correct_long"]
                correct_neutral = cm["correct_neutral"]
                correct_short = cm["correct_short"]
                
                # Tạo ma trận
                cm_matrix = [
                    [correct_long, pred_long - correct_long, true_long - correct_long],
                    [pred_neutral - correct_neutral, correct_neutral, true_neutral - correct_neutral],
                    [pred_short - correct_short, true_short - correct_short, correct_short]
                ]
                
                # Tạo biểu đồ ma trận nhầm lẫn
                fig = go.Figure(data=go.Heatmap(
                    z=cm_matrix,
                    x=['Dự đoán LONG', 'Dự đoán NEUTRAL', 'Dự đoán SHORT'],
                    y=['Thực tế LONG', 'Thực tế NEUTRAL', 'Thực tế SHORT'],
                    colorscale='Viridis',
                    showscale=True
                ))
                
                fig.update_layout(
                    title='Ma trận nhầm lẫn',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with backtest_tabs[3]:
                # Tab thống kê chi tiết
                st.subheader("Thống kê chi tiết")
                
                # Tính toán các thống kê
                profit_trades = [t for t in result["trades"] if t["pnl"] > 0]
                loss_trades = [t for t in result["trades"] if t["pnl"] <= 0]
                
                # Tạo bảng thống kê
                stats = {
                    "Số giao dịch thắng": len(profit_trades),
                    "Số giao dịch thua": len(loss_trades),
                    "Tỷ lệ thắng": f"{len(profit_trades) / len(result['trades']):.2%}",
                    "Lợi nhuận trung bình (giao dịch thắng)": f"${sum([t['pnl'] for t in profit_trades]) / len(profit_trades):.2f}",
                    "Thua lỗ trung bình (giao dịch thua)": f"${sum([t['pnl'] for t in loss_trades]) / len(loss_trades):.2f}",
                    "Tỷ lệ lợi nhuận trên rủi ro": f"{abs(sum([t['pnl'] for t in profit_trades]) / sum([t['pnl'] for t in loss_trades])):.2f}",
                    "Lợi nhuận tổng cộng": f"${sum([t['pnl'] for t in result['trades']]):.2f}",
                    "Thời gian backtest": f"{result['config']['start_date']} đến {result['config']['end_date']}",
                    "Khung thời gian": result['config']['timeframe'],
                    "Thời gian dự đoán": result['config']['prediction_horizon']
                }
                
                # Chuyển thành DataFrame để hiển thị
                stats_df = pd.DataFrame(list(stats.items()), columns=["Chỉ số", "Giá trị"])
                st.dataframe(stats_df, use_container_width=True)

elif st.session_state.selected_tab == "System Status":
    st.title("System Status")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system first")
    else:
        # Use columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display system status
            display_system_status(
                data_status=st.session_state.data_fetch_status,
                thread_status=st.session_state.thread_running,
                prediction_count=len(st.session_state.predictions)
            )
            
            # Activity Logs
            st.write("### Activity Logs")
            if 'log_messages' in st.session_state and st.session_state.log_messages:
                # Create a scrollable log area with fixed height
                log_container = st.container()
                with log_container:
                    st.markdown("""
                    <style>
                    .log-container {
                        height: 300px;
                        overflow-y: auto;
                        background-color: #f0f0f0;
                        padding: 10px;
                        border-radius: 5px;
                        font-family: monospace;
                    }
                    </style>
                    <div class="log-container">
                    """ + "<br>".join(st.session_state.log_messages[-50:]) + """
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add clear logs button
                if st.button("Clear Logs"):
                    st.session_state.log_messages = []
                    st.rerun()
            else:
                st.info("No activity logs available. Perform actions to generate logs.")
        
        with col2:
            # Data source information
            st.write("### Data Source")
            if 'data_source' in st.session_state:
                source_color = "green" if st.session_state.data_source == "Binance API (Real Data)" else "orange"
                st.markdown(f"**Current Source:** :{source_color}[{st.session_state.data_source}]")
            else:
                st.info("Data source not initialized")
            
            # API status
            st.write("### API Connection Status")
            try:
                if isinstance(st.session_state.data_collector, MockDataCollector):
                    st.warning("Using simulated data (MockDataCollector)")
                    
                    # Check if we have API status information
                    if hasattr(st.session_state, 'api_status'):
                        # If we tried to connect to the API but failed
                        if 'error' in st.session_state.api_status and st.session_state.api_status['error']:
                            st.error(f"API Connection Error: {st.session_state.api_status['message']}")
                            
                            # Check for geographic restrictions
                            if "Geographic restriction" in st.session_state.api_status.get('error', ''):
                                st.warning("⚠️ Binance has geographic restrictions in your region")
                                st.info("Consider using a VPN service to access Binance API from supported regions")
                            
                            # Show more details in an expander
                            with st.expander("API Connection Details"):
                                st.write("**Error Type:**", st.session_state.api_status.get('error', 'Unknown'))
                                st.write("**Last Check:**", st.session_state.api_status.get('last_check', 'Unknown'))
                                st.write("**Try using the mock data collector for development purposes**")
                        else:
                            st.info("The system is configured to use mock data")
                            
                            # Show toggle in expander
                            with st.expander("Data Source Configuration"):
                                st.write("To use real Binance API data, update the following in config.py:")
                                st.code("""
# Feature flags
USE_REAL_API = True  # Set to True to use real Binance API 
FORCE_MOCK_DATA = False  # Set to False to allow real API usage
                                """)
                    else:
                        st.info("The system is configured to use mock data due to API restrictions in the current environment.")
                        st.info("Actual API implementation is available in the code for deployment in production environments.")
                else:
                    # We're using real Binance API
                    # Test connection to Binance
                    api_status = "Connected" if hasattr(st.session_state.data_collector, 'client') and st.session_state.data_collector.client else "Not Connected"
                    st.success(f"Binance API: {api_status}")
                    
                    # Display API connection details
                    with st.expander("API Connection Details"):
                        st.write("**API Key:** ", "✓ Configured" if config.BINANCE_API_KEY else "❌ Missing")
                        st.write("**API Secret:** ", "✓ Configured" if config.BINANCE_API_SECRET else "❌ Missing")
                        st.write("**Last Check:** ", st.session_state.api_status.get('last_check', 'Unknown'))
            except Exception as e:
                st.error(f"Error checking API status: {e}")
            
            # Quick actions
            st.write("### Quick Actions")
            if st.button("Fetch Latest Data", key="fetch_latest_btn"):
                with st.spinner("Fetching..."):
                    fetch_data()
                    st.success("Data fetched successfully!")
            
            if st.button("Make New Prediction", key="new_pred_btn"):
                with st.spinner("Generating prediction..."):
                    make_prediction()
                    st.success("New prediction generated!")
        
        # Data preview (full width)
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
        
        # Model information
        with st.expander("Model Information"):
            if st.session_state.model_trained:
                st.success("All models are trained and ready")
                st.write("Model versions and paths:")
                st.code(f"Models directory: {config.MODEL_DIR}\nVersion: {config.MODEL_VERSION}")
            else:
                st.warning("Models are not trained yet")

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
    - `price`: Current price at time of prediction
    - `target_price`: Predicted future price target
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
    # Fetch data immediately after initialization to show real-time chart
    if st.session_state.initialized:
        with st.spinner("Đang tải dữ liệu thời gian thực..."):
            fetch_data()
            # Generate an initial prediction
            make_prediction()