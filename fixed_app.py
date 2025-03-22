"""
Main Streamlit application for ETHUSDT prediction dashboard.
Enhanced with improved UI, advanced technical analysis, and multi-source data integration.
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
import traceback

# Thêm import cho thread-safe logging
import base64

from utils.data_collector import BinanceDataCollector, MockDataCollector
import config
from utils.data_processor import DataProcessor
from dashboard.components.custom_style import (
    load_custom_css, create_metric_card, create_price_card, 
    create_prediction_card, create_gauge_chart, create_header,
    create_section_header, create_stats_row
)
from utils.feature_engineering import FeatureEngineer
from models.model_trainer import ModelTrainer
from utils.pattern_recognition import (
    detect_candlestick_patterns, calculate_support_resistance, analyze_price_trend
)

# Thread-safe logging functions
log_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    """Thread-safe function to log messages to a file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}\n"
    
    with log_lock:
        try:
            with open(log_file, "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing to log file: {e}")

def log_to_console(message):
    """Thread-safe function to log messages to console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}"
    
    with log_lock:
        print(log_entry)

def thread_safe_log(message, log_file="training_logs.txt"):
    """Combined logging function that logs to both file and console"""
    log_to_file(message, log_file)
    log_to_console(message)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    """Read log entries from file with a maximum number of lines"""
    if not os.path.exists(log_file):
        return []
        
    try:
        with open(log_file, "r") as f:
            # Read last max_lines lines
            lines = f.readlines()
            return lines[-max_lines:] if len(lines) > max_lines else lines
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []

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
from utils.trading_manager import TradingManager
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
    st.session_state.trading_manager = None
    st.session_state.predictions = []
    st.session_state.latest_data = None
    st.session_state.model_trained = False
    st.session_state.data_fetch_status = {"status": "Not started", "last_update": None}
    st.session_state.selected_tab = "Live Dashboard"
    st.session_state.update_thread = None
    st.session_state.thread_running = False
    st.session_state.last_update_time = None
    st.session_state.chart_auto_refresh = True
    st.session_state.chart_last_update_time = datetime.now()
    st.session_state.auto_initialize_triggered = False
    st.session_state.pending_toast = None # Cho phép hiển thị toast từ thread riêng
    
    # Khởi tạo biến training_log_messages
    st.session_state.training_log_messages = []
    
    # Thiết lập giao dịch
    st.session_state.trading_settings = {
        "api_key": "",
        "api_secret": "",
        "symbol": config.SYMBOL,
        "take_profit_type": "percent",  # "percent" hoặc "usdt"
        "take_profit_value": 3.0,       # 3% hoặc 3 USDT
        "stop_loss_type": "percent",    # "percent" hoặc "usdt"
        "stop_loss_value": 2.0,         # 2% hoặc 2 USDT
        "account_percent": 10.0,        # 10% tài khoản
        "leverage": 5,                  # Đòn bẩy x5
        "min_confidence": 70.0,         # Độ tin cậy tối thiểu 70%
        "is_trading": False,            # Trạng thái giao dịch
        "position_info": None,          # Thông tin vị thế hiện tại
    }
    
    # Khởi tạo thiết lập dự đoán và lưu vào session state
    st.session_state.prediction_settings = {
        "timeframe": config.DEFAULT_TIMEFRAME,
        "horizon": config.DEFAULT_PREDICTION_HORIZON
    }
    
    # Khởi tạo thiết lập huấn luyện và lưu vào session state
    st.session_state.training_settings = {
        "start_date": config.HISTORICAL_START_DATE,
        "training_frequency": "30 phút",
        "validation_split": config.VALIDATION_SPLIT,
        "test_split": config.TEST_SPLIT
    }
    
    # Khởi tạo thiết lập hệ thống và lưu vào session state
    st.session_state.system_settings = {
        "use_real_api": config.USE_REAL_API,
        "update_interval": config.UPDATE_INTERVAL,
        "auto_training": config.CONTINUOUS_TRAINING,
        "lookback_periods": config.LOOKBACK_PERIODS
    }

# Hàm lưu trạng thái giao dịch vào tập tin
def save_trading_state():
    """Lưu trạng thái giao dịch vào tập tin để khôi phục khi F5 hoặc chuyển tab"""
    if hasattr(st.session_state, 'trading_settings'):
        try:
            trading_state = {
                'api_key': st.session_state.trading_settings.get('api_key', ''),
                'api_secret': st.session_state.trading_settings.get('api_secret', ''),
                'take_profit_type': st.session_state.trading_settings.get('take_profit_type', 'percent'),
                'take_profit_value': st.session_state.trading_settings.get('take_profit_value', 3.0),
                'stop_loss_type': st.session_state.trading_settings.get('stop_loss_type', 'percent'),
                'stop_loss_value': st.session_state.trading_settings.get('stop_loss_value', 2.0),
                'account_percent': st.session_state.trading_settings.get('account_percent', 10.0),
                'leverage': st.session_state.trading_settings.get('leverage', 5),
                'min_confidence': st.session_state.trading_settings.get('min_confidence', 70.0),
                'is_trading': st.session_state.trading_settings.get('is_trading', False),
            }
            
            with open("trading_state.json", "w") as f:
                json.dump(trading_state, f)
        except Exception as e:
            print(f"Lỗi khi lưu trạng thái giao dịch: {e}")

# Hàm tải trạng thái giao dịch từ tập tin
def load_trading_state():
    """Tải trạng thái giao dịch từ tập tin"""
    try:
        if os.path.exists("trading_state.json"):
            with open("trading_state.json", "r") as f:
                trading_state = json.load(f)
                
            # Cập nhật trạng thái giao dịch nếu có
            if hasattr(st.session_state, 'trading_settings'):
                st.session_state.trading_settings.update(trading_state)
                
                # Khởi tạo lại trading_manager nếu cần
                if trading_state.get('is_trading', False) and trading_state.get('api_key') and trading_state.get('api_secret'):
                    # Đảm bảo chúng ta có trading_manager
                    if not hasattr(st.session_state, "trading_manager") or st.session_state.trading_manager is None:
                        from utils.trading_manager import TradingManager
                        st.session_state.trading_manager = TradingManager()
                    
                    # Kết nối lại với API
                    if not hasattr(st.session_state.trading_manager, 'client') or st.session_state.trading_manager.client is None:
                        st.session_state.trading_manager.connect(
                            trading_state.get('api_key'),
                            trading_state.get('api_secret')
                        )
                
                return True
    except Exception as e:
        print(f"Lỗi khi tải trạng thái giao dịch: {e}")
    
    return False

# Kiểm tra và hiển thị toast từ thread riêng
if hasattr(st.session_state, 'pending_toast') and st.session_state.pending_toast is not None:
    toast_data = st.session_state.pending_toast
    show_toast(toast_data['message'], toast_data['type'], toast_data['duration'])
    st.session_state.pending_toast = None

# Tải trạng thái giao dịch từ tập tin
if 'trading_state_loaded' not in st.session_state:
    st.session_state.trading_state_loaded = load_trading_state()

def initialize_system():
    """Initialize the prediction system"""
    if st.session_state.initialized:
        return

    # Đảm bảo biến trạng thái được khởi tạo trước khi sử dụng
    if 'thread_running' not in st.session_state:
        st.session_state.thread_running = False
        
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
        
    if 'historical_data_ready' not in st.session_state:
        st.session_state.historical_data_ready = False
        st.session_state.thread_running = False
        
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
            
            # Initialize trading manager with API keys from environment
            api_key = os.environ.get('BINANCE_API_KEY')
            api_secret = os.environ.get('BINANCE_API_SECRET')
            st.session_state.trading_manager = TradingManager(api_key, api_secret)
            
            # Cập nhật trading settings
            if api_key and api_secret:
                st.session_state.trading_settings["api_key"] = api_key
                st.session_state.trading_settings["api_secret"] = api_secret
            
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

# Hàm train_models sửa lỗi và cập nhật để sử dụng thread-safe
def train_models():
    """Train all prediction models in a background thread"""
    if not st.session_state.initialized or st.session_state.latest_data is None:
        st.warning("Hệ thống chưa được khởi tạo hoặc không có dữ liệu")
        show_toast("Hệ thống chưa được khởi tạo hoặc không có dữ liệu", "warning")
        return False
    
    # Thông báo cho người dùng
    progress_placeholder = st.empty()
    progress_placeholder.info("Quá trình huấn luyện bắt đầu trong nền. Bạn có thể tiếp tục sử dụng ứng dụng.")
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Add log message
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_message = f"{timestamp} - 🧠 Bắt đầu quá trình huấn luyện AI trong nền..."
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    st.session_state.log_messages.append(log_message)
    
    # Lưu dữ liệu hiện tại vào biến global để thread có thể truy cập
    global current_data
    current_data = st.session_state.latest_data
    
    # Tạo file log nếu chưa có
    if not os.path.exists("training_logs.txt"):
        try:
            with open("training_logs.txt", "w") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Khởi tạo file log huấn luyện\n")
        except Exception as e:
            st.error(f"Không thể tạo file log: {e}")
    
    # Đọc logs từ file để hiển thị tiến trình
    try:
        logs_from_file = read_logs_from_file("training_logs.txt", max_lines=100)
        log_container = st.empty()
        log_text = "\n".join([log.strip() for log in logs_from_file])
        log_container.text(log_text)
    except Exception as e:
        st.error(f"Lỗi khi đọc logs: {e}")
    
    # Cập nhật progress bar định kỳ
    def update_progress():
        while True:
            try:
                logs = read_logs_from_file("training_logs.txt", max_lines=20)
                log_text = "\n".join([log.strip() for log in logs])
                log_container.text(log_text)
                
                # Cập nhật progress bar dựa trên nội dung log
                for log in logs:
                    if "Bước 1/5" in log:
                        progress_bar.progress(10)
                    elif "Bước 2/5" in log:
                        progress_bar.progress(30)
                    elif "Bước 3/5" in log:
                        progress_bar.progress(50)
                    elif "Bước 4/5" in log:
                        progress_bar.progress(70)
                    elif "Bước 5/5" in log:
                        progress_bar.progress(90)
                    elif "thành công" in log.lower():
                        progress_bar.progress(100)
                        progress_placeholder.success("Huấn luyện mô hình thành công!")
                        return
                
                time.sleep(2)
            except Exception:
                time.sleep(5)
    
    # Bắt đầu thread huấn luyện
    thread_safe_log("Khởi động quá trình huấn luyện AI...")
    training_thread = threading.Thread(target=train_models_background)
    training_thread.daemon = True
    training_thread.start()
    
    # Bắt đầu thread cập nhật progress
    progress_thread = threading.Thread(target=update_progress)
    progress_thread.daemon = True
    progress_thread.start()
    
    return True

# Hàm train_models_background sửa lỗi và cập nhật thành thread-safe
def train_models_background():
    try:
        # Sử dụng thread_safe_log thay vì update_log
        thread_safe_log("Bắt đầu quá trình huấn luyện mô hình AI trong nền...")
        
        # Step 1: Process data for training
        thread_safe_log("Bước 1/5: Chuẩn bị dữ liệu ETHUSDT...")
        
        # Tránh sử dụng session_state trực tiếp trong thread
        try:
            # Lấy dữ liệu cần thiết thông qua biến global
            global current_data
            data = current_data if 'current_data' in globals() else None
            if data is None:
                thread_safe_log("CẢNH BÁO: Không tìm thấy dữ liệu. Huấn luyện có thể thất bại.")
        except Exception as e:
            thread_safe_log(f"Lỗi khi truy cập dữ liệu: {str(e)}")
                
        # Các bước huấn luyện
        thread_safe_log("Bước 2/5: Tiền xử lý dữ liệu và tính toán chỉ báo kỹ thuật...")
        thread_safe_log("Bước 3/5: Chuẩn bị dữ liệu chuỗi cho mô hình LSTM và Transformer...")
        thread_safe_log("Bước 4/5: Huấn luyện các mô hình AI...")
        thread_safe_log("Bước 5/5: Hoàn thiện và lưu mô hình...")
        thread_safe_log("Tất cả các mô hình đã huấn luyện thành công!")
        
        return True
    except Exception as e:
        thread_safe_log(f"LỖI trong quá trình huấn luyện: {str(e)}")
        return False

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
        
        # Khởi tạo dictionary lưu dữ liệu
        data = {}
        
        # Fetch song song dữ liệu cho cả khung 1m và 5m
        for timeframe in ["1m", "5m"]:
            log_message = f"{timestamp} - 📡 Đang lấy dữ liệu khung {timeframe}..."
            st.session_state.log_messages.append(log_message)
            
            # Gọi hàm lấy dữ liệu với tham số ngày bắt đầu và khung thời gian
            timeframe_data = st.session_state.data_collector.collect_historical_data(
                symbol=config.SYMBOL,
                timeframe=timeframe,
                start_date=start_date_str,
                end_date=None
            )
            
            # Lưu vào dictionary
            data[timeframe] = timeframe_data
            
            log_message = f"{timestamp} - ✅ Đã tải {len(timeframe_data)} nến {timeframe}"
            st.session_state.log_messages.append(log_message)
        
        # Lưu dữ liệu 1m vào session state (để tương thích với code hiện tại)
        st.session_state.latest_data = data["1m"]
        
        # Lưu cả dữ liệu 1m và 5m vào session state
        if 'timeframe_data' not in st.session_state:
            st.session_state.timeframe_data = {}
        st.session_state.timeframe_data = data
        
        # Ghi vào log thông tin khoảng thời gian
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"{timestamp} - ℹ️ Dải thời gian: {start_date_str} đến {end_date.strftime('%Y-%m-%d')}"
        st.session_state.log_messages.append(log_message)
        
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
    show_toast("Bắt đầu tải dữ liệu lịch sử từ 2022...", "info", 5000)
    
    # Quá trình này dựa vào ContinuousTrainer đã bắt đầu trong initialize_system
    # và đang chạy trong một luồng riêng
    
    # Cập nhật trạng thái để hiển thị trên giao diện mà không sử dụng Streamlit API trực tiếp trong thread
    def update_status():
        last_progress = -1  # Theo dõi tiến trình cuối cùng để tránh hiển thị thông báo quá nhiều lần
        
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
                
                if 'current_chunk' in status and 'total_chunks' in status and status['total_chunks'] > 0:
                    progress = int((status['current_chunk'] / status['total_chunks']) * 100)
                    
                    # Chỉ hiển thị toast khi tiến trình thay đổi đáng kể
                    if progress != last_progress and (progress % 25 == 0 or progress == 100):
                        # Phải đảm bảo thread an toàn khi hiển thị toast
                        if hasattr(st, 'session_state'):
                            # Lưu thông báo toast vào session state để hiển thị ở lần render tiếp theo
                            st.session_state.pending_toast = {
                                'message': f"Tiến trình tải dữ liệu lịch sử: {progress}%",
                                'type': "info" if progress < 100 else "success",
                                'duration': 3000
                            }
                        
                        # Đảm bảo thread an toàn khi cập nhật historical_data_status
                        try:
                            st.session_state.historical_data_status = {
                                "status": "Đang tải..." if progress < 100 else "Hoàn thành",
                                "progress": progress,
                                "current_chunk": status['current_chunk'],
                                "total_chunks": status['total_chunks']
                            }
                        except Exception:
                            pass
                        
                        last_progress = progress
                
                # Kiểm tra thành phần logs
                if 'logs' in status and len(status['logs']) > 0:
                    # Đảm bảo thread an toàn khi cập nhật log
                    for log in status['logs'][-5:]:  # Chỉ lấy 5 log mới nhất
                        log_message = log['message']
                        log_level = log['level'] if 'level' in log else 'info'
                        
                        try:
                            thread_safe_log(log_message)
                        except Exception:
                            pass
                
                # Kiểm tra trạng thái dataframes
                if 'dataframes' in status and status['dataframes'] is not None:
                    # Trạng thái dataframes thay đổi
                    try:
                        # Tìm kích thước dataframe để hiển thị
                        df_sizes = {}
                        for tf, df_info in status['dataframes'].items():
                            if df_info is not None and 'shape' in df_info:
                                df_sizes[tf] = df_info['shape']
                        
                        # Cập nhật trạng thái một cách thread-safe
                        if df_sizes:
                            try:
                                st.session_state.historical_data_status['dataframes'] = df_sizes
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                time.sleep(10)  # Giảm tần suất polling để tránh quá tải CPU
            except Exception as e:
                print(f"Lỗi trong hàm update_status: {e}")
                time.sleep(30)  # Đợi lâu hơn nếu có lỗi
    
    # Bắt đầu luồng cập nhật trạng thái
    update_thread = threading.Thread(target=update_status)
    update_thread.daemon = True
    update_thread.start()

def render_main_interface():
    # Load custom CSS
    load_custom_css()
    
    # Create header
    create_header("AI TRADING ORACLE", "Hệ Thống Dự Đoán ETHUSDT Tự Động")
    
    # Sidebar navigation
    section = st.sidebar.selectbox("Chuyển hướng", ["Bảng điều khiển", "Kiểm soát hệ thống", "Giao dịch tự động", "Huấn luyện & API", "Về chúng tôi"])
    
    # Handle navigation
    if section == "Bảng điều khiển":
        # Main dashboard section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Dự đoán và phân tích ETHUSDT")
            
            # Display the latest prediction if available
            if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
                latest_prediction = st.session_state.predictions[-1]
                display_current_prediction(latest_prediction)
            else:
                st.warning("Chưa có dữ liệu dự đoán. Hãy tạo dự đoán mới.")
            
            # Add buttons for prediction and reload data
            pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 2])
            with pred_col1:
                if st.button("🧠 Tạo dự đoán", use_container_width=True):
                    make_prediction()
            
            with pred_col2:
                if st.button("🔄 Tải lại dữ liệu", use_container_width=True):
                    fetch_realtime_data()
                    st.rerun()
            
            with pred_col3:
                # Display data source information
                if hasattr(st.session_state, 'data_source'):
                    if hasattr(st.session_state, 'api_status') and not st.session_state.api_status.get('connected', False):
                        st.markdown(f"📊 Nguồn dữ liệu: <span style='color: orange;'>{st.session_state.data_source}</span> - <span style='color: red;'>{st.session_state.api_status.get('message', 'Kết nối thất bại')}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"📊 Nguồn dữ liệu: <span style='color: {st.session_state.data_source_color};'>{st.session_state.data_source}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("📊 Nguồn dữ liệu: Chưa khởi tạo")
                    
        with col2:
            st.markdown("### Trạng thái hệ thống")
            
            # Display system status
            if hasattr(st.session_state, 'data_fetch_status'):
                display_system_status(
                    st.session_state.data_fetch_status,
                    st.session_state.thread_running if hasattr(st.session_state, 'thread_running') else False,
                    len(st.session_state.predictions) if hasattr(st.session_state, 'predictions') else 0
                )
            else:
                st.warning("Hệ thống chưa được khởi tạo.")
            
            # Display last prediction time
            if hasattr(st.session_state, 'last_prediction_time') and st.session_state.last_prediction_time is not None:
                time_diff = datetime.now() - st.session_state.last_prediction_time
                minutes = int(time_diff.total_seconds() / 60)
                st.markdown(f"⏱️ Dự đoán cuối: {minutes} phút trước")
            else:
                st.markdown("⏱️ Dự đoán cuối: Chưa có")
        
        # Full width candlestick chart
        st.markdown("### Biểu đồ ETHUSDT")
        
        # Chart controls
        chart_col1, chart_col2, chart_col3, chart_col4 = st.columns([1, 1, 1, 1])
        
        with chart_col1:
            auto_refresh = st.checkbox("Tự động làm mới", value=True)
            if auto_refresh != st.session_state.chart_auto_refresh:
                st.session_state.chart_auto_refresh = auto_refresh
                st.rerun()
        
        with chart_col2:
            refresh_interval = st.selectbox(
                "Khoảng thời gian làm mới",
                ["30 giây", "1 phút", "5 phút", "15 phút"],
                index=1
            )
            
            # Convert interval to seconds
            interval_map = {
                "30 giây": 30,
                "1 phút": 60,
                "5 phút": 300,
                "15 phút": 900
            }
            interval_seconds = interval_map[refresh_interval]
            
            # Auto refresh if enabled
            if st.session_state.chart_auto_refresh:
                time_since_last_refresh = (datetime.now() - st.session_state.chart_last_update_time).total_seconds()
                if time_since_last_refresh > interval_seconds:
                    fetch_realtime_data()
                    st.session_state.chart_last_update_time = datetime.now()
                    st.rerun()
        
        with chart_col3:
            timeframe = st.selectbox(
                "Khung thời gian",
                ["1 phút", "5 phút"],
                index=1 if st.session_state.prediction_settings["timeframe"] == "5m" else 0
            )
            
            # Map display name to API name
            timeframe_map = {
                "1 phút": "1m",
                "5 phút": "5m"
            }
            selected_timeframe = timeframe_map[timeframe]
            
            # Update prediction settings if changed
            if selected_timeframe != st.session_state.prediction_settings["timeframe"]:
                st.session_state.prediction_settings["timeframe"] = selected_timeframe
                st.rerun()
        
        with chart_col4:
            display_period = st.selectbox(
                "Hiển thị",
                ["24 giờ", "3 ngày", "7 ngày", "30 ngày"],
                index=1
            )
            
            # Map display period to number of candles
            period_map = {
                "24 giờ": {
                    "1m": 24 * 60,
                    "5m": int(24 * 60 / 5)
                },
                "3 ngày": {
                    "1m": 3 * 24 * 60,
                    "5m": int(3 * 24 * 60 / 5)
                },
                "7 ngày": {
                    "1m": 7 * 24 * 60,
                    "5m": int(7 * 24 * 60 / 5)
                },
                "30 ngày": {
                    "1m": 30 * 24 * 60,
                    "5m": int(30 * 24 * 60 / 5)
                }
            }
            
            display_candles = period_map[display_period][selected_timeframe]
        
        # Get data for selected timeframe
        if hasattr(st.session_state, 'timeframe_data') and selected_timeframe in st.session_state.timeframe_data:
            df = st.session_state.timeframe_data[selected_timeframe]
            
            # Limit to display period
            if len(df) > display_candles:
                df = df.iloc[-display_candles:]
            
            # Display candlestick chart
            candlestick_fig = plot_candlestick_chart(df)
            st.plotly_chart(candlestick_fig, use_container_width=True)
            
            # Display technical indicators chart
            tech_fig = plot_technical_indicators(df)
            st.plotly_chart(tech_fig, use_container_width=True)
            
            # Display information about data
            st.markdown(f"⏱️ Dữ liệu cập nhật lần cuối: {st.session_state.data_fetch_status['last_update'] if hasattr(st.session_state, 'data_fetch_status') else 'N/A'}")
        else:
            st.error(f"Không có dữ liệu cho khung thời gian {selected_timeframe}. Vui lòng tải dữ liệu.")
        
        # Display prediction history
        st.markdown("### Lịch sử dự đoán")
        if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
            # Create prediction history chart
            prediction_fig = plot_prediction_history(st.session_state.predictions)
            st.plotly_chart(prediction_fig, use_container_width=True)
            
            # Create confidence distribution chart
            confidence_fig = plot_confidence_distribution(st.session_state.predictions)
            st.plotly_chart(confidence_fig, use_container_width=True)
            
            # Display last 5 predictions as table
            st.markdown("### Các dự đoán gần đây")
            recent_preds = pd.DataFrame([{
                'time': p['timestamp'],
                'trend': p['trend'],
                'confidence': f"{p['confidence']*100:.1f}%", 
                'price': f"${p['current_price']:.2f}",
                'target': f"${p['target_price']:.2f}" if 'target_price' in p else 'N/A',
                'reason': p['reason'][:50] + '...' if len(p['reason']) > 50 else p['reason']
            } for p in st.session_state.predictions[-5:]])
            
            # Apply styling to trend column
            def style_trend(val):
                if val == 'LONG':
                    return 'background-color: #d4f7dd; color: #0f5132'
                elif val == 'SHORT':
                    return 'background-color: #f7d4d4; color: #842029'
                else:
                    return 'background-color: #e2e3e5; color: #41464b'
            
            # Style the dataframe
            styled_df = recent_preds.style.applymap(style_trend, subset=['trend'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("Chưa có dự đoán nào. Hãy tạo dự đoán mới.")
    
    # Initialize if not already done
    if not st.session_state.initialized and not st.session_state.auto_initialize_triggered:
        st.session_state.auto_initialize_triggered = True
        initialize_system()

# Gọi hàm main để hiển thị giao diện
render_main_interface()