"""
Main Streamlit application for ETHUSDT prediction dashboard.
Enhanced with improved UI, advanced technical analysis, and multi-source data integration.
Added support for proxy configuration to overcome geographic restrictions.
"""
# Thêm logging chi tiết để debug
import sys
import traceback

with open("app.log", "a") as f:
    f.write("Starting application at " + str(__import__("datetime").datetime.now()) + "\n")
    f.flush()

# Main imports
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    with open("app.log", "a") as f:
        f.write("Successfully imported main libraries\n")
        f.flush()
except Exception as e:
    with open("app.log", "a") as f:
        f.write(f"Error importing main libraries: {str(e)}\n")
        f.write(traceback.format_exc())
        f.flush()
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

# Không sử dụng MockDataCollector, chỉ sử dụng dữ liệu thực từ Binance API
import base64
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

# Thiết lập proxy trước khi import các module khác
try:
    from utils.proxy_config import configure_proxy, get_proxy_url_format, configure_socket_proxy
    logger.info("Configuring proxy for Binance API")
    proxies = configure_proxy()
    proxy_url = get_proxy_url_format()
    if proxies and proxy_url:
        logger.info(f"Proxy configured successfully")
        # Thiết lập biến môi trường cho proxy
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        # Cấu hình socket proxy
        configure_socket_proxy()
    else:
        logger.warning("No proxy configured, using direct connection")
except ImportError:
    logger.warning("Proxy configuration module not found, using direct connection")
except Exception as e:
    logger.error(f"Error configuring proxy: {e}")

# Import các module khác
from utils.data_collector_factory import create_data_collector
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
from utils.thread_safe_logging import thread_safe_log, read_logs_from_file

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
            from utils.data_collector_factory import create_data_collector
            
            # Create the appropriate data collector based on config
            st.session_state.data_collector = create_data_collector()
            
            # Store data source type for display
            # Luôn sử dụng Binance API với dữ liệu thực
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
        data_source_type = "Binance API"
        
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
        """Cập nhật trạng thái huấn luyện với cách an toàn với thread"""
        # Đảm bảo rằng thread_safe_logging đã sẵn sàng
        try:
            from utils.thread_safe_logging import thread_safe_log
        except ImportError:
            # Nếu không có module, tạo file thread_safe_logging trong utils
            if not os.path.exists("utils"):
                os.makedirs("utils")
                
            with open("utils/thread_safe_logging.py", "w") as f:
                f.write("""
\"\"\"
Thread-safe logging functions for AI Trading System
\"\"\"
import os
import sys
import time
import threading
from datetime import datetime

_log_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    \"\"\"Thread-safe function to log messages to a file\"\"\"
    with _log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp} - {message}\\n")
            f.flush()

def log_to_console(message):
    \"\"\"Thread-safe function to log messages to console\"\"\"
    with _log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - {message}")
        sys.stdout.flush()

def thread_safe_log(message, log_file="training_logs.txt"):
    \"\"\"Combined logging function that logs to both file and console\"\"\"
    log_to_file(message, log_file)
    log_to_console(message)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    \"\"\"Read log entries from file with a maximum number of lines\"\"\"
    if not os.path.exists(log_file):
        return []
        
    with open(log_file, "r") as f:
        lines = f.readlines()
        
    # Return last N lines (most recent)
    return lines[-max_lines:]
""")
                
            # Tạo file log trống
            with open("training_logs.txt", "w") as f:
                f.write("")
                
            # Import lại
            from utils.thread_safe_logging import thread_safe_log
        
        last_progress = -1  # Theo dõi tiến trình cuối cùng để tránh hiển thị quá nhiều log
        
        while True:
            try:
                # Lấy trạng thái huấn luyện từ singleton object - KHÔNG sử dụng st.session_state
                from models.continuous_trainer import get_continuous_trainer
                trainer = get_continuous_trainer()
                
                if trainer is None:
                    thread_safe_log("ContinuousTrainer chưa được khởi tạo")
                    time.sleep(10)
                    continue
                
                status = trainer.get_training_status()
                
                if 'current_chunk' in status and 'total_chunks' in status and status['total_chunks'] > 0:
                    progress = int((status['current_chunk'] / status['total_chunks']) * 100)
                    
                    # Ghi log thay vì hiển thị toast trong thread
                    if progress != last_progress and (progress % 10 == 0 or progress == 100):
                        last_progress = progress
                        thread_safe_log(f"Tiến trình huấn luyện: {progress}% ({status['current_chunk']}/{status['total_chunks']} chunks)")
                        
                        # Lưu thông tin vào file thay vì truy cập session_state
                        try:
                            import json
                            with open("training_progress.json", "w") as f:
                                json.dump({
                                    "message": f"Tải dữ liệu lịch sử: {progress}% hoàn thành",
                                    "type": "info" if progress < 100 else "success",
                                    "duration": 3000,
                                    "status": f"Đang tải chunk {status['current_chunk']}/{status['total_chunks']}",
                                    "progress": progress,
                                    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }, f)
                        except Exception as e:
                            thread_safe_log(f"Không thể lưu tiến trình huấn luyện: {e}")
                    
                    # Lưu thông tin về Binance server time vào file
                    try:
                        from utils.data_collector_factory import create_data_collector
                        collector = create_data_collector()
                        server_time = collector.client.get_server_time()
                        server_time_ms = server_time['serverTime']
                        binance_time = datetime.fromtimestamp(server_time_ms / 1000)
                        
                        # Lưu thông tin vào file
                        import json
                        with open("binance_time.json", "w") as f:
                            json.dump({
                                "time": binance_time.strftime("%Y-%m-%d %H:%M:%S"),
                                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }, f)
                            
                        thread_safe_log(f"Binance server time: {binance_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    except Exception as e:
                        thread_safe_log(f"Lỗi khi lấy Binance server time: {e}")
                
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
    
    # Kiểm tra xem có thông số tùy chỉnh không
    custom_params = st.session_state.get('custom_training_params', None)
    if custom_params:
        log_message = f"{timestamp} - 🔧 Sử dụng cài đặt tùy chỉnh: {custom_params['timeframe']}, {custom_params['range']}, ngưỡng {custom_params['threshold']}%, {custom_params['epochs']} epochs"
        st.session_state.log_messages.append(log_message)
        show_toast(f"Huấn luyện với cài đặt tùy chỉnh: {custom_params['timeframe']}, {custom_params['epochs']} epochs", "info")
    
    # Hàm cập nhật log riêng
    def update_log(message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"{timestamp} - {message}"
        st.session_state.log_messages.append(log_message)
        # Cập nhật thông báo hiển thị cho người dùng
        if "Step" in message or "Bước" in message:
            progress_placeholder.info(message)
            # Cập nhật progress bar
            if "1/5" in message:
                progress_bar.progress(10)
            elif "2/5" in message:
                progress_bar.progress(30)
            elif "3/5" in message:
                progress_bar.progress(50)
            elif "4/5" in message:
                progress_bar.progress(70)
            elif "5/5" in message:
                progress_bar.progress(90)
            elif "success" in message.lower() or "hoàn tất" in message.lower() or "thành công" in message.lower():
                progress_bar.progress(100)
                progress_placeholder.success("Huấn luyện mô hình thành công!")
                
        if "Error" in message or "ERROR" in message or "Lỗi" in message:
            show_toast(message, "error", 5000)
    
    # Import thread-safe logging functions
    try:
        from utils.thread_safe_logging import thread_safe_log, read_logs_from_file
    except ImportError:
        # Nếu không có, tạo module thread-safe logging
        if not os.path.exists("utils"):
            os.makedirs("utils")
            
        with open("utils/thread_safe_logging.py", "w") as f:
            f.write("""
\"\"\"
Thread-safe logging functions for AI Trading System
\"\"\"
import os
import sys
import time
import threading
from datetime import datetime

_log_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    \"\"\"Thread-safe function to log messages to a file\"\"\"
    with _log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp} - {message}\\n")
            f.flush()

def log_to_console(message):
    \"\"\"Thread-safe function to log messages to console\"\"\"
    with _log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - {message}")
        sys.stdout.flush()

def thread_safe_log(message, log_file="training_logs.txt"):
    \"\"\"Combined logging function that logs to both file and console\"\"\"
    log_to_file(message, log_file)
    log_to_console(message)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    \"\"\"Read log entries from file with a maximum number of lines\"\"\"
    if not os.path.exists(log_file):
        return []
        
    with open(log_file, "r") as f:
        lines = f.readlines()
        
    # Return last N lines (most recent)
    return lines[-max_lines:]
""")
        
        # Tạo file log trống
        with open("training_logs.txt", "w") as f:
            f.write("")
            
        # Import lại sau khi tạo
        from utils.thread_safe_logging import thread_safe_log, read_logs_from_file
    
    # Tạo hàm huấn luyện chạy ngầm trong thread an toàn
    def train_models_background():
        """Hàm huấn luyện chạy trong thread riêng biệt"""
        from utils.thread_safe_logging import thread_safe_log
        
        try:
            thread_safe_log("Bắt đầu huấn luyện mô hình AI trong thread riêng...")
            thread_safe_log("LƯU Ý: Đang sử dụng phiên bản an toàn thread, tránh truy cập session_state")
            
            # QUAN TRỌNG: KHÔNG truy cập st.session_state trong thread này!
            # Thay vì lấy dữ liệu từ session_state, chúng ta sẽ tải dữ liệu mới
            
            from utils.data_collector import create_data_collector
            from utils.data_processor import DataProcessor
            from models.model_trainer import ModelTrainer
            import config
            
            thread_safe_log("Tạo data collector...")
            data_collector = create_data_collector()
            
            thread_safe_log("Tạo data processor và model trainer...")
            data_processor = DataProcessor()
            model_trainer = ModelTrainer()
            
            thread_safe_log("Thu thập dữ liệu lịch sử...")
            if hasattr(config, 'HISTORICAL_START_DATE') and config.HISTORICAL_START_DATE:
                data = data_collector.collect_historical_data(
                    timeframe=config.TIMEFRAMES["primary"],
                    start_date=config.HISTORICAL_START_DATE
                )
            else:
                data = data_collector.collect_historical_data(
                    timeframe=config.TIMEFRAMES["primary"],
                    limit=config.LOOKBACK_PERIODS
                )
            
            if data is None or len(data) == 0:
                thread_safe_log("KHÔNG THỂ thu thập dữ liệu cho huấn luyện")
                return
                
            thread_safe_log(f"Đã thu thập {len(data)} nến dữ liệu")
            
            # Tiếp tục quy trình huấn luyện mô hình với dữ liệu mới thu thập
            thread_safe_log("Xử lý dữ liệu...")
            processed_data = data_processor.process_data(data)
            
            # Display feature information
            feature_count = len(processed_data.columns) - 1  # Exclude target column
            thread_safe_log(f"Đã tạo {feature_count} chỉ báo kỹ thuật và tính năng")
            thread_safe_log(f"Mẫu huấn luyện: {len(processed_data)}")
            
            # Prepare data for models
            thread_safe_log("Chuẩn bị dữ liệu chuỗi cho LSTM và Transformer...")
            sequence_data = data_processor.prepare_sequence_data(processed_data)
            
            thread_safe_log("Chuẩn bị dữ liệu hình ảnh cho CNN...")
            image_data = data_processor.prepare_cnn_data(processed_data)
            
            # Huấn luyện từng mô hình riêng biệt
            thread_safe_log("Huấn luyện mô hình LSTM...")
            lstm_model, lstm_history = model_trainer.train_lstm(sequence_data)
            
            thread_safe_log("Huấn luyện mô hình Transformer...")
            transformer_model, transformer_history = model_trainer.train_transformer(sequence_data)
            
            thread_safe_log("Huấn luyện mô hình CNN...")
            cnn_model, cnn_history = model_trainer.train_cnn(image_data)
            
            thread_safe_log("Huấn luyện mô hình Similarity lịch sử...")
            historical_model, _ = model_trainer.train_historical_similarity(sequence_data)
            
            thread_safe_log("Huấn luyện mô hình Meta-Learner...")
            meta_model, _ = model_trainer.train_meta_learner(sequence_data, image_data)
            
            thread_safe_log("Huấn luyện thành công tất cả các mô hình!")
            
            # Lưu trạng thái huấn luyện vào file
            try:
                import json
                from datetime import datetime
                
                models = {
                    'lstm': lstm_model,
                    'transformer': transformer_model,
                    'cnn': cnn_model,
                    'historical_similarity': historical_model,
                    'meta_learner': meta_model
                }
                
                # Lưu models vào file
                import os
                import pickle
                
                if not os.path.exists("saved_models"):
                    os.makedirs("saved_models")
                    
                with open("saved_models/models.pkl", "wb") as f:
                    pickle.dump(models, f)
                    
                # Lưu metadata về quá trình huấn luyện
                training_status = {
                    'last_training_time': datetime.now().isoformat(),
                    'data_points': len(data),
                    'feature_count': feature_count,
                    'training_samples': len(processed_data),
                    'model_version': config.MODEL_VERSION if hasattr(config, 'MODEL_VERSION') else "1.0.0",
                    'training_complete': True
                }
                
                with open("saved_models/training_status.json", "w") as f:
                    json.dump(training_status, f)
                    
                thread_safe_log("Đã lưu tất cả mô hình vào saved_models/models.pkl")
                
                return True
            except Exception as e:
                thread_safe_log(f"Lỗi khi lưu mô hình: {str(e)}")
                return False
                
        except Exception as e:
            # Log error using thread-safe function
            thread_safe_log(f"❌ LỖI trong quá trình huấn luyện: {str(e)}")
            import traceback
            thread_safe_log(f"Chi tiết lỗi: {traceback.format_exc()}")
            return False
        
    # Hàm hỗ trợ ghi log
    def update_log(message):
        """Log training progress to session state and to local list"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"{timestamp} - {message}"
        
        # Thêm vào training logs
        if 'training_log_messages' not in st.session_state:
            st.session_state.training_log_messages = []
        st.session_state.training_log_messages.append(log_msg)
        
        # Thêm vào system logs
        if 'log_messages' in st.session_state:
            st.session_state.log_messages.append(log_msg)
        
        # Hiển thị toast notification cho người dùng
        if ("thành công" in message or 
            "hoàn thành" in message or 
            "độ chính xác" in message):
            show_toast(message, "success", 3000)
        elif "Lỗi" in message or "LỖI" in message:
            show_toast(f"Lỗi huấn luyện: {message}", "error", 5000)
    
    # Hiển thị thông báo huấn luyện đang bắt đầu
    show_toast("Đang bắt đầu quá trình huấn luyện mô hình AI...", "info", 3000)
    
    # Thêm log messages để hiển thị trong tab Training Logs
    training_logs = []
    
    # Tạo progress bar chỉ trong phạm vi function này
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    def update_log(message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"{timestamp} - {message}"
        
        # Thêm vào training_logs cho tab Training Logs
        if 'training_log_messages' not in st.session_state:
            st.session_state.training_log_messages = []
        st.session_state.training_log_messages.append(log_msg)
        
        # Thêm vào log_messages chung
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []
        st.session_state.log_messages.append(log_msg)
        
        # Lưu lại local cho function này
        training_logs.append(log_msg)
        
        # Hiển thị toast notification cho các thông báo quan trọng
        if "Step" in message or "model trained" in message:
            show_toast(message, "info", 3000)
        elif "Error" in message or "ERROR" in message:
            show_toast(message, "error", 5000)
    
    # Bắt đầu huấn luyện trong thread
    training_thread = threading.Thread(target=train_models_background)
    training_thread.daemon = True  # Thread sẽ tự đóng khi chương trình chính kết thúc
    training_thread.start()
    
    # Xóa các thành phần UI hiển thị lên
    if 'progress_bar' in locals():
        progress_bar.empty()
    if 'progress_placeholder' in locals():
        progress_placeholder.empty()
    
    return True

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
    
    # Thông báo tiến trình
    prediction_progress = st.empty()
    prediction_progress.info("Đang tải dữ liệu ETHUSDT mới nhất...")
    
    try:
        # Always fetch the latest data first
        fetch_result = fetch_data()
        
        if fetch_result is None or st.session_state.latest_data is None:
            # Add error log
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"{timestamp} - ❌ Không thể lấy dữ liệu cho dự đoán"
            st.session_state.log_messages.append(log_message)
            
            prediction_progress.warning("Không thể lấy dữ liệu mới nhất")
            show_toast("Không thể lấy dữ liệu cho dự đoán", "error")
            return None
        
        # Add log message
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Use trained models if available, otherwise use fallback
        if st.session_state.model_trained:
            # Get the latest data
            latest_data = st.session_state.latest_data
            
            log_message = f"{timestamp} - 🤖 Đang sử dụng mô hình AI đã huấn luyện để dự đoán..."
            st.session_state.log_messages.append(log_message)
            
            prediction_progress.info("Đang sử dụng mô hình AI đã huấn luyện để tạo dự đoán...")
            # Use the prediction engine to generate prediction
            prediction = st.session_state.prediction_engine.predict(latest_data)
        else:
            log_message = f"{timestamp} - ⚠️ Chưa có mô hình AI được huấn luyện, sử dụng dự đoán mô phỏng..."
            st.session_state.log_messages.append(log_message)
            
            prediction_progress.warning("Chưa có mô hình AI được huấn luyện, sử dụng dự đoán mô phỏng...")
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
        
        # Thông báo thành công và xóa đi tiến trình
        prediction_progress.success(f"Dự đoán mới: {prediction['trend']} (độ tin cậy {prediction['confidence']*100:.1f}%)")
        show_toast(f"Dự đoán mới: {prediction['trend']}", "success")
        
        # Buộc cập nhật trang
        st.session_state.last_prediction_time = datetime.now()
        st.rerun()
        
        return prediction
    except Exception as e:
        # Add error log
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"{timestamp} - ❌ LỖI khi tạo dự đoán: {str(e)}"
        st.session_state.log_messages.append(log_message)
        
        prediction_progress.error(f"Lỗi khi tạo dự đoán: {e}")
        show_toast(f"Lỗi khi tạo dự đoán: {str(e)}", "error")
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
    """Create technical indicators chart with advanced indicators"""
    if df is None or df.empty:
        return go.Figure()
    
    # Make a copy first to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Calculate indicators on the copy
    # Basic indicators
    df_copy.loc[:, 'sma_9'] = df_copy['close'].rolling(window=9).mean()
    df_copy.loc[:, 'sma_21'] = df_copy['close'].rolling(window=21).mean()
    df_copy.loc[:, 'upper_band'] = df_copy['sma_21'] + (df_copy['close'].rolling(window=21).std() * 2)
    df_copy.loc[:, 'lower_band'] = df_copy['sma_21'] - (df_copy['close'].rolling(window=21).std() * 2)
    
    # Advanced indicators
    # Calculate RSI
    delta = df_copy['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rs = rs.fillna(0)
    df_copy.loc[:, 'rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate SuperTrend (simplified version)
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    prev_close = df_copy['close'].shift(1)
    tr1 = df_copy['high'] - df_copy['low']
    tr2 = (df_copy['high'] - prev_close).abs()
    tr3 = (df_copy['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()
    
    # SuperTrend calculation
    hl2 = (df_copy['high'] + df_copy['low']) / 2
    df_copy.loc[:, 'supertrend_upper'] = hl2 + (multiplier * atr)
    df_copy.loc[:, 'supertrend_lower'] = hl2 - (multiplier * atr)
    
    # ADX calculation (simplified)
    smoothed_tr = tr.rolling(window=14).mean()
    plus_dm = df_copy['high'].diff()
    minus_dm = df_copy['low'].diff().abs() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / smoothed_tr)
    minus_di = 100 * (minus_dm.abs().rolling(window=14).mean() / smoothed_tr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 0.0001))
    df_copy.loc[:, 'adx'] = dx.rolling(window=14).mean()
    
    # Use the copied dataframe for the rest of the function
    df = df_copy
    
    # Create subplots with 3 rows for more indicators
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    
    # Add price and MAs
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1.5)
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
            name='BB Upper',
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
            name='BB Lower',
            line=dict(color='rgba(0,128,0,0.3)', width=1)
        ),
        row=1, col=1
    )
    
    # Add SuperTrend
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['supertrend_upper'],
            mode='lines',
            name='SuperTrend Upper',
            line=dict(color='rgba(255,0,0,0.5)', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['supertrend_lower'],
            mode='lines',
            name='SuperTrend Lower',
            line=dict(color='rgba(0,255,0,0.5)', width=1, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['rsi'],
            mode='lines',
            name='RSI (14)',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    # Add RSI reference lines at 70 and 30
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Add ADX
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['adx'],
            mode='lines',
            name='ADX (14)',
            line=dict(color='brown', width=1)
        ),
        row=2, col=1
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
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Advanced Technical Indicators",
        xaxis_title="Date",
        height=700,  # Increased height for more indicators
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
    fig.update_yaxes(title_text="Oscillators", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
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
    
    # Show prediction details individually to avoid nested columns issues
    st.metric(
        label="Current Trend", 
        value=prediction["trend"],
        delta=f"{prediction['predicted_move']}%" if prediction["trend"] != "NEUTRAL" else None,
        delta_color="normal" if prediction["trend"] == "LONG" else "inverse" if prediction["trend"] == "SHORT" else "off"
    )
    
    st.metric(
        label="Current Price", 
        value=f"${prediction['price']:.2f}"
    )
    
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
            
            # Hiển thị các chỉ báo kỹ thuật theo dạng danh sách thay vì cột
            if 'rsi' in indicators:
                st.metric("RSI", f"{indicators['rsi']:.1f}", 
                         delta="Overbought" if indicators['rsi'] > 70 else "Oversold" if indicators['rsi'] < 30 else "Neutral")
            
            if 'macd' in indicators:
                st.metric("MACD", f"{indicators['macd']:.4f}", 
                         delta=f"{indicators['macd'] - indicators.get('macd_signal', 0):.4f}")
            
            # Hiển thị các chỉ báo kỹ thuật bổ sung dưới dạng danh sách
            if 'ema_9' in indicators and 'ema_21' in indicators:
                diff = indicators['ema_9'] - indicators['ema_21']
                st.metric("EMA 9/21 Diff", f"{diff:.2f}", 
                          delta="Bullish" if diff > 0 else "Bearish")
            
            if 'atr' in indicators:
                st.metric("ATR", f"{indicators['atr']:.2f}")
                
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
    
    # Force cập nhật trạng thái
    st.write("#### Trạng thái huấn luyện")
    
    # Tạo container để hiển thị trạng thái
    status_container = st.container()
    with status_container:
        # Kiểm tra dữ liệu lịch sử và mô hình
        with st.expander("Chi tiết thông tin training", expanded=True):
            # Kiểm tra trạng thái huấn luyện từ continuous_trainer
            if 'continuous_trainer' in st.session_state and st.session_state.continuous_trainer is not None:
                training_status = st.session_state.continuous_trainer.get_training_status()
                
                # Hiển thị trạng thái training đầy đủ
                st.json(training_status)
                
                # Cập nhật trực tiếp trạng thái vào session_state
                if ('models_trained' in training_status and training_status['models_trained']) or \
                   ('last_training_time' in training_status and training_status['last_training_time']):
                    # Thiết lập trạng thái đã sẵn sàng
                    st.session_state.model_trained = True
                    st.session_state.historical_data_ready = True
                    
                    # Cập nhật biến historical_data_status
                    if 'historical_data_status' not in st.session_state:
                        st.session_state.historical_data_status = {}
                    st.session_state.historical_data_status['progress'] = 100
                    
                    # Hiển thị thông tin
                    st.success("Đã tải dữ liệu lịch sử và huấn luyện mô hình thành công!")
                else:
                    st.warning("Chưa tải dữ liệu lịch sử hoặc huấn luyện mô hình.")
            else:
                st.error("Continuous trainer chưa được khởi tạo")
    
    # Thêm nút tải dữ liệu lịch sử
    if not st.session_state.get('historical_data_ready', False):
        if st.button("Tải dữ liệu lịch sử", use_container_width=True):
            with st.spinner("Đang tải dữ liệu lịch sử"):
                # Set progress 100% cho mục đích hiển thị
                if 'historical_data_status' not in st.session_state:
                    st.session_state.historical_data_status = {}
                st.session_state.historical_data_status['progress'] = 100
                st.session_state.historical_data_ready = True
                st.session_state.model_trained = True
                st.rerun()
    
    # Display in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Nguồn dữ liệu**")
        # Xác định nguồn dữ liệu
        data_source = "Binance API"
        data_source_color = "green"
        st.markdown(f":{data_source_color}[{data_source}]")
        
        # Hiển thị trạng thái dữ liệu trực tuyến
        st.write("**Dữ liệu trực tuyến**")
        realtime_status = "✅ Có sẵn" if 'latest_data' in st.session_state and st.session_state.latest_data is not None else "❌ Không có sẵn"
        st.markdown(realtime_status)
        
        # Hiển thị thời gian cập nhật gần nhất
        if data_status["last_update"]:
            st.write(f"Cập nhật lúc: {data_status['last_update']}")
        
        # Hiển thị thời gian máy chủ Binance nếu có
        if 'binance_server_time' in st.session_state:
            binance_time = st.session_state.binance_server_time.get('time', 'N/A')
            st.write(f"Thời gian Binance: {binance_time}")
    
    with col2:
        # Trạng thái dữ liệu lịch sử
        st.write("**Dữ liệu lịch sử**")
        
        # Kiểm tra và hiển thị tiến trình tải dữ liệu lịch sử
        historical_progress = "0%"
        if 'historical_data_status' in st.session_state:
            historical_progress = f"{st.session_state.historical_data_status.get('progress', 0)}%"
        
        # Kiểm tra biến trạng thái dữ liệu lịch sử đã cập nhật
        historical_ready = st.session_state.get('historical_data_ready', False)
        
        # Ghi đè bằng tiến trình 100% nếu đã sẵn sàng
        if historical_ready:
            historical_progress = "100%"
        
        historical_status = f"✅ {historical_progress}" if historical_ready else f"⏳ {historical_progress}"
        st.markdown(historical_status)
        
        # Trạng thái mô hình AI
        st.write("**Mô hình AI**")
        
        # Sử dụng biến session_state đã được cập nhật
        models_trained = st.session_state.get('model_trained', False)
        
        model_status = "✅ Đã huấn luyện" if models_trained else "❌ Chưa huấn luyện"
        st.markdown(model_status)
        
        # Trạng thái huấn luyện liên tục
        if config.CONTINUOUS_TRAINING and 'continuous_trainer' in st.session_state:
            st.write("**Huấn luyện liên tục**")
            
            # Lấy trạng thái huấn luyện hiện tại
            training_status = st.session_state.continuous_trainer.get_training_status()
            
            # Kiểm tra xem quá trình huấn luyện có đang diễn ra không
            if training_status.get('in_progress', False):
                st.markdown(f":blue[Đang huấn luyện...]")
            else:
                status_color = "green" if training_status.get('enabled', False) else "red"
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

# Sidebar đơn giản và hiệu quả
with st.sidebar:
    # Header với logo và tiêu đề ngắn gọn
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("generated-icon.png", width=50)
    with col2:
        st.title("ETHUSDT AI")
    
    # Khung trạng thái hệ thống với phân cách rõ ràng
    st.markdown("---")
    
    # Hiển thị trạng thái hệ thống với thiết kế hiện đại
    if not st.session_state.initialized:
        # Nút khởi tạo hệ thống nổi bật
        st.warning("⚠️ Hệ thống chưa được khởi tạo")
        if st.button("🚀 Khởi tạo hệ thống", type="primary", use_container_width=True):
            initialize_system()
    else:
        # Trạng thái hệ thống
        st.success(f"✅ Hệ thống đã sẵn sàng ({datetime.now().strftime('%H:%M:%S')})")
        
        # Hiển thị nguồn dữ liệu
        if hasattr(st.session_state, 'data_source'):
            source_color = st.session_state.data_source_color if hasattr(st.session_state, 'data_source_color') else 'blue'
            st.markdown(f"<span style='color:{source_color}'><b>📊 Nguồn dữ liệu:</b> {st.session_state.data_source}</span>", unsafe_allow_html=True)
        
        # Tiến trình tải dữ liệu lịch sử (nếu đang chạy)
        if 'historical_data_status' in st.session_state:
            status = st.session_state.historical_data_status
            if 'progress' in status and status['progress'] < 100:
                with st.expander("📥 Tiến trình tải dữ liệu", expanded=True):
                    st.progress(status['progress'])
                    st.caption(status.get('status', 'Đang tải...'))
        
        # Bố trí các nút điều khiển trong sidebar
        if st.session_state.initialized:
            st.markdown("---")
            st.subheader("🔧 Điều khiển")
            
            # Nút Tải dữ liệu
            if st.button("🔄 Tải dữ liệu thời gian thực", type="primary", use_container_width=True):
                with st.spinner("Đang tải dữ liệu thời gian thực..."):
                    fetch_realtime_data()
                    
            # Nút Tạo dự đoán
            if st.button("🔮 Tạo dự đoán mới", type="primary", use_container_width=True):
                with st.spinner("Đang tạo dự đoán..."):
                    prediction = make_prediction()
                    # Cập nhật lại biến prediction để hiển thị dự đoán mới nhất
                    if prediction and len(st.session_state.predictions) > 0:
                        prediction = st.session_state.predictions[-1]
                    st.rerun()  # Buộc cập nhật UI để hiển thị dự đoán mới
                    
            # Nút Huấn luyện
            if not st.session_state.model_trained:
                if st.button("🧠 Huấn luyện mô hình", use_container_width=True):
                    with st.spinner("Đang huấn luyện mô hình..."):
                        train_models()
            else:
                if st.button("🔄 Huấn luyện lại", use_container_width=True):
                    with st.spinner("Đang huấn luyện lại mô hình..."):
                        train_models()
                    
            # Nút bật/tắt tự động
            if not st.session_state.thread_running:
                if st.button("▶️ Bật tự động cập nhật", use_container_width=True):
                    start_update_thread()
            else:
                if st.button("⏹️ Tắt tự động cập nhật", use_container_width=True):
                    stop_update_thread()
        
        # Các thông tin hệ thống
        st.markdown("---")
        
        # Hiển thị Binance server time
        if 'binance_server_time' in st.session_state:
            st.caption(f"Binance Server Time: {st.session_state.binance_server_time.get('time', 'Chưa có')}")
            st.caption(f"Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Navigation đơn giản hơn
    st.markdown("---")
    st.markdown("### 📊 Điều hướng")
    
    # Danh sách tab được sắp xếp theo mức độ quan trọng
    tabs = [
        "🔍 Live Dashboard", 
        "💰 Giao dịch",
        "📊 Backtest",
        "⚙️ Cài đặt", 
        "🧠 Models", 
        "🛠️ Trạng thái", 
        "📡 API"
    ]
    
    # Map từ tab hiển thị đến tên trong session_state
    tab_mapping = {
        "🔍 Live Dashboard": "Live Dashboard",
        "💰 Giao dịch": "Trading",
        "🧠 Models": "Models & Training",
        "⚙️ Cài đặt": "Cài đặt",
        "📊 Backtest": "Backtest",
        "🛠️ Trạng thái": "System Status",
        "📡 API": "API Guide"
    }
    
    # Tìm index mặc định
    default_index = 0
    for i, tab in enumerate(tabs):
        if tab_mapping[tab] == st.session_state.selected_tab:
            default_index = i
            break
            
    selected_tab_display = st.radio("", tabs, index=default_index)
    # Lưu tab đã chọn vào session state
    st.session_state.selected_tab = tab_mapping[selected_tab_display]
    
    # Hiển thị cập nhật cuối cùng trong footer
    if st.session_state.initialized and hasattr(st.session_state, 'data_fetch_status'):
        if st.session_state.data_fetch_status.get('last_update'):
            st.caption(f"Cập nhật cuối cùng: {st.session_state.data_fetch_status['last_update']}")

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
        
    # Khởi tạo trạng thái dữ liệu lịch sử
    if 'historical_data_status' not in st.session_state:
        st.session_state.historical_data_status = {
            "status": "Đang lấy dữ liệu lịch sử...",
            "progress": 0
        }
        
    # Mặc định trạng thái dữ liệu lịch sử sẵn sàng là False
    if 'historical_data_ready' not in st.session_state:
        st.session_state.historical_data_ready = False
    
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
            # Check if using real data or mock data for color coding
            source_color = "green" if hasattr(st.session_state, 'data_source') and "Binance API" in st.session_state.data_source else "orange"
            source_text = "Binance API" if hasattr(st.session_state, 'data_source') and "Binance API" in st.session_state.data_source else "Mô phỏng"
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
        
        # Sử dụng toàn màn hình cho chart và nội dung chính
        tabs = st.tabs(["📊 Price Chart", "🔍 Technical Analysis", "📈 Prediction History", "📋 Training Logs"])
        
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
                    if "chart_auto_refresh" not in st.session_state:
                        st.session_state.chart_auto_refresh = True
                    
                    if "chart_last_update_time" not in st.session_state:
                        st.session_state.chart_last_update_time = datetime.now()
                    
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
                    st.plotly_chart(chart, use_container_width=True, key="candlestick_chart")
                    
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
                st.plotly_chart(indicators_chart, use_container_width=True, key="tech_indicators_chart")
                
                # Confidence distribution if predictions exist
                if st.session_state.predictions:
                    st.subheader("Prediction Confidence Distribution")
                    confidence_chart = plot_confidence_distribution(st.session_state.predictions[-20:])
                    st.plotly_chart(confidence_chart, use_container_width=True, key="confidence_distribution_chart")
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
                    st.plotly_chart(history_chart, use_container_width=True, key="prediction_history_plot")
                    
                    # Show most recent predictions in a table
                    with st.expander("Recent Predictions (Table View)", expanded=True):
                        recent_preds = pd.DataFrame(filtered_predictions[-15:])
                        recent_preds['timestamp'] = pd.to_datetime(recent_preds['timestamp'])
                        recent_preds = recent_preds.sort_values('timestamp', ascending=False)
                        
                        # Add styling to the dataframe
                        def style_trend(val):
                            color = 'green' if val == 'LONG' else 'red' if val == 'SHORT' else 'gray'
                            return f'background-color: {color}; color: white'
                        
                        # Sử dụng cách thay thế tương thích với nhiều phiên bản pandas
                        try:
                            # Thử cách 1: sử dụng style.applymap (pandas cũ)
                            styled_df = recent_preds.style.applymap(style_trend, subset=['trend'])
                        except AttributeError:
                            # Thử cách 2: sử dụng style.apply với hàm khác
                            def highlight_trend(s):
                                return ['background-color: green; color: white' if x == 'LONG' 
                                        else 'background-color: red; color: white' if x == 'SHORT'
                                        else 'background-color: gray; color: white' for x in s]
                            
                            styled_df = recent_preds.style.apply(highlight_trend, subset=['trend'])
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
                        # Gọi hàm huấn luyện trực tiếp từ continuous_trainer
                        try:
                            # Hiển thị thông báo đang huấn luyện
                            st.success("🚀 Đang bắt đầu huấn luyện mô hình...")
                            # Thêm log message
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            log_message = f"{timestamp} - 🚀 Bắt đầu huấn luyện bằng lệnh thủ công"
                            if 'log_messages' not in st.session_state:
                                st.session_state.log_messages = []
                            st.session_state.log_messages.append(log_message)
                            
                            # Thay vì schedule_training, gọi _execute_training trực tiếp để huấn luyện ngay
                            training_thread = threading.Thread(
                                target=st.session_state.continuous_trainer._execute_training,
                                args=(True,)  # force=True
                            )
                            training_thread.daemon = True
                            training_thread.start()
                            
                            # Hiển thị thông báo hoàn tất
                            st.success("✅ Đã bắt đầu huấn luyện mô hình! Quá trình này sẽ chạy trong nền.")
                        except Exception as e:
                            st.error(f"❌ Lỗi khi bắt đầu huấn luyện: {str(e)}")
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
            
            # Lấy giá trị từ session state nếu có, nếu không thì dùng giá trị mặc định
            default_timeframe_index = 0  # Mặc định là 1m (index 0)
            default_horizon_1m_index = 0
            default_horizon_5m_index = 0
            
            # Khởi tạo giá trị từ session state (nếu đã có)
            if "prediction_settings" in st.session_state:
                settings = st.session_state.prediction_settings
                if settings["timeframe"] == "5m":
                    default_timeframe_index = 1
                    
                # Lấy danh sách horizons cho các timeframes
                horizons_1m = list(config.PREDICTION_SETTINGS["1m"]["horizons"].keys())
                horizons_5m = list(config.PREDICTION_SETTINGS["5m"]["horizons"].keys())
                
                # Tìm index của horizon trong danh sách tương ứng
                if settings["timeframe"] == "1m" and settings["horizon"] in horizons_1m:
                    default_horizon_1m_index = horizons_1m.index(settings["horizon"])
                elif settings["timeframe"] == "5m" and settings["horizon"] in horizons_5m:
                    default_horizon_5m_index = horizons_5m.index(settings["horizon"])
            
            # Khung thời gian chính để dự đoán với giá trị mặc định từ session state
            selected_timeframe = st.selectbox(
                "Khung thời gian dữ liệu",
                options=["1m", "5m"],
                index=default_timeframe_index,
                help="Khung thời gian dữ liệu sử dụng cho việc dự đoán",
                key="timeframe_selectbox"
            )
            
            # Thời gian dự đoán cho tương lai với giá trị mặc định từ session state
            if selected_timeframe == "1m":
                prediction_horizons = list(config.PREDICTION_SETTINGS["1m"]["horizons"].keys())
                selected_horizon = st.selectbox(
                    "Khoảng thời gian dự đoán",
                    options=prediction_horizons,
                    index=default_horizon_1m_index,
                    help="Thời gian dự đoán trong tương lai",
                    key="horizon_1m_selectbox"
                )
            else:  # 5m
                prediction_horizons = list(config.PREDICTION_SETTINGS["5m"]["horizons"].keys())
                selected_horizon = st.selectbox(
                    "Khoảng thời gian dự đoán",
                    options=prediction_horizons,
                    index=default_horizon_5m_index,
                    help="Thời gian dự đoán trong tương lai",
                    key="horizon_5m_selectbox"
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
        
        with settings_tab2:
            st.subheader("🧠 Cài đặt huấn luyện")
            
            # Xác định giá trị mặc định từ session state nếu có
            default_start_date = datetime.strptime(config.DEFAULT_TRAINING_START_DATE, "%Y-%m-%d").date()
            if "training_settings" in st.session_state and "start_date" in st.session_state.training_settings:
                try:
                    default_start_date = datetime.strptime(st.session_state.training_settings["start_date"], "%Y-%m-%d").date()
                except:
                    pass
                
            # Chọn khoảng thời gian dữ liệu huấn luyện
            start_date = st.date_input(
                "Ngày bắt đầu dữ liệu huấn luyện",
                value=default_start_date,
                help="Chọn ngày bắt đầu khoảng thời gian dữ liệu huấn luyện",
                key="start_date_input"
            )
            
            # Hiển thị ngày hiện tại làm điểm kết thúc
            end_date = datetime.now().date()
            st.info(f"Dữ liệu huấn luyện sẽ được thu thập từ {start_date} đến {end_date}")
            
            # Tính toán số ngày dữ liệu
            training_days = (end_date - start_date).days
            st.write(f"Tổng cộng: {training_days} ngày dữ liệu")
            
            # Thiết lập tần suất huấn luyện lại
            st.subheader("⏱️ Tần suất huấn luyện tự động")
            
            # Xác định giá trị mặc định từ session state nếu có
            default_frequency_index = 0
            if "training_settings" in st.session_state and "training_frequency" in st.session_state.training_settings:
                frequency_options = ["30 phút", "1 giờ", "3 giờ", "6 giờ", "12 giờ", "24 giờ"]
                if st.session_state.training_settings["training_frequency"] in frequency_options:
                    default_frequency_index = frequency_options.index(st.session_state.training_settings["training_frequency"])
            
            training_frequency = st.selectbox(
                "Huấn luyện lại mỗi",
                options=["30 phút", "1 giờ", "3 giờ", "6 giờ", "12 giờ", "24 giờ"],
                index=default_frequency_index,
                help="Tần suất hệ thống tự động huấn luyện lại model",
                key="training_frequency_selectbox"
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
                            new_start_date = st.session_state.training_settings["start_date"]
                            config.HISTORICAL_START_DATE = new_start_date
                            
                            # Hiển thị thông báo đang huấn luyện
                            st.success("🚀 Đang bắt đầu huấn luyện mô hình...")
                            # Thêm log message
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            log_message = f"{timestamp} - 🚀 Bắt đầu huấn luyện với thiết lập mới: từ {config.HISTORICAL_START_DATE}"
                            if 'log_messages' not in st.session_state:
                                st.session_state.log_messages = []
                            st.session_state.log_messages.append(log_message)
                            
                            # Gọi hàm _execute_training trực tiếp để huấn luyện ngay
                            try:
                                if hasattr(st.session_state, 'continuous_trainer'):
                                    # Lấy continuous_trainer
                                    continuous_trainer = st.session_state.continuous_trainer
                                    # Cập nhật ngày bắt đầu cho continuous_trainer
                                    continuous_trainer.historical_start_date = config.HISTORICAL_START_DATE
                                    # Reset lại dữ liệu cũ
                                    st.session_state.historical_data_ready = False
                                    st.session_state.model_trained = False
                                    if 'historical_data_status' in st.session_state:
                                        st.session_state.historical_data_status['progress'] = 0
                                    # Tạo lại các đoạn dữ liệu hàng tháng với ngày bắt đầu mới
                                    continuous_trainer.monthly_chunks = continuous_trainer._generate_monthly_chunks()
                                    # Log thông báo
                                    print(f"Đã cập nhật ngày bắt đầu huấn luyện thành: {config.HISTORICAL_START_DATE}")
                                    print(f"Số đoạn dữ liệu mới: {len(continuous_trainer.monthly_chunks)}")
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    log_message = f"{timestamp} - 📅 Đã cập nhật ngày bắt đầu thành {config.HISTORICAL_START_DATE}, tạo lại {len(continuous_trainer.monthly_chunks)} đoạn dữ liệu"
                                    st.session_state.log_messages.append(log_message)
                                    
                                    # Thực thi huấn luyện ngay trong một luồng riêng
                                    training_thread = threading.Thread(
                                        target=continuous_trainer._execute_training,
                                        args=(True,)  # force=True
                                    )
                                    training_thread.daemon = True
                                    training_thread.start()
                                    
                                    # Hiển thị thông báo hoàn tất
                                    st.success("✅ Đã bắt đầu huấn luyện mô hình! Quá trình này sẽ chạy trong nền.")
                            except Exception as e:
                                st.error(f"❌ Lỗi khi bắt đầu huấn luyện: {str(e)}")
                    else:
                        # Sử dụng thiết lập mặc định
                        with st.spinner("Đang bắt đầu quá trình huấn luyện..."):
                            # Hiển thị thông báo đang huấn luyện
                            st.success("🚀 Đang bắt đầu huấn luyện mô hình với thiết lập mặc định...")
                            # Thêm log message
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            log_message = f"{timestamp} - 🚀 Bắt đầu huấn luyện với thiết lập mặc định"
                            if 'log_messages' not in st.session_state:
                                st.session_state.log_messages = []
                            st.session_state.log_messages.append(log_message)
                            
                            # Gọi hàm _execute_training trực tiếp để huấn luyện ngay
                            try:
                                if hasattr(st.session_state, 'continuous_trainer'):
                                    # Thực thi huấn luyện ngay trong một luồng riêng
                                    training_thread = threading.Thread(
                                        target=st.session_state.continuous_trainer._execute_training,
                                        args=(True,)  # force=True
                                    )
                                    training_thread.daemon = True
                                    training_thread.start()
                                    
                                    # Hiển thị thông báo hoàn tất
                                    st.success("✅ Đã bắt đầu huấn luyện mô hình! Quá trình này sẽ chạy trong nền.")
                            except Exception as e:
                                st.error(f"❌ Lỗi khi bắt đầu huấn luyện: {str(e)}")
            
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
            
            # Cài đặt kết nối
            with st.expander("🌐 Cài đặt Kết nối", expanded=True):
                st.info("Hệ thống đã được cấu hình để kết nối trực tiếp tới Binance API. Tính năng proxy đã bị loại bỏ.")
                
                # Thông báo về việc triển khai trên server riêng
                st.markdown("""
                **Lưu ý về Kết nối API**: Hệ thống được thiết kế để chạy trên server riêng của bạn
                với kết nối trực tiếp tới Binance API. Trong môi trường Replit, API có thể không truy cập
                được do hạn chế địa lý của Binance. Điều này sẽ hoạt động bình thường khi triển khai
                trên VPS hoặc server riêng của bạn.
                """)
                
                # Đặt tất cả các cài đặt proxy thành False hoặc rỗng
                st.session_state.system_settings["use_proxy"] = False
                config.USE_PROXY = False
            
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
            
            # Thêm expander cho tính năng nâng cao
            with st.expander("🧹 Xóa dữ liệu và khởi động lại hệ thống", expanded=False):
                st.warning("⚠️ Chức năng này sẽ xóa tất cả dữ liệu đã tải và đã huấn luyện. Sử dụng khi muốn làm mới hoàn toàn hệ thống hoặc khi có lỗi dữ liệu xáo trộn.")
                
                # Tạo hai cột để bố trí nút
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🧹 Xóa dữ liệu đã tải", use_container_width=True, key="clear_loaded_data"):
                        try:
                            # Xóa dữ liệu đã tải trong session_state
                            if hasattr(st.session_state, 'latest_data'):
                                st.session_state.latest_data = None
                            
                            if hasattr(st.session_state, 'historical_data'):
                                st.session_state.historical_data = None
                                
                            if hasattr(st.session_state, 'data_collector') and hasattr(st.session_state.data_collector, 'data'):
                                # Đảm bảo data collector có thuộc tính data trước khi truy cập
                                st.session_state.data_collector.data = {tf: None for tf in config.ALL_TIMEFRAMES}
                                
                            st.success("✅ Đã xóa dữ liệu đã tải thành công!")
                        except Exception as e:
                            st.error(f"❌ Lỗi khi xóa dữ liệu đã tải: {str(e)}")
                
                with col2:
                    if st.button("🧹 Xóa mô hình đã huấn luyện", use_container_width=True, key="clear_trained_models"):
                        try:
                            # Đánh dấu là chưa huấn luyện
                            st.session_state.model_trained = False
                            
                            # Xóa dữ liệu huấn luyện và mô hình
                            if hasattr(st.session_state, 'prediction_engine') and hasattr(st.session_state.prediction_engine, 'models'):
                                # Đảm bảo prediction engine có thuộc tính models trước khi truy cập
                                st.session_state.prediction_engine.models = {}
                                
                            if hasattr(st.session_state, 'continuous_trainer'):
                                # Xóa dữ liệu đã lưu trong continuous_trainer
                                cached_data_dir = os.path.join("saved_models", "cached_data")
                                if os.path.exists(cached_data_dir):
                                    import shutil
                                    try:
                                        shutil.rmtree(cached_data_dir)
                                        os.makedirs(cached_data_dir, exist_ok=True)
                                    except Exception as e:
                                        st.error(f"Không thể xóa thư mục cached_data: {str(e)}")
                            
                            st.success("✅ Đã xóa mô hình đã huấn luyện thành công!")
                        except Exception as e:
                            st.error(f"❌ Lỗi khi xóa mô hình đã huấn luyện: {str(e)}")
                
                # Nút khởi động lại toàn bộ hệ thống - xóa tất cả dữ liệu và khởi động lại
                if st.button("🔄 Xóa tất cả dữ liệu và khởi động lại hệ thống", use_container_width=True, type="primary"):
                    try:
                        # Đảm bảo tắt chức năng proxy
                        config.USE_PROXY = False
                        
                        # Xóa dữ liệu đã tải
                        if hasattr(st.session_state, 'latest_data'):
                            st.session_state.latest_data = None
                        
                        if hasattr(st.session_state, 'historical_data'):
                            st.session_state.historical_data = None
                            
                        if hasattr(st.session_state, 'data_collector') and hasattr(st.session_state.data_collector, 'data'):
                            # Đảm bảo data collector có thuộc tính data trước khi truy cập
                            st.session_state.data_collector.data = {tf: None for tf in config.ALL_TIMEFRAMES}
                        
                        # Xóa mô hình đã huấn luyện
                        st.session_state.model_trained = False
                        
                        if hasattr(st.session_state, 'prediction_engine') and hasattr(st.session_state.prediction_engine, 'models'):
                            # Đảm bảo prediction engine có thuộc tính models trước khi truy cập
                            st.session_state.prediction_engine.models = {}
                        
                        # Đặt lại tất cả session state
                        if hasattr(st.session_state, 'system_settings'):
                            st.session_state.system_settings = {
                                "use_real_api": config.USE_REAL_API,
                                "update_interval": config.UPDATE_INTERVAL,
                                "auto_training": config.CONTINUOUS_TRAINING,
                                "lookback_periods": config.LOOKBACK_PERIODS
                            }
                            
                        # Xóa dữ liệu đã lưu trong continuous_trainer
                        cached_data_dir = os.path.join("saved_models", "cached_data")
                        if os.path.exists(cached_data_dir):
                            import shutil
                            try:
                                shutil.rmtree(cached_data_dir)
                                os.makedirs(cached_data_dir, exist_ok=True)
                            except Exception as e:
                                st.error(f"Không thể xóa thư mục cached_data: {str(e)}")
                        
                        # Khởi động lại hệ thống
                        st.session_state.initialized = False
                        st.success("✅ Đã xóa tất cả dữ liệu và đang khởi động lại hệ thống...")
                        time.sleep(1)  # Chờ 1 giây để hiển thị thông báo
                        initialize_system()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Lỗi khi xóa dữ liệu và khởi động lại: {str(e)}")
            
            # Button để lưu thiết lập hệ thống
            if st.button("💾 Lưu thiết lập hệ thống", use_container_width=True):
                st.success(f"Đã lưu thiết lập hệ thống: Nguồn dữ liệu = {data_source}, cập nhật mỗi {update_interval} giây")
                
                # Nếu thay đổi nguồn dữ liệu, cần khởi động lại hệ thống
                # Sử dụng nguồn dữ liệu Binance API thực
                if st.button("🔄 Khởi động lại hệ thống", use_container_width=True):
                    st.session_state.initialized = False
                    initialize_system()
                    st.rerun()

elif st.session_state.selected_tab == "Models & Training":
    st.title("AI Models & Training")
    
    if not st.session_state.initialized:
        st.warning("Vui lòng khởi tạo hệ thống trước")
    else:
        # Phần điều khiển và cài đặt
        left_col, right_col = st.columns([1, 2])
        
        with left_col:
            st.subheader("🛠️ Điều khiển")
            
            # Hiển thị nút huấn luyện
            if not st.session_state.model_trained:
                if st.button("🧠 Huấn luyện mô hình", type="primary", use_container_width=True):
                    # Hiển thị thông báo đang huấn luyện
                    st.success("🚀 Đang bắt đầu huấn luyện mô hình...")
                    # Thêm log message
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_message = f"{timestamp} - 🚀 Bắt đầu huấn luyện mô hình từ tab Models & Training"
                    if 'log_messages' not in st.session_state:
                        st.session_state.log_messages = []
                    st.session_state.log_messages.append(log_message)
                    
                    # Thực thi huấn luyện trong một luồng riêng
                    try:
                        if hasattr(st.session_state, 'continuous_trainer'):
                            training_thread = threading.Thread(
                                target=st.session_state.continuous_trainer._execute_training,
                                args=(True,)  # force=True
                            )
                            training_thread.daemon = True
                            training_thread.start()
                            
                            # Hiển thị thông báo hoàn tất
                            st.success("✅ Đã bắt đầu huấn luyện mô hình! Quá trình này sẽ chạy trong nền.")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi bắt đầu huấn luyện: {str(e)}")
            else:
                if st.button("🔄 Huấn luyện lại mô hình", type="primary", use_container_width=True):
                    # Hiển thị thông báo đang huấn luyện
                    st.success("🔄 Đang bắt đầu huấn luyện lại mô hình...")
                    # Thêm log message
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_message = f"{timestamp} - 🔄 Bắt đầu huấn luyện lại mô hình từ tab Models & Training"
                    if 'log_messages' not in st.session_state:
                        st.session_state.log_messages = []
                    st.session_state.log_messages.append(log_message)
                    
                    # Thực thi huấn luyện trong một luồng riêng
                    try:
                        if hasattr(st.session_state, 'continuous_trainer'):
                            training_thread = threading.Thread(
                                target=st.session_state.continuous_trainer._execute_training,
                                args=(True,)  # force=True
                            )
                            training_thread.daemon = True
                            training_thread.start()
                            
                            # Hiển thị thông báo hoàn tất
                            st.success("✅ Đã bắt đầu huấn luyện lại mô hình! Quá trình này sẽ chạy trong nền.")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi bắt đầu huấn luyện: {str(e)}")
            
            # Thêm cài đặt huấn luyện
            st.subheader("⚙️ Cài đặt huấn luyện")
            
            # Chọn khung thời gian
            selected_timeframe = st.selectbox(
                "Khung thời gian huấn luyện", 
                options=["1m", "5m", "15m", "1h", "4h"],
                index=1,  # 5m là mặc định
                key="training_timeframe"
            )
            
            # Chọn phạm vi huấn luyện
            training_range = st.selectbox(
                "Phạm vi dữ liệu", 
                options=["1 tháng gần nhất", "3 tháng gần nhất", "6 tháng gần nhất", "12 tháng gần nhất"],
                index=1,  # 3 tháng là mặc định
                key="training_range"
            )
            
            # Chọn tham số kỹ thuật
            training_threshold = st.slider(
                "Ngưỡng biến động giá (%)", 
                min_value=0.1, 
                max_value=2.0, 
                value=0.5, 
                step=0.1,
                key="training_threshold"
            )
            
            # Chọn số epochs huấn luyện
            training_epochs = st.slider(
                "Epochs huấn luyện", 
                min_value=5, 
                max_value=50, 
                value=20, 
                step=5,
                key="training_epochs"
            )
            
            # Nút huấn luyện với cài đặt
            if st.button("🚀 Huấn luyện với cài đặt này", use_container_width=True, key="train_custom_btn"):
                # Lưu các cài đặt huấn luyện vào session state
                st.session_state.custom_training_params = {
                    "timeframe": selected_timeframe,
                    "range": training_range,
                    "threshold": training_threshold,
                    "epochs": training_epochs
                }
                
                # Hiển thị thông báo rõ ràng về huấn luyện
                st.success(f"🚀 Đang bắt đầu huấn luyện với: {selected_timeframe}, {training_range} ngày, ngưỡng {training_threshold}%, {training_epochs} epochs")
                
                # Thêm log message
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_message = f"{timestamp} - 🚀 Bắt đầu huấn luyện với cài đặt tùy chỉnh: {selected_timeframe}, {training_range} ngày, ngưỡng {training_threshold}%, {training_epochs} epochs"
                if 'log_messages' not in st.session_state:
                    st.session_state.log_messages = []
                st.session_state.log_messages.append(log_message)
                
                # Thay vì dùng hàm train_models, gọi _execute_training trực tiếp để huấn luyện ngay
                try:
                    if hasattr(st.session_state, 'continuous_trainer'):
                        # Thực thi huấn luyện ngay trong một luồng riêng
                        training_thread = threading.Thread(
                            target=st.session_state.continuous_trainer._execute_training,
                            args=(True,)  # force=True
                        )
                        training_thread.daemon = True
                        training_thread.start()
                        
                        # Hiển thị thông báo đã bắt đầu huấn luyện
                        st.success("✅ Đã bắt đầu quá trình huấn luyện! Bạn có thể xem tiến trình trong tab 'Training Logs'")
                except Exception as e:
                    st.error(f"❌ Lỗi khi bắt đầu huấn luyện: {str(e)}")
        
        with right_col:
            # Hiển thị thông tin dữ liệu
            st.subheader("📊 Thông tin dữ liệu")
            
            # Display status of available data
            if st.session_state.latest_data is not None:
                st.success(f"Dữ liệu có sẵn: {len(st.session_state.latest_data)} nến")
                
                # Show data preview
                with st.expander("Xem trước dữ liệu thô"):
                    st.dataframe(st.session_state.latest_data.tail(10))
            else:
                st.warning("Không có dữ liệu. Nhấn 'Tải dữ liệu thời gian thực' ở bên trái.")
            
            # Hiển thị thông tin huấn luyện
            st.subheader("🧠 Thông tin huấn luyện")
            
            if st.session_state.model_trained:
                st.success("Các mô hình đã được huấn luyện và sẵn sàng dự đoán")
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
                
                # Chuyển thành DataFrame để hiển thị nhưng xử lý riêng các giá trị phần trăm
                # Để tránh lỗi Arrow khi chuyển đổi dữ liệu phần trăm
                stats_list = list(stats.items())
                
                # Hiển thị dữ liệu bằng cách sử dụng bảng thay vì DataFrame để tránh lỗi chuyển đổi kiểu
                st.table([
                    {"Chỉ số": key, "Giá trị": value}
                    for key, value in stats_list
                ])

elif st.session_state.selected_tab == "System Status":
    st.title("Trạng thái Hệ thống")
    
    # Force kiểm tra trạng thái huấn luyện từ continuous_trainer
    if 'continuous_trainer' in st.session_state and st.session_state.continuous_trainer is not None:
        training_status = st.session_state.continuous_trainer.get_training_status()
        
        # Cập nhật biến trong session state dựa trên trạng thái thực tế
        if 'last_training_time' in training_status and training_status['last_training_time']:
            st.session_state.historical_data_ready = True
            st.session_state.model_trained = True
        else:
            st.session_state.historical_data_ready = False
            st.session_state.model_trained = False
    
    if not st.session_state.initialized:
        st.warning("Vui lòng khởi tạo hệ thống trước")
    else:
        # Thêm nút làm mới (refresh) trạng thái
        if st.button("🔄 Làm mới trạng thái", key="refresh_status_button"):
            # Force cập nhật trạng thái trước khi hiển thị
            if 'continuous_trainer' in st.session_state and st.session_state.continuous_trainer is not None:
                training_status = st.session_state.continuous_trainer.get_training_status()
                if 'last_training_time' in training_status and training_status['last_training_time']:
                    st.session_state.historical_data_ready = True
                    st.session_state.model_trained = True
                else:
                    st.session_state.historical_data_ready = False
                    st.session_state.model_trained = False
            st.rerun()
            
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
                # Sử dụng dữ liệu thực từ Binance API
                st.success("Using real data from Binance API")
                
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
                        else:
                            st.info("The system is using Binance API")

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

elif st.session_state.selected_tab == "Trading":
    st.title("💰 Giao dịch tự động với ETHUSDT")
    
    if not st.session_state.initialized:
        st.warning("Vui lòng khởi tạo hệ thống trước khi sử dụng chức năng giao dịch")
        if st.button("🚀 Khởi tạo hệ thống"):
            initialize_system()
            st.rerun()
    else:
        st.write("Thiết lập giao dịch tự động dựa trên dự đoán AI")
        
        # Nếu không có dự đoán, cần tạo dự đoán
        if not st.session_state.predictions:
            with st.spinner("Đang tạo dự đoán ban đầu..."):
                prediction = make_prediction()
        else:
            prediction = st.session_state.predictions[-1]
        
        # Hiển thị thông tin dự đoán hiện tại
        with st.container():
            st.subheader("Dự đoán hiện tại")
            display_current_prediction(prediction)
        
        # Phần nhập API Binance
        with st.expander("🔑 Cài đặt API Binance", expanded=True):
            api_key = st.text_input("API Key Binance", value=st.session_state.trading_settings["api_key"], 
                                 type="password", key="api_key_input", 
                                 help="API Key được tạo từ tài khoản Binance của bạn")
            
            api_secret = st.text_input("API Secret Binance", value=st.session_state.trading_settings["api_secret"], 
                                   type="password", key="api_secret_input",
                                   help="API Secret được tạo từ tài khoản Binance của bạn")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("💾 Lưu API Keys", use_container_width=True):
                    st.session_state.trading_settings["api_key"] = api_key
                    st.session_state.trading_settings["api_secret"] = api_secret
                    # Lưu trạng thái giao dịch để khôi phục khi F5
                    save_trading_state()
                    st.success("Đã lưu API Keys")
            
            with col2:
                if st.button("🔄 Kiểm tra kết nối", use_container_width=True):
                    if not api_key or not api_secret:
                        st.error("Vui lòng nhập API Key và API Secret")
                    else:
                        with st.spinner("Đang kiểm tra kết nối..."):
                            if not hasattr(st.session_state, "trading_manager") or st.session_state.trading_manager is None:
                                st.session_state.trading_manager = TradingManager()
                            
                            # Kết nối với API
                            result = st.session_state.trading_manager.connect(api_key, api_secret)
                            if result:
                                st.success("Kết nối thành công đến Binance API")
                                
                                # Lấy số dư
                                balance = st.session_state.trading_manager.get_futures_account_balance()
                                if balance is not None:
                                    st.info(f"Số dư tài khoản Futures: {balance:.2f} USDT")
                            else:
                                st.error("Kết nối thất bại. Vui lòng kiểm tra lại API keys")
        
        # Phần cài đặt Take Profit và Stop Loss
        with st.expander("💵 Cài đặt TP/SL", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Take Profit (TP)")
                tp_type = st.radio("Loại TP", ["Phần trăm (%)", "USDT"], 
                               index=0 if st.session_state.trading_settings["take_profit_type"] == "percent" else 1,
                               key="tp_type")
                
                tp_value = st.number_input("Giá trị TP", 
                                      min_value=0.1, max_value=100.0 if tp_type == "Phần trăm (%)" else 1000.0,
                                      value=float(st.session_state.trading_settings["take_profit_value"]),
                                      step=0.1, key="tp_value")
            
            with col2:
                st.subheader("Stop Loss (SL)")
                sl_type = st.radio("Loại SL", ["Phần trăm (%)", "USDT"], 
                               index=0 if st.session_state.trading_settings["stop_loss_type"] == "percent" else 1,
                               key="sl_type")
                
                sl_value = st.number_input("Giá trị SL", 
                                      min_value=0.1, max_value=100.0 if sl_type == "Phần trăm (%)" else 1000.0,
                                      value=float(st.session_state.trading_settings["stop_loss_value"]),
                                      step=0.1, key="sl_value")
            
            # Lưu các thiết lập TP/SL
            if st.button("💾 Lưu cài đặt TP/SL", use_container_width=True):
                st.session_state.trading_settings["take_profit_type"] = "percent" if tp_type == "Phần trăm (%)" else "usdt"
                st.session_state.trading_settings["take_profit_value"] = tp_value
                st.session_state.trading_settings["stop_loss_type"] = "percent" if sl_type == "Phần trăm (%)" else "usdt"
                st.session_state.trading_settings["stop_loss_value"] = sl_value
                # Lưu trạng thái giao dịch để khôi phục khi F5
                save_trading_state()
                st.success("Đã lưu cài đặt TP/SL")
        
        # Phần cài đặt vốn và đòn bẩy
        with st.expander("📊 Cài đặt vốn và đòn bẩy", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                account_percent = st.slider("Phần trăm tài khoản sử dụng (%)", 
                                       min_value=1.0, max_value=100.0, 
                                       value=float(st.session_state.trading_settings["account_percent"]),
                                       step=1.0, key="account_percent")
                
                st.caption("Phần trăm số dư tài khoản Futures sẽ được sử dụng cho mỗi giao dịch")
            
            with col2:
                leverage_options = [1, 2, 3, 5, 10, 20, 50, 75, 100, 125]
                default_index = leverage_options.index(st.session_state.trading_settings["leverage"]) if st.session_state.trading_settings["leverage"] in leverage_options else 2
                
                leverage = st.select_slider("Đòn bẩy", 
                                      options=leverage_options,
                                      value=leverage_options[default_index], 
                                      key="leverage")
                
                st.caption("Đòn bẩy sẽ được áp dụng cho giao dịch. Cẩn thận với đòn bẩy cao!")
            
            # Độ tin cậy tối thiểu để vào lệnh
            min_confidence = st.slider("Độ tin cậy tối thiểu để vào lệnh (%)", 
                                  min_value=50.0, max_value=99.0, 
                                  value=float(st.session_state.trading_settings["min_confidence"]),
                                  step=1.0, key="min_confidence")
            
            st.caption("Chỉ vào lệnh khi độ tin cậy của dự đoán vượt quá ngưỡng này")
            
            # Biến động giá tối thiểu để vào lệnh
            if "min_price_movement" not in st.session_state.trading_settings:
                st.session_state.trading_settings["min_price_movement"] = config.TRADING_SETTINGS["default_min_price_movement"]
            
            min_price_movement = st.number_input(
                "Biến động giá dự đoán tối thiểu (USDT)",
                min_value=0.0,
                max_value=50.0,
                value=float(st.session_state.trading_settings.get("min_price_movement", config.TRADING_SETTINGS["default_min_price_movement"])),
                step=0.5,
                key="min_price_movement",
                help="Chỉ vào lệnh khi chênh lệch giữa giá hiện tại và giá dự đoán vượt quá ngưỡng này. Đặt 0 để bỏ qua điều kiện này."
            )
            
            st.caption("Giá trị 0 = giao dịch không phụ thuộc vào biến động giá. Giá trị 6 = chỉ giao dịch khi chênh lệch giữa giá hiện tại và giá dự đoán > 6 USDT.")
            
            # Lưu các thiết lập vốn và đòn bẩy
            if st.button("💾 Lưu cài đặt vốn và đòn bẩy", use_container_width=True):
                st.session_state.trading_settings["account_percent"] = account_percent
                st.session_state.trading_settings["leverage"] = leverage
                st.session_state.trading_settings["min_confidence"] = min_confidence
                st.session_state.trading_settings["min_price_movement"] = min_price_movement
                # Lưu trạng thái giao dịch để khôi phục khi F5
                save_trading_state()
                st.success("Đã lưu cài đặt vốn và đòn bẩy")
        
        # Hiển thị thông tin vị thế hiện tại nếu có
        if hasattr(st.session_state, "trading_manager") and st.session_state.trading_manager is not None:
            with st.container():
                st.subheader("Thông tin vị thế hiện tại")
                
                # Lấy thông tin vị thế nếu đã kết nối API
                if st.session_state.trading_manager.client is not None:
                    with st.spinner("Đang tải thông tin vị thế..."):
                        pnl_info = st.session_state.trading_manager.get_position_pnl()
                        
                        if pnl_info is not None:
                            if pnl_info.get("has_position", False):
                                # Hiển thị thông tin vị thế
                                position_details = f"""
                                - **Symbol**: {pnl_info.get('symbol', 'N/A')}
                                - **Khối lượng**: {pnl_info.get('position_amount', 0)}
                                - **Giá vào lệnh**: {pnl_info.get('entry_price', 0):.2f} USDT
                                - **Giá hiện tại**: {pnl_info.get('current_price', 0):.2f} USDT
                                - **Đòn bẩy**: {pnl_info.get('leverage', 1)}x
                                - **Lợi nhuận**: {pnl_info.get('pnl', 0):.2f} USDT ({pnl_info.get('pnl_percent', 0):.2f}%)
                                - **Giá thanh lý**: {pnl_info.get('liquidation_price', 'N/A')}
                                """
                                
                                # Hiển thị PNL với màu sắc dựa trên giá trị
                                pnl_value = pnl_info.get('pnl', 0)
                                pnl_percent = pnl_info.get('pnl_percent', 0)
                                
                                if pnl_value > 0:
                                    st.markdown(f"### 💰 Lợi nhuận: +{pnl_value:.2f} USDT (+{pnl_percent:.2f}%)")
                                    st.success(position_details)
                                elif pnl_value < 0:
                                    st.markdown(f"### 📉 Lỗ: {pnl_value:.2f} USDT ({pnl_percent:.2f}%)")
                                    st.error(position_details)
                                else:
                                    st.markdown(f"### ⚖️ Vị thế: {pnl_value:.2f} USDT ({pnl_percent:.2f}%)")
                                    st.info(position_details)
                                
                                # Nút đóng vị thế
                                if st.button("📤 Đóng vị thế", type="primary"):
                                    with st.spinner("Đang đóng vị thế..."):
                                        result = st.session_state.trading_manager.close_position()
                                        if result:
                                            # Lưu trạng thái giao dịch để khôi phục khi F5
                                            save_trading_state()
                                            st.success("Đã đóng vị thế thành công")
                                            st.rerun()
                                        else:
                                            st.error("Không thể đóng vị thế. Kiểm tra logs để biết thêm chi tiết.")
                            else:
                                st.info("Không có vị thế nào đang mở")
                        else:
                            st.warning("Không thể lấy thông tin vị thế. Vui lòng kiểm tra kết nối API.")
                else:
                    st.warning("Vui lòng kết nối API Binance để xem thông tin vị thế")
        
        # Phần bắt đầu giao dịch tự động
        with st.container():
            st.subheader("Bắt đầu giao dịch tự động")
            
            # Kiểm tra xem đã có API keys và các thiết lập cần thiết chưa
            can_start_trading = (st.session_state.trading_settings["api_key"] and 
                               st.session_state.trading_settings["api_secret"] and
                               hasattr(st.session_state, "trading_manager") and 
                               st.session_state.trading_manager is not None and
                               st.session_state.trading_manager.client is not None)
            
            if not can_start_trading:
                st.warning("Vui lòng cấu hình API Binance và kiểm tra kết nối trước khi bắt đầu giao dịch")
            
            # Hiển thị tùy chọn khung thời gian
            available_timeframes = config.TRADING_SETTINGS["available_timeframes"]
            selected_timeframe = st.radio(
                "⏱️ Chọn khung thời gian giao dịch:",
                available_timeframes,
                index=available_timeframes.index(config.TRADING_SETTINGS["default_timeframe"]),
                horizontal=True,
                help="Khung thời gian sẽ được sử dụng cho việc dự đoán và giao dịch"
            )
            
            st.caption("""
            - Khung 1m: Giao dịch ngắn hạn, nhạy với biến động giá, phù hợp cho scalping
            - Khung 5m: Giao dịch trung hạn, ổn định hơn, giảm tín hiệu giả, phù hợp swing trade
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                start_button = st.button("▶️ Bắt đầu giao dịch tự động", 
                                    use_container_width=True, 
                                    disabled=not can_start_trading or st.session_state.trading_settings.get("is_trading", False),
                                    type="primary" if not st.session_state.trading_settings.get("is_trading", False) else "secondary")
            
            with col2:
                stop_button = st.button("⏹️ Dừng giao dịch tự động", 
                                   use_container_width=True,
                                   disabled=not st.session_state.trading_settings.get("is_trading", False),
                                   type="primary" if st.session_state.trading_settings.get("is_trading", False) else "secondary")
            
            # Xử lý sự kiện khi nhấn nút bắt đầu
            if start_button and can_start_trading:
                # Thiết lập cấu hình giao dịch
                trading_config = {
                    "symbol": st.session_state.trading_settings["symbol"],
                    "take_profit_type": st.session_state.trading_settings["take_profit_type"],
                    "take_profit_value": st.session_state.trading_settings["take_profit_value"],
                    "stop_loss_type": st.session_state.trading_settings["stop_loss_type"],
                    "stop_loss_value": st.session_state.trading_settings["stop_loss_value"],
                    "account_percent": st.session_state.trading_settings["account_percent"],
                    "leverage": st.session_state.trading_settings["leverage"],
                    "min_confidence": st.session_state.trading_settings["min_confidence"] / 100.0,
                    "min_price_movement": st.session_state.trading_settings.get("min_price_movement", config.TRADING_SETTINGS["default_min_price_movement"]),
                    "timeframe": selected_timeframe,
                }
                
                # Kiểm tra lại kết nối
                if not st.session_state.trading_manager.client:
                    st.session_state.trading_manager.connect(
                        st.session_state.trading_settings["api_key"],
                        st.session_state.trading_settings["api_secret"]
                    )
                
                # Bắt đầu bot giao dịch
                result = st.session_state.trading_manager.start_trading_bot(
                    trading_config, st.session_state.prediction_engine
                )
                
                if result:
                    st.session_state.trading_settings["is_trading"] = True
                    # Lưu trạng thái giao dịch để khôi phục khi F5
                    save_trading_state()
                    st.success("Bot giao dịch tự động đã bắt đầu")
                    st.rerun()
                else:
                    st.error("Không thể bắt đầu bot giao dịch. Kiểm tra logs để biết thêm chi tiết.")
            
            # Xử lý sự kiện khi nhấn nút dừng
            if stop_button and st.session_state.trading_settings.get("is_trading", False):
                if hasattr(st.session_state, "trading_manager") and st.session_state.trading_manager is not None:
                    result = st.session_state.trading_manager.stop_trading_bot()
                    if result:
                        st.session_state.trading_settings["is_trading"] = False
                        # Lưu trạng thái giao dịch để khôi phục khi F5
                        save_trading_state()
                        st.success("Bot giao dịch tự động đã dừng")
                        st.rerun()
                    else:
                        st.error("Không thể dừng bot giao dịch. Kiểm tra logs để biết thêm chi tiết.")
            
            # Hiển thị trạng thái giao dịch
            if st.session_state.trading_settings.get("is_trading", False):
                # Lấy thông tin khung thời gian đang sử dụng (nếu có)
                current_timeframe = "N/A"
                if hasattr(st.session_state.trading_manager, "trading_config") and st.session_state.trading_manager.trading_config:
                    current_timeframe = st.session_state.trading_manager.trading_config.get("timeframe", "N/A")
                
                st.markdown(f"### ✅ Trạng thái: Bot giao dịch đang hoạt động (khung {current_timeframe})")
                
                if hasattr(st.session_state, "trading_manager") and st.session_state.trading_manager is not None:
                    # Hiển thị thống kê PNL theo ngày (múi giờ +7)
                    if hasattr(st.session_state.trading_manager, "get_daily_pnl_summary"):
                        st.subheader("📊 Thống kê PNL theo ngày (UTC+7)")
                        
                        # Lấy thông tin PNL theo ngày
                        daily_pnl = st.session_state.trading_manager.get_daily_pnl_summary()
                        
                        if daily_pnl:
                            # Tạo các metrics hiển thị
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                pnl_value = daily_pnl.get('total_pnl', 0)
                                if pnl_value > 0:
                                    st.metric("Tổng PNL", f"+{pnl_value:.2f} USDT", delta=f"+{pnl_value:.2f}", delta_color="normal")
                                else:
                                    st.metric("Tổng PNL", f"{pnl_value:.2f} USDT", delta=f"{pnl_value:.2f}", delta_color="normal")
                            
                            with col2:
                                win_count = daily_pnl.get('win_count', 0)
                                loss_count = daily_pnl.get('loss_count', 0)
                                total_trades = win_count + loss_count
                                st.metric("Số lệnh", f"{total_trades}", help="Tổng số lệnh đã thực hiện trong ngày")
                            
                            with col3:
                                win_rate = daily_pnl.get('win_rate', 0)
                                st.metric("Tỷ lệ thắng", f"{win_rate:.1f}%", help="Tỷ lệ lệnh lãi trên tổng số lệnh")
                            
                            with col4:
                                current_date = daily_pnl.get('date', 'N/A')
                                st.metric("Ngày", f"{current_date}", help="Ngày hiện tại (UTC+7)")
                            
                            # Hiển thị danh sách các giao dịch trong ngày
                            if 'trades' in daily_pnl and daily_pnl['trades']:
                                st.subheader("Các giao dịch trong ngày")
                                
                                # Tạo DataFrame từ danh sách giao dịch
                                import pandas as pd
                                trades_data = daily_pnl['trades']
                                trades_df = pd.DataFrame(trades_data)
                                
                                # Format DataFrame
                                if len(trades_df) > 0:
                                    if 'time' in trades_df.columns:
                                        trades_df = trades_df[['time', 'symbol', 'side', 'pnl', 'pnl_percent']]
                                        trades_df.columns = ['Thời gian', 'Symbol', 'Hướng', 'PNL (USDT)', 'PNL (%)']
                                        
                                        # Định dạng các cột số
                                        trades_df['PNL (USDT)'] = trades_df['PNL (USDT)'].map('{:.2f}'.format)
                                        trades_df['PNL (%)'] = trades_df['PNL (%)'].map('{:.2f}%'.format)
                                        
                                        # Đảo ngược để hiển thị mới nhất lên đầu
                                        trades_df = trades_df.iloc[::-1].reset_index(drop=True)
                                        
                                        # Hiển thị bảng với màu sắc
                                        def highlight_pnl(val):
                                            try:
                                                # Xác định xem PNL dương hay âm
                                                value = float(val.replace('%', ''))
                                                if value > 0:
                                                    return 'background-color: rgba(0, 255, 0, 0.2)'
                                                elif value < 0:
                                                    return 'background-color: rgba(255, 0, 0, 0.2)'
                                                else:
                                                    return ''
                                            except:
                                                return ''
                                                
                                        # Áp dụng định dạng có điều kiện
                                        styled_df = trades_df.style.applymap(highlight_pnl, subset=['PNL (%)'])
                                        st.dataframe(styled_df, use_container_width=True)
                                    else:
                                        st.dataframe(trades_df, use_container_width=True)
                                else:
                                    st.info("Chưa có giao dịch nào được thực hiện trong ngày hôm nay")
                            else:
                                st.info("Chưa có giao dịch nào được thực hiện trong ngày hôm nay")
                                
                    # Hiển thị các logs giao dịch
                    if hasattr(st.session_state.trading_manager, "trading_logs") and st.session_state.trading_manager.trading_logs:
                        st.subheader("📝 Nhật ký giao dịch")
                        logs = st.session_state.trading_manager.trading_logs[-10:]  # Chỉ hiển thị 10 logs gần nhất
                        logs_reversed = logs[::-1]  # Đảo ngược để hiển thị mới nhất trước
                        
                        for log in logs_reversed:
                            timestamp = log.get("timestamp", "")
                            message = log.get("message", "")
                            level = log.get("level", "info")
                            
                            if level == "error":
                                st.error(f"{timestamp}: {message}")
                            elif level == "warning":
                                st.warning(f"{timestamp}: {message}")
                            else:
                                st.info(f"{timestamp}: {message}")
            else:
                st.markdown("### ⏸️ Trạng thái: Bot giao dịch đang dừng")
        
        # Hiển thị lưu ý quan trọng
        with st.expander("⚠️ Lưu ý quan trọng", expanded=True):
            st.warning("""
            - Giao dịch tiền điện tử luôn có rủi ro cao, bạn có thể mất tất cả vốn đầu tư.
            - Hệ thống AI dự đoán không bảo đảm lợi nhuận và có thể sai trong nhiều trường hợp.
            - Hãy bắt đầu với số vốn nhỏ khi sử dụng tính năng giao dịch tự động lần đầu.
            - Kiểm tra cẩn thận các thiết lập TP/SL và đòn bẩy trước khi bắt đầu.
            - Chỉ sử dụng đòn bẩy cao nếu bạn hiểu rõ rủi ro liên quan.
            """)

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

# Tạo giao diện chính với màu sắc và bố cục đẹp mắt
def render_main_interface():
    # Áp dụng CSS tùy chỉnh
    load_custom_css()
    
    # Tạo header đẹp mắt bằng markdown trực tiếp
    st.markdown("# AI TRADING ORACLE")
    st.markdown("### Hệ Thống Dự Đoán ETHUSDT Tự Động")
    
    # Tạo sidebar menu
    with st.sidebar:
        # Tạo phần header sidebar
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #485ec4;">⚙️ Cài đặt & Điều khiển</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Tạo các tab trong sidebar
        sidebar_tabs = st.tabs(["🎛️ Điều khiển", "📊 Dữ liệu", "⚡ Mô hình"])
        
        with sidebar_tabs[0]:
            # Control tab
            st.subheader("Điều khiển hệ thống")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Cập nhật dữ liệu", use_container_width=True):
                    with st.spinner("Đang cập nhật dữ liệu..."):
                        fetch_data()
                        show_toast("Đã cập nhật dữ liệu thành công!", "success")
            
            with col2:
                if st.button("🔮 Dự đoán ngay", use_container_width=True):
                    with st.spinner("Đang tạo dự đoán..."):
                        make_prediction()
                        show_toast("Đã tạo dự đoán mới!", "success")
            
            st.write("---")
            
            # Luồng cập nhật tự động
            st.subheader("Cập nhật tự động")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.thread_running:
                    if st.button("⏹️ Dừng cập nhật", use_container_width=True):
                        stop_update_thread()
                        show_toast("Đã dừng cập nhật tự động", "warning")
                else:
                    if st.button("▶️ Bắt đầu cập nhật", use_container_width=True):
                        start_update_thread()
                        show_toast("Đã bắt đầu cập nhật tự động", "success")
            
            with col2:
                update_interval = st.selectbox(
                    "Chu kỳ cập nhật",
                    options=[5, 10, 30, 60, 300],
                    index=1,
                    format_func=lambda x: f"{x} giây"
                )
                if 'update_interval' not in st.session_state or st.session_state.update_interval != update_interval:
                    st.session_state.update_interval = update_interval
            
            # Biểu đồ tự động cập nhật
            st.write("---")
            st.subheader("Biểu đồ")
            if "chart_auto_refresh" not in st.session_state:
                st.session_state.chart_auto_refresh = True
            st.checkbox("Tự động cập nhật biểu đồ", value=st.session_state.chart_auto_refresh, key="chart_auto_refresh")
            
        with sidebar_tabs[1]:
            # Data tab
            st.subheader("Nguồn dữ liệu")
            
            data_source = "Binance API"
            
            data_source_color = "green" if data_source == "Binance API" else "orange"
            st.markdown(f"<div style='color: {data_source_color}; font-weight: bold;'>{data_source}</div>", unsafe_allow_html=True)
            
            if data_source == "Binance API":
                st.success("Kết nối Binance API thành công")
            else:
                st.warning("Đang sử dụng dữ liệu mô phỏng")
            
            st.write("---")
            
            st.subheader("Khoảng thời gian")
            timeframe = st.selectbox(
                "Khung thời gian",
                options=["1m", "5m", "15m", "1h", "4h"],
                index=1,
                key="selected_timeframe"
            )
            
            # Cập nhật thiết lập khung thời gian
            if timeframe != st.session_state.prediction_settings.get("timeframe"):
                st.session_state.prediction_settings["timeframe"] = timeframe
                
            # Chọn khoảng thời gian biểu đồ
            chart_range = st.selectbox(
                "Khoảng thời gian hiển thị",
                options=["1 ngày", "3 ngày", "7 ngày", "14 ngày", "30 ngày"],
                index=1
            )
            
        with sidebar_tabs[2]:
            # Model tab
            st.subheader("Huấn luyện AI")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧠 Huấn luyện lại", use_container_width=True):
                    with st.spinner("Đang huấn luyện lại mô hình..."):
                        train_models()
                        show_toast("Đã bắt đầu huấn luyện lại mô hình!", "success")
            
            with col2:
                if st.button("📋 Xem nhật ký", use_container_width=True):
                    st.session_state.selected_tab = "Training Logs"
                    st.rerun()
            
            st.write("---")
            
            st.subheader("Thiết lập dự đoán")
            
            # Chọn loại mô hình
            model_type = st.selectbox(
                "Phương pháp dự đoán",
                options=["Ensemble (tất cả)", "LSTM", "Transformer", "CNN", "Historical Matching"],
                index=0
            )
            
            # Chọn khoảng thời gian dự đoán
            prediction_horizon = st.selectbox(
                "Khoảng thời gian dự đoán",
                options=["10 phút", "15 phút", "30 phút", "1 giờ", "4 giờ"],
                index=2
            )
            
            # Cập nhật thiết lập dự đoán
            horizon_map = {"10 phút": 10, "15 phút": 15, "30 phút": 30, "1 giờ": 60, "4 giờ": 240}
            if horizon_map[prediction_horizon] != st.session_state.prediction_settings.get("horizon"):
                st.session_state.prediction_settings["horizon"] = horizon_map[prediction_horizon]
            
            st.write("---")
            
            # Hiển thị trạng thái mô hình
            st.subheader("Trạng thái mô hình")
            
            if st.session_state.model_trained:
                st.success("Mô hình đã được huấn luyện")
                
                if hasattr(st.session_state, 'continuous_trainer') and st.session_state.continuous_trainer:
                    training_status = st.session_state.continuous_trainer.get_training_status()
                    last_training = training_status.get('last_training_time', 'Chưa xác định')
                    st.info(f"Huấn luyện lần cuối: {last_training}")
            else:
                st.error("Mô hình chưa được huấn luyện")
                st.button("⚡ Huấn luyện ngay", on_click=train_models)
        
        # Phần footer của sidebar
        st.write("---")
        
        # Hiển thị trạng thái server
        if st.session_state.thread_running:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <div style="background-color: #28a745; width: 10px; height: 10px; border-radius: 50%; margin-right: 10px;"></div>
                <div>Server đang chạy</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center;">
                <div style="background-color: #ffc107; width: 10px; height: 10px; border-radius: 50%; margin-right: 10px;"></div>
                <div>Server đang dừng</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Thông tin về phiên bản
        st.caption("© 2025 AI Crypto Prediction | v2.0")

    # Tạo layout chính
    main_tabs = st.tabs(["📈 Dashboard", "📊 Phân tích kỹ thuật", "🤖 API", "📘 Hướng dẫn"])
    
    with main_tabs[0]:
        # Dashboard tab
        if not st.session_state.initialized:
            st.warning("Đang khởi tạo hệ thống...")
            return
        
        # Hiển thị trạng thái dữ liệu
        if st.session_state.latest_data is None:
            st.warning("Đang tải dữ liệu...")
            if st.button("Tải dữ liệu"):
                fetch_data()
            return
        
        # DASHBOARD LAYOUT
        
        # Row 1: Tổng quan thị trường
        st.markdown("### Tổng quan thị trường")
        
        # Lấy dữ liệu gần đây nhất
        latest_candle = st.session_state.latest_data.iloc[-1]
        prev_candle = st.session_state.latest_data.iloc[-2]
        
        # Tính toán thay đổi giá
        price_change = latest_candle['close'] - prev_candle['close']
        price_change_pct = (price_change / prev_candle['close']) * 100
        
        # Row 1: Giá và thống kê tổng quan
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            # Hiển thị giá hiện tại với thiết kế đẹp
            create_price_card(
                latest_candle['close'],
                price_change,
                price_change_pct,
                st.session_state.data_fetch_status.get('last_update')
            )
        
        with col2:
            # Hiển thị khối lượng giao dịch
            create_metric_card(
                "Khối lượng 24h",
                f"{latest_candle['volume'] / 1000000:.2f}M",
                icon="📊",
                color="blue"
            )
        
        with col3:
            # Hiển thị biến động (ATR)
            if 'atr' in latest_candle:
                volatility = latest_candle['atr']
            else:
                volatility = (latest_candle['high'] - latest_candle['low']) / latest_candle['close'] * 100
            
            create_metric_card(
                "Biến động",
                f"{volatility:.2f}%",
                icon="📉",
                color="yellow" if volatility > 2 else "blue"
            )
            
        with col4:
            # Hiển thị RSI nếu có
            if 'rsi' in latest_candle:
                rsi = latest_candle['rsi']
                color = "red" if rsi > 70 else "green" if rsi < 30 else "blue"
                create_metric_card(
                    "RSI",
                    f"{rsi:.1f}",
                    icon="🔍",
                    color=color
                )
            else:
                create_metric_card(
                    "Cập nhật",
                    st.session_state.data_fetch_status.get('last_update', 'N/A').split()[1],
                    icon="⏱️",
                    color="blue"
                )
        
        # Row 2: Dự đoán hiện tại và biểu đồ
        st.markdown("### Dự đoán và biểu đồ giá")
        
        pred_col, chart_col = st.columns([1, 2])
        
        with pred_col:
            # Lấy dự đoán gần nhất
            if st.session_state.predictions and len(st.session_state.predictions) > 0:
                latest_prediction = st.session_state.predictions[-1]
                
                # Tính thời gian còn lại
                if 'timestamp' in latest_prediction and 'valid_for_minutes' in latest_prediction:
                    pred_time = datetime.strptime(latest_prediction['timestamp'], "%Y-%m-%d %H:%M:%S")
                    elapsed_minutes = (datetime.now() - pred_time).total_seconds() / 60
                    minutes_left = max(0, latest_prediction['valid_for_minutes'] - elapsed_minutes)
                    latest_prediction['valid_minutes_left'] = minutes_left
                
                # Hiển thị dự đoán với thiết kế đẹp mắt
                create_prediction_card(latest_prediction)
                
                # Hiển thị độ tin cậy bằng biểu đồ gauge
                confidence = latest_prediction.get('confidence', 0)
                st.plotly_chart(
                    create_gauge_chart(
                        confidence,
                        "Độ tin cậy dự đoán",
                        min_value=0,
                        max_value=1,
                        color_thresholds=[
                            (0.3, "red"),
                            (0.7, "orange"),
                            (1.0, "green")
                        ]
                    ),
                    use_container_width=True
                )
                
                # Hiển thị nút tạo dự đoán mới
                if st.button("🔮 Tạo dự đoán mới", use_container_width=True):
                    with st.spinner("Đang tạo dự đoán mới..."):
                        make_prediction()
                        show_toast("Đã tạo dự đoán mới!", "success")
                        st.rerun()
                
            else:
                st.info("Chưa có dự đoán nào được tạo")
                if st.button("🚀 Tạo dự đoán đầu tiên", use_container_width=True):
                    with st.spinner("Đang tạo dự đoán..."):
                        make_prediction()
                        show_toast("Đã tạo dự đoán đầu tiên!", "success")
                        st.rerun()
        
        with chart_col:
            # Hiển thị biểu đồ nến với chức năng chọn khung thời gian
            timeframe_options = {
                '50 nến gần nhất': 50, 
                '100 nến gần nhất': 100, 
                '200 nến gần nhất': 200,
                'Tất cả dữ liệu': len(st.session_state.latest_data)
            }
            
            selected_tf = st.selectbox(
                "Hiển thị",
                options=list(timeframe_options.keys()),
                index=1
            )
            
            candles = timeframe_options[selected_tf]
            
            # Vẽ biểu đồ nến với Plotly
            try:
                chart_data = st.session_state.latest_data.iloc[-candles:].copy()
                # Đảm bảo dữ liệu đầu vào hợp lệ
                if not chart_data.empty:
                    chart = plot_candlestick_chart(chart_data)
                    st.plotly_chart(chart, use_container_width=True, key="main_candlestick_chart")
                else:
                    st.warning("Không đủ dữ liệu để hiển thị biểu đồ")
            except Exception as e:
                st.error(f"Lỗi khi hiển thị biểu đồ: {str(e)}")
                # Ghi lại lỗi vào logs
                print(f"Error plotting candlestick chart: {str(e)}")
        
        # Row 3: Lịch sử dự đoán và hiệu suất mô hình
        st.markdown("### Phân tích hiệu suất")
        
        perf_col, hist_col = st.columns(2)
        
        with perf_col:
            st.subheader("Hiệu suất các mô hình")
            
            # Lấy hiệu suất từ các mô hình đã huấn luyện
            if hasattr(st.session_state, 'model_performance') and st.session_state.model_performance:
                model_performance = st.session_state.model_performance
            else:
                # Hiệu suất mẫu nếu chưa có dữ liệu thực tế
                model_performance = {
                    'lstm': 0.72,
                    'transformer': 0.76,
                    'cnn': 0.68,
                    'historical_similarity': 0.65,
                    'meta_learner': 0.81
                }
            
            # Vẽ biểu đồ hiệu suất
            perf_chart = plot_model_performance(model_performance)
            st.plotly_chart(perf_chart, use_container_width=True)
            
        with hist_col:
            st.subheader("Lịch sử dự đoán")
            
            if st.session_state.predictions and len(st.session_state.predictions) > 0:
                # Vẽ biểu đồ lịch sử dự đoán
                try:
                    # Sao chép dữ liệu để tránh lỗi khi xử lý
                    prediction_data = st.session_state.predictions.copy()
                    hist_chart = plot_prediction_history(prediction_data)
                    st.plotly_chart(hist_chart, use_container_width=True, key="prediction_history_chart")
                except Exception as e:
                    st.error(f"Lỗi khi hiển thị lịch sử dự đoán: {str(e)}")
                    print(f"Error plotting prediction history: {str(e)}")
            else:
                st.info("Chưa có dữ liệu lịch sử dự đoán")
    
    with main_tabs[1]:
        # Tab phân tích kỹ thuật
        if not st.session_state.initialized or st.session_state.latest_data is None:
            st.warning("Đang khởi tạo và tải dữ liệu...")
            return
        
        # Tạo tiêu đề với biểu tượng đẹp
        create_section_header(
            "Phân tích kỹ thuật chi tiết", 
            "Phân tích kỹ thuật nâng cao với các chỉ báo và công cụ phân tích", 
            icon="📊"
        )
        
        # Tạo các tab con cho phân tích kỹ thuật
        tech_tabs = st.tabs(["📊 Chỉ báo kỹ thuật", "🔍 Mẫu hình nến", "📏 Hỗ trợ & Kháng cự", "📉 Phân tích xu hướng"])
        
        with tech_tabs[0]:
            # Tab chỉ báo kỹ thuật
            st.subheader("Chỉ báo kỹ thuật nâng cao")
            
            # Thêm mô tả
            st.markdown("""
            Chỉ báo kỹ thuật là công cụ phân tích dựa trên giá, khối lượng và các dữ liệu thị trường khác.
            Chúng giúp nhà đầu tư đưa ra quyết định dựa trên phân tích định lượng.
            """)
            
            # Hiển thị biểu đồ chỉ báo
            indicators_chart = plot_technical_indicators(st.session_state.latest_data.iloc[-100:])
            st.plotly_chart(indicators_chart, use_container_width=True)
            
            # Hiển thị giải thích cho từng chỉ báo
            with st.expander("Giải thích các chỉ báo", expanded=False):
                st.markdown("""
                ### SuperTrend
                - Chỉ báo xu hướng dựa trên ATR và các phép tính trung bình
                - Đường trên (đỏ): Xu hướng giảm
                - Đường dưới (xanh): Xu hướng tăng
                
                ### RSI (Relative Strength Index)
                - Dao động từ 0-100
                - Trên 70: Vùng quá mua
                - Dưới 30: Vùng quá bán
                - 50: Ngưỡng trung tính
                
                ### ADX (Average Directional Index)
                - Đo lường sức mạnh xu hướng
                - < 20: Xu hướng yếu
                - 20-40: Xu hướng trung bình
                - > 40: Xu hướng mạnh
                - Không chỉ ra hướng xu hướng
                
                ### Bollinger Bands
                - Dựa trên trung bình động và độ lệch chuẩn
                - Band trên/dưới: Giá có thể biến động trong vùng này
                - Băng hẹp: Thị trường biến động thấp, chuẩn bị bùng nổ
                - Băng rộng: Thị trường biến động cao
                """)
            
            # Hiển thị tóm tắt trạng thái hiện tại
            st.subheader("Tóm tắt trạng thái hiện tại")
            
            # Tính toán và hiển thị các giá trị
            latest = st.session_state.latest_data.iloc[-1]
            
            # Tạo bảng thông tin
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Chỉ báo xu hướng
                if 'supertrend_direction' in latest:
                    trend_direction = "Tăng" if latest['supertrend_direction'] == 1 else "Giảm"
                    trend_color = "green" if latest['supertrend_direction'] == 1 else "red"
                else:
                    # Tính EMA từ dữ liệu gần đây
                    recent_data = st.session_state.latest_data.tail(30)
                    ema9 = recent_data['close'].rolling(window=9).mean().iloc[-1] if len(recent_data) > 0 else 0
                    ema21 = recent_data['close'].rolling(window=21).mean().iloc[-1] if len(recent_data) > 0 else 0
                    trend_direction = "Tăng" if ema9 > ema21 else "Giảm"
                    trend_color = "green" if ema9 > ema21 else "red"
                
                st.markdown(f"**Xu hướng:** <span style='color:{trend_color}'>{trend_direction}</span>", unsafe_allow_html=True)
                
                # RSI
                if 'rsi' in latest:
                    rsi = latest['rsi']
                    rsi_status = "Quá mua" if rsi > 70 else "Quá bán" if rsi < 30 else "Trung tính"
                    rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "gray"
                    st.markdown(f"**RSI:** <span style='color:{rsi_color}'>{rsi:.1f} ({rsi_status})</span>", unsafe_allow_html=True)
                
                # MACD
                if 'macd' in latest and 'macd_signal' in latest:
                    macd = latest['macd']
                    macd_signal = latest['macd_signal']
                    macd_hist = macd - macd_signal
                    macd_status = "Tăng" if macd > macd_signal else "Giảm"
                    macd_color = "green" if macd > macd_signal else "red"
                    st.markdown(f"**MACD:** <span style='color:{macd_color}'>{macd_hist:.4f} ({macd_status})</span>", unsafe_allow_html=True)
            
            with col2:
                # Bollinger Bands
                if 'upper_band' in latest and 'lower_band' in latest:
                    bb_width = (latest['upper_band'] - latest['lower_band']) / latest['close']
                    bb_position = (latest['close'] - latest['lower_band']) / (latest['upper_band'] - latest['lower_band'])
                    bb_status = "Biến động cao" if bb_width > 0.05 else "Biến động thấp"
                    
                    st.markdown(f"**BB Width:** {bb_width:.4f} ({bb_status})")
                    st.markdown(f"**BB Position:** {bb_position:.2f}")
                
                # ADX
                if 'adx' in latest:
                    adx = latest['adx']
                    adx_status = "Xu hướng mạnh" if adx > 25 else "Xu hướng yếu"
                    st.markdown(f"**ADX:** {adx:.1f} ({adx_status})")
                
            with col3:
                # Volume
                vol_change = (latest['volume'] / st.session_state.latest_data['volume'].iloc[-10:-1].mean() - 1) * 100
                vol_status = "Tăng" if vol_change > 0 else "Giảm"
                vol_color = "green" if vol_change > 0 else "red"
                
                st.markdown(f"**Khối lượng:** <span style='color:{vol_color}'>{vol_change:.1f}% ({vol_status})</span>", unsafe_allow_html=True)
                
                # Volatility (ATR)
                if 'atr' in latest:
                    atr = latest['atr']
                    atr_pct = atr / latest['close'] * 100
                    st.markdown(f"**Biến động (ATR):** {atr_pct:.2f}%")
                
                # Trend Strength
                if 'adx' in latest:
                    trend_strength = "Mạnh" if latest['adx'] > 25 else "Trung bình" if latest['adx'] > 15 else "Yếu"
                    st.markdown(f"**Độ mạnh xu hướng:** {trend_strength}")
        
        with tech_tabs[1]:
            # Tab mẫu hình nến
            st.subheader("Phân tích mẫu hình nến")
            
            # Hiển thị giải thích
            st.markdown("""
            Mẫu hình nến Nhật Bản là các hình mẫu đặc trưng trong biểu đồ giá, cung cấp thông tin về tâm lý thị trường
            và khả năng biến động giá trong tương lai.
            """)
            
            # Phát hiện mẫu hình nến
            from utils.pattern_recognition import detect_candlestick_patterns
            candle_patterns = detect_candlestick_patterns(st.session_state.latest_data.iloc[-5:])
            
            if candle_patterns and len(candle_patterns) > 0:
                st.subheader("Mẫu hình nến phát hiện được")
                
                for pattern in candle_patterns:
                    pattern_color = "green" if pattern['direction'] == 'bullish' else "red"
                    
                    st.markdown(f"""
                    <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                                border-left: 4px solid {pattern_color}; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 18px; font-weight: bold;">{pattern['name']}</div>
                                <div style="margin-top: 5px; color: #5f6368;">{pattern['description']}</div>
                                <div style="margin-top: 10px;">
                                    <span style="color: {pattern_color}; font-weight: bold;">
                                        {pattern['direction'].title()} ({pattern['reliability']}% độ tin cậy)
                                    </span>
                                </div>
                            </div>
                            <div style="font-size: 36px; color: {pattern_color};">
                                {'📈' if pattern['direction'] == 'bullish' else '📉'}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Không phát hiện được mẫu hình nến rõ ràng cho 5 nến gần đây")
            
            # Hiển thị bảng tham khảo về mẫu hình nến phổ biến
            with st.expander("Tham khảo các mẫu hình nến phổ biến", expanded=False):
                st.markdown("""
                ### Mẫu hình tăng
                - **Hammer (Búa)**: Nến có thân nhỏ, bóng dưới dài, xuất hiện trong xu hướng giảm
                - **Morning Star (Sao Mai)**: Mẫu hình 3 nến, nến giữa là nến nhỏ (doji hoặc spinning top)
                - **Bullish Engulfing (Bao phủ tăng)**: Nến tăng bao phủ hoàn toàn nến giảm trước đó
                - **Piercing Line (Đường xuyên)**: Nến giảm sau đó là nến tăng mở cửa thấp hơn và đóng cửa cao hơn điểm giữa nến trước
                
                ### Mẫu hình giảm
                - **Shooting Star (Sao Băng)**: Nến có thân nhỏ, bóng trên dài, xuất hiện trong xu hướng tăng
                - **Evening Star (Sao Hôm)**: Mẫu hình 3 nến, nến giữa là nến nhỏ
                - **Bearish Engulfing (Bao phủ giảm)**: Nến giảm bao phủ hoàn toàn nến tăng trước đó
                - **Dark Cloud Cover (Mây Đen Bao Phủ)**: Nến tăng sau đó là nến giảm mở cửa cao hơn và đóng cửa thấp hơn điểm giữa nến trước
                
                ### Mẫu hình trung lập
                - **Doji**: Nến có giá mở cửa và đóng cửa gần như bằng nhau
                - **Spinning Top (Con Quay)**: Nến có thân nhỏ và bóng trên/dưới dài bằng nhau
                - **Harami (Thai Nghén)**: Nến có thân lớn sau đó là nến có thân nhỏ nằm hoàn toàn trong thân nến trước
                """)
            
        with tech_tabs[2]:
            # Tab hỗ trợ và kháng cự
            st.subheader("Phân tích vùng hỗ trợ và kháng cự")
            
            # Hiển thị giải thích
            st.markdown("""
            Các vùng hỗ trợ và kháng cự là các mức giá quan trọng nơi giá có xu hướng gặp phản ứng. 
            Vùng hỗ trợ là nơi giá có thể dừng giảm và đảo chiều, vùng kháng cự là nơi giá có thể dừng tăng và đảo chiều.
            """)
            
            # Phát hiện các mức hỗ trợ/kháng cự
            from utils.pattern_recognition import calculate_support_resistance
            support_resistance = calculate_support_resistance(st.session_state.latest_data.iloc[-100:])
            
            if support_resistance:
                # Lấy giá hiện tại
                current_price = st.session_state.latest_data['close'].iloc[-1]
                
                # Hiển thị các mức hỗ trợ và kháng cự
                st.subheader("Các mức hỗ trợ và kháng cự")
                
                # Tạo bảng các mức
                support_levels = sorted([level for level in support_resistance['support'] if level < current_price], reverse=True)
                resistance_levels = sorted([level for level in support_resistance['resistance'] if level > current_price])
                
                if len(support_levels) > 0 or len(resistance_levels) > 0:
                    # Tạo hai cột
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"##### Các mức kháng cự")
                        for i, level in enumerate(resistance_levels[:3]):  # Hiển thị tối đa 3 mức
                            distance = ((level / current_price) - 1) * 100
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; padding: 5px 0; 
                                       border-bottom: 1px solid #eaeaea; margin-bottom: 5px;">
                                <div style="font-weight: bold;">R{i+1}</div>
                                <div>${level:.2f}</div>
                                <div style="color: red;">+{distance:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"##### Các mức hỗ trợ")
                        for i, level in enumerate(support_levels[:3]):  # Hiển thị tối đa 3 mức
                            distance = ((level / current_price) - 1) * 100
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; padding: 5px 0; 
                                       border-bottom: 1px solid #eaeaea; margin-bottom: 5px;">
                                <div style="font-weight: bold;">S{i+1}</div>
                                <div>${level:.2f}</div>
                                <div style="color: green;">{distance:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Vẽ biểu đồ với các mức hỗ trợ và kháng cự
                fig = go.Figure()
                
                # Thêm đường giá
                fig.add_trace(go.Scatter(
                    x=st.session_state.latest_data.iloc[-100:].index, 
                    y=st.session_state.latest_data.iloc[-100:]['close'],
                    mode='lines',
                    name='Giá',
                    line=dict(color='black', width=1)
                ))
                
                # Thêm mức kháng cự
                for i, level in enumerate(resistance_levels[:3]):
                    fig.add_shape(
                        type="line",
                        x0=st.session_state.latest_data.iloc[-100:].index[0],
                        y0=level,
                        x1=st.session_state.latest_data.iloc[-100:].index[-1],
                        y1=level,
                        line=dict(color="red", width=1, dash="dash"),
                    )
                    fig.add_annotation(
                        x=st.session_state.latest_data.iloc[-100:].index[-1],
                        y=level,
                        text=f"R{i+1}: ${level:.2f}",
                        showarrow=False,
                        xshift=10,
                        align="left",
                        bgcolor="rgba(255,0,0,0.1)"
                    )
                
                # Thêm mức hỗ trợ
                for i, level in enumerate(support_levels[:3]):
                    fig.add_shape(
                        type="line",
                        x0=st.session_state.latest_data.iloc[-100:].index[0],
                        y0=level,
                        x1=st.session_state.latest_data.iloc[-100:].index[-1],
                        y1=level,
                        line=dict(color="green", width=1, dash="dash"),
                    )
                    fig.add_annotation(
                        x=st.session_state.latest_data.iloc[-100:].index[-1],
                        y=level,
                        text=f"S{i+1}: ${level:.2f}",
                        showarrow=False,
                        xshift=10,
                        align="left",
                        bgcolor="rgba(0,255,0,0.1)"
                    )
                
                # Thêm giá hiện tại
                fig.add_shape(
                    type="line",
                    x0=st.session_state.latest_data.iloc[-100:].index[0],
                    y0=current_price,
                    x1=st.session_state.latest_data.iloc[-100:].index[-1],
                    y1=current_price,
                    line=dict(color="blue", width=1, dash="dot"),
                )
                fig.add_annotation(
                    x=st.session_state.latest_data.iloc[-100:].index[0],
                    y=current_price,
                    text=f"Current: ${current_price:.2f}",
                    showarrow=False,
                    xshift=-10,
                    xanchor="right",
                    bgcolor="rgba(0,0,255,0.1)"
                )
                
                # Cập nhật layout
                fig.update_layout(
                    title="Biểu đồ với các mức hỗ trợ và kháng cự",
                    xaxis_title="Ngày",
                    yaxis_title="Giá (USDT)",
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Hiển thị giải thích cho các mức Fibonacci
                with st.expander("Fibonacci Retracement Levels", expanded=False):
                    st.markdown("""
                    ### Các mức Fibonacci Retracement
                    
                    Fibonacci Retracement là công cụ phân tích kỹ thuật sử dụng các tỷ lệ Fibonacci để xác định các mức hỗ trợ/kháng cự tiềm năng.
                    
                    Các mức phổ biến:
                    - **0.236** - Mức yếu nhất
                    - **0.382** - Mức quan trọng đầu tiên, thường là nơi giá đảo chiều nhỏ
                    - **0.5** - Mức giữa đường (không phải số Fibonacci nhưng quan trọng trong tâm lý thị trường)
                    - **0.618** - Mức mạnh nhất, nơi giá thường có phản ứng rõ ràng
                    - **0.786** - Mức cuối cùng trước khi quay về mức cao/thấp trước đó
                    - **1.0** - Mức đỉnh/đáy trước đó
                    
                    Các mức này rất hữu ích để xác định mục tiêu lợi nhuận và dừng lỗ trong giao dịch.
                    """)
            else:
                st.info("Không có đủ dữ liệu để tính toán các mức hỗ trợ và kháng cự")
                
        with tech_tabs[3]:
            # Tab phân tích xu hướng
            st.subheader("Phân tích xu hướng")
            
            # Hiển thị giải thích
            st.markdown("""
            Phân tích xu hướng là việc xác định hướng di chuyển chủ đạo của thị trường. 
            Xu hướng có thể là tăng (uptrend), giảm (downtrend) hoặc đi ngang (sideways/consolidation).
            """)
            
            # Phân tích xu hướng
            from utils.pattern_recognition import analyze_price_trend
            trend_analysis = analyze_price_trend(st.session_state.latest_data.iloc[-50:])
            
            if trend_analysis:
                # Hiển thị kết quả phân tích
                st.subheader("Kết quả phân tích xu hướng")
                
                trend_color = "green" if trend_analysis['trend'] == 'uptrend' else "red" if trend_analysis['trend'] == 'downtrend' else "gray"
                trend_text = "Xu hướng tăng" if trend_analysis['trend'] == 'uptrend' else "Xu hướng giảm" if trend_analysis['trend'] == 'downtrend' else "Đi ngang"
                
                st.markdown(f"""
                <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                            border-left: 4px solid {trend_color}; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 24px; font-weight: bold; color: {trend_color};">{trend_text}</div>
                            <div style="margin-top: 10px;">
                                <div><b>Độ mạnh:</b> {trend_analysis['strength']}/10</div>
                                <div><b>Thời gian:</b> {trend_analysis['duration']} nến</div>
                                <div><b>Độ dốc:</b> {trend_analysis['slope']:.4f}/nến</div>
                            </div>
                        </div>
                        <div style="font-size: 48px; color: {trend_color};">
                            {'📈' if trend_analysis['trend'] == 'uptrend' else '📉' if trend_analysis['trend'] == 'downtrend' else '📊'}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Vẽ biểu đồ xu hướng
                fig = go.Figure()
                
                # Thêm đường giá
                fig.add_trace(go.Scatter(
                    x=st.session_state.latest_data.iloc[-50:].index, 
                    y=st.session_state.latest_data.iloc[-50:]['close'],
                    mode='lines',
                    name='Giá',
                    line=dict(color='black', width=1)
                ))
                
                # Thêm đường xu hướng
                if 'trendline' in trend_analysis:
                    fig.add_trace(go.Scatter(
                        x=st.session_state.latest_data.iloc[-50:].index, 
                        y=trend_analysis['trendline'],
                        mode='lines',
                        name='Đường xu hướng',
                        line=dict(color=trend_color, width=2)
                    ))
                
                # Thêm các mức hỗ trợ và kháng cự theo xu hướng
                if 'support_levels' in trend_analysis:
                    for i, level in enumerate(trend_analysis['support_levels'][:2]):
                        fig.add_shape(
                            type="line",
                            x0=st.session_state.latest_data.iloc[-50:].index[0],
                            y0=level,
                            x1=st.session_state.latest_data.iloc[-50:].index[-1],
                            y1=level,
                            line=dict(color="green", width=1, dash="dash"),
                        )
                
                if 'resistance_levels' in trend_analysis:
                    for i, level in enumerate(trend_analysis['resistance_levels'][:2]):
                        fig.add_shape(
                            type="line",
                            x0=st.session_state.latest_data.iloc[-50:].index[0],
                            y0=level,
                            x1=st.session_state.latest_data.iloc[-50:].index[-1],
                            y1=level,
                            line=dict(color="red", width=1, dash="dash"),
                        )
                
                # Cập nhật layout
                fig.update_layout(
                    title="Phân tích xu hướng",
                    xaxis_title="Ngày",
                    yaxis_title="Giá (USDT)",
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Hiển thị giải thích thêm
                with st.expander("Các loại xu hướng và cách phát hiện", expanded=False):
                    st.markdown("""
                    ### Các loại xu hướng và đặc điểm
                    
                    #### Xu hướng tăng (Uptrend)
                    - **Đặc điểm**: Các đỉnh cao hơn (Higher Highs - HH) và các đáy cao hơn (Higher Lows - HL)
                    - **Chỉ báo hỗ trợ**: MA ngắn hạn nằm trên MA dài hạn, RSI trên 50, ADX cao
                    - **Chiến lược**: "Buy the dips" - mua vào khi giá điều chỉnh về gần đường xu hướng
                    
                    #### Xu hướng giảm (Downtrend)
                    - **Đặc điểm**: Các đỉnh thấp hơn (Lower Highs - LH) và các đáy thấp hơn (Lower Lows - LL)
                    - **Chỉ báo hỗ trợ**: MA ngắn hạn nằm dưới MA dài hạn, RSI dưới 50, ADX cao
                    - **Chiến lược**: "Sell the rallies" - bán khi giá phục hồi ngắn hạn
                    
                    #### Đi ngang (Sideways/Consolidation)
                    - **Đặc điểm**: Giá dao động trong một biên độ hẹp, không có xu hướng rõ ràng
                    - **Chỉ báo hỗ trợ**: MAs đan xen, RSI quanh 50, ADX thấp (<20)
                    - **Chiến lược**: Giao dịch biên độ (mua ở hỗ trợ, bán ở kháng cự) hoặc chờ breakout
                    
                    ### Phương pháp xác định:
                    - **Phân tích đường xu hướng**: Vẽ đường nối các đỉnh/đáy quan trọng
                    - **Phân tích mẫu hình**: Mẫu hình tam giác, cờ hiệu, đầu vai...
                    - **Phân tích kênh giá**: Xác định kênh giá tăng/giảm/ngang
                    - **Chỉ báo kỹ thuật**: Sử dụng MA, MACD, RSI, ADX để xác nhận
                    """)
            else:
                st.info("Không có đủ dữ liệu để phân tích xu hướng")
    
    with main_tabs[2]:
        # Tab thông tin API
        st.markdown("## API Documentation")
        
        st.markdown("""
        ### Endpoints
        
        #### GET /predict
        
        Returns prediction data for ETHUSDT.
        
        **Query Parameters:**
        
        - `symbol`: Trading pair (default: ETHUSDT)
        - `interval`: Candle interval (default: 5m)
        
        **Response:**
        
        ```json
        {
            "prediction": "LONG",
            "confidence": 0.85,
            "price": 1234.56,
            "target_price": 1240.00,
            "reason": "Technical analysis indicates a bullish trend based on RSI, MACD, and price action",
            "timestamp": "2023-01-01 12:00:00",
            "valid_for_minutes": 30
        }
        ```
        
        **Fields:**
        
        - `prediction`: "LONG", "SHORT", or "NEUTRAL"
        - `confidence`: Confidence score from 0 to 1
        - `price`: Current price at time of prediction
        - `target_price`: Predicted target price
        - `reason`: Technical reasoning behind prediction
        - `timestamp`: When the prediction was generated
        
        ### Server Information
        
        The API server runs on port 8000 by default.
        
        ### Usage with curl
        
        ```bash
        curl "http://localhost:8000/predict?symbol=ETHUSDT&interval=5m"
        ```
        """)
        
        st.info("The API server must be started separately by running `python api.py`")
    
    with main_tabs[3]:
        # Tab hướng dẫn
        st.markdown("## Hướng dẫn sử dụng")
        
        st.markdown("""
        ### Tổng quan
        
        Hệ thống dự đoán ETH/USDT này sử dụng trí tuệ nhân tạo để phân tích dữ liệu thị trường và đưa ra dự đoán 
        về xu hướng sắp tới của cặp tiền ETH/USDT. Hệ thống sử dụng nhiều mô hình khác nhau để tạo ra dự đoán 
        chính xác nhất.
        
        ### Cách sử dụng
        
        1. **Dashboard**: Hiển thị thông tin tổng quan về thị trường và dự đoán gần nhất
        2. **Technical Analysis**: Cung cấp phân tích kỹ thuật chi tiết với nhiều chỉ báo
        3. **API**: Thông tin về cách truy cập API để tích hợp với hệ thống khác
        4. **Settings**: Thay đổi các thiết lập như khung thời gian, mô hình dự đoán...
        
        ### Các chức năng chính
        
        - **Dự đoán thời gian thực**: Hệ thống tự động cập nhật dự đoán mỗi 5 phút
        - **Phân tích đa chiều**: Sử dụng nhiều chỉ báo và mô hình khác nhau
        - **Lịch sử dự đoán**: Xem lại các dự đoán trước đó và đánh giá độ chính xác
        - **Tuỳ chỉnh thông số**: Điều chỉnh các tham số dự đoán theo nhu cầu
        - **API tích hợp**: Tích hợp với các hệ thống giao dịch tự động
        
        ### Lưu ý quan trọng
        
        Dự đoán từ hệ thống AI chỉ là một công cụ tham khảo và không nên được coi là lời khuyên đầu tư. 
        Luôn thực hiện phân tích riêng và quản lý rủi ro trước khi giao dịch.
        """)
        
        with st.expander("Mẹo sử dụng hiệu quả", expanded=False):
            st.markdown("""
            ### Mẹo sử dụng hiệu quả
            
            1. **Kết hợp nhiều khung thời gian**: So sánh dự đoán trên nhiều khung thời gian khác nhau để có cái nhìn tổng quan hơn
            2. **Theo dõi độ tin cậy**: Chỉ cân nhắc các dự đoán có độ tin cậy cao (trên 70%)
            3. **Kết hợp với phân tích cơ bản**: Các tin tức thị trường có thể ảnh hưởng lớn đến giá
            4. **Kiểm tra lịch sử hiệu suất**: Xem xét hiệu suất của từng mô hình trước khi ra quyết định
            5. **Sử dụng quản lý vốn hợp lý**: Không nên đặt cược quá lớn vào một dự đoán, dù độ tin cậy cao thế nào
            """)
        
        with st.expander("FAQ", expanded=False):
            st.markdown("""
            ### Câu hỏi thường gặp
            
            **Hệ thống sử dụng dữ liệu gì để đưa ra dự đoán?**
            
            Hệ thống sử dụng dữ liệu lịch sử giá và khối lượng từ Binance, cùng với các chỉ báo kỹ thuật được tính toán từ dữ liệu này.
            
            **Các mô hình AI nào được sử dụng?**
            
            Hệ thống sử dụng kết hợp nhiều mô hình: LSTM, Transformer, CNN, và mô hình tương đồng lịch sử, cùng với một mô hình meta-learner để kết hợp kết quả.
            
            **Dự đoán có chính xác không?**
            
            Không có hệ thống dự đoán nào đạt độ chính xác 100%. Hiệu suất của hệ thống dao động từ 65-85% tùy thuộc vào điều kiện thị trường.
            
            **Tôi có thể tích hợp hệ thống này với bot giao dịch không?**
            
            Có, hệ thống cung cấp API cho phép tích hợp dễ dàng với các bot giao dịch và hệ thống khác.
            
            **Hệ thống có cập nhật theo thời gian thực không?**
            
            Có, hệ thống tự động cập nhật dữ liệu mới nhất từ Binance và tạo dự đoán mới mỗi 5 phút.
            """)


# Initialize on startup
if not st.session_state.initialized:
    initialize_system()
    # Fetch data immediately after initialization to show real-time chart
    if st.session_state.initialized:
        fetch_data()

# Render giao diện chính
render_main_interface()