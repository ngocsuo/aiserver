"""
Main Streamlit application for ETHUSDT prediction dashboard.
PHIÊN BẢN SỬA ĐỔI: Thêm hỗ trợ proxy cho Binance API
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import datetime
import os
import logging
import json
import requests
from datetime import datetime, timedelta

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

# Thiết lập proxy trước khi import các module khác
from utils.proxy_config import configure_proxy, get_proxy_url_format
logger.info("Configuring proxy for Binance API")
proxies = configure_proxy()
proxy_url = get_proxy_url_format()
if proxies and proxy_url:
    logger.info(f"Proxy configured successfully: {proxy_url}")
    # Thiết lập biến môi trường cho proxy (cho các thư viện sử dụng)
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
else:
    logger.warning("No proxy configured, using direct connection")

# Import các module khác sau khi thiết lập proxy
import config
from utils.data_collector_factory import create_data_collector
from utils.data_processor import DataProcessor
from utils.feature_engineering import FeatureEngineer
from utils.thread_safe_logging import thread_safe_log, read_logs_from_file

# Import các module liên quan đến dự đoán và giao dịch
from prediction.prediction_engine import PredictionEngine
from utils.trading_manager import TradingManager
from utils.market_filter import MarketFilter, DecisionSupport

# Đặt title và layout
st.set_page_config(
    page_title="AI ETHUSDT Trading Oracle",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thiết lập session state 
if 'data_status' not in st.session_state:
    st.session_state.data_status = {
        "last_update": None,
        "update_count": 0,
        "is_updating": False,
        "update_time": 0
    }

if 'thread_status' not in st.session_state:
    st.session_state.thread_status = {
        "update_thread_running": False,
        "training_thread_running": False, 
        "last_training": None
    }

if 'notification' not in st.session_state:
    st.session_state.notification = None

if 'notification_type' not in st.session_state:
    st.session_state.notification_type = "info"

if 'training_log_messages' not in st.session_state:
    st.session_state.training_log_messages = []

if 'trading_state' not in st.session_state:
    st.session_state.trading_state = {
        "bot_running": False,
        "positions": [],
        "pnl": 0.0,
        "balance": 1000.0,
        "trades_history": [],
        "daily_pnl": {},
        "win_rate": 0.0,
        "current_position": None,
        "auto_trade": False
    }

# Khởi tạo các đối tượng
@st.cache_resource
def initialize_engines():
    prediction_engine = PredictionEngine()
    market_filter = MarketFilter()
    decision_support = DecisionSupport(market_filter)
    data_processor = DataProcessor()
    
    return {
        "prediction_engine": prediction_engine,
        "market_filter": market_filter,
        "decision_support": decision_support,
        "data_processor": data_processor
    }

engines = initialize_engines()

# Hàm hiển thị thông báo toast
def show_toast(message, type="info", duration=3000):
    """
    Display a toast notification that fades out.
    
    Args:
        message (str): Message to display
        type (str): Type of notification ('info', 'success', 'warning', 'error')
        duration (int): Duration in milliseconds before fading out
    """
    st.session_state.notification = message
    st.session_state.notification_type = type

# Hàm lưu trạng thái giao dịch
def save_trading_state():
    """Lưu trạng thái giao dịch vào tập tin để khôi phục khi F5 hoặc chuyển tab"""
    try:
        state_file = "trading_state.json"
        with open(state_file, "w") as f:
            # Chỉ lưu các trường cần thiết và có thể serialize
            save_state = {
                "bot_running": st.session_state.trading_state["bot_running"],
                "pnl": st.session_state.trading_state["pnl"],
                "balance": st.session_state.trading_state["balance"],
                "win_rate": st.session_state.trading_state["win_rate"],
                "auto_trade": st.session_state.trading_state["auto_trade"],
                "trades_history": st.session_state.trading_state["trades_history"][-50:] if st.session_state.trading_state["trades_history"] else [],
                "daily_pnl": st.session_state.trading_state["daily_pnl"]
            }
            json.dump(save_state, f)
    except Exception as e:
        logger.error(f"Error saving trading state: {e}")

# Hàm tải trạng thái giao dịch
def load_trading_state():
    """Tải trạng thái giao dịch từ tập tin"""
    try:
        state_file = "trading_state.json"
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                saved_state = json.load(f)
                for key, value in saved_state.items():
                    st.session_state.trading_state[key] = value
    except Exception as e:
        logger.error(f"Error loading trading state: {e}")

# Khởi tạo hệ thống
def initialize_system():
    """Initialize the prediction system"""
    # Load trading state from file
    load_trading_state()
    
    # Thiết lập API keys
    if hasattr(config, 'BINANCE_API_KEY') and config.BINANCE_API_KEY:
        logger.info("Binance API keys detected in config")
    
    # Khởi tạo trading manager
    trading_manager = TradingManager(
        api_key=config.BINANCE_API_KEY, 
        api_secret=config.BINANCE_API_SECRET
    )
    
    # Lưu vào session_state
    if 'trading_manager' not in st.session_state:
        st.session_state.trading_manager = trading_manager
        
    # Tạo data collector
    if 'data_collector' not in st.session_state:
        logger.info("Creating data collector with proxy...")
        st.session_state.data_collector = create_data_collector()
    
    # Cung cấp data collector cho các đối tượng khác
    engines["market_filter"].set_data_collector(st.session_state.data_collector)
    engines["decision_support"].set_data_collector(st.session_state.data_collector)
    
    # Khởi tạo prediction count
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    
    # Khởi tạo historical predictions
    if 'historical_predictions' not in st.session_state:
        st.session_state.historical_predictions = []
    
    # Start data update thread if not running
    if not st.session_state.thread_status["update_thread_running"]:
        start_update_thread()

# Hàm lấy dữ liệu thời gian thực 
def fetch_realtime_data():
    """Fetch the latest real-time data from Binance for the dashboard"""
    try:
        st.session_state.data_status["is_updating"] = True
        start_time = time.time()
        
        # Cập nhật dữ liệu từ data collector
        updated_data = st.session_state.data_collector.update_data()
        
        # Lấy dữ liệu khung thời gian chính
        primary_data = updated_data.get(config.TIMEFRAMES["primary"], None)
        
        # Cập nhật trạng thái
        st.session_state.data_status["last_update"] = datetime.now().strftime("%H:%M:%S")
        st.session_state.data_status["update_count"] += 1
        st.session_state.data_status["update_time"] = time.time() - start_time
        st.session_state.data_status["is_updating"] = False
        
        return primary_data
    except Exception as e:
        logger.error(f"Error fetching realtime data: {e}")
        st.session_state.data_status["is_updating"] = False
        return None

# Thread để fetch dữ liệu lịch sử
def fetch_historical_data_thread():
    """Fetch historical data from Binance for training in a separate thread"""
    try:
        thread_safe_log("Bắt đầu tải dữ liệu lịch sử...")
        
        # Lấy collector từ factory
        data_collector = create_data_collector()
        
        # Tạo một thread riêng để cập nhật trạng thái
        update_thread = threading.Thread(target=update_status)
        update_thread.daemon = True
        update_thread.start()
        
        # Nếu có ngày bắt đầu lịch sử, lấy dữ liệu từ đó đến hiện tại
        if hasattr(config, 'HISTORICAL_START_DATE') and config.HISTORICAL_START_DATE:
            thread_safe_log(f"Lấy dữ liệu lịch sử từ {config.HISTORICAL_START_DATE} đến hiện tại...")
            
            end_date = datetime.now().strftime("%Y-%m-%d")
            data = data_collector.collect_historical_data(
                timeframe=config.TIMEFRAMES["primary"],
                start_date=config.HISTORICAL_START_DATE,
                end_date=end_date
            )
        else:
            # Lấy dữ liệu với số lượng nến được cấu hình
            thread_safe_log(f"Lấy {config.LOOKBACK_PERIODS} nến gần nhất...")
            data = data_collector.collect_historical_data(
                timeframe=config.TIMEFRAMES["primary"],
                limit=config.LOOKBACK_PERIODS
            )
            
        if data is not None:
            thread_safe_log(f"Đã thu thập {len(data)} nến dữ liệu.")
            
            # Lưu dữ liệu vào file
            data_file = f"historical_data_{config.TIMEFRAMES['primary']}.pkl"
            data.to_pickle(data_file)
            thread_safe_log(f"Đã lưu dữ liệu vào {data_file}")
            
            return data
        else:
            thread_safe_log("Không tải được dữ liệu lịch sử.")
            return None
    except Exception as e:
        thread_safe_log(f"Lỗi khi tải dữ liệu lịch sử: {str(e)}")
        return None
    
    def update_status():
        """Cập nhật trạng thái tải dữ liệu"""
        dots = 0
        while True:
            try:
                thread_safe_log(f"Đang tải dữ liệu{'.' * dots}")
                dots = (dots + 1) % 4
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error updating status: {e}")
                break

# Hàm train models sử dụng thread-safe logging
def train_models():
    """Train all prediction models in a background thread"""
    if st.session_state.thread_status["training_thread_running"]:
        st.warning("Quá trình huấn luyện đang diễn ra. Vui lòng đợi đến khi hoàn tất.")
        return False
    
    # Tạo file training_logs.txt nếu chưa tồn tại
    if not os.path.exists("training_logs.txt"):
        with open("training_logs.txt", "w") as f:
            f.write("# Training logs started\n")
    
    thread_safe_log("Bắt đầu quá trình huấn luyện mô hình...")
    
    # Tạo thread huấn luyện
    training_thread = threading.Thread(
        target=train_models_background,
        name="train_models_background"
    )
    training_thread.daemon = True
    training_thread.start()
    
    st.session_state.thread_status["training_thread_running"] = True
    st.session_state.thread_status["last_training"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Hiển thị thông báo cho user
    st.success("Đã bắt đầu huấn luyện mô hình trong nền. Kiểm tra logs để theo dõi tiến trình.")
    
    return True

def train_models_background():
    """Hàm huấn luyện chạy trong thread riêng biệt"""
    from utils.thread_safe_logging import thread_safe_log
    
    try:
        thread_safe_log("Bắt đầu huấn luyện mô hình AI trong thread riêng...")
        thread_safe_log("LƯU Ý: Đang sử dụng phiên bản an toàn thread, tránh truy cập session_state")
        
        # QUAN TRỌNG: KHÔNG truy cập st.session_state trong thread này!
        # Thay vì lấy dữ liệu từ session_state, chúng ta sẽ tải dữ liệu mới
        
        from utils.data_collector_factory import create_data_collector
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
        
        if data is None or data.empty:
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
        
        # Train all models
        thread_safe_log("Huấn luyện tất cả các mô hình...")
        # QUAN TRỌNG: CHỈ nhận giá trị models, không nhận histories
        models = model_trainer.train_all_models(sequence_data, image_data)
        
        thread_safe_log("Huấn luyện thành công tất cả các mô hình!")
        
        # Lưu trạng thái huấn luyện vào file
        try:
            import json
            training_result = {
                "success": True,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": "Huấn luyện thành công tất cả các mô hình"
            }
            with open('training_result.json', 'w') as f:
                json.dump(training_result, f)
        except Exception as e:
            thread_safe_log(f"Lỗi lưu kết quả huấn luyện: {str(e)}")
                
    except Exception as e:
        from utils.thread_safe_logging import thread_safe_log
        thread_safe_log(f"LỖI trong quá trình huấn luyện: {str(e)}")
    finally:
        # QUAN TRỌNG: KHÔNG truy cập st.session_state ở đây!
        # Thay vào đó, ghi log về việc hoàn thành
        thread_safe_log("Thread huấn luyện đã kết thúc.")

# Hàm cập nhật dữ liệu liên tục
def update_data_continuously():
    """Update data continuously in a separate thread"""
    while st.session_state.thread_status["update_thread_running"]:
        try:
            start_time = time.time()
            
            # Fetch data
            data = fetch_realtime_data()
            
            # Sleep to maintain update interval
            elapsed = time.time() - start_time
            sleep_time = max(1, config.UPDATE_INTERVAL - elapsed)
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in update thread: {e}")
            time.sleep(5)  # Sleep a bit longer on error

# Hàm bắt đầu thread cập nhật
def start_update_thread():
    """Start the continuous update thread"""
    if not st.session_state.thread_status["update_thread_running"]:
        update_thread = threading.Thread(target=update_data_continuously)
        update_thread.daemon = True
        update_thread.start()
        st.session_state.thread_status["update_thread_running"] = True
        logger.info("Data update thread started")
        return True
    return False

# Hàm dừng thread cập nhật
def stop_update_thread():
    """Stop the continuous update thread"""
    st.session_state.thread_status["update_thread_running"] = False
    logger.info("Data update thread stopped")
    return True

# Thêm các hàm hiển thị biểu đồ và thông tin khác theo cần thiết
# ...

# Hiển thị phần giao diện
def render_main_interface():
    """
    Render the main Streamlit interface
    """
    # Khởi tạo hệ thống nếu cần
    initialize_system()
    
    # Hiển thị tiêu đề chính
    st.title("🤖 AI ETHUSDT Trading Oracle")
    
    # Hiển thị thông báo toast nếu có
    if st.session_state.notification:
        if st.session_state.notification_type == "info":
            st.info(st.session_state.notification)
        elif st.session_state.notification_type == "success":
            st.success(st.session_state.notification)
        elif st.session_state.notification_type == "warning":
            st.warning(st.session_state.notification)
        elif st.session_state.notification_type == "error":
            st.error(st.session_state.notification)
        
        # Xóa thông báo sau khi hiển thị
        st.session_state.notification = None
    
    # Chia layout thành các tab
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧠 Huấn luyện", "💰 Giao dịch", "⚙️ Cài đặt"])
    
    with tab1:
        st.header("Bảng điều khiển")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Biểu đồ giá & Phân tích kỹ thuật")
            
            # Lấy dữ liệu giá hiện tại
            primary_data = None
            if hasattr(st.session_state, 'data_collector'):
                if hasattr(st.session_state.data_collector, 'data'):
                    primary_data = st.session_state.data_collector.data.get(config.TIMEFRAMES["primary"], None)
            
            if primary_data is not None and not primary_data.empty:
                # Tạo biểu đồ giá
                fig = go.Figure(data=[go.Candlestick(
                    x=primary_data.index,
                    open=primary_data['open'],
                    high=primary_data['high'],
                    low=primary_data['low'],
                    close=primary_data['close']
                )])
                
                # Cấu hình biểu đồ
                fig.update_layout(
                    title="ETHUSDT Candlestick Chart",
                    xaxis_title="Time",
                    yaxis_title="Price (USDT)",
                    template="plotly_dark"
                )
                
                # Hiển thị biểu đồ
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Không có dữ liệu để hiển thị biểu đồ")
        
        with col2:
            st.subheader("Trạng thái hệ thống")
            
            # Hiển thị trạng thái kết nối
            if hasattr(st.session_state, 'data_collector') and hasattr(st.session_state.data_collector, 'connection_status'):
                connection_status = st.session_state.data_collector.connection_status
                
                if connection_status["connected"]:
                    st.success("✅ Kết nối đến Binance API thành công")
                    if connection_status.get("using_proxy", False):
                        st.info("🔄 Đang sử dụng proxy")
                else:
                    st.error(f"❌ Kết nối thất bại: {connection_status['message']}")
                    
                    # Kiểm tra nếu lỗi là do hạn chế địa lý
                    if "geographic restriction" in connection_status.get("message", "").lower() or "restricted location" in connection_status.get("message", "").lower():
                        st.warning("⚠️ Lỗi hạn chế địa lý. Hệ thống sẽ hoạt động bình thường khi triển khai trên server riêng của bạn.")
            else:
                st.warning("⚠️ Chưa có thông tin kết nối")
            
            # Hiển thị thông tin cập nhật dữ liệu
            st.subheader("Thông tin cập nhật")
            st.write(f"Cập nhật lần cuối: {st.session_state.data_status['last_update'] or 'Chưa cập nhật'}")
            st.write(f"Số lần cập nhật: {st.session_state.data_status['update_count']}")
            st.write(f"Thời gian cập nhật: {st.session_state.data_status['update_time']:.2f}s")
            
            # Hiển thị thống kê huấn luyện
            st.subheader("Thông tin huấn luyện")
            st.write(f"Huấn luyện lần cuối: {st.session_state.thread_status['last_training'] or 'Chưa huấn luyện'}")
            
            # Nút huấn luyện thủ công
            st.button("🔄 Huấn luyện lại", type="primary", on_click=train_models)
    
    with tab2:
        st.header("Huấn luyện mô hình AI")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Tải dữ liệu huấn luyện")
            
            if st.button("📥 Tải dữ liệu lịch sử", key="load_historical_data"):
                with st.spinner("Đang tải dữ liệu lịch sử..."):
                    # Khởi động thread tải dữ liệu
                    threading.Thread(target=fetch_historical_data_thread).start()
            
            st.subheader("Huấn luyện thủ công")
            
            # Nút huấn luyện
            train_button = st.button("🧠 Huấn luyện mô hình", type="primary", key="train_models_button")
            if train_button:
                if train_models():
                    st.success("Đã bắt đầu huấn luyện mô hình. Quá trình này có thể mất vài phút.")
        
        with col2:
            st.subheader("Nhật ký huấn luyện")
            
            # Đọc logs từ file
            logs = read_logs_from_file(log_file="training_logs.txt", max_lines=20) 
            
            # Hiển thị logs
            if logs:
                for log in logs:
                    st.text(log)
            else:
                st.info("Chưa có nhật ký huấn luyện")
            
            # Nút làm mới nhật ký
            if st.button("🔄 Làm mới nhật ký"):
                st.experimental_rerun()
    
    with tab3:
        st.header("Giao dịch tự động")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Cài đặt giao dịch")
            
            # Kiểm tra kết nối
            trading_manager = st.session_state.trading_manager if 'trading_manager' in st.session_state else None
            
            if trading_manager:
                if trading_manager.connection_status["connected"]:
                    st.success("✅ Kết nối API giao dịch thành công")
                    
                    # Hiển thị thông tin tài khoản
                    try:
                        balance = trading_manager.get_futures_account_balance()
                        st.metric("Số dư USDT", f"{balance:.2f}" if balance else "N/A")
                    except Exception as e:
                        st.error(f"Lỗi khi lấy thông tin tài khoản: {str(e)}")
                    
                    # Thiết lập đòn bẩy
                    leverage = st.slider("Đòn bẩy", min_value=1, max_value=20, value=5, step=1)
                    
                    # Thiết lập % vốn
                    capital_percent = st.slider("Phần trăm vốn (%)", min_value=1, max_value=100, value=10, step=1)
                    
                    # Thiết lập TP/SL
                    col_tp, col_sl = st.columns(2)
                    with col_tp:
                        take_profit = st.slider("Take Profit (%)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
                    with col_sl:
                        stop_loss = st.slider("Stop Loss (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
                    
                    # Bật/tắt giao dịch tự động
                    auto_trade = st.checkbox("Giao dịch tự động", value=st.session_state.trading_state["auto_trade"])
                    
                    # Lưu cấu hình
                    if auto_trade != st.session_state.trading_state["auto_trade"]:
                        st.session_state.trading_state["auto_trade"] = auto_trade
                        save_trading_state()
                        
                        if auto_trade:
                            st.success("Đã bật giao dịch tự động")
                        else:
                            st.warning("Đã tắt giao dịch tự động")
                else:
                    st.error(f"❌ Kết nối API giao dịch thất bại: {trading_manager.connection_status['message']}")
                    
                    # Hiển thị thông tin chi tiết về lỗi
                    st.info("Đảm bảo đã cấu hình đúng API key và API secret trong config.py")
                    
                    if "geographic restriction" in trading_manager.connection_status.get("message", "").lower() or "restricted location" in trading_manager.connection_status.get("message", "").lower():
                        st.warning("⚠️ Lỗi hạn chế địa lý. Hệ thống sẽ hoạt động bình thường khi triển khai trên server riêng của bạn.")
        
        with col2:
            st.subheader("Lịch sử giao dịch")
            
            # Hiển thị PnL
            st.metric("Lợi nhuận", f"{st.session_state.trading_state['pnl']:.2f} USDT", 
                    delta=f"{st.session_state.trading_state['pnl']:.2f}" if st.session_state.trading_state['pnl'] != 0 else None)
            
            # Hiển thị win rate
            st.metric("Tỷ lệ thắng", f"{st.session_state.trading_state['win_rate']:.1f}%")
            
            # Hiển thị lịch sử giao dịch dưới dạng bảng
            trades = st.session_state.trading_state["trades_history"]
            if trades:
                # Convert to DataFrame
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df)
            else:
                st.info("Chưa có giao dịch nào")
    
    with tab4:
        st.header("Cài đặt hệ thống")
        
        # Hiển thị thông tin cấu hình
        st.subheader("Cấu hình kết nối")
        
        api_key_placeholder = "***" + config.BINANCE_API_KEY[-4:] if hasattr(config, 'BINANCE_API_KEY') and config.BINANCE_API_KEY else "Chưa cài đặt"
        api_secret_placeholder = "***" + config.BINANCE_API_SECRET[-4:] if hasattr(config, 'BINANCE_API_SECRET') and config.BINANCE_API_SECRET else "Chưa cài đặt"
        
        st.write(f"🔑 API Key: {api_key_placeholder}")
        st.write(f"🔒 API Secret: {api_secret_placeholder}")
        
        # Hiển thị thông tin proxy
        st.subheader("Cấu hình proxy")
        
        proxy_url = get_proxy_url_format()
        
        if proxy_url:
            st.success(f"✅ Proxy: Đã cấu hình")
            st.write(f"🔄 URL: {proxy_url.replace('hvnteam:matkhau123', 'username:******')}")
        else:
            st.warning("⚠️ Proxy: Chưa cấu hình")
        
        # Hiển thị mẹo triển khai
        st.subheader("Mẹo triển khai")
        
        st.info("""
        **Để triển khai trên server riêng:**
        
        1. Clone repository về server
        2. Cài đặt các gói phụ thuộc: `pip install -r requirements.txt`
        3. Cấu hình API keys trong file .env hoặc config.py
        4. Khởi động với: `streamlit run app.py --server.port=5000 --server.address=0.0.0.0`
        
        **Để khắc phục lỗi hạn chế địa lý:**
        
        1. Sử dụng server riêng ở vùng không bị chặn (ví dụ: Singapore, Nhật, Úc)
        2. Hoặc sử dụng proxy như đã cấu hình trong ứng dụng này
        """)

# Chạy ứng dụng
if __name__ == "__main__":
    render_main_interface()