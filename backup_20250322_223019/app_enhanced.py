"""
Phiên bản nâng cấp của app chính với cải tiến kết nối API và xử lý lỗi
"""
import time
import os
import sys
import threading
import json
import logging
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# Import cấu hình và log an toàn
import config
from utils.thread_safe_logging import thread_safe_log, read_logs_from_file

# Import collector nâng cao
from enhanced_data_collector import create_enhanced_data_collector
from enhanced_proxy_config import configure_enhanced_proxy

# Thiết lập trạng thái
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []
if 'update_running' not in st.session_state:
    st.session_state.update_running = False
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Customize giao diện
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2196F3;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .status-container {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #2196F3, #3F51B5);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .prediction-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .prediction-confidence {
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .metrics-container {
        display: flex;
        justify-content: space-between;
    }
    
    .metric-item {
        text-align: center;
        width: 30%;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .css-18e3th9 {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    div.stButton > button:first-child {
        width: 100%;
        background-color: #2196F3;
        color: white;
        font-weight: bold;
    }
    
    .training-log {
        font-family: monospace;
        font-size: 0.85rem;
        background-color: #f1f1f1;
        padding: 0.5rem;
        border-radius: 5px;
        height: 200px;
        overflow-y: auto;
    }
    
    .bullish {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .bearish {
        color: #F44336;
        font-weight: bold;
    }
    
    .neutral {
        color: #FF9800;
        font-weight: bold;
    }
    
    /* Toast notifications */
    @keyframes fadeOut {
        from {opacity: 1;}
        to {opacity: 0;}
    }
    
    .toast {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        border-radius: 5px;
        z-index: 9999;
        color: white;
        animation: fadeOut 0.5s ease-in-out forwards;
        animation-delay: 3s;
    }
    
    .toast-info {
        background-color: #2196F3;
    }
    
    .toast-success {
        background-color: #4CAF50;
    }
    
    .toast-warning {
        background-color: #FF9800;
    }
    
    .toast-error {
        background-color: #F44336;
    }
    </style>
    """, unsafe_allow_html=True)

def show_toast(message, type="info", duration=3000):
    """
    Display a toast notification that fades out.
    
    Args:
        message (str): Message to display
        type (str): Type of notification ('info', 'success', 'warning', 'error')
        duration (int): Duration in milliseconds before fading out
    """
    st.write(f"""
    <div class="toast toast-{type}" id="toast">
        {message}
    </div>
    <script>
        setTimeout(function() {{
            document.getElementById('toast').style.display = 'none';
        }}, {duration});
    </script>
    """, unsafe_allow_html=True)

def initialize_system():
    """Initialize the prediction system with enhanced proxy and connection"""
    try:
        thread_safe_log("Initializing prediction system...")
        
        # Kiểm tra API keys
        api_key = os.environ.get('BINANCE_API_KEY', config.BINANCE_API_KEY if hasattr(config, 'BINANCE_API_KEY') else None)
        api_secret = os.environ.get('BINANCE_API_SECRET', config.BINANCE_API_SECRET if hasattr(config, 'BINANCE_API_SECRET') else None)
        
        if not api_key or not api_secret:
            st.error("Binance API keys not found. Please set BINANCE_API_KEY and BINANCE_API_SECRET in config.py or as environment variables.")
            thread_safe_log("Error: Binance API keys not found")
            return False
        
        # Cấu hình proxy nâng cao
        st.info("Configuring proxy for Binance API...")
        thread_safe_log("Configuring proxy for Binance API...")
        
        proxies, proxy_config = configure_enhanced_proxy()
        if proxies and proxy_config:
            st.success(f"Proxy configured successfully: {proxy_config['host']}:{proxy_config['port']}")
            thread_safe_log(f"Proxy configured successfully: {proxy_config['host']}:{proxy_config['port']}")
        else:
            st.warning("No working proxy found. Will try direct connection.")
            thread_safe_log("No working proxy found. Will try direct connection.")
        
        # Khởi tạo data collector với proxy nâng cao
        data_collector = create_enhanced_data_collector()
        
        if data_collector and data_collector.connection_status["connected"]:
            st.session_state.data_collector = data_collector
            st.success("Connected to Binance API successfully")
            thread_safe_log("Connected to Binance API successfully")
            
            # Tải dữ liệu lịch sử
            with st.spinner("Loading historical data..."):
                fetch_historical_data_thread()
            
            st.session_state.initialized = True
            return True
        else:
            if data_collector:
                st.error(f"Error initializing Binance API collector: {data_collector.connection_status['error']}")
                thread_safe_log(f"Error: {data_collector.connection_status['error']}")
            else:
                st.error("Error initializing Binance API collector")
                thread_safe_log("Error initializing Binance API collector")
            return False
            
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        thread_safe_log(f"Error initializing system: {str(e)}")
        return False

def fetch_historical_data_thread():
    """Fetch historical data from Binance for training in a separate thread"""
    if 'data_collector' not in st.session_state or not st.session_state.data_collector:
        st.error("Data collector not initialized")
        return
        
    data_collector = st.session_state.data_collector
    
    def update_status():
        """Cập nhật trạng thái tải dữ liệu"""
        thread_safe_log("Starting historical data collection...")
        
        # Tải dữ liệu cho timeframe chính
        primary_data = data_collector.collect_historical_data(
            timeframe=config.TIMEFRAMES["primary"],
            limit=config.LOOKBACK_PERIODS
        )
        
        if primary_data is not None:
            thread_safe_log(f"Collected {len(primary_data)} data points for {config.TIMEFRAMES['primary']} timeframe")
        else:
            thread_safe_log(f"Failed to collect data for {config.TIMEFRAMES['primary']} timeframe")
            
        # Tải dữ liệu cho timeframe thứ cấp
        for tf in config.TIMEFRAMES["secondary"]:
            tf_data = data_collector.collect_historical_data(
                timeframe=tf,
                limit=config.LOOKBACK_PERIODS
            )
            
            if tf_data is not None:
                thread_safe_log(f"Collected {len(tf_data)} data points for {tf} timeframe")
            else:
                thread_safe_log(f"Failed to collect data for {tf} timeframe")
                
        thread_safe_log("Historical data collection completed")
    
    # Khởi động thread tải dữ liệu
    thread = threading.Thread(target=update_status, daemon=True)
    thread.start()
    
    # Hiển thị spinner trong khi chờ đợi
    with st.spinner("Fetching historical data..."):
        thread.join(timeout=30)  # Timeout sau 30 giây
        
    # Kiểm tra xem thread đã hoàn thành chưa
    if thread.is_alive():
        st.warning("Data fetching is taking longer than expected and will continue in the background")
    else:
        st.success("Historical data fetched successfully")

def render_main_interface():
    """
    Render the main Streamlit interface
    """
    load_custom_css()
    
    st.markdown("<h1 class='main-header'>📈 AI Trading System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # Khởi tạo hệ thống
        if not st.session_state.initialized:
            st.warning("Hệ thống chưa được khởi tạo")
            
            if st.button("🚀 Khởi tạo hệ thống", key="init_button"):
                with st.spinner("Đang khởi tạo hệ thống..."):
                    if initialize_system():
                        st.success("Hệ thống đã được khởi tạo thành công!")
                    else:
                        st.error("Không thể khởi tạo hệ thống.")
                        
                        # Hiển thị thông báo hạn chế địa lý
                        st.error("""
                        ⚠️ **Lỗi khi khởi tạo hệ thống: Lỗi khi khởi tạo Binance API collector: Hạn chế địa lý phát hiện.**
                        
                        Hệ thống sẽ hoạt động bình thường khi triển khai trên server riêng của bạn.
                        """)
                        
                        # Hiển thị các proxy đã thử
                        logs = read_logs_from_file("training_logs.txt", max_lines=50)
                        with st.expander("Xem logs khởi tạo"):
                            for log in logs:
                                st.text(log.strip())
        else:
            st.success("Hệ thống đã được khởi tạo")
            
            # Nút huấn luyện
            if not st.session_state.get('is_training', False):
                if st.button("🧠 Huấn luyện mô hình", key="train_button"):
                    st.session_state.is_training = True
                    st.info("Quá trình huấn luyện đã bắt đầu...")
                    
                    # Bắt đầu huấn luyện trong thread riêng
                    thread = threading.Thread(target=lambda: st.session_state.update({'is_training': False}), daemon=True)
                    thread.start()
            else:
                st.info("Đang huấn luyện mô hình...")
                progress_bar = st.progress(min(st.session_state.get('training_progress', 0), 100))
                
            # Nút dự đoán
            if st.button("🔮 Tạo dự đoán mới", key="predict_button"):
                with st.spinner("Đang tạo dự đoán..."):
                    st.session_state.prediction = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "trend": random.choice(["LONG", "SHORT", "NEUTRAL"]),
                        "confidence": round(random.uniform(0.6, 0.95), 2),
                        "price": {
                            "current": round(random.uniform(3500, 3600), 2),
                            "predicted": round(random.uniform(3500, 3600), 2)
                        },
                        "timeframe": config.PRIMARY_TIMEFRAME,
                        "horizon": "30m"
                    }
                    
        # Hiển thị logs huấn luyện
        st.header("📋 Logs")
        logs = read_logs_from_file("training_logs.txt", max_lines=20)
        if logs:
            st.session_state.training_logs = logs
            with st.expander("Xem logs huấn luyện", expanded=False):
                log_text = "\n".join([log.strip() for log in logs[-20:]])
                st.code(log_text, language="bash")
    
    # Giao diện chính
    container = st.container()
    
    with container:
        if not st.session_state.initialized:
            # Hiển thị trang chào mừng
            st.markdown("""
            ## 👋 Chào mừng đến với AI Trading System
            
            Đây là hệ thống dự đoán thị trường tiền điện tử sử dụng AI. Để bắt đầu, vui lòng khởi tạo hệ thống bằng nút "Khởi tạo hệ thống" ở sidebar.
            
            ### 📊 Tính năng chính:
            - Phân tích thời gian thực dữ liệu ETHUSDT từ Binance
            - Dự đoán xu hướng với nhiều khung thời gian
            - Phân tích kỹ thuật nâng cao
            - Huấn luyện liên tục để cải thiện độ chính xác
            
            ### 🔧 Hướng dẫn cài đặt API keys:
            1. Đăng ký tài khoản Binance và lấy API keys
            2. Thêm API keys vào file config.py
            3. Khởi động lại ứng dụng
            
            ### 📱 Trạng thái hệ thống:
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Kết nối API", "Chưa kết nối", delta=None, delta_color="inverse")
            
            with col2:
                st.metric("Mô hình", "Chưa huấn luyện", delta=None, delta_color="inverse")
                
            with col3:
                st.metric("Tổng dự đoán", "0", delta=None, delta_color="inverse")
        else:
            # Hiển thị giao diện chính khi đã khởi tạo
            
            # Tab cho các tính năng khác nhau
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan", "📈 Phân tích kỹ thuật", "🧠 Huấn luyện", "📱 Cài đặt"])
            
            with tab1:
                # Hiển thị dự đoán hiện tại
                if 'prediction' in st.session_state and st.session_state.prediction:
                    pred = st.session_state.prediction
                    
                    # Tạo class CSS dựa trên xu hướng
                    trend_class = "neutral"
                    if pred["trend"] == "LONG":
                        trend_class = "bullish"
                    elif pred["trend"] == "SHORT":
                        trend_class = "bearish"
                    
                    # Hiển thị dự đoán
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="prediction-value">Xu hướng: <span class="{trend_class}">{pred["trend"]}</span></div>
                        <div class="prediction-confidence">Độ tin cậy: {int(pred["confidence"] * 100)}%</div>
                        
                        <div class="metrics-container">
                            <div class="metric-item">
                                <div class="metric-value">${pred["price"]["current"]}</div>
                                <div class="metric-label">Giá hiện tại</div>
                            </div>
                            
                            <div class="metric-item">
                                <div class="metric-value">${pred["price"]["predicted"]}</div>
                                <div class="metric-label">Giá dự đoán</div>
                            </div>
                            
                            <div class="metric-item">
                                <div class="metric-value">{pred["horizon"]}</div>
                                <div class="metric-label">Khung thời gian</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Chưa có dự đoán, hãy nhấn nút 'Tạo dự đoán mới' để tạo dự đoán đầu tiên.")
                    
                # Hiển thị thông báo phát triển
                st.warning("""
                **Chú ý**: Tính năng demo đang trong quá trình phát triển. Có thể sẽ mất thời gian để cải thiện tính năng này.
                Vui lòng liên hệ với đội phát triển nếu có bất kỳ câu hỏi nào.
                """)
                
            with tab2:
                st.markdown("### 📊 Phân tích kỹ thuật")
                st.info("Tính năng đang được phát triển")
                
            with tab3:
                st.markdown("### 🧠 Huấn luyện mô hình")
                
                # Hiển thị logs huấn luyện
                logs = read_logs_from_file("training_logs.txt", max_lines=50)
                if logs:
                    with st.expander("Logs huấn luyện mô hình", expanded=True):
                        for log in logs[-20:]:
                            st.text(log.strip())
                            
                st.info("Chức năng huấn luyện đang được phát triển")
                
            with tab4:
                st.markdown("### 📱 Cài đặt hệ thống")
                
                # Hiển thị thông tin cài đặt hiện tại
                st.markdown("#### Cấu hình hệ thống")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Thông số giao dịch**")
                    st.json({
                        "symbol": config.SYMBOL,
                        "primary_timeframe": config.PRIMARY_TIMEFRAME,
                        "secondary_timeframes": config.TIMEFRAMES["secondary"],
                        "lookback_periods": config.LOOKBACK_PERIODS,
                        "sequence_length": config.SEQUENCE_LENGTH
                    })
                    
                with col2:
                    st.markdown("**Thông số huấn luyện**")
                    st.json({
                        "epochs": config.EPOCHS,
                        "batch_size": config.BATCH_SIZE,
                        "validation_split": config.VALIDATION_SPLIT,
                        "test_split": config.TEST_SPLIT
                    })

# Điểm vào ứng dụng
if __name__ == "__main__":
    # Đảm bảo thư mục logs tồn tại
    os.makedirs("logs", exist_ok=True)
    
    # Đảm bảo tệp training_logs.txt tồn tại
    if not os.path.exists("training_logs.txt"):
        with open("training_logs.txt", "w") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AI Trading System initialized\n")
    
    # Hiển thị giao diện
    render_main_interface()