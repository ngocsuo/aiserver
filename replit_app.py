"""
Phiên bản nhẹ của ETHUSDT Dashboard tối ưu cho Replit.

Phiên bản này:
1. Giảm thiểu sử dụng tài nguyên hệ thống
2. Hạn chế số lượng kết nối đồng thời đến Binance API
3. Tắt các tính năng không cần thiết
4. Lưu trữ dữ liệu trong bộ nhớ đệm để tránh tải lại thường xuyên
"""

import os
import sys
import time
import logging
import streamlit as st
import pandas as pd
import numpy as np
import threading
import datetime
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
import warnings
import random

# Tắt cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# Thiết lập logging đơn giản
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("replit_app.log"),
        logging.StreamHandler()
    ]
)

# Tạo biến môi trường để chỉ định đây là phiên bản Replit
os.environ["REPLIT_VERSION"] = "lite"

# Giảm số lượng khung thời gian được sử dụng
TIMEFRAMES = ["5m"]  # Chỉ sử dụng khung thời gian 5 phút
PREDICTION_HORIZONS = ["30m"]  # Chỉ dự đoán trước 30 phút

# Đặt giới hạn dữ liệu lịch sử
MAX_HISTORICAL_DAYS = 30  # Chỉ lấy dữ liệu 30 ngày gần nhất

# Đặt khoảng thời gian cập nhật dữ liệu
UPDATE_INTERVAL = 300  # Cập nhật dữ liệu mỗi 5 phút thay vì mỗi phút

# Biến toàn cục
data_cache = {}  # Cache dữ liệu để tránh phải tải lại
model_predictions = {}  # Cache dự đoán mô hình
last_update_time = None  # Thời gian cập nhật dữ liệu cuối cùng
update_thread = None  # Thread cập nhật dữ liệu
update_running = False  # Trạng thái cập nhật

# Cấu hình CSS cho giao diện đơn giản
def load_css():
    """Tải CSS đơn giản cho giao diện"""
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .dataframe {
        font-size: 14px;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 1200px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Kết nối đến Binance API
def connect_to_binance():
    """Kết nối đơn giản đến Binance API không sử dụng thư viện"""
    try:
        # Kiểm tra kết nối
        response = requests.get("https://api.binance.com/api/v3/ping")
        if response.status_code == 200:
            # Lấy thời gian máy chủ
            time_response = requests.get("https://api.binance.com/api/v3/time")
            if time_response.status_code == 200:
                server_time = time_response.json()["serverTime"]
                logging.info(f"Kết nối thành công đến Binance API, server time: {server_time}")
                return True
            else:
                logging.error(f"Lỗi khi lấy thời gian máy chủ: {time_response.status_code}")
                return False
        else:
            logging.error(f"Lỗi khi ping Binance API: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Lỗi khi kết nối đến Binance API: {str(e)}")
        return False

# Lấy dữ liệu OHLCV từ Binance
def get_klines(symbol="ETHUSDT", interval="5m", limit=100):
    """Lấy dữ liệu nến từ Binance API"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url)
        if response.status_code == 200:
            klines = response.json()
            # Chuyển đổi dữ liệu thành DataFrame
            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", 
                                              "close_time", "quote_asset_volume", "number_of_trades",
                                              "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
            # Chuyển đổi kiểu dữ liệu
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Chuyển đổi timestamp thành datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            return df
        else:
            logging.error(f"Lỗi khi lấy dữ liệu nến: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Lỗi khi lấy dữ liệu nến: {str(e)}")
        return None

# Tính toán chỉ báo kỹ thuật đơn giản
def add_indicators(df):
    """Thêm một số chỉ báo kỹ thuật cơ bản"""
    # SMA
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    
    # EMA
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    
    # MACD
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    return df

# Hàm dự đoán đơn giản
def generate_prediction(df):
    """Tạo dự đoán đơn giản dựa trên chỉ báo kỹ thuật"""
    # Lấy giá trị gần nhất
    last_close = df["close"].iloc[-1]
    last_rsi = df["rsi"].iloc[-1]
    last_macd = df["macd"].iloc[-1]
    last_macd_signal = df["macd_signal"].iloc[-1]
    last_bb_upper = df["bb_upper"].iloc[-1]
    last_bb_lower = df["bb_lower"].iloc[-1]
    
    # Đánh giá RSI
    rsi_signal = 1 if last_rsi > 70 else (-1 if last_rsi < 30 else 0)
    
    # Đánh giá MACD
    macd_signal = 1 if last_macd > last_macd_signal else -1
    
    # Đánh giá Bollinger Bands
    bb_signal = 1 if last_close > last_bb_upper else (-1 if last_close < last_bb_lower else 0)
    
    # Tổng hợp tín hiệu
    total_signal = rsi_signal + macd_signal + bb_signal
    
    # Xác định xu hướng
    if total_signal > 1:
        trend = "LONG"
        confidence = 0.7 + random.uniform(0, 0.2)  # Mô phỏng độ tin cậy 70-90%
    elif total_signal < -1:
        trend = "SHORT"
        confidence = 0.7 + random.uniform(0, 0.2)  # Mô phỏng độ tin cậy 70-90%
    else:
        trend = "NEUTRAL"
        confidence = 0.5 + random.uniform(0, 0.2)  # Mô phỏng độ tin cậy 50-70%
    
    # Tạo kết quả dự đoán
    prediction = {
        "trend": trend,
        "confidence": confidence,
        "signals": {
            "rsi": rsi_signal,
            "macd": macd_signal,
            "bollinger": bb_signal
        },
        "price": {
            "current": last_close,
            "predicted": last_close * (1 + 0.005 * total_signal)  # Mô phỏng giá dự đoán
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return prediction

# Hàm cập nhật dữ liệu
def update_data():
    """Cập nhật dữ liệu từ Binance API"""
    global data_cache, last_update_time, model_predictions
    
    # Kiểm tra xem đã đến lúc cập nhật chưa
    current_time = time.time()
    if last_update_time is not None and current_time - last_update_time < UPDATE_INTERVAL:
        return  # Chưa đến lúc cập nhật
    
    # Cập nhật dữ liệu
    for timeframe in TIMEFRAMES:
        try:
            # Lấy dữ liệu
            df = get_klines(symbol="ETHUSDT", interval=timeframe, limit=100)
            if df is not None:
                # Thêm chỉ báo
                df = add_indicators(df)
                
                # Lưu vào cache
                data_cache[timeframe] = df
                
                # Tạo dự đoán
                prediction = generate_prediction(df)
                model_predictions[timeframe] = prediction
                
                logging.info(f"Đã cập nhật dữ liệu cho {timeframe}")
            else:
                logging.warning(f"Không thể lấy dữ liệu cho {timeframe}")
        except Exception as e:
            logging.error(f"Lỗi khi cập nhật dữ liệu cho {timeframe}: {str(e)}")
    
    # Cập nhật thời gian
    last_update_time = current_time

# Hàm cập nhật dữ liệu liên tục trong thread
def update_data_continuously():
    """Cập nhật dữ liệu liên tục trong thread"""
    global update_running
    
    while update_running:
        try:
            update_data()
        except Exception as e:
            logging.error(f"Lỗi khi cập nhật dữ liệu: {str(e)}")
        
        # Đợi đến lần cập nhật tiếp theo
        time.sleep(UPDATE_INTERVAL)

# Hàm bắt đầu thread cập nhật
def start_update_thread():
    """Bắt đầu thread cập nhật dữ liệu"""
    global update_thread, update_running
    
    if update_thread is None or not update_thread.is_alive():
        update_running = True
        update_thread = threading.Thread(target=update_data_continuously)
        update_thread.daemon = True
        update_thread.start()
        logging.info("Đã bắt đầu thread cập nhật dữ liệu")

# Hàm dừng thread cập nhật
def stop_update_thread():
    """Dừng thread cập nhật dữ liệu"""
    global update_running
    
    update_running = False
    logging.info("Đã dừng thread cập nhật dữ liệu")

# Vẽ biểu đồ nến
def plot_candlestick_chart(df):
    """Vẽ biểu đồ nến với volumn"""
    # Tạo biểu đồ
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.02, row_heights=[0.7, 0.3])
    
    # Thêm nến
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLC"
    ), row=1, col=1)
    
    # Thêm đường EMA
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["ema_20"],
        name="EMA 20",
        line=dict(color="rgba(255, 0, 0, 0.7)")
    ), row=1, col=1)
    
    # Thêm Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["bb_upper"],
        name="BB Upper",
        line=dict(color="rgba(0, 0, 255, 0.5)")
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["bb_middle"],
        name="BB Middle",
        line=dict(color="rgba(0, 0, 255, 0.5)")
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["bb_lower"],
        name="BB Lower",
        line=dict(color="rgba(0, 0, 255, 0.5)")
    ), row=1, col=1)
    
    # Thêm volumn
    colors = ["red" if row["close"] < row["open"] else "green" for _, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df["timestamp"],
        y=df["volume"],
        name="Volume",
        marker_color=colors
    ), row=2, col=1)
    
    # Cập nhật layout
    fig.update_layout(
        title="ETHUSDT Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Vẽ biểu đồ chỉ báo kỹ thuật
def plot_indicators(df):
    """Vẽ biểu đồ chỉ báo kỹ thuật"""
    # Tạo biểu đồ
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, row_heights=[0.5, 0.5])
    
    # Thêm RSI
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["rsi"],
        name="RSI"
    ), row=1, col=1)
    
    # Thêm đường 70 và 30
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=[70] * len(df),
        name="RSI 70",
        line=dict(color="red", dash="dash")
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=[30] * len(df),
        name="RSI 30",
        line=dict(color="green", dash="dash")
    ), row=1, col=1)
    
    # Thêm MACD
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["macd"],
        name="MACD"
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["macd_signal"],
        name="Signal"
    ), row=2, col=1)
    
    # Thêm histogram
    colors = ["red" if val < 0 else "green" for val in df["macd_hist"]]
    fig.add_trace(go.Bar(
        x=df["timestamp"],
        y=df["macd_hist"],
        name="Histogram",
        marker_color=colors
    ), row=2, col=1)
    
    # Cập nhật layout
    fig.update_layout(
        title="Technical Indicators",
        xaxis_title="Date",
        height=500
    )
    
    # Cập nhật y-axis
    fig.update_yaxes(title_text="RSI", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    
    return fig

# Hiển thị dự đoán
def display_prediction(prediction):
    """Hiển thị dự đoán"""
    # Tạo màu sắc dựa trên xu hướng
    if prediction["trend"] == "LONG":
        trend_color = "green"
        trend_icon = "📈"
    elif prediction["trend"] == "SHORT":
        trend_color = "red"
        trend_icon = "📉"
    else:
        trend_color = "gray"
        trend_icon = "⟷"
    
    # Tạo nội dung hiển thị
    st.markdown(f"<h3 style='color: {trend_color};'>{trend_icon} {prediction['trend']} ({prediction['confidence']:.2%})</h3>", 
               unsafe_allow_html=True)
    
    # Hiển thị giá
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Giá hiện tại", f"${prediction['price']['current']:.2f}")
    with col2:
        price_change = prediction['price']['predicted'] - prediction['price']['current']
        st.metric("Giá dự đoán", f"${prediction['price']['predicted']:.2f}", 
                 f"{price_change:.2f} ({price_change / prediction['price']['current']:.2%})")
    
    # Hiển thị thời gian
    st.caption(f"Cập nhật lúc: {prediction['timestamp']}")
    
    # Hiển thị tín hiệu
    st.subheader("Tín hiệu")
    signals = prediction["signals"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if signals["rsi"] > 0:
            st.success("RSI: Quá mua")
        elif signals["rsi"] < 0:
            st.error("RSI: Quá bán")
        else:
            st.info("RSI: Trung tính")
    
    with col2:
        if signals["macd"] > 0:
            st.success("MACD: Tăng")
        else:
            st.error("MACD: Giảm")
    
    with col3:
        if signals["bollinger"] > 0:
            st.success("BB: Vượt dải trên")
        elif signals["bollinger"] < 0:
            st.error("BB: Dưới dải dưới")
        else:
            st.info("BB: Trong dải")

# Hiển thị thông tin hệ thống
def display_system_info():
    """Hiển thị thông tin hệ thống"""
    st.subheader("Thông tin hệ thống")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("Phiên bản Replit Lite")
    
    with col2:
        if last_update_time is not None:
            last_update = datetime.fromtimestamp(last_update_time).strftime("%H:%M:%S")
            st.info(f"Cập nhật gần nhất: {last_update}")
        else:
            st.warning("Chưa có dữ liệu")
    
    with col3:
        if update_thread is not None and update_thread.is_alive():
            st.success("Hệ thống đang chạy")
        else:
            st.error("Hệ thống không chạy")

# Hàm chính
def main():
    """Hàm chính của ứng dụng"""
    global data_cache, model_predictions
    
    # Thiết lập tiêu đề và favicon
    st.set_page_config(
        page_title="ETHUSDT Dashboard Lite",
        page_icon="📊",
        layout="wide"
    )
    
    # Tải CSS
    load_css()
    
    # Tiêu đề
    st.title("📊 ETHUSDT Dashboard - Phiên bản Replit")
    st.caption("Phiên bản tối ưu hóa cho Replit với tài nguyên giới hạn")
    
    # Kết nối đến Binance API
    if not connect_to_binance():
        st.error("Không thể kết nối đến Binance API. Vui lòng thử lại sau.")
        return
    
    # Bắt đầu thread cập nhật
    start_update_thread()
    
    # Cập nhật dữ liệu
    with st.spinner("Đang tải dữ liệu..."):
        update_data()
    
    # Kiểm tra xem đã có dữ liệu chưa
    if not data_cache:
        st.warning("Không thể tải dữ liệu. Vui lòng thử lại sau.")
        return
    
    # Tạo tabs
    tab1, tab2, tab3 = st.tabs(["📈 Biểu đồ", "🔍 Chỉ báo", "ℹ️ Thông tin"])
    
    # Tab biểu đồ
    with tab1:
        # Chọn khung thời gian
        timeframe = st.selectbox("Khung thời gian", TIMEFRAMES)
        
        # Lấy dữ liệu
        df = data_cache.get(timeframe)
        if df is None:
            st.warning(f"Không có dữ liệu cho khung thời gian {timeframe}")
            return
        
        # Hiển thị biểu đồ nến
        fig = plot_candlestick_chart(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị dự đoán
        st.subheader("Dự đoán")
        prediction = model_predictions.get(timeframe)
        if prediction:
            display_prediction(prediction)
        else:
            st.warning("Không có dự đoán cho khung thời gian này")
    
    # Tab chỉ báo
    with tab2:
        # Chọn khung thời gian
        timeframe = st.selectbox("Khung thời gian", TIMEFRAMES, key="tab2_timeframe")
        
        # Lấy dữ liệu
        df = data_cache.get(timeframe)
        if df is None:
            st.warning(f"Không có dữ liệu cho khung thời gian {timeframe}")
            return
        
        # Hiển thị biểu đồ chỉ báo
        fig = plot_indicators(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị dữ liệu gần đây
        st.subheader("Dữ liệu gần đây")
        st.dataframe(df.tail(10))
    
    # Tab thông tin
    with tab3:
        # Hiển thị thông tin hệ thống
        display_system_info()
        
        # Hiển thị thông tin về phiên bản nhẹ
        st.subheader("Thông tin phiên bản Replit")
        st.markdown("""
        Đây là phiên bản tối ưu hóa cho Replit với các tính năng giảm thiểu:
        - Giảm số lượng kết nối đến Binance API
        - Giảm lưu lượng dữ liệu với bộ nhớ đệm
        - Chỉ dùng một khung thời gian (5m)
        - Đơn giản hóa giao diện người dùng
        - Sử dụng bộ dự đoán nhẹ
        
        Các tính năng đầy đủ có thể được mở khi chạy trên môi trường có nhiều tài nguyên hơn.
        """)
        
        # Hiển thị nút làm mới dữ liệu
        if st.button("Làm mới dữ liệu", use_container_width=True):
            with st.spinner("Đang làm mới dữ liệu..."):
                update_data()
            st.success("Đã làm mới dữ liệu!")
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Lỗi không mong muốn: {str(e)}")
        logging.error(f"Lỗi không mong muốn: {str(e)}", exc_info=True)
    finally:
        # Dừng thread khi ứng dụng kết thúc
        stop_update_thread()