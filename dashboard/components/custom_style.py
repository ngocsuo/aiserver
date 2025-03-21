"""
Tệp này chứa các phần CSS tùy chỉnh và hàm tạo các thành phần giao diện đẹp mắt
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

def load_custom_css():
    """
    Tải CSS tùy chỉnh cho giao diện
    """
    custom_css = """
    <style>
    /* Cải thiện màu sắc chủ đạo */
    .main {
        background-color: #f7f9fc;
    }
    
    /* Tùy chỉnh thanh tiêu đề */
    .css-1dp5vir {
        background-image: linear-gradient(90deg, #4f8bf9, #485ec4);
    }
    
    /* Làm cho các thẻ và bảng hiển thị đẹp hơn */
    .stTabs {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        padding: 10px;
    }
    
    /* Tùy chỉnh kiểu dáng các thẻ */
    .stTab {
        background-color: #f0f2f6;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .stTab:hover {
        background-color: #e1e5ed;
    }
    
    /* Làm cho các thẻ active nổi bật */
    .stTab [data-baseweb="tab"][aria-selected="true"] {
        background-color: #485ec4;
        color: white;
    }
    
    /* Style cho các container */
    div[data-testid="stVerticalBlock"] > div {
        background-color: white;
        border-radius: 8px;
        padding: 5px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Làm cho metrics đẹp hơn */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Styling cho bảng */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .dataframe th {
        background-color: #485ec4;
        color: white;
        font-weight: normal;
        padding: 12px;
        text-align: left;
    }
    
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid #eaeaea;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    .dataframe tr:hover {
        background-color: #f0f2f6;
    }
    
    /* Đẹp hơn cho các expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    
    /* Tùy chỉnh thanh progress */
    .stProgress > div > div {
        background-image: linear-gradient(to right, #4f8bf9, #485ec4);
    }
    
    /* Tùy chỉnh các card */
    div.card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    div.card-buy {
        border-left: 4px solid #2ecc71;
    }
    
    div.card-sell {
        border-left: 4px solid #e74c3c;
    }
    
    div.card-neutral {
        border-left: 4px solid #95a5a6;
    }
    
    /* Tùy chỉnh màu sắc cho các chỉ báo dự đoán */
    .indicator-long {
        color: #2ecc71;
        font-weight: bold;
    }
    
    .indicator-short {
        color: #e74c3c;
        font-weight: bold;
    }
    
    .indicator-neutral {
        color: #95a5a6;
        font-weight: bold;
    }
    
    /* Animation cho loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s infinite ease-in-out;
    }
    
    /* Custom gauge chart */
    .gauge-value {
        font-size: 24px;
        font-weight: bold;
    }
    
    /* Nút đẹp */
    div.stButton > button {
        background-color: #485ec4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 15px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: #4f8bf9;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Thông báo đẹp */
    div.stAlert {
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def create_metric_card(title, value, subtitle=None, icon=None, color="blue", is_percent=False):
    """
    Tạo một card hiển thị số liệu đẹp mắt
    
    Args:
        title (str): Tiêu đề card
        value: Giá trị chính hiển thị
        subtitle (str): Tiêu đề phụ (có thể là None)
        icon (str): Biểu tượng (ví dụ: "📈")
        color (str): Màu sắc của card
        is_percent (bool): Có hiển thị dạng % không
    """
    # Xử lý giá trị value đầu vào để tránh các lỗi khi hiển thị
    if isinstance(value, str) and "<div>" in value or "</div>" in value:
        # Nếu chứa HTML tags, sử dụng giá trị đơn giản thay thế
        value_str = "N/A"
    else:
        # Định dạng giá trị bình thường
        try:
            value_str = f"{float(value):.2f}%" if is_percent else f"{value}"
        except (ValueError, TypeError):
            # Nếu không thể chuyển thành số, hiển thị nguyên dạng
            value_str = f"{value}%" if is_percent else f"{value}"
    
    # Xử lý icon
    icon_html = f"<span style='font-size: 24px;'>{icon}</span>" if icon else ""
    
    color_map = {
        "blue": "#485ec4",
        "green": "#2ecc71",
        "red": "#e74c3c",
        "yellow": "#f1c40f",
        "gray": "#95a5a6"
    }
    
    border_color = color_map.get(color, "#485ec4")
    
    # Tạo HTML an toàn
    card_html = f"""
    <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-left: 4px solid {border_color}; margin-bottom: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #7f8c8d; font-size: 14px;">{title}</div>
                <div style="font-size: 28px; font-weight: bold; color: #2c3e50;">{value_str}</div>
                {f'<div style="color: #95a5a6; font-size: 12px;">{subtitle}</div>' if subtitle else ''}
            </div>
            <div>
                {icon_html}
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_price_card(price, change, change_percent, last_update=None):
    """
    Tạo card hiển thị giá và % thay đổi đẹp mắt
    
    Args:
        price (float): Giá hiện tại
        change (float): Thay đổi so với giá trước
        change_percent (float): Phần trăm thay đổi
        last_update (str): Thời gian cập nhật cuối
    """
    color = "green" if change >= 0 else "red"
    icon = "📈" if change >= 0 else "📉"
    
    price_html = f"""
    <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #7f8c8d; font-size: 16px;">ETHUSDT</div>
                <div style="font-size: 32px; font-weight: bold;">${price:.2f}</div>
                <div style="color: {'#2ecc71' if change >= 0 else '#e74c3c'}; font-size: 14px; display: flex; align-items: center;">
                    <span>${change:.2f} ({change_percent:.2f}%)</span>
                </div>
                {f'<div style="color: #95a5a6; font-size: 12px; margin-top: 5px;">Cập nhật: {last_update}</div>' if last_update else ''}
            </div>
            <div>
                <span style="font-size: 48px;">{icon}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(price_html, unsafe_allow_html=True)

def create_prediction_card(prediction):
    """
    Tạo card hiển thị kết quả dự đoán đẹp mắt
    
    Args:
        prediction (dict): Thông tin dự đoán
    """
    if not prediction:
        st.info("Chưa có dự đoán")
        return
    
    # Xác định màu sắc và biểu tượng dựa trên xu hướng
    colors = {"LONG": "#2ecc71", "SHORT": "#e74c3c", "NEUTRAL": "#95a5a6"}
    icons = {"LONG": "📈", "SHORT": "📉", "NEUTRAL": "📊"}
    
    trend = prediction.get("trend", "NEUTRAL")
    color = colors.get(trend, "#95a5a6")
    icon = icons.get(trend, "📊")
    
    # Tính toán hiển thị thời gian
    timestamp = prediction.get("timestamp", "")
    time_diff = ""
    if timestamp:
        try:
            pred_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            diff = (now - pred_time).total_seconds() / 60
            if diff < 60:
                time_diff = f"{int(diff)} phút trước"
            else:
                time_diff = f"{int(diff/60)} giờ {int(diff%60)} phút trước"
        except:
            time_diff = timestamp
    
    # Tạo card HTML
    card_html = f"""
    <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-left: 4px solid {color}; margin-bottom: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="flex-grow: 1;">
                <div style="font-size: 24px; font-weight: bold; color: {color};">
                    {trend}
                    <span style="font-size: 14px; color: #7f8c8d; font-weight: normal; margin-left: 10px;">
                        {time_diff}
                    </span>
                </div>
                
                <div style="display: flex; margin-top: 10px;">
                    <div style="flex: 1;">
                        <div style="color: #7f8c8d; font-size: 14px;">Giá hiện tại</div>
                        <div style="font-size: 18px; font-weight: bold;">${prediction.get('price', 0):.2f}</div>
                    </div>
                    <div style="flex: 1;">
                        <div style="color: #7f8c8d; font-size: 14px;">Giá mục tiêu</div>
                        <div style="font-size: 18px; font-weight: bold;">${prediction.get('target_price', 0):.2f}</div>
                    </div>
                    <div style="flex: 1;">
                        <div style="color: #7f8c8d; font-size: 14px;">Độ tin cậy</div>
                        <div style="font-size: 18px; font-weight: bold;">{prediction.get('confidence', 0)*100:.1f}%</div>
                    </div>
                </div>
                
                <div style="margin-top: 15px;">
                    <div style="color: #7f8c8d; font-size: 14px;">Thời gian dự đoán</div>
                    <div style="margin-top: 5px;">
                        <div style="background-color: #f0f2f6; height: 8px; border-radius: 4px; position: relative;">
                            <div style="position: absolute; height: 8px; border-radius: 4px; width: {min(prediction.get('valid_minutes_left', 0)/prediction.get('valid_for_minutes', 30)*100, 100)}%; background-color: {color};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px; color: #7f8c8d;">
                            <span>0 phút</span>
                            <span>{prediction.get('valid_for_minutes', 30)} phút</span>
                        </div>
                    </div>
                </div>
            </div>
            <div style="font-size: 36px; margin-left: 15px; color: {color};">
                {icon}
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_gauge_chart(value, title="Độ tin cậy", min_value=0, max_value=1, color_thresholds=None):
    """
    Tạo biểu đồ đồng hồ đo gauge để hiển thị độ tin cậy
    
    Args:
        value (float): Giá trị cần hiển thị (0-1)
        title (str): Tiêu đề của biểu đồ
        min_value (float): Giá trị tối thiểu của thang đo
        max_value (float): Giá trị tối đa của thang đo
        color_thresholds (list): Danh sách các ngưỡng màu sắc [(giá_trị, màu),...]
    """
    if color_thresholds is None:
        color_thresholds = [
            (0.3, "red"),
            (0.7, "orange"),
            (1.0, "green")
        ]
    
    # Xác định màu dựa trên ngưỡng
    color = color_thresholds[-1][1]  # Màu mặc định (màu cuối cùng)
    for threshold, threshold_color in color_thresholds:
        if value <= threshold:
            color = threshold_color
            break
    
    # Tạo biểu đồ gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_value, max_value], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [threshold_range[0], threshold_range[1]], 'color': threshold_color}
                for threshold_range, threshold_color in zip(
                    [(min_value, t) for t, _ in color_thresholds],
                    [c for _, c in color_thresholds]
                )
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'suffix': "", 'font': {'size': 20}}
    ))
    
    # Cập nhật layout cho phù hợp
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_header():
    """
    Tạo header đẹp mắt cho ứng dụng
    """
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #4f8bf9, #485ec4); padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="color: white; margin: 0;">AI CRYPTO PREDICTION</h1>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Dự đoán tự động ETH/USDT với AI nâng cao</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 50px; padding: 5px 15px; color: white; font-size: 14px;">
                    Phiên bản 2.0
                </div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

def create_section_header(title, subtitle=None, icon=None):
    """
    Tạo tiêu đề cho một phần trong giao diện
    
    Args:
        title (str): Tiêu đề chính
        subtitle (str): Tiêu đề phụ (có thể là None)
        icon (str): Biểu tượng (ví dụ: "📈")
    """
    icon_html = f'<span style="font-size: 24px; margin-right: 10px;">{icon}</span>' if icon else ''
    subtitle_html = f'<p style="color: #7f8c8d; margin: 5px 0 15px 0;">{subtitle}</p>' if subtitle else ''
    
    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #2c3e50; margin: 0; display: flex; align-items: center;">
                {icon_html}{title}
            </h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def create_stats_row(stats):
    """
    Tạo một hàng hiển thị các thống kê
    
    Args:
        stats (list): Danh sách các thống kê dạng [(label, value, icon, color), ...]
    """
    # Tính toán số cột
    cols = st.columns(len(stats))
    
    # Hiển thị từng thống kê trong một cột
    for i, (label, value, icon, color) in enumerate(stats):
        with cols[i]:
            create_metric_card(label, value, icon=icon, color=color)