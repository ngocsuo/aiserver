"""
Module cung cấp các hàm định dạng CSS cho giao diện
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def load_custom_css():
    """
    Tải CSS tùy chỉnh từ file
    """
    try:
        with open("dashboard/styles/custom.css", "r") as f:
            custom_css = f.read()
            st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Không thể tải CSS: {e}")
        # Sử dụng CSS mặc định khi không tìm thấy file
        default_css = """
        div[data-testid="stDataFrame"] table {
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        div[data-testid="stDataFrame"] th {
            background-color: #485ec4 !important;
            color: white !important;
            font-weight: normal;
            padding: 10px;
            text-align: left;
        }
        
        div[data-testid="stDataFrame"] tr:hover td {
            background-color: #e5e9f5;
        }
        """
        st.markdown(f"<style>{default_css}</style>", unsafe_allow_html=True)

def create_metric_card(label, value, delta=None, help_text=None, prefix="", suffix="", icon=None, color=None):
    """
    Tạo card hiển thị số liệu với định dạng đẹp
    
    Args:
        label (str): Nhãn của card
        value (str/float): Giá trị hiển thị
        delta (float, optional): Giá trị thay đổi
        help_text (str, optional): Văn bản hỗ trợ
        prefix (str): Tiền tố cho giá trị
        suffix (str): Hậu tố cho giá trị
        icon (str, optional): Biểu tượng hiển thị (emoji)
        color (str, optional): Màu sắc của card
    """
    # Nếu có icon, thêm vào label
    if icon:
        label = f"{icon} {label}"
        
    # Tạo metric card với hoặc không có màu tùy chỉnh
    if color:
        st.markdown(f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; border-left: 4px solid {color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <p style="color: gray; margin: 0; font-size: 0.8em;">{label}</p>
            <h3 style="margin: 0; color: #333;">{prefix}{value}{suffix}</h3>
            {f'<p style="margin: 0; color: {"green" if float(delta) > 0 else "red"};">{delta}</p>' if delta else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.metric(
            label=label,
            value=f"{prefix}{value}{suffix}",
            delta=delta,
            help=help_text
        )

def create_price_card(price, change_24h, change_7d, high_24h, low_24h):
    """
    Tạo card hiển thị giá với các thông tin chi tiết
    
    Args:
        price (float): Giá hiện tại
        change_24h (float): Thay đổi 24 giờ
        change_7d (float): Thay đổi 7 ngày
        high_24h (float): Giá cao nhất 24 giờ
        low_24h (float): Giá thấp nhất 24 giờ
    """
    st.markdown(f"""
    <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h3 style="margin: 0; color: #333;">${price:.2f}</h3>
        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
            <div>
                <p style="margin: 0; color: {'green' if change_24h >= 0 else 'red'};">
                    {change_24h:.2f}% (24h)
                </p>
                <p style="margin: 0; color: {'green' if change_7d >= 0 else 'red'};">
                    {change_7d:.2f}% (7d)
                </p>
            </div>
            <div>
                <p style="margin: 0; color: #555;">
                    High: ${high_24h:.2f}
                </p>
                <p style="margin: 0; color: #555;">
                    Low: ${low_24h:.2f}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_prediction_card(prediction_data, confidence=None, time_horizon=None, reasoning=None):
    """
    Tạo card hiển thị dự đoán với định dạng đẹp
    
    Args:
        prediction_data: Có thể là một chuỗi ("Tăng"/"Giảm") hoặc một dict chứa thông tin dự đoán
        confidence (float, optional): Độ tin cậy (0-100%)
        time_horizon (str, optional): Khung thời gian dự đoán
        reasoning (str, optional): Lý do dự đoán
    """
    # Xử lý trường hợp prediction_data là một dictionary
    if isinstance(prediction_data, dict):
        pred_text = prediction_data.get('prediction', 'N/A')
        confidence = prediction_data.get('confidence', confidence or 0)
        time_horizon = prediction_data.get('time_horizon', time_horizon or "N/A")
        reasoning = prediction_data.get('reasoning', reasoning)
    else:
        pred_text = prediction_data
    
    # Xác định màu sắc dựa trên dự đoán
    color = "green" if pred_text == "Tăng" else "red"
    
    # Định dạng độ tin cậy
    confidence_text = f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else confidence
    
    # Tạo thẻ dự đoán
    st.markdown(f"""
    <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: {color};">{pred_text}</h3>
                <p style="margin: 0; color: #666;">Khung thời gian: {time_horizon}</p>
            </div>
            <div style="text-align: right;">
                <h4 style="margin: 0; color: #333;">Độ tin cậy</h4>
                <p style="margin: 0; font-size: 18px; color: {color};">{confidence_text}</p>
            </div>
        </div>
        {f'<div style="margin-top: 10px;"><p style="margin: 0; color: #555;">{reasoning}</p></div>' if reasoning else ''}
    </div>
    """, unsafe_allow_html=True)

def create_gauge_chart(confidence, trend="Tăng", min_value=0, max_value=100, color_thresholds=None):
    """
    Tạo biểu đồ đồng hồ hiển thị độ tin cậy dự đoán
    
    Args:
        confidence (float): Giá trị độ tin cậy để hiển thị
        trend (str): Xu hướng dự đoán ("Tăng" hoặc "Giảm")
        min_value (float): Giá trị nhỏ nhất trên thang đo
        max_value (float): Giá trị lớn nhất trên thang đo
        color_thresholds (list): Danh sách các ngưỡng màu [(ngưỡng, màu), ...]
    """
    # Xác định màu sắc dựa trên xu hướng
    color = "green" if trend == "Tăng" else "red"
    
    # Tỉ lệ giá trị hiển thị theo thang đo
    # Nếu thang đo là 0-1, nhưng muốn hiển thị % (0-100), thì nhân với 100
    display_value = confidence
    if max_value <= 1 and confidence <= 1:
        display_value = confidence * 100
    
    # Tạo các ngưỡng màu
    if color_thresholds is None:
        steps = [
            {"range": [min_value, min_value + (max_value - min_value) * 0.3], "color": "#ffcccc"},
            {"range": [min_value + (max_value - min_value) * 0.3, min_value + (max_value - min_value) * 0.7], "color": "#ffff99"},
            {"range": [min_value + (max_value - min_value) * 0.7, max_value], "color": "#b3ffb3"}
        ]
    else:
        steps = []
        prev_threshold = min_value
        for threshold, color_val in color_thresholds:
            steps.append({"range": [prev_threshold, threshold * (max_value / 1.0)], "color": color_val})
            prev_threshold = threshold * (max_value / 1.0)
    
    # Tạo biểu đồ đồng hồ
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_value,
        title={"text": trend, "font": {"color": color}},
        number={"suffix": "%", "font": {"color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": steps,
        }
    ))
    
    # Cấu hình bố cục biểu đồ
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={"color": "darkblue", "family": "Arial"}
    )
    
    return fig

def create_header(title, subtitle=None):
    """Tạo tiêu đề trang với định dạng tùy chỉnh"""
    st.markdown(f"""
    <h1 style="color: #1E3A8A; margin-bottom: 0;">{title}</h1>
    {f'<p style="color: #64748B; font-size: 1.2em; margin-top: 0;">{subtitle}</p>' if subtitle else ''}
    <hr style="margin: 0.5em 0 1em 0; border: none; height: 2px; background-color: #E2E8F0;">
    """, unsafe_allow_html=True)

def create_section_header(title, description=None, icon=None):
    """Tạo tiêu đề phần với định dạng tùy chỉnh"""
    # Thêm icon vào title nếu được cung cấp
    if icon:
        title = f"{icon} {title}"
        
    st.markdown(f"""
    <h2 style="color: #2563EB; margin-bottom: 0; font-size: 1.5em;">{title}</h2>
    {f'<p style="color: #64748B; margin-top: 0.2em;">{description}</p>' if description else ''}
    """, unsafe_allow_html=True)

def create_stats_row(stats_data):
    """
    Tạo hàng hiển thị thống kê
    
    Args:
        stats_data (list): Danh sách các dict với các khóa:
                          'label', 'value', 'delta' (optional), 'color' (optional)
    """
    cols = st.columns(len(stats_data))
    for i, stat in enumerate(stats_data):
        color = stat.get('color', 'blue')
        delta = stat.get('delta', None)
        delta_str = f" ({delta:+.2f}%)" if delta else ""
        
        with cols[i]:
            st.markdown(f"""
            <div style="background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <p style="margin: 0; color: #666; font-size: 0.9em;">{stat['label']}</p>
                <h3 style="margin: 0; color: {color};">{stat['value']}{delta_str}</h3>
            </div>
            """, unsafe_allow_html=True)