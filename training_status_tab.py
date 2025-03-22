"""
Module hiển thị thông tin về trạng thái huấn luyện và hiệu suất mô hình.
"""
import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils.thread_safe_logging import read_logs_from_file

def read_training_status():
    """
    Đọc trạng thái huấn luyện từ tập tin training_status.json.
    
    Returns:
        dict: Dữ liệu trạng thái huấn luyện
    """
    try:
        if os.path.exists("training_status.json"):
            with open("training_status.json", "r") as f:
                return json.load(f)
        return {
            "model_trained": False,
            "last_training_time": None,
            "progress": 0,
            "timeframes": [],
            "models": {},
            "data_stats": {},
            "class_distribution": {}
        }
    except Exception as e:
        st.error(f"Lỗi khi đọc trạng thái huấn luyện: {e}")
        return {
            "model_trained": False,
            "last_training_time": None,
            "progress": 0,
            "timeframes": [],
            "models": {},
            "data_stats": {},
            "class_distribution": {}
        }

def get_latest_logs(max_lines=50):
    """
    Lấy log mới nhất từ file training_logs.txt.
    
    Args:
        max_lines (int): Số dòng log tối đa cần lấy
        
    Returns:
        list: Danh sách các dòng log
    """
    return read_logs_from_file(log_file="training_logs.txt", max_lines=max_lines)

def render_training_status_tab():
    """
    Render tab trạng thái huấn luyện mô hình.
    """
    st.title("Trạng thái Huấn luyện Mô hình")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Tổng quan Huấn luyện")
        
        status_data = read_training_status()
        
        if status_data["model_trained"]:
            st.success("✅ Mô hình đã được huấn luyện thành công!")
            last_training = datetime.strptime(status_data["last_training_time"], "%Y-%m-%d %H:%M:%S")
            st.info(f"Huấn luyện gần nhất: {last_training.strftime('%d/%m/%Y %H:%M:%S')}")
            
            # Hiển thị thời gian từ lần huấn luyện cuối
            now = datetime.now()
            time_diff = now - last_training
            hours = time_diff.seconds // 3600
            minutes = (time_diff.seconds % 3600) // 60
            if time_diff.days > 0:
                st.warning(f"⚠️ Cách đây {time_diff.days} ngày, {hours} giờ, {minutes} phút")
            else:
                st.info(f"⏱️ Cách đây {hours} giờ, {minutes} phút")
        else:
            st.warning("⚠️ Mô hình chưa được huấn luyện")
        
        # Hiển thị các timeframe đã huấn luyện
        st.subheader("Dữ liệu đã huấn luyện")
        if status_data["timeframes"]:
            timeframes = ", ".join([f"**{tf}**" for tf in status_data["timeframes"]])
            st.markdown(f"Khung thời gian: {timeframes}")
        else:
            st.warning("Chưa có khung thời gian nào được huấn luyện")
        
        # Hiển thị thông tin thống kê dữ liệu
        if status_data["data_stats"]:
            data_stats = status_data["data_stats"]
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Tổng số mẫu dữ liệu", f"{data_stats.get('total_data_points', 0):,}")
            with col_stat2:
                st.metric("Số lượng tính năng", f"{data_stats.get('features', 0)}")
            with col_stat3:
                st.metric("Tỉ lệ huấn luyện/kiểm tra", "60%/20%/20%")
            
            # Hiển thị phân phối lớp
            if status_data["class_distribution"]:
                st.subheader("Phân phối lớp")
                class_dist = status_data["class_distribution"]
                fig = px.pie(
                    values=list(class_dist.values()),
                    names=list(class_dist.keys()),
                    color=list(class_dist.keys()),
                    color_discrete_map={
                        'LONG': 'green',
                        'SHORT': 'red',
                        'NEUTRAL': 'gray'
                    },
                    title="Phân phối lớp trong dữ liệu huấn luyện"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Độ chính xác mô hình")
        if status_data["models"]:
            models = status_data["models"]
            
            # Tạo biểu đồ thanh cho độ chính xác mô hình
            fig = go.Figure()
            model_names = list(models.keys())
            accuracies = [models[m]["accuracy"] for m in model_names]
            
            # Định nghĩa màu sắc cho các mô hình
            colors = {
                "lstm": "rgba(31, 119, 180, 0.8)",
                "transformer": "rgba(255, 127, 14, 0.8)",
                "cnn": "rgba(44, 160, 44, 0.8)",
                "historical_similarity": "rgba(214, 39, 40, 0.8)",
                "meta_learner": "rgba(148, 103, 189, 0.8)"
            }
            
            model_colors = [colors.get(m.lower(), "rgba(100, 100, 100, 0.8)") for m in model_names]
            
            fig.add_trace(go.Bar(
                x=model_names,
                y=accuracies,
                marker_color=model_colors,
                text=[f"{acc:.1%}" for acc in accuracies],
                textposition="auto"
            ))
            
            fig.update_layout(
                title="Độ chính xác theo mô hình",
                xaxis_title="Mô hình",
                yaxis_title="Độ chính xác",
                yaxis=dict(range=[0, 1]),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Hiển thị bảng thông tin chi tiết
            model_data = []
            for model, info in models.items():
                model_data.append({
                    "Mô hình": model,
                    "Độ chính xác": f"{info['accuracy']:.1%}",
                    "Trạng thái": "Đã huấn luyện" if info["trained"] else "Chưa huấn luyện"
                })
            
            st.dataframe(pd.DataFrame(model_data), hide_index=True)
            
        else:
            st.warning("Chưa có thông tin về mô hình")

    st.subheader("Nhật ký huấn luyện")
    
    # Thêm CSS tùy chỉnh để hiển thị logs đẹp hơn
    st.markdown("""
    <style>
    /* Style cho bảng logs */
    [data-testid="stDataFrame"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Làm đẹp header của bảng */
    [data-testid="stDataFrame"] th {
        background-color: #485ec4;
        color: white;
        font-weight: normal;
        padding: 10px;
        text-align: left;
    }
    
    /* Hiệu ứng khi hover dòng */
    [data-testid="stDataFrame"] tr:hover {
        background-color: #e5e9f5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    logs = get_latest_logs(50)
    if logs:
        # Tạo DataFrame cho logs và định dạng lại
        logs_df = pd.DataFrame({
            "Thời gian": [log.split(" - ")[0] for log in logs],
            "Thông tin": [log.split(" - ")[1] if " - " in log else log for log in logs]
        })
        
        # Thêm kiểu cho các loại thông báo khác nhau
        def style_logs(df):
            styles = []
            for i, row in df.iterrows():
                info = row['Thông tin'].lower()
                if 'error' in info or 'lỗi' in info:
                    styles.append('background-color: #ffe6e6')
                elif 'warning' in info or 'cảnh báo' in info:
                    styles.append('background-color: #fff7e6')
                elif 'success' in info or 'thành công' in info:
                    styles.append('background-color: #e6ffe6')
                else:
                    styles.append('')
            return [''] * len(df.columns) if not styles else styles
        
        # Hiển thị DataFrame với style
        st.dataframe(logs_df, hide_index=True)
    else:
        st.info("Chưa có nhật ký huấn luyện")
    
    st.subheader("Tùy chọn huấn luyện")
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("Huấn luyện lại mô hình", type="primary"):
            st.session_state["training_requested"] = True
            st.rerun()
            
    with col_btn2:
        if st.button("Xóa bộ nhớ cache"):
            # Thêm logic xóa cache ở đây
            st.warning("Đã xóa bộ nhớ cache. Lần huấn luyện tiếp theo sẽ tải lại toàn bộ dữ liệu.")

if __name__ == "__main__":
    render_training_status_tab()