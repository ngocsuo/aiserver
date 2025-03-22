"""
Component hiển thị thông tin tài nguyên hệ thống (CPU, RAM, Disk)
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import os
import json

from utils.system_monitor import get_system_stats, start_monitoring

def render_system_stats():
    """
    Hiển thị thông tin tài nguyên hệ thống
    """
    # Đảm bảo hệ thống giám sát đã được khởi động
    if 'system_monitor_started' not in st.session_state:
        start_monitoring()
        st.session_state.system_monitor_started = True
    
    # Đọc thông tin tài nguyên
    system_stats = get_system_stats()
    
    # Tạo UI cho thông tin tài nguyên
    st.markdown("## 🖥️ Tài nguyên hệ thống")
    st.markdown(f"*Cập nhật lần cuối: {system_stats['last_update']}*")
    
    # Container chứa các metrics
    metrics_container = st.container()
    
    with metrics_container:
        col1, col2, col3 = st.columns(3)
        
        # CPU Usage
        with col1:
            st.metric(
                label="CPU Usage",
                value=f"{system_stats['cpu']['percent']}%",
                delta=f"{system_stats['cpu']['count']} cores"
            )
        
        # RAM Usage
        with col2:
            st.metric(
                label="RAM Usage",
                value=f"{system_stats['memory']['percent']}%",
                delta=f"{system_stats['memory']['used']:.1f}/{system_stats['memory']['total']:.1f} GB"
            )
        
        # Disk Usage
        with col3:
            st.metric(
                label="Disk Usage",
                value=f"{system_stats['disk']['percent']}%",
                delta=f"{system_stats['disk']['used']:.1f}/{system_stats['disk']['total']:.1f} GB"
            )
    
    # Hiển thị biểu đồ CPU và RAM
    with st.expander("Chi tiết tài nguyên", expanded=False):
        tab1, tab2 = st.tabs(["CPU", "RAM"])
        
        with tab1:
            # Biểu đồ CPU
            if system_stats['cpu']['stats']:
                df_cpu = pd.DataFrame(system_stats['cpu']['stats'])
                df_cpu['time'] = pd.to_datetime(df_cpu['time'])
                
                # Tạo biểu đồ
                fig = px.line(
                    df_cpu, 
                    x='time', 
                    y='value',
                    labels={'value': 'CPU Usage (%)', 'time': 'Thời gian'},
                    title='CPU Usage (%)'
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Đang thu thập dữ liệu CPU...")
        
        with tab2:
            # Biểu đồ RAM
            if system_stats['memory']['stats']:
                df_mem = pd.DataFrame(system_stats['memory']['stats'])
                df_mem['time'] = pd.to_datetime(df_mem['time'])
                
                # Tạo biểu đồ
                fig = px.line(
                    df_mem, 
                    x='time', 
                    y='value',
                    labels={'value': 'RAM Usage (%)', 'time': 'Thời gian'},
                    title='RAM Usage (%)'
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Đang thu thập dữ liệu RAM...")