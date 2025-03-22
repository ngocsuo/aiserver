import streamlit as st
import psutil
import time
import pandas as pd
import plotly.express as px
from datetime import datetime

def display_system_resources():
    """Hiển thị thông tin tài nguyên hệ thống (CPU, RAM, Disk)"""
    st.title("Thông tin tài nguyên hệ thống")
    
    # Thông tin CPU
    cpu_usage = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Thông tin RAM
    memory = psutil.virtual_memory()
    ram_total = memory.total / (1024 ** 3)  # Chuyển đổi sang GB
    ram_used = memory.used / (1024 ** 3)
    ram_avail = memory.available / (1024 ** 3)
    
    # Thông tin Disk
    disk = psutil.disk_usage('/')
    disk_total = disk.total / (1024 ** 3)
    disk_used = disk.used / (1024 ** 3)
    disk_free = disk.free / (1024 ** 3)
    
    # Hiển thị thông tin trong cột
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="CPU Usage",
            value=f"{cpu_usage}%",
            delta=f"{cpu_count} cores"
        )
    
    with col2:
        st.metric(
            label="RAM Usage",
            value=f"{memory.percent}%",
            delta=f"{ram_used:.1f}/{ram_total:.1f} GB"
        )
    
    with col3:
        st.metric(
            label="Disk Usage",
            value=f"{disk.percent}%",
            delta=f"{disk_used:.1f}/{disk_total:.1f} GB"
        )
    
    # Hiển thị thời gian cập nhật
    st.caption(f"Cập nhật lần cuối: {datetime.now().strftime('%H:%M:%S')}")
    
    if st.button("Cập nhật"):
        st.rerun()

if __name__ == "__main__":
    display_system_resources()