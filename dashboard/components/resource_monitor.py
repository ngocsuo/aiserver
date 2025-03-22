"""
Component hiển thị thông tin tài nguyên hệ thống (CPU, RAM, Disk)
"""
import streamlit as st
import psutil
import time
from datetime import datetime

def get_system_resources():
    """Lấy thông tin tài nguyên hệ thống hiện tại"""
    # Thông tin CPU
    cpu_usage = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Thông tin RAM
    memory = psutil.virtual_memory()
    ram_total = memory.total / (1024 ** 3)  # Chuyển đổi sang GB
    ram_used = memory.used / (1024 ** 3)
    ram_percent = memory.percent
    
    # Thông tin Disk
    disk = psutil.disk_usage('/')
    disk_total = disk.total / (1024 ** 3)
    disk_used = disk.used / (1024 ** 3)
    disk_percent = disk.percent
    
    # Tạo đối tượng chứa thông tin
    resources = {
        'cpu': {
            'percent': cpu_usage,
            'count': cpu_count,
            'text': f"{cpu_usage}% (of {cpu_count} cores)"
        },
        'memory': {
            'percent': ram_percent,
            'total': ram_total,
            'used': ram_used,
            'text': f"{ram_percent}% ({ram_used:.1f}/{ram_total:.1f} GB)"
        },
        'disk': {
            'percent': disk_percent,
            'total': disk_total,
            'used': disk_used,
            'text': f"{disk_percent}% ({disk_used:.1f}/{disk_total:.1f} GB)"
        },
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }
    
    return resources

def render_system_resources():
    """Hiển thị thông tin tài nguyên hệ thống"""
    # Lấy thông tin tài nguyên
    resources = get_system_resources()
    
    # Hiển thị tiêu đề
    st.subheader("🖥️ Thông tin máy chủ")
    st.caption(f"Cập nhật lần cuối: {resources['timestamp']}")
    
    # Hiển thị thông tin trong cột
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="CPU Usage",
            value=f"{resources['cpu']['percent']}%",
            delta=f"{resources['cpu']['count']} cores"
        )
    
    with col2:
        st.metric(
            label="RAM Usage",
            value=f"{resources['memory']['percent']}%",
            delta=f"{resources['memory']['used']:.1f}/{resources['memory']['total']:.1f} GB"
        )
    
    with col3:
        st.metric(
            label="Disk Usage",
            value=f"{resources['disk']['percent']}%",
            delta=f"{resources['disk']['used']:.1f}/{resources['disk']['total']:.1f} GB"
        )
        
    # Hiển thị nút cập nhật thủ công
    if st.button("🔄 Cập nhật thông tin tài nguyên", use_container_width=True):
        st.rerun()