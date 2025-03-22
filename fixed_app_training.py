"""
Main Streamlit application for ETHUSDT prediction dashboard - FIXED VERSION.
Sửa lỗi thread-safety trong huấn luyện mô hình.
"""
import os
import json
import datetime
import threading
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from utils.thread_safe_logging import thread_safe_log, read_logs_from_file

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
            import datetime
            training_result = {
                "success": True,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": "Huấn luyện thành công tất cả các mô hình"
            }
            with open('training_result.json', 'w') as f:
                json.dump(training_result, f)
                
            # Thông báo đã huấn luyện thành công - set flag cho main thread
            with open('training_completed.txt', 'w') as f:
                f.write('success')
        except Exception as e:
            thread_safe_log(f"Lỗi lưu kết quả huấn luyện: {str(e)}")
                
    except Exception as e:
        from utils.thread_safe_logging import thread_safe_log
        thread_safe_log(f"LỖI trong quá trình huấn luyện: {str(e)}")
        
        # Lưu thông tin lỗi vào file
        try:
            training_result = {
                "success": False,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }
            with open('training_result.json', 'w') as f:
                json.dump(training_result, f)
                
            # Thông báo lỗi cho main thread
            with open('training_completed.txt', 'w') as f:
                f.write('error')
        except Exception:
            pass

def train_models():
    """Khởi động quá trình huấn luyện trong thread riêng biệt"""
    import threading
    from utils.thread_safe_logging import thread_safe_log
    
    # Tạo file training_logs.txt nếu chưa tồn tại
    import os
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
    
    # Hiển thị thông báo cho user
    st.success("Đã bắt đầu huấn luyện mô hình trong nền. Kiểm tra logs để theo dõi tiến trình.")
    
    return True

def get_training_result():
    """Đọc kết quả huấn luyện từ file"""
    if os.path.exists('training_result.json'):
        try:
            with open('training_result.json', 'r') as f:
                result = json.load(f)
            return result
        except Exception:
            return None
    return None

def is_training_complete():
    """Kiểm tra xem quá trình huấn luyện đã hoàn tất chưa"""
    if os.path.exists('training_completed.txt'):
        try:
            with open('training_completed.txt', 'r') as f:
                status = f.read().strip()
            # Xóa file để tránh đọc lại trạng thái cũ
            os.remove('training_completed.txt')
            return status
        except Exception:
            return None
    return None

# Thêm hàm hiển thị nhật ký huấn luyện từ file
def display_training_logs():
    """Hiển thị logs huấn luyện từ file"""
    # Đọc logs từ file
    training_logs = read_logs_from_file("training_logs.txt", max_lines=100)
    
    # Hiển thị logs
    if training_logs:
        # Format the logs with color highlighting
        formatted_logs = []
        for log in training_logs:
            if "ERROR" in log or "error" in log or "LỖI" in log:
                formatted_logs.append(f'<span style="color: red;">{log}</span>')
            elif "WARNING" in log or "warning" in log:
                formatted_logs.append(f'<span style="color: yellow;">{log}</span>')
            elif "SUCCESS" in log or "success" in log or "thành công" in log:
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

# Ví dụ thêm một phần UI đơn giản để kiểm tra
if __name__ == "__main__":
    st.title("Kiểm tra huấn luyện mô hình (Thread-safe)")
    
    # Kiểm tra trạng thái huấn luyện
    training_status = is_training_complete()
    if training_status == 'success':
        st.success("Huấn luyện đã hoàn thành thành công!")
    elif training_status == 'error':
        st.error("Huấn luyện gặp lỗi!")
    
    # Hiển thị logs
    st.subheader("Nhật ký huấn luyện")
    display_training_logs()
    
    # Nút bắt đầu huấn luyện
    if st.button("Bắt đầu huấn luyện"):
        train_models()