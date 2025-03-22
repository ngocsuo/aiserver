"""
Phiên bản đã sửa lỗi của train_models và train_models_background trong app.py.
Sửa lỗi liên quan đến session_state trong thread riêng

Hướng dẫn:
1. Sao chép toàn bộ mã các hàm train_models_background() và train_models() vào app.py
2. Thay thế (hoặc bình luận) TOÀN BỘ hàm cũ trong app.py 
3. Tạo thư mục utils nếu chưa có và đảm bảo có module thread_safe_logging.py
4. Khởi động lại streamlit để áp dụng thay đổi
"""

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
            from datetime import datetime
            
            models = {
                'lstm': lstm_model,
                'transformer': transformer_model,
                'cnn': cnn_model,
                'historical_similarity': historical_model,
                'meta_learner': meta_model
            }
            
            # Lưu models vào file
            import os
            import pickle
            
            if not os.path.exists("saved_models"):
                os.makedirs("saved_models")
                
            with open("saved_models/models.pkl", "wb") as f:
                pickle.dump(models, f)
                
            # Lưu metadata về quá trình huấn luyện
            training_status = {
                'last_training_time': datetime.now().isoformat(),
                'data_points': len(data),
                'model_version': config.MODEL_VERSION if hasattr(config, 'MODEL_VERSION') else "1.0.0",
                'training_complete': True
            }
            
            with open("saved_models/training_status.json", "w") as f:
                json.dump(training_status, f)
                
            thread_safe_log("Đã lưu tất cả mô hình vào saved_models/models.pkl")
        except Exception as e:
            thread_safe_log(f"Lỗi khi lưu mô hình: {str(e)}")
            
    except Exception as e:
        thread_safe_log(f"❌ Lỗi trong quá trình huấn luyện: {str(e)}")
        import traceback
        thread_safe_log(f"Chi tiết lỗi: {traceback.format_exc()}")

def train_models():
    """Train all prediction models in a background thread"""
    import os
    import sys
    import time
    import threading
    from datetime import datetime
    import streamlit as st
    
    if not st.session_state.initialized or st.session_state.latest_data is None:
        st.warning("Hệ thống chưa được khởi tạo hoặc không có dữ liệu")
        show_toast("Hệ thống chưa được khởi tạo hoặc không có dữ liệu", "warning")
        return False
    
    # Đảm bảo thư mục utils/thread_safe_logging.py tồn tại
    try:
        from utils.thread_safe_logging import thread_safe_log, read_logs_from_file
    except ImportError:
        # Nếu không có, tạo module thread-safe logging
        if not os.path.exists("utils"):
            os.makedirs("utils")
            
        with open("utils/thread_safe_logging.py", "w") as f:
            f.write("""
\"\"\"
Thread-safe logging functions for AI Trading System
\"\"\"
import os
import sys
import time
import threading
from datetime import datetime

_log_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    \"\"\"Thread-safe function to log messages to a file\"\"\"
    with _log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp} - {message}\\n")
            f.flush()

def log_to_console(message):
    \"\"\"Thread-safe function to log messages to console\"\"\"
    with _log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - {message}")
        sys.stdout.flush()

def thread_safe_log(message, log_file="training_logs.txt"):
    \"\"\"Combined logging function that logs to both file and console\"\"\"
    log_to_file(message, log_file)
    log_to_console(message)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    \"\"\"Read log entries from file with a maximum number of lines\"\"\"
    if not os.path.exists(log_file):
        return []
        
    with open(log_file, "r") as f:
        lines = f.readlines()
        
    # Return last N lines (most recent)
    return lines[-max_lines:]
""")
        
        # Tạo file log trống
        with open("training_logs.txt", "w") as f:
            f.write("")
            
        # Import lại sau khi tạo
        from utils.thread_safe_logging import thread_safe_log, read_logs_from_file
    
    # Thông báo cho người dùng
    progress_placeholder = st.empty()
    progress_placeholder.info("Quá trình huấn luyện bắt đầu trong nền. Bạn có thể tiếp tục sử dụng ứng dụng.")
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Khởi tạo session state nếu chưa có
    if 'training_log_messages' not in st.session_state:
        st.session_state.training_log_messages = []
        
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    
    # Thêm message khởi động huấn luyện
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_message = f"{timestamp} - 🧠 Bắt đầu quá trình huấn luyện AI trong nền..."
    st.session_state.log_messages.append(log_message)
    st.session_state.training_log_messages.append(log_message)
    
    # Ghi log vào file
    thread_safe_log("Bắt đầu quá trình huấn luyện mô hình AI")
    
    # Bắt đầu huấn luyện trong thread
    training_thread = threading.Thread(target=train_models_background)
    training_thread.daemon = True  # Thread sẽ tự đóng khi chương trình chính kết thúc
    training_thread.start()
    
    # Cập nhật UI
    def update_progress():
        placeholder = st.empty()
        log_container = placeholder.container()
        
        while training_thread.is_alive():
            # Đọc logs từ file
            logs = read_logs_from_file("training_logs.txt", 20)
            
            # Hiển thị logs
            if logs:
                log_display = "\n".join(logs)
                log_container.text_area("Tiến trình huấn luyện:", log_display, height=300)
                
                # Cập nhật progress bar
                progress = 0
                for log in logs:
                    if "LSTM" in log:
                        progress = max(progress, 20)
                    elif "Transformer" in log:
                        progress = max(progress, 40)
                    elif "CNN" in log:
                        progress = max(progress, 60)
                    elif "Similarity" in log:
                        progress = max(progress, 80)
                    elif "Meta-Learner" in log:
                        progress = max(progress, 90)
                    elif "thành công" in log or "Đã lưu" in log:
                        progress = 100
                
                progress_bar.progress(progress)
                
                # Hiển thị toast cho thông báo quan trọng
                last_log = logs[-1].lower() if logs else ""
                if "lỗi" in last_log:
                    show_toast("Có lỗi trong quá trình huấn luyện", "error")
                elif "thành công" in last_log:
                    show_toast("Huấn luyện mô hình thành công!", "success")
            
            # Tạm dừng 2 giây trước khi cập nhật lại
            time.sleep(2)
        
        # Khi thread kết thúc, hiển thị thông báo hoàn tất
        final_logs = read_logs_from_file("training_logs.txt", 20)
        success = any("thành công" in log.lower() for log in final_logs)
        
        if success:
            progress_bar.progress(100)
            progress_placeholder.success("Huấn luyện mô hình thành công!")
        else:
            progress_placeholder.error("Huấn luyện mô hình thất bại. Xem logs để biết chi tiết.")
    
    # Chạy update_progress trong một thread riêng biệt
    update_thread = threading.Thread(target=update_progress)
    update_thread.daemon = True
    update_thread.start()
    
    return True