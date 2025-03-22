# Sửa lỗi khẩn cấp cho lỗi session_state

## Lỗi hiện tại trên server:
```
LỖI trong quá trình huấn luyện: st.session_state has no attribute "latest_data". Did you forget to initialize it?
Thread 'Thread-21 (train_models_background)': missing ScriptRunContext
```

## Các vấn đề cần kiểm tra:

1. **Hàm train_models và train_models_background có được cập nhật chưa?**
   - Cần đảm bảo bạn đã thay thế TOÀN BỘ hai hàm này với phiên bản không sử dụng session_state
   - Các hàm mới nhận dữ liệu qua tham số thay vì truy cập session_state

2. **File thread_safe_logging.py đã được tạo chưa?**
   - Cần đảm bảo file utils/thread_safe_logging.py đã được tạo đúng

3. **Các file cần thiết đã được tạo chưa?**
   - File training_logs.txt cần được tạo trước khi chạy

## GIẢI PHÁP KHẨN CẤP:

### 1. Tạo thread_safe_logging.py
Tạo file utils/thread_safe_logging.py với nội dung:

```python
"""
Thread-safe logging functions for AI Trading System
"""
import threading
import datetime
import os

# Thread-safe lock for logging
log_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    """Thread-safe function to log messages to a file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}\n"
    
    with log_lock:
        try:
            with open(log_file, "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing to log file: {e}")

def log_to_console(message):
    """Thread-safe function to log messages to console"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}"
    
    with log_lock:
        print(log_entry)

def thread_safe_log(message, log_file="training_logs.txt"):
    """Combined logging function that logs to both file and console"""
    log_to_file(message, log_file)
    log_to_console(message)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    """Read log entries from file with a maximum number of lines"""
    if not os.path.exists(log_file):
        return []
        
    try:
        with open(log_file, "r") as f:
            # Read last max_lines lines
            lines = f.readlines()
            return lines[-max_lines:] if len(lines) > max_lines else lines
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []
```

### 2. Tạo file training_logs.txt
```bash
touch training_logs.txt && chmod 666 training_logs.txt
```

### 3. CÁCH NHANH NHẤT: Vô hiệu hóa tính năng huấn luyện background

Nếu bạn không có thời gian sửa đầy đủ, đây là giải pháp tạm thời:

```python
def train_models():
    """Train all prediction models in a background thread"""
    st.error("Tính năng huấn luyện tự động đang được bảo trì. Vui lòng thử lại sau.")
    return False
```

Thay thế hàm train_models hiện tại bằng hàm này sẽ vô hiệu hóa tính năng huấn luyện cho đến khi bạn có thời gian sửa hoàn chỉnh.

### 4. GIẢI PHÁP ĐÚNG: Bắt lỗi để tránh truy cập session_state

Sửa lại hàm train_models_background để bắt lỗi trước khi truy cập session_state:

```python
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
        
        if data is None or data.empty:
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
        model_trainer.train_lstm(sequence_data)
        
        thread_safe_log("Huấn luyện mô hình Transformer...")
        model_trainer.train_transformer(sequence_data)
        
        thread_safe_log("Huấn luyện mô hình CNN...")
        model_trainer.train_cnn(image_data)
        
        thread_safe_log("Huấn luyện mô hình Similarity lịch sử...")
        model_trainer.train_historical_similarity(sequence_data)
        
        thread_safe_log("Huấn luyện mô hình Meta-Learner...")
        model_trainer.train_meta_learner(sequence_data, image_data)
        
        thread_safe_log("Huấn luyện thành công tất cả các mô hình!")
        
        # Lưu trạng thái huấn luyện vào file
        try:
            import json
            training_result = {
                "success": True,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": "Huấn luyện thành công tất cả các mô hình"
            }
            with open('training_result.json', 'w') as f:
                json.dump(training_result, f)
        except Exception as e:
            thread_safe_log(f"Lỗi lưu kết quả huấn luyện: {str(e)}")
                
    except Exception as e:
        from utils.thread_safe_logging import thread_safe_log
        thread_safe_log(f"LỖI trong quá trình huấn luyện: {str(e)}")

def train_models():
    """Bắt đầu huấn luyện model trong thread riêng"""
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
    import streamlit as st
    st.success("Đã bắt đầu huấn luyện mô hình trong nền. Kiểm tra logs để theo dõi tiến trình.")
    
    return True
```

### 5. Khởi động lại ứng dụng
```bash
cd /đường/dẫn/tới/ứng/dụng
touch training_logs.txt && chmod 666 training_logs.txt
streamlit run app.py
```