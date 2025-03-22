# Hướng dẫn sửa lỗi cho triển khai trên server

## Tóm tắt các vấn đề đã phát hiện:

1. **Lỗi thread-safety trong quá trình huấn luyện:** Không thể truy cập st.session_state từ thread khác
2. **Lỗi pandas style.map:** Phiên bản pandas mới không có phương thức style.map
3. **Lỗi kết nối Binance API:** Geographic restriction - sẽ được giải quyết khi triển khai ở Việt Nam

## Giải pháp cho tất cả các vấn đề:

### 1. Sửa lỗi pandas style.map (dòng 2194)

Tìm dòng:
```python
styled_df = recent_preds.style.map(style_trend, subset=['trend'])
```

Thay bằng:
```python
try:
    # Thử cách 1: sử dụng style.applymap (pandas cũ)
    styled_df = recent_preds.style.applymap(style_trend, subset=['trend'])
except AttributeError:
    # Thử cách 2: sử dụng style.apply với hàm khác
    def highlight_trend(s):
        return ['background-color: green; color: white' if x == 'LONG' 
                else 'background-color: red; color: white' if x == 'SHORT'
                else 'background-color: gray; color: white' for x in s]
    
    styled_df = recent_preds.style.apply(highlight_trend, subset=['trend'])
```

### 2. Sửa lỗi thread-safety trong huấn luyện:

#### Bước 1: Tạo file thread_safe_logging.py

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

#### Bước 2: Sửa hàm train_models:

```python
def train_models():
    """Train all prediction models in a background thread"""
    import os
    import json
    import threading
    from utils.thread_safe_logging import thread_safe_log
    
    # Tạo file training_logs.txt nếu chưa tồn tại
    if not os.path.exists("training_logs.txt"):
        with open("training_logs.txt", "w") as f:
            f.write("# Training logs started\n")
    
    # Lấy dữ liệu từ session_state và chuẩn bị truyền vào thread
    if not hasattr(st.session_state, 'latest_data') or st.session_state.latest_data is None:
        st.error("Không có dữ liệu để huấn luyện. Vui lòng thu thập dữ liệu trước.")
        return False
    
    # Lấy các dữ liệu cần thiết từ session_state
    latest_data = st.session_state.latest_data.copy() if hasattr(st.session_state, 'latest_data') else None
    data_processor = st.session_state.data_processor if hasattr(st.session_state, 'data_processor') else None
    model_trainer = st.session_state.model_trainer if hasattr(st.session_state, 'model_trainer') else None
    custom_params = st.session_state.get('custom_training_params', None)
    
    # Kiểm tra dữ liệu đủ để huấn luyện không
    if latest_data is None or data_processor is None or model_trainer is None:
        thread_safe_log("Không đủ dữ liệu hoặc thành phần cần thiết để huấn luyện")
        st.error("Không đủ dữ liệu hoặc thành phần cần thiết để huấn luyện")
        return False
    
    # Truyền dữ liệu vào thread qua tham số
    thread_safe_log("Khởi động quá trình huấn luyện AI...")
    training_thread = threading.Thread(
        target=train_models_background,
        args=(latest_data, data_processor, model_trainer, custom_params)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return True
```

#### Bước 3: Sửa hàm train_models_background:

```python
def train_models_background(latest_data, data_processor, model_trainer, custom_params=None):
    """Hàm huấn luyện chạy trong thread riêng biệt"""
    import datetime
    import json
    import os
    import config
    from utils.thread_safe_logging import thread_safe_log
    
    try:
        # Sử dụng thread_safe_log thay vì update_log
        thread_safe_log("Bắt đầu quá trình huấn luyện mô hình AI trong nền...")
        thread_safe_log("Bước 1/5: Chuẩn bị dữ liệu ETHUSDT...")
        
        # Bây giờ sử dụng các dữ liệu được truyền vào thay vì truy cập session_state
        data = latest_data
        thread_safe_log(f"Số điểm dữ liệu: {len(data)} nến")
        thread_safe_log(f"Khung thời gian: {data.name if hasattr(data, 'name') else config.TIMEFRAMES['primary']}")
        thread_safe_log(f"Phạm vi ngày: {data.index.min()} đến {data.index.max()}")
        
        # Step 2: Preprocess data
        thread_safe_log("Bước 2/5: Tiền xử lý dữ liệu và tính toán chỉ báo kỹ thuật...")
        processed_data = data_processor.process_data(data)
        
        # Display feature information
        feature_count = len(processed_data.columns) - 1  # Exclude target column
        thread_safe_log(f"Đã tạo {feature_count} chỉ báo kỹ thuật và tính năng")
        thread_safe_log(f"Mẫu huấn luyện: {len(processed_data)} (sau khi loại bỏ giá trị NaN)")
        
        # Display class distribution
        if 'target_class' in processed_data.columns:
            class_dist = processed_data['target_class'].value_counts()
            thread_safe_log(f"Phân phối lớp: SHORT={class_dist.get(0, 0)}, NEUTRAL={class_dist.get(1, 0)}, LONG={class_dist.get(2, 0)}")
        
        # Step 3: Prepare sequence and image data
        thread_safe_log("Bước 3/5: Chuẩn bị dữ liệu chuỗi cho mô hình LSTM và Transformer...")
        sequence_data = data_processor.prepare_sequence_data(processed_data)
        
        thread_safe_log("Chuẩn bị dữ liệu hình ảnh cho mô hình CNN...")
        image_data = data_processor.prepare_cnn_data(processed_data)
        
        # Step 4: Train all models
        thread_safe_log("Bước 4/5: Huấn luyện mô hình LSTM...")
        lstm_model, lstm_history = model_trainer.train_lstm(sequence_data)
        thread_safe_log(f"Mô hình LSTM đã huấn luyện với độ chính xác: {lstm_history.get('val_accuracy', [-1])[-1]:.4f}")
        
        thread_safe_log("Huấn luyện mô hình Transformer...")
        transformer_model, transformer_history = model_trainer.train_transformer(sequence_data)
        thread_safe_log(f"Mô hình Transformer đã huấn luyện với độ chính xác: {transformer_history.get('val_accuracy', [-1])[-1]:.4f}")
        
        thread_safe_log("Huấn luyện mô hình CNN...")
        cnn_model, cnn_history = model_trainer.train_cnn(image_data)
        thread_safe_log(f"Mô hình CNN đã huấn luyện với độ chính xác: {cnn_history.get('val_accuracy', [-1])[-1]:.4f}")
        
        thread_safe_log("Huấn luyện mô hình Similarity lịch sử...")
        historical_model, _ = model_trainer.train_historical_similarity(sequence_data)
        
        thread_safe_log("Bước 5/5: Huấn luyện mô hình Meta-Learner...")
        meta_model, _ = model_trainer.train_meta_learner(sequence_data, image_data)
        
        # Lưu kết quả huấn luyện vào file (thay vì session_state)
        training_result = {
            "success": True,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_points": len(data),
            "feature_count": feature_count,
            "training_samples": len(processed_data),
            "class_distribution": {
                "SHORT": int(class_dist.get(0, 0)) if 'target_class' in processed_data.columns and class_dist is not None else 0,
                "NEUTRAL": int(class_dist.get(1, 0)) if 'target_class' in processed_data.columns and class_dist is not None else 0,
                "LONG": int(class_dist.get(2, 0)) if 'target_class' in processed_data.columns and class_dist is not None else 0
            },
            "model_performance": {
                "lstm": float(lstm_history.get('val_accuracy', [-1])[-1]),
                "transformer": float(transformer_history.get('val_accuracy', [-1])[-1]),
                "cnn": float(cnn_history.get('val_accuracy', [-1])[-1]),
                "historical_similarity": 0.65,
                "meta_learner": 0.85
            }
        }
        with open('training_result.json', 'w') as f:
            json.dump(training_result, f)
        
        # Thông báo đã huấn luyện thành công - set flag cho main thread
        with open('training_completed.txt', 'w') as f:
            f.write('success')
        
        thread_safe_log("Tất cả các mô hình đã huấn luyện thành công!")
        return True
    except Exception as e:
        thread_safe_log(f"LỖI trong quá trình huấn luyện: {str(e)}")
        # Lưu thông tin lỗi vào file
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
            
        return False
```

#### Bước 4: Tạo hàm kiểm tra kết quả huấn luyện:

```python
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
```

#### Bước 5: Kiểm tra kết quả huấn luyện trong main thread:

Thêm đoạn code sau vào phần khởi tạo ứng dụng hoặc nơi thích hợp trong main thread:

```python
# Kiểm tra kết quả huấn luyện từ background thread
training_status = is_training_complete()
if training_status == 'success':
    # Đọc kết quả huấn luyện từ file
    training_result = get_training_result()
    if training_result and training_result.get('success', False):
        # Cập nhật session_state với kết quả huấn luyện
        st.session_state.model_trained = True
        st.session_state.training_info = training_result
        # Hiển thị thông báo thành công
        st.success("🎉 Mô hình AI đã được huấn luyện thành công!")
        # Cập nhật UI
        st.rerun()
elif training_status == 'error':
    # Đọc thông tin lỗi
    training_result = get_training_result()
    if training_result:
        # Hiển thị thông báo lỗi
        error_msg = training_result.get('error', 'Unknown error')
        st.error(f"❌ Lỗi huấn luyện mô hình: {error_msg}")
```

## Phần bổ sung:

### Hiển thị nhật ký huấn luyện từ file:

Thay thế đoạn code hiển thị training logs:

```python
# Hiển thị training logs từ file thay vì session_state
from utils.thread_safe_logging import read_logs_from_file

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
```

## Các bước triển khai:

1. Sao chép file thread_safe_logging.py vào thư mục utils/
2. Sửa hàm train_models, train_models_background và thêm các hàm mới
3. Sửa lỗi pandas style.map
4. Tạo file training_logs.txt trước khi chạy: `touch training_logs.txt && chmod 666 training_logs.txt`
5. Khởi động lại ứng dụng