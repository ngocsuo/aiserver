# Hướng dẫn triển khai và sửa lỗi trên Server

## 1. Sửa lỗi "too many values to unpack (expected 2)"

### Bước 1: Tìm và sửa file `continuous_trainer.py`
```bash
# SSH vào server của bạn
ssh user@your-server-ip

# Tìm vị trí của file continuous_trainer.py trong dự án
cd /đường/dẫn/đến/dự/án
find . -name "continuous_trainer.py"
```

### Bước 2: Sửa đoạn code gây lỗi
Mở file đã tìm thấy và tìm phương thức `_train_for_timeframe`:

```bash
nano /đường/dẫn/đến/continuous_trainer.py
```

Tìm dòng:
```python
models, histories = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
```

Sửa thành:
```python
models = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
```

Lưu file (Ctrl+O, Enter, Ctrl+X).

## 2. Sửa lỗi "Thread missing ScriptRunContext"

### Bước 1: Tạo file `utils/thread_safe_logging.py` nếu chưa có
```bash
mkdir -p utils  # Tạo thư mục nếu chưa tồn tại
nano utils/thread_safe_logging.py
```

### Bước 2: Dán nội dung sau vào file:
```python
"""
Thread-safe logging functions for AI Trading System
"""
import os
import sys
import threading
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Threading lock for file operations
_log_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    """Thread-safe function to log messages to a file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {message}\n"
        
        with _log_lock:
            with open(log_file, "a") as f:
                f.write(log_entry)
                f.flush()
        return True
    except Exception as e:
        print(f"Error writing to log file: {e}", file=sys.stderr)
        return False

def log_to_console(message):
    """Thread-safe function to log messages to console"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - {message}")
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"Error logging to console: {e}", file=sys.stderr)
        return False

def thread_safe_log(message, log_file="training_logs.txt"):
    """Combined logging function that logs to both file and console"""
    log_to_console(message)
    return log_to_file(message, log_file)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    """Read log entries from file with a maximum number of lines"""
    try:
        if not os.path.exists(log_file):
            return ["No logs found"]
            
        with _log_lock:
            with open(log_file, "r") as f:
                lines = f.readlines()
                
        # Return the last max_lines entries
        return [line.strip() for line in lines[-max_lines:]]
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        return [f"Error reading logs: {e}"]
```

### Bước 3: Sửa file app.py để sử dụng thread_safe_logging

Tìm hàm `train_models_background` và `train_models` trong file app.py:

```bash
nano app.py
```

Thay thế phần update_log và các gọi st.session_state trong thread bằng thread_safe_log:

```python
def train_models_background():
    """Hàm huấn luyện chạy trong thread riêng biệt"""
    from utils.thread_safe_logging import thread_safe_log
    
    try:
        thread_safe_log("Bắt đầu quá trình huấn luyện mô hình AI trong nền...")
        
        # Phần còn lại của code, thay thế các gọi update_log bằng thread_safe_log
        # ...
        
    except Exception as e:
        thread_safe_log(f"LỖI trong quá trình huấn luyện: {str(e)}")
```

## 3. Đảm bảo training_logs.txt có thể ghi được

```bash
# Tạo file nếu không tồn tại
touch training_logs.txt

# Thiết lập quyền đọc và ghi
chmod 666 training_logs.txt
```

## 4. Khởi động lại ứng dụng

### Nếu sử dụng Supervisor

```bash
sudo supervisorctl restart your_app_name
```

### Nếu sử dụng Systemd

```bash
sudo systemctl restart your_app_name
```

### Nếu chạy trực tiếp

```bash
# Tắt tiến trình hiện tại
pkill -f "streamlit run app.py"

# Khởi động lại
cd /đường/dẫn/đến/dự/án
nohup streamlit run app.py --server.port=5000 --server.address=0.0.0.0 > streamlit.log 2>&1 &
```

## 5. Theo dõi tiến trình huấn luyện

```bash
# Xem log huấn luyện
tail -f training_logs.txt

# Hoặc sử dụng công cụ theo dõi tiến trình
python view_training_progress.py
```

## Các lỗi khác có thể gặp phải và cách khắc phục

### Lỗi "Geographic restriction detected"
- Lỗi này chỉ xuất hiện khi chạy trong môi trường Replit, sẽ không xuất hiện trên server riêng của bạn
- Đảm bảo đã cấu hình API key Binance chính xác trong file .env hoặc biến môi trường

### Lỗi bộ nhớ trong quá trình huấn luyện
- Nếu server có ít RAM, hãy giảm BATCH_SIZE trong config.py
- Hoặc thêm swap memory:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Lỗi khi huấn luyện với GPU
- Nếu sử dụng GPU, đảm bảo đã cài đặt CUDA và cuDNN đúng phiên bản
- Kiểm tra xem TensorFlow/PyTorch có nhận diện được GPU không:

```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

## Lưu ý quan trọng
- Luôn sao lưu các file quan trọng trước khi sửa chữa
- Kiểm tra logs để theo dõi và phát hiện lỗi mới
- Đảm bảo hệ thống có đủ dung lượng đĩa cho việc lưu trữ mô hình đã huấn luyện