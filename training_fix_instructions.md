# Hướng dẫn sửa lỗi thread-safety trong quá trình huấn luyện mô hình

## Mô tả vấn đề

Hiện tại, ứng dụng đang gặp lỗi trong quá trình huấn luyện mô hình vì cố gắng truy cập `st.session_state` từ một thread khác. Lỗi cụ thể là:

```
AttributeError: st.session_state has no attribute "training_log_messages". Did you forget to initialize it?
```

Đây là vấn đề thread-safety vì Streamlit không cho phép truy cập `session_state` từ thread không phải là main thread.

## Giải pháp

Sử dụng cơ chế ghi log thread-safe thay vì trực tiếp truy cập session_state. Các bước thực hiện:

1. Tạo module `thread_safe_logging.py` (đã tạo sẵn)
2. Sửa hàm `train_models_background()` để sử dụng thread_safe_log() thay vì update_log()
3. Đọc các log từ file để hiển thị trong UI

## Các bước thực hiện chi tiết

### Bước 1: Đã tạo module utils/thread_safe_logging.py
```python
def thread_safe_log(message, log_file="training_logs.txt"):
    """Combined logging function that logs to both file and console"""
    # Implementation... (đã code sẵn)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    """Read log entries from file with a maximum number of lines"""
    # Implementation... (đã code sẵn)
```

### Bước 2: Thay thế hàm train_models_background() trong app.py

1. Thêm import:
```python
from utils.thread_safe_logging import thread_safe_log, read_logs_from_file
```

2. Thay thế tất cả lệnh update_log() trong hàm train_models_background() bằng thread_safe_log()
   - Xem file `fix_train_model.py` để tham khảo cách thay thế chi tiết

3. Tạo file log nếu chưa tồn tại:
```python
if not os.path.exists("training_logs.txt"):
    with open("training_logs.txt", "w") as f:
        f.write("# Training logs file created\n")
```

### Bước 3: Thay đổi cách hiển thị log trong UI

1. Khi cần hiển thị logs trong UI, sử dụng:
```python
logs = read_logs_from_file("training_logs.txt")
for log in logs:
    st.text(log.strip())
```

## Triển khai trên server thực tế

Khi triển khai trên server thực tế, cần thực hiện các bước sau:

1. Đảm bảo file `utils/thread_safe_logging.py` đã được copy lên server
2. Đảm bảo đã tạo file `training_logs.txt` trên server với quyền ghi: 
```bash
touch training_logs.txt && chmod 666 training_logs.txt
```
3. Sửa hàm `train_models_background()` như hướng dẫn ở trên

## Xử lý lỗi hạn chế địa lý (Geographic restriction)

Hiện tại, ứng dụng gặp lỗi "Geographic restriction" khi kết nối đến Binance API. Đây là do hạn chế khu vực địa lý từ Binance.

Lỗi này sẽ được giải quyết khi triển khai trên máy chủ riêng của bạn ở Việt Nam. Không cần thực hiện thay đổi gì vì code đã xử lý trường hợp này.