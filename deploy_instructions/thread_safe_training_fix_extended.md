# Sửa lỗi Thread Safety trong Huấn luyện (Mở rộng)

## Mô tả lỗi mới:

```
LỖI trong quá trình huấn luyện: st.session_state has no attribute "latest_data". Did you forget to initialize it?
```

Đây tiếp tục là lỗi thread-safety: không thể truy cập st.session_state từ thread khác.

## Giải pháp nâng cao:

Thay vì chỉ sửa hàm ghi log, chúng ta cần phải thiết kế lại toàn bộ quá trình huấn luyện để:
1. Không sử dụng st.session_state từ thread background
2. Truyền dữ liệu cần thiết vào thread qua tham số

## Cách triển khai:

### 1. Sửa đổi hàm train_models():

```python
def train_models():
    """Train all prediction models in a background thread"""
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

### 2. Sửa đổi hàm train_models_background():

```python
def train_models_background(latest_data, data_processor, model_trainer, custom_params=None):
    """Hàm huấn luyện chạy trong thread riêng biệt"""
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
        
        # Step 3 và các bước tiếp theo... (tiếp tục cùng logic nhưng sử dụng dữ liệu được truyền vào)
        # ...
        
        # Lưu kết quả huấn luyện vào file (thay vì session_state)
        training_result = {
            "success": True,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": "Huấn luyện thành công"
        }
        with open('training_result.json', 'w') as f:
            json.dump(training_result, f)
        
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
        return False
```

### 3. Thêm function đọc kết quả huấn luyện:

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
```

## Triển khai trên server:

1. Sao chép cả 3 hàm trên vào file app.py trên server
2. Đảm bảo bạn đã cài đặt và sử dụng thread_safe_logging.py
3. Tạo file training_logs.txt trống trước khi chạy

## Lưu ý bổ sung:

- Đây là một thay đổi lớn trong kiến trúc của quá trình huấn luyện
- Sau khi thực hiện thay đổi này, bạn cần cập nhật các phần UI hiển thị kết quả huấn luyện để đọc từ file thay vì session_state
- Một cách tiếp cận khác là sử dụng global variables thay vì session_state, nhưng cách sử dụng file như trên bền vững hơn