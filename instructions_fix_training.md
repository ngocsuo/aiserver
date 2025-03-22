# Hướng dẫn sửa lỗi huấn luyện mô hình

Tôi đã phân tích vấn đề với hàm huấn luyện mô hình trong app.py và xác định được hai lỗi chính:

1. Lỗi "too many values to unpack (expected 2)" trong continuous_trainer.py
2. Lỗi "Thread missing ScriptRunContext" khi cố gắng truy cập st.session_state từ thread nền

## Cách sửa lỗi thread-safety

1. Tập tin `utils/thread_safe_logging.py` đã được tạo. Mô-đun này cung cấp các hàm ghi log an toàn cho thread.
2. Tập tin `training_logs.txt` đã được tạo để lưu trữ log huấn luyện.
3. Tập tin `fixed_training_functions.py` chứa phiên bản đã sửa của hai hàm:
   - `train_models_background()`
   - `train_models()`

### Để hoàn thành việc sửa lỗi

1. Mở tập tin app.py và tìm hai hàm:
   ```python
   def train_models():
   ```
   và
   ```python
   def train_models_background():
   ```

2. Thay thế toàn bộ nội dung hai hàm này bằng mã từ tập tin `fixed_training_functions.py`.

3. Lưu ý rằng phiên bản mới đã:
   - Không truy cập st.session_state trong thread nền
   - Sử dụng hàm thread_safe_log() để ghi log
   - Tạo data_collector, data_processor và model_trainer mới trong thread
   - Lưu mô hình vào thư mục saved_models/
   - Xử lý trường hợp train_all_models() trả về tuple hoặc dict

## Cách sửa lỗi "too many values to unpack"

Nếu bạn vẫn gặp lỗi trong continuous_trainer.py, hãy xem tập tin `fixed_train_models.py` để sửa phương thức _train_for_timeframe().

## Kiểm tra kết nối Binance API

Tôi cũng đã tạo tập tin simple_app.py để kiểm tra kết nối Binance API. Tập tin này xác nhận rằng:
- API keys đã được cấu hình đúng
- Kết nối trực tiếp đến Binance có thể bị chặn do giới hạn địa lý
- Kết nối thông qua proxy (64.176.51.107:3128:hvnteam:matkhau123) hoạt động tốt

## Để chạy phiên bản đơn giản

```bash
streamlit run simple_app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
```

## Để khởi động lại ứng dụng chính sau khi sửa lỗi

```bash
streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
```