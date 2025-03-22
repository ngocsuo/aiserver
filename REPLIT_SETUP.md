# Hướng dẫn thiết lập ETHUSDT Dashboard trên Replit

## Phiên bản Replit Lite

Trên môi trường Replit có giới hạn tài nguyên, chúng tôi cung cấp phiên bản "Lite" của ETHUSDT Dashboard. Phiên bản này được tối ưu hóa để chạy ổn định trên Replit với các tính năng sau:

- Giảm số lượng kết nối API đồng thời
- Chỉ sử dụng khung thời gian 5 phút
- Tần suất cập nhật dữ liệu thấp hơn (5 phút thay vì 1 phút)
- Chỉ hiển thị một số chỉ báo kỹ thuật cơ bản
- Lưu trữ dữ liệu trong bộ nhớ đệm để giảm tải

## Hướng dẫn cài đặt

### 1. Sử dụng phiên bản Lite

```bash
# Chạy phiên bản Lite
streamlit run replit_app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
```

Phiên bản này được thiết kế để chạy với tài nguyên tối thiểu và ổn định hơn.

### 2. Thiết lập thủ công

Nếu bạn muốn chạy phiên bản đầy đủ với workflow giám sát, sử dụng lệnh sau:

```bash
# Thiết lập thư mục logs
mkdir -p logs deployment/logs data

# Khởi động ứng dụng với giám sát
python run_clean.py --mode direct
```

**Lưu ý:** Phiên bản đầy đủ có thể gặp vấn đề về tài nguyên trên Replit.

## Sự khác biệt giữa các phiên bản

| Tính năng | Phiên bản Lite | Phiên bản Đầy đủ |
|-----------|----------------|------------------|
| Khung thời gian | Chỉ 5m | 1m và 5m |
| Mô hình AI | Đơn giản, dựa trên chỉ báo | Nhiều mô hình (LSTM, Transformer, CNN, ...) |
| Huấn luyện mô hình | Không | Có, tự động mỗi 30 phút |
| Cập nhật dữ liệu | Mỗi 5 phút | Mỗi 1 phút |
| Giao diện | Đơn giản | Đầy đủ, nhiều tùy chọn |
| Lưu lịch sử | Không | Có |

## Khắc phục sự cố

### 1. Lỗi "Quá nhiều kết nối"

```bash
# Khởi động lại ứng dụng với phiên bản Lite
streamlit run replit_app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
```

### 2. Lỗi "Không đủ bộ nhớ"

```bash
# Xóa bộ nhớ cache và khởi động lại
rm -rf ./.streamlit/cache
streamlit run replit_app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
```

### 3. Lỗi "Dừng đột ngột"

Đây là vấn đề phổ biến trên Replit do giới hạn tài nguyên. Phiên bản Lite được thiết kế đặc biệt để tránh vấn đề này. Nếu vẫn gặp phải, hãy thử:

```bash
# Khởi động lại ứng dụng với bộ nhớ cache được xóa
rm -rf ./.streamlit/cache
streamlit run replit_app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
```

## Triển khai lâu dài

Để triển khai lâu dài trên Replit, chúng tôi khuyến nghị:

1. Sử dụng phiên bản Lite
2. Thiết lập Replit để tự động khởi động lại khi gặp sự cố
3. Sử dụng các dịch vụ bên ngoài (nếu có thể) để theo dõi uptime

## Cách mở rộng trong tương lai

Nếu Replit nâng cấp giới hạn tài nguyên hoặc bạn chuyển sang nền tảng khác, bạn có thể dễ dàng chuyển sang phiên bản đầy đủ:

```bash
python run_clean.py --mode service
```

Điều này sẽ kích hoạt hệ thống đầy đủ với tất cả các tính năng nâng cao.