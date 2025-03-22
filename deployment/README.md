# Hướng dẫn triển khai ETHUSDT Dashboard

## Giới thiệu

Tài liệu này hướng dẫn cách triển khai và duy trì ứng dụng ETHUSDT Dashboard một cách ổn định trên server. Các công cụ và script được cung cấp để đảm bảo ứng dụng chạy liên tục và tự động khắc phục các lỗi phổ biến.

## Các vấn đề đã biết và cách khắc phục

1. **Lỗi "Empty dataset received for normalization"**
   - Nguyên nhân: Module xử lý dữ liệu không xử lý tốt trường hợp dataset trống
   - Giải pháp: Sử dụng script `feature_engineering_fix.py` để vá lỗi

2. **Tự động dừng với thông báo "main done, exiting"**
   - Nguyên nhân: Thread chính kết thúc khi các thread nền vẫn chạy
   - Giải pháp: Sử dụng dịch vụ giám sát liên tục `deployment/deploy_service.py`

3. **Lỗi kết nối Binance API**
   - Nguyên nhân: Vấn đề về kết nối mạng hoặc proxy
   - Giải pháp: Sử dụng cấu hình proxy trong `enhanced_proxy_config.py` và retry tự động

## Cấu trúc thư mục triển khai

```
deployment/
├── README.md               # Tài liệu hướng dẫn triển khai
├── startup.sh              # Script khởi động ứng dụng với các bản vá
├── deploy_service.py       # Dịch vụ giám sát và tự động khởi động lại
├── keep_alive_process.py   # Tiến trình duy trì hoạt động liên tục
└── logs/                   # Thư mục chứa log triển khai
```

## Các cách chạy ứng dụng

### 1. Chạy với dịch vụ giám sát đầy đủ (Khuyến nghị cho môi trường production)

```bash
python run_with_monitoring.py --mode service
```

Chế độ này sẽ:
- Áp dụng các bản vá lỗi feature_engineering
- Sửa các vấn đề dữ liệu
- Khởi động dịch vụ giám sát liên tục
- Tự động khởi động lại nếu ứng dụng gặp sự cố
- Ghi log đầy đủ về tài nguyên hệ thống

### 2. Chạy với script khởi động

```bash
python run_with_monitoring.py --mode script
```

Chế độ này sẽ:
- Áp dụng các bản vá lỗi
- Sử dụng `deployment/startup.sh` để khởi động ứng dụng
- Phù hợp cho môi trường phát triển hoặc kiểm thử

### 3. Chạy trực tiếp

```bash
python run_with_monitoring.py --mode direct
```

Chế độ này sẽ:
- Áp dụng các bản vá lỗi
- Khởi động Streamlit trực tiếp không qua dịch vụ giám sát
- Phù hợp cho phát triển nhanh và kiểm tra

### 4. Chạy trực tiếp với dịch vụ giám sát

```bash
python deployment/deploy_service.py
```

## Xử lý sự cố

### 1. Kiểm tra log

Các file log được lưu tại:
- `deployment/deploy.log`: Log của dịch vụ triển khai
- `deployment/logs/app.log`: Log của ứng dụng chính
- `deployment/logs/resources.log`: Log về tài nguyên hệ thống
- `feature_engineering_fix.log`: Log của quá trình vá lỗi feature_engineering
- `data_fix.log`: Log của quá trình sửa lỗi dữ liệu

### 2. Kiểm tra trạng thái

Trạng thái hệ thống được lưu tại `system_stats.json` và được cập nhật liên tục.

### 3. Kiểm tra kết nối

Để kiểm tra kết nối đến ứng dụng:

```bash
curl -s -o /dev/null -w '%{http_code}' http://localhost:5000
```

Nếu trả về 200, ứng dụng đang hoạt động bình thường.

## Bảo trì định kỳ

### 1. Dọn dẹp log cũ

Dịch vụ giám sát tự động dọn dẹp log cũ (>7 ngày), nhưng bạn cũng có thể thực hiện thủ công:

```bash
find deployment/logs -name "*.log.*" -type f -mtime +7 -delete
```

### 2. Kiểm tra tài nguyên

```bash
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, RAM: {psutil.virtual_memory().percent}%')"
```

### 3. Khởi động lại định kỳ

Để đảm bảo ứng dụng hoạt động ổn định, nên khởi động lại ít nhất mỗi tuần một lần:

```bash
python run_with_monitoring.py --mode service
```

## Nâng cấp hệ thống

Khi cập nhật mã nguồn, hãy luôn đảm bảo:

1. Sao lưu cấu hình và dữ liệu
2. Áp dụng lại các bản vá lỗi 
3. Kiểm tra kỹ trước khi đưa vào sử dụng

## Liên hệ hỗ trợ

Nếu gặp vấn đề không thể tự khắc phục, vui lòng liên hệ:

- Email: <email_hỗ_trợ>
- Telegram: <telegram_hỗ_trợ>