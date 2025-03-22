# Hướng dẫn triển khai ETHUSDT Dashboard lên Server

## Chuẩn bị

1. Cài đặt permissions cho các script:
```bash
chmod +x sync_to_server.sh
chmod +x server_setup.sh
```

## Bước 1: Thiết lập server

Kết nối SSH vào server:
```bash
ssh root@45.76.196.13
```

Tải script cài đặt trực tiếp:
```bash
wget -O setup.sh https://raw.githubusercontent.com/yourusername/ethusdt_dashboard/main/server_setup.sh
chmod +x setup.sh
```

Hoặc bạn có thể tạo file `setup.sh` và sao chép nội dung từ file `server_setup.sh`.

Sau đó chạy:
```bash
./setup.sh
```

Script này sẽ cài đặt:
- Python và các gói phụ thuộc
- Thư viện TA-Lib cho phân tích kỹ thuật
- Môi trường ảo Python
- Cấu trúc thư mục cần thiết
- Systemd service để tự động khởi động ứng dụng

## Bước 2: Đồng bộ code từ Replit

Từ Replit, chạy script đồng bộ:
```bash
./sync_to_server.sh
```

Script này sẽ:
- Chuyển toàn bộ code từ Replit sang server
- Cấu hình API keys trên server
- Khởi động lại ứng dụng

## Bước 3: Kiểm tra hoạt động

Sau khi đồng bộ, kiểm tra trạng thái của ứng dụng trên server:
```bash
systemctl status ethusdt-dashboard
```

Truy cập ứng dụng tại:
```
http://45.76.196.13:5000
```

## Các lệnh quản lý hữu ích

Khởi động ứng dụng:
```bash
systemctl start ethusdt-dashboard
```

Dừng ứng dụng:
```bash
systemctl stop ethusdt-dashboard
```

Khởi động lại ứng dụng:
```bash
systemctl restart ethusdt-dashboard
```

Xem logs:
```bash
journalctl -u ethusdt-dashboard -f
```

Đường dẫn logs cụ thể:
```bash
cat /root/ethusdt_dashboard/logs/streamlit.log
```

## Vấn đề thường gặp

### Không kết nối được với Binance API
Kiểm tra lại các API keys đã được cấu hình đúng:
```bash
nano /root/ethusdt_dashboard/restart.sh
```

### Không thể truy cập ứng dụng từ trình duyệt
Kiểm tra firewall:
```bash
ufw status
```

Nếu cần, mở port 5000:
```bash
ufw allow 5000/tcp
```

### Cập nhật code từ Replit
Đơn giản là chạy lại script đồng bộ từ Replit:
```bash
./sync_to_server.sh
```

## Cấu trúc thư mục trên server

```
/root/ethusdt_dashboard/
├── app.py                # File chính của ứng dụng
├── config.py             # Cấu hình
├── requirements_server.txt  # Danh sách thư viện
├── restart.sh            # Script khởi động lại
├── setup_api_keys.sh     # Script cấu hình API keys
├── utils/                # Thư mục tiện ích
├── models/               # Thư mục mô hình
├── data/                 # Thư mục dữ liệu
├── logs/                 # Thư mục logs
└── saved_models/         # Thư mục lưu mô hình
```