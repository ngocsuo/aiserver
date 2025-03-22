# Hướng dẫn triển khai ETHUSDT Dashboard lên server

## 1. Chuẩn bị package triển khai

Để đóng gói ứng dụng và triển khai lên server của bạn, hãy sử dụng script `prepare_for_server.py`:

```bash
# Tạo package cơ bản
./prepare_for_server.py

# Tạo package bao gồm dữ liệu hiện có
./prepare_for_server.py --include-data

# Chỉ định tên file đầu ra
./prepare_for_server.py --output ethusdt_dashboard.zip
```

Script này sẽ tạo một file zip có tên theo định dạng `ethusdt_dashboard_deploy_YYYYMMDD_HHMMSS.zip` hoặc theo tên bạn chỉ định.

## 2. Chuyển package lên server

Sử dụng SCP hoặc SFTP để chuyển file zip lên server của bạn:

```bash
scp ethusdt_dashboard_deploy_YYYYMMDD_HHMMSS.zip user@your-server-ip:/path/to/destination/
```

## 3. Triển khai trên server

Đăng nhập vào server và thực hiện các bước sau:

```bash
# Di chuyển đến thư mục đích
cd /path/to/destination

# Giải nén file zip
unzip ethusdt_dashboard_deploy_YYYYMMDD_HHMMSS.zip

# Di chuyển vào thư mục giải nén
cd ethusdt_dashboard

# Chạy script cài đặt
./install.sh
```

Nếu bạn có quyền root, script cài đặt sẽ tự động cấu hình systemd service để ứng dụng chạy như một dịch vụ và tự động khởi động lại khi cần.

## 4. Kiểm tra trạng thái (nếu đã cài đặt systemd service)

```bash
systemctl status ethusdt-dashboard.service
```

## 5. Xem log

```bash
# Xem log ứng dụng
tail -f logs/app.log

# Xem log dịch vụ triển khai
tail -f deployment/deploy.log

# Xem log hệ thống
tail -f deployment/logs/system.log
```

## 6. Cấu hình

Các file cấu hình chính:

- `config.py`: Cấu hình tổng thể của ứng dụng
- `deployment/deploy_service.py`: Cấu hình dịch vụ triển khai

## 7. Truy cập ứng dụng

Sau khi triển khai, ứng dụng sẽ chạy tại:

```
http://your-server-ip:5000
```

## 8. Khắc phục sự cố

### 8.1. Nếu ứng dụng không khởi động

```bash
# Kiểm tra logs
tail -f logs/app.log
tail -f deployment/deploy.log

# Khởi động thủ công
python run_clean.py --mode direct
```

### 8.2. Nếu gặp vấn đề về bộ nhớ

```bash
# Dọn dẹp logs và cache
python deployment/optimize_logs.py --all

# Khởi động lại với ít tính năng hơn
python run_clean.py --mode direct
```

### 8.3. Nếu gặp vấn đề về kết nối API

```bash
# Kiểm tra kết nối Binance API
python test_binance_connection.py
```

## 9. Nâng cấp

Để nâng cấp ứng dụng, tạo package mới và thực hiện lại các bước triển khai:

```bash
# Dừng dịch vụ hiện tại
systemctl stop ethusdt-dashboard.service

# Triển khai phiên bản mới
# (Giải nén và cài đặt như các bước ở trên)

# Khởi động lại dịch vụ
systemctl start ethusdt-dashboard.service
```