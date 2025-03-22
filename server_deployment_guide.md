# Hướng dẫn triển khai ETHUSDT Dashboard lên Server

Tài liệu này cung cấp hướng dẫn chi tiết để triển khai ứng dụng ETHUSDT Dashboard từ Replit lên server Ubuntu của bạn.

## Yêu cầu hệ thống

- Server Ubuntu 20.04 hoặc cao hơn
- Quyền truy cập SSH với tài khoản root
- Đã cài đặt các công cụ cơ bản: git, curl, wget, rsync
- Cổng 5000 đã được mở trong firewall
- API key Binance hợp lệ

## Quy trình triển khai

### 1. Đồng bộ mã nguồn từ Replit lên server

Thực hiện lệnh sau trong Replit để đồng bộ mã nguồn lên server:

```bash
./sync_to_server.sh
```

Script này sẽ:
- Kiểm tra kết nối đến server
- Sao chép tất cả các file nguồn cần thiết
- Tự động chuyển API keys từ Replit sang server
- Khởi động lại ứng dụng trên server

### 2. Cài đặt môi trường lần đầu trên server

Nếu đây là lần đầu bạn triển khai ứng dụng lên server, cần thực hiện script cài đặt để chuẩn bị môi trường. SSH vào server và chạy:

```bash
cd /root/ethusdt_dashboard
./server_setup.sh
```

Script này sẽ:
- Cập nhật hệ thống
- Cài đặt Python và các gói phụ thuộc
- Tạo môi trường ảo Python
- Cài đặt các thư viện từ requirements_server.txt
- Cấu hình Streamlit
- Tạo và kích hoạt service systemd
- Tạo script khởi động lại

### 3. Kiểm tra kết nối Binance API trên server

Để đảm bảo Binance API hoạt động đúng trên server, chạy lệnh sau trong Replit:

```bash
./automation_scripts/test_binance_connection.sh
```

### 4. Kiểm tra trạng thái ứng dụng trên server

Để kiểm tra trạng thái của ứng dụng và server, chạy:

```bash
./automation_scripts/check_server_status.sh
```

### 5. Truy cập ứng dụng

Sau khi triển khai, bạn có thể truy cập ứng dụng tại:

```
http://SERVER_IP:5000
```

Với SERVER_IP là địa chỉ IP của server của bạn (45.76.196.13).

## Quản lý ứng dụng trên server

### Khởi động lại ứng dụng

```bash
ssh root@45.76.196.13 "/root/ethusdt_dashboard/restart.sh"
```

### Kiểm tra logs

```bash
ssh root@45.76.196.13 "tail -f /root/ethusdt_dashboard/logs/streamlit.log"
```

### Kiểm tra trạng thái service

```bash
ssh root@45.76.196.13 "systemctl status ethusdt-dashboard"
```

### Dừng ứng dụng

```bash
ssh root@45.76.196.13 "systemctl stop ethusdt-dashboard"
```

### Khởi động ứng dụng

```bash
ssh root@45.76.196.13 "systemctl start ethusdt-dashboard"
```

## Cập nhật ứng dụng

Khi bạn muốn cập nhật ứng dụng với phiên bản mới từ Replit:

1. Cập nhật mã nguồn trên Replit
2. Chạy `./sync_to_server.sh` để đồng bộ thay đổi lên server
3. Script sẽ tự động khởi động lại ứng dụng

## Xử lý sự cố

### Ứng dụng không khởi động

Kiểm tra logs:

```bash
ssh root@45.76.196.13 "journalctl -u ethusdt-dashboard -n 50"
```

### Vấn đề kết nối Binance API

Kiểm tra trạng thái proxy và API key:

```bash
./automation_scripts/test_binance_connection.sh
```

### Sửa lỗi hệ thống

Nếu cần khởi động lại toàn bộ server:

```bash
ssh root@45.76.196.13 "reboot"
```

## Bảo mật

- Đảm bảo cổng 22 (SSH) được bảo vệ bằng key authentication
- Chỉ mở cổng 5000 cho Streamlit
- Đặt mật khẩu mạnh cho tài khoản root
- Cân nhắc thiết lập tường lửa nếu cần

## Thông tin liên hệ

Nếu bạn gặp vấn đề khi triển khai, vui lòng liên hệ hỗ trợ.