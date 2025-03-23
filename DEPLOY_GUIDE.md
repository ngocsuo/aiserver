# Hướng dẫn triển khai ETHUSDT Dashboard

## Giới thiệu

Tài liệu này hướng dẫn chi tiết quy trình triển khai ETHUSDT Dashboard lên server mới. Hệ thống chạy một ứng dụng Streamlit kết nối với Binance API để cung cấp dự đoán ETH/USDT dựa trên mô hình AI.

## Yêu cầu hệ thống

- **Hệ điều hành**: Ubuntu 22.04 hoặc mới hơn
- **RAM**: Tối thiểu 4GB (Khuyến nghị: 8GB+)
- **CPU**: 2 cores trở lên
- **Dung lượng ổ cứng**: Tối thiểu 20GB
- **Kết nối mạng**: Ổn định, băng thông tốt

## Phương pháp triển khai

Có hai phương pháp triển khai chính:

### Phương pháp 1: Sử dụng Package Cài Đặt Tự Động

1. **Tạo Package**:
   ```bash
   python prepare_for_server.py
   ```
   - Lệnh này tạo file `ethusdt_dashboard.zip` chứa toàn bộ mã nguồn và script cài đặt

2. **Upload lên Server**:
   ```bash
   scp ethusdt_dashboard.zip root@your_server_ip:/root/
   ```

3. **Cài đặt trên Server**:
   ```bash
   ssh root@your_server_ip
   cd /root
   unzip ethusdt_dashboard.zip
   bash server_install.sh
   ```

### Phương pháp 2: Sử dụng Script Đồng Bộ Hóa

1. **Cập nhật thông tin server**:
   Chỉnh sửa file `sync_to_server.sh`, thay `your_actual_server_ip` bằng địa chỉ IP thực của server

2. **Chạy script đồng bộ**:
   ```bash
   ./sync_to_server.sh
   ```

3. **Kiểm tra trạng thái dịch vụ**:
   ```bash
   ssh root@your_server_ip
   systemctl status ethusdt-dashboard
   ```

## Cấu hình API Binance

Để dashboard hoạt động, cần cung cấp API key Binance:

1. **Cài đặt qua biến môi trường**:
   ```bash
   BINANCE_API_KEY="your_api_key" BINANCE_API_SECRET="your_api_secret" bash server_install.sh
   ```

2. **Cập nhật thủ công sau khi cài đặt**:
   ```bash
   nano /root/ethusdt_dashboard/.env
   ```
   Thêm hoặc cập nhật các dòng:
   ```
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   ```

## Quản lý Dịch Vụ

### Kiểm tra trạng thái:
```bash
systemctl status ethusdt-dashboard
```

### Khởi động/dừng/khởi động lại:
```bash
systemctl start ethusdt-dashboard
systemctl stop ethusdt-dashboard
systemctl restart ethusdt-dashboard
```

### Xem logs:
```bash
journalctl -fu ethusdt-dashboard
```

## Xử lý Sự Cố

### Lỗi không thể kết nối Binance API:

1. Kiểm tra API keys:
   ```bash
   cat /root/ethusdt_dashboard/.env
   ```

2. Kiểm tra kết nối mạng:
   ```bash
   ping api.binance.com
   ```

3. Kiểm tra logs:
   ```bash
   journalctl -fu ethusdt-dashboard | grep "error"
   ```

### Lỗi dịch vụ không khởi động:

1. Kiểm tra cài đặt Python và các thư viện:
   ```bash
   cd /root/ethusdt_dashboard
   source venv/bin/activate
   pip list | grep streamlit
   ```

2. Thử chạy ứng dụng thủ công:
   ```bash
   cd /root/ethusdt_dashboard
   source venv/bin/activate
   streamlit run app.py
   ```

## Cập Nhật Phiên Bản

Để cập nhật lên phiên bản mới:

1. **Sử dụng script đồng bộ**:
   ```bash
   ./sync_to_server.sh
   ```

2. **Hoặc tải lên package mới**:
   ```bash
   scp ethusdt_dashboard.zip root@your_server_ip:/root/
   ssh root@your_server_ip
   cd /root
   unzip -o ethusdt_dashboard.zip
   bash server_install.sh
   ```

## Tham Khảo

- [Tài liệu Streamlit](https://docs.streamlit.io/)
- [Tài liệu Binance API](https://binance-docs.github.io/apidocs/)
- [Quản lý Systemd](https://www.digitalocean.com/community/tutorials/how-to-use-systemctl-to-manage-systemd-services-and-units)