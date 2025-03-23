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

Có nhiều phương pháp triển khai tùy theo nhu cầu của bạn:

### Phương pháp 1: Sử dụng Script Triển Khai Tự Động từ GitHub (Khuyến nghị)

Script này tự động hóa toàn bộ quá trình: chuẩn bị môi trường, clone repository, cài đặt dependencies và khởi động dịch vụ.

1. **Tải script triển khai**:
   ```bash
   wget https://raw.githubusercontent.com/yourusername/ethusdt-dashboard/main/deploy_from_github.sh
   chmod +x deploy_from_github.sh
   ```

2. **Chạy script triển khai**:
   ```bash
   ./deploy_from_github.sh [GitHub_URL] [user@server_ip] [password]
   ```

   Ví dụ:
   ```bash
   ./deploy_from_github.sh https://github.com/yourusername/ethusdt-dashboard.git root@192.168.1.100 your_password
   ```

   - `GitHub_URL`: Đường dẫn đến repository (mặc định là "https://github.com/yourusername/ethusdt-dashboard.git")
   - `user@server_ip`: Thông tin đăng nhập server (bắt buộc)
   - `password`: Mật khẩu SSH (tùy chọn, nếu không cung cấp sẽ sử dụng SSH key hoặc hỏi mật khẩu)

### Phương pháp 2: Sử dụng Các Script Riêng Lẻ

Nếu bạn muốn có nhiều kiểm soát hơn hoặc đã thực hiện một số bước, bạn có thể sử dụng các script riêng lẻ:

1. **Cài đặt môi trường**:
   ```bash
   ./prepare_server_env.sh
   ```
   Script này sẽ:
   - Cài đặt các gói phụ thuộc hệ thống
   - Tạo thư mục và người dùng hệ thống
   - Thiết lập môi trường Python và dịch vụ systemd

2. **Kiểm tra sức khỏe hệ thống**:
   ```bash
   ./server_health_check.sh [--detailed]
   ```
   Script này sẽ kiểm tra:
   - Tài nguyên hệ thống (CPU, RAM, disk)
   - Trạng thái dịch vụ và port
   - Kết nối Binance API
   - Thư mục dữ liệu và logs
   - Trạng thái ứng dụng web

### Phương pháp 3: Sử dụng Package Cài Đặt Tự Động

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

### Phương pháp 4: Sử dụng Script Đồng Bộ Hóa

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

1. **Cài đặt qua biến môi trường (với deploy_from_github.sh)**:
   ```bash
   BINANCE_API_KEY="your_api_key" BINANCE_API_SECRET="your_api_secret" ./deploy_from_github.sh [GitHub_URL] [user@server_ip]
   ```

2. **Cập nhật thủ công sau khi cài đặt**:
   ```bash
   sudo nano /opt/ethusdt-dashboard/.env
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

### Kiểm tra sức khỏe hệ thống:
```bash
cd /opt/ethusdt-dashboard
./server_health_check.sh
```

Để xem thông tin chi tiết hơn:
```bash
./server_health_check.sh --detailed
```

## Xử lý Sự Cố

### Lỗi không thể kết nối Binance API:

1. Kiểm tra API keys:
   ```bash
   cat /opt/ethusdt-dashboard/.env
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
   cd /opt/ethusdt-dashboard
   source venv/bin/activate
   pip list | grep streamlit
   ```

2. Thử chạy ứng dụng thủ công:
   ```bash
   cd /opt/ethusdt-dashboard
   source venv/bin/activate
   streamlit run app.py
   ```

3. Kiểm tra lỗi dependencies:
   ```bash
   cd /opt/ethusdt-dashboard
   source venv/bin/activate
   pip install -r requirements_server.txt
   ```

## Cập Nhật Phiên Bản

### Cập nhật bằng deploy_from_github.sh:
```bash
./deploy_from_github.sh [GitHub_URL] [user@server_ip]
```

### Cập nhật bằng script đồng bộ:
```bash
./sync_to_server.sh
```

### Cập nhật bằng package mới:
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