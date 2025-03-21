# Hướng Dẫn Triển Khai Hệ Thống Dự Đoán ETHUSDT

Tài liệu này cung cấp hướng dẫn chi tiết để triển khai hệ thống dự đoán ETHUSDT trên máy chủ riêng hoặc VPS của bạn.

## Yêu Cầu Hệ Thống

- **VPS/Máy Chủ**: Ubuntu 20.04 LTS hoặc mới hơn
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB để tối ưu hiệu suất)
- **CPU**: 2 core hoặc cao hơn
- **Lưu Trữ**: Tối thiểu 20GB SSD
- **Kết Nối Internet**: Ổn định và không bị chặn kết nối đến Binance
- **Python**: Phiên bản 3.9 hoặc cao hơn

## Thiết Lập Môi Trường

### 1. Cài Đặt Các Phần Mềm Cần Thiết

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git supervisor nginx
```

### 2. Tải Mã Nguồn

```bash
# Tạo thư mục cho ứng dụng
mkdir -p /opt/ethusdt_predictor
cd /opt/ethusdt_predictor

# Sao chép mã nguồn từ bản sao lưu hoặc từ Git (nếu bạn lưu trữ trên Git)
# Ví dụ sử dụng Git:
# git clone https://your-repository-url.git .

# Hoặc sao chép thủ công bằng các công cụ như SCP, SFTP, v.v.
```

### 3. Tạo Môi Trường Ảo Python

```bash
cd /opt/ethusdt_predictor
python3 -m venv venv
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

Nếu không có file requirements.txt, bạn có thể cài đặt các thư viện sau:

```bash
pip install streamlit pandas numpy plotly python-binance scikit-learn tensorflow
```

## Cấu Hình Hệ Thống

### 1. Thiết Lập API Binance

Để hệ thống có thể kết nối đến API Binance, bạn cần cung cấp API key và secret của Binance. Bạn có thể thiết lập chúng bằng biến môi trường:

```bash
# Tạo file .env trong thư mục /opt/ethusdt_predictor
cat > /opt/ethusdt_predictor/.env << EOF
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
EOF

# Đảm bảo quyền truy cập phù hợp
chmod 600 /opt/ethusdt_predictor/.env
```

### 2. Cấu Hình Hệ Thống Chạy Tự Động Với Supervisor

```bash
# Tạo file cấu hình supervisor
sudo tee /etc/supervisor/conf.d/ethusdt_predictor.conf > /dev/null << EOF
[program:ethusdt_predictor]
command=/opt/ethusdt_predictor/venv/bin/streamlit run /opt/ethusdt_predictor/app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
directory=/opt/ethusdt_predictor
user=root
autostart=true
autorestart=true
startretries=10
stderr_logfile=/var/log/ethusdt_predictor.err.log
stdout_logfile=/var/log/ethusdt_predictor.out.log
environment=BINANCE_API_KEY="%(ENV_BINANCE_API_KEY)s",BINANCE_API_SECRET="%(ENV_BINANCE_API_SECRET)s"
EOF

# Cập nhật và khởi động dịch vụ
sudo supervisorctl reread
sudo supervisorctl update
```

### 3. Cấu Hình Nginx (Tùy Chọn)

Nếu bạn muốn truy cập ứng dụng qua web với domain và HTTPS, bạn có thể cấu hình Nginx:

```bash
# Tạo cấu hình Nginx
sudo tee /etc/nginx/sites-available/ethusdt_predictor > /dev/null << EOF
server {
    listen 80;
    server_name your-domain.com; # Thay bằng tên miền của bạn

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Kích hoạt cấu hình
sudo ln -s /etc/nginx/sites-available/ethusdt_predictor /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Cấu Hình HTTPS với Certbot (Tùy Chọn)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Khởi Động Và Kiểm Tra Hệ Thống

```bash
# Khởi động lại supervisord để đảm bảo ứng dụng đang chạy
sudo supervisorctl restart ethusdt_predictor

# Kiểm tra logs
sudo tail -f /var/log/ethusdt_predictor.out.log
```

Sau khi hoàn tất, bạn có thể truy cập hệ thống thông qua:
- http://your-server-ip:5000 (nếu không dùng Nginx)
- https://your-domain.com (nếu đã cấu hình Nginx và HTTPS)

## Khắc Phục Sự Cố

### Vấn Đề Kết Nối Binance API

Nếu gặp lỗi kết nối đến Binance API, hãy kiểm tra:
1. API key và secret đã được cấu hình chính xác
2. API key có đủ quyền truy cập cần thiết (đọc dữ liệu futures market)
3. VPS/máy chủ của bạn không bị chặn địa lý khi truy cập Binance

### Lỗi Khi Khởi Động Ứng Dụng

Nếu ứng dụng không khởi động, kiểm tra logs:

```bash
sudo tail -f /var/log/ethusdt_predictor.err.log
```

### Tối Ưu Hiệu Suất

Nếu hệ thống chạy chậm:
1. Tăng RAM và CPU cho VPS/máy chủ
2. Điều chỉnh các thông số trong file config.py (giảm LOOKBACK_PERIODS, tăng UPDATE_INTERVAL)
3. Tắt các chức năng không cần thiết (ví dụ: các indicators phức tạp)

## Sao Lưu Và Phục Hồi

### Sao Lưu Dữ Liệu

```bash
# Sao lưu thư mục dữ liệu và mô hình
cd /opt
tar -czf ethusdt_backup_$(date +%Y%m%d).tar.gz ethusdt_predictor/saved_models
```

### Khôi Phục Dữ Liệu

```bash
# Khôi phục từ bản sao lưu
cd /opt
tar -xzf ethusdt_backup_YYYYMMDD.tar.gz
```

## Bảo Mật

- **API Keys**: Luôn đảm bảo API keys chỉ có quyền đọc dữ liệu trừ khi bạn cần chức năng giao dịch tự động
- **Firewall**: Cấu hình ufw để chỉ mở port cần thiết (SSH, HTTP/HTTPS)
- **Tài Khoản**: Không chạy ứng dụng với quyền root trong môi trường production (điều chỉnh file supervisor tương ứng)

## Cập Nhật Hệ Thống

Để cập nhật mã nguồn:

```bash
cd /opt/ethusdt_predictor
# Sao lưu file cấu hình hiện tại
cp config.py config.py.backup

# Tải mã nguồn mới (từ Git hoặc copy thủ công)
# git pull

# Khôi phục cấu hình cá nhân nếu cần
# cp config.py.backup config.py

# Khởi động lại ứng dụng
sudo supervisorctl restart ethusdt_predictor
```

## Liên Hệ Hỗ Trợ

Nếu bạn gặp vấn đề trong quá trình triển khai, vui lòng liên hệ qua:
- Email: your-email@example.com
- Telegram: @your_telegram_username