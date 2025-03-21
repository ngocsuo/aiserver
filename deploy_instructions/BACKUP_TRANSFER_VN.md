# Hướng Dẫn Sao Lưu và Chuyển Mã Nguồn

Tài liệu này hướng dẫn cách sao lưu mã nguồn hệ thống dự đoán ETHUSDT từ Replit và chuyển nó sang máy chủ của bạn.

## Sao Lưu Từ Replit

### 1. Sao Lưu Bằng Tệp ZIP

**Phương pháp 1: Sử dụng giao diện Replit**
1. Trong giao diện Replit, click vào biểu tượng menu (3 dấu gạch ngang) ở góc trái trên cùng
2. Chọn "Export Repl"
3. Chọn "Download as ZIP"
4. Lưu tệp ZIP vào máy tính của bạn

**Phương pháp 2: Sử dụng lệnh trong Replit Shell**
1. Mở Shell trong Replit
2. Chạy lệnh sau để tạo tệp ZIP của toàn bộ dự án:
   ```bash
   zip -r ethusdt_predictor.zip . -x "*.git*" -x "saved_models/data_cache/*" -x "__pycache__/*" -x "venv/*" -x "node_modules/*"
   ```
3. Tải xuống tệp ZIP bằng cách nhấp chuột phải vào nó trong Files panel và chọn "Download"

### 2. Sao Lưu Bằng Git (Nếu Bạn Muốn Quản Lý Phiên Bản)

1. Tạo một repository mới trên GitHub, GitLab hoặc dịch vụ Git khác
2. Trong Replit Shell, chạy các lệnh sau:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://your-repository-url.git
   git push -u origin main
   ```

## Chuyển Mã Nguồn Sang Máy Chủ

### 1. Sử Dụng Tệp ZIP

1. Kết nối đến máy chủ của bạn qua SSH:
   ```bash
   ssh username@your-server-ip
   ```

2. Tạo thư mục cho ứng dụng:
   ```bash
   mkdir -p /opt/ethusdt_predictor
   ```

3. Tải tệp ZIP lên máy chủ (từ máy tính cục bộ của bạn):
   ```bash
   scp ethusdt_predictor.zip username@your-server-ip:/opt/ethusdt_predictor/
   ```

4. Giải nén tệp ZIP trên máy chủ:
   ```bash
   cd /opt/ethusdt_predictor
   unzip ethusdt_predictor.zip
   ```

### 2. Sử Dụng Git

1. Kết nối đến máy chủ của bạn qua SSH:
   ```bash
   ssh username@your-server-ip
   ```

2. Cài đặt Git nếu chưa có:
   ```bash
   sudo apt update
   sudo apt install -y git
   ```

3. Clone repository:
   ```bash
   mkdir -p /opt/ethusdt_predictor
   cd /opt/ethusdt_predictor
   git clone https://your-repository-url.git .
   ```

## Thiết Lập Môi Trường Sau Khi Chuyển

1. Tạo môi trường ảo Python:
   ```bash
   cd /opt/ethusdt_predictor
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r deploy_instructions/requirements.txt
   ```

3. Thiết lập cấu hình Binance API:
   ```bash
   # Tạo file .env
   cat > /opt/ethusdt_predictor/.env << EOF
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_api_secret_here
   EOF
   
   # Đảm bảo quyền truy cập phù hợp
   chmod 600 /opt/ethusdt_predictor/.env
   ```

4. Tạo thư mục cho dữ liệu:
   ```bash
   mkdir -p /opt/ethusdt_predictor/saved_models/data_cache
   ```

## Xác Minh Cài Đặt

1. Chạy ứng dụng thử nghiệm:
   ```bash
   cd /opt/ethusdt_predictor
   source venv/bin/activate
   streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
   ```

2. Truy cập ứng dụng thông qua trình duyệt web:
   ```
   http://your-server-ip:5000
   ```

3. Kiểm tra log để đảm bảo kết nối đến Binance API thành công:
   ```bash
   # Nhấn Ctrl+C để dừng ứng dụng, sau đó
   cat nohup.out  # Nếu bạn chạy với nohup
   ```

## Thiết Lập Cấu Hình Tự Động Khởi Động

Sau khi xác minh rằng ứng dụng hoạt động, thiết lập cấu hình để tự động khởi động như đã mô tả trong file README_VN.md:

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

## Khắc Phục Sự Cố Chuyển Giao

### Thay Đổi Quyền Sở Hữu Tệp

Nếu bạn gặp vấn đề về quyền, hãy chạy:
```bash
sudo chown -R your-username:your-username /opt/ethusdt_predictor
```

### Kiểm Tra Tính Khả Dụng của Cổng

Nếu ứng dụng không thể truy cập được thông qua trình duyệt, kiểm tra tường lửa:
```bash
sudo ufw status
# Nếu UFW đang hoạt động, hãy mở cổng 5000
sudo ufw allow 5000/tcp
```

### Kiểm Tra Các Phụ Thuộc Bị Thiếu

Nếu ứng dụng không khởi động được do thiếu các phụ thuộc:
```bash
# Kiểm tra log
sudo tail -f /var/log/ethusdt_predictor.err.log

# Cài đặt thêm các phụ thuộc nếu cần
source venv/bin/activate
pip install tên-thư-viện-bị-thiếu
```

## Lưu Ý Về Bảo Mật

1. Đặt mọi khóa API và bí mật trong tệp .env và đảm bảo chúng không được thêm vào Git
2. Thay đổi cấu hình supervisor để chạy dưới một người dùng không phải root trong môi trường sản xuất
3. Cấu hình HTTPS thông qua Nginx và Let's Encrypt nếu ứng dụng có thể truy cập từ Internet

## Giữ Mã Nguồn Luôn Cập Nhật

Nếu bạn sử dụng Git để quản lý mã nguồn, bạn có thể dễ dàng cập nhật khi có các tính năng mới:
```bash
cd /opt/ethusdt_predictor
git pull
sudo supervisorctl restart ethusdt_predictor
```