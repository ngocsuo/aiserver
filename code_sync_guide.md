# Hướng dẫn đồng bộ code ETHUSDT Dashboard với server

## 1. Sử dụng Git (Khuyến nghị)

Nếu server của bạn có cài đặt Git, đây là cách tốt nhất để đồng bộ code:

```bash
# Trên máy local, đẩy code lên repository
git add .
git commit -m "Cập nhật phiên bản mới nhất"
git push origin main

# Trên server, kéo code mới nhất về
git pull origin main
```

## 2. Sử dụng rsync

Nếu không sử dụng Git, bạn có thể dùng rsync để đồng bộ code:

```bash
# Trên máy local, đồng bộ code lên server
rsync -avz --exclude 'logs' --exclude 'data' --exclude '.git' --exclude 'venv' --exclude '__pycache__' /path/to/local/ethusdt_dashboard/ user@server-ip:/path/to/server/ethusdt_dashboard/
```

## 3. Sử dụng SCP

```bash
# Tạo một file zip (không bao gồm thư mục logs, data và các file tạm)
zip -r ethusdt_dashboard.zip . -x "*.git*" -x "logs/*" -x "data/*.csv" -x "*.pyc" -x "__pycache__/*" -x "venv/*"

# Chuyển file zip lên server
scp ethusdt_dashboard.zip user@server-ip:/path/to/server/

# Trên server, giải nén file
unzip -o ethusdt_dashboard.zip -d /path/to/server/ethusdt_dashboard/
```

## 4. Sử dụng gói triển khai đã tạo sẵn

Bạn có thể sử dụng gói triển khai đã tạo sẵn bằng script `prepare_for_server.py`:

```bash
# Tạo package triển khai
python prepare_for_server.py --output ethusdt_dashboard.zip

# Chuyển file zip lên server
scp ethusdt_dashboard.zip user@server-ip:/path/to/server/

# Trên server, giải nén và cài đặt
unzip ethusdt_dashboard.zip
cd ethusdt_dashboard
chmod +x install.sh
./install.sh
```

## 5. Đảm bảo các secrets được cấu hình đúng

Trên server, đảm bảo các biến môi trường được thiết lập:

```bash
# Thiết lập Binance API keys
export BINANCE_API_KEY=your_api_key
export BINANCE_API_SECRET=your_api_secret

# Hoặc thêm vào .bashrc hoặc .profile để tự động load khi đăng nhập
echo 'export BINANCE_API_KEY=your_api_key' >> ~/.bashrc
echo 'export BINANCE_API_SECRET=your_api_secret' >> ~/.bashrc
source ~/.bashrc
```

## 6. Script kiểm tra đồng bộ

Sử dụng script sau để kiểm tra xem code đã được đồng bộ thành công chưa:

```bash
#!/bin/bash
# Tạo file trên server: check_sync.sh

echo "Kiểm tra phiên bản code..."
echo "Thời gian cập nhật gần nhất: $(stat -c %y app.py)"
echo "MD5 của file chính: $(md5sum app.py)"
echo "MD5 của file cấu hình: $(md5sum config.py)"
echo "Số lượng file Python: $(find . -name "*.py" | wc -l)"
echo "Kiểm tra các thư mục quan trọng:"
echo "- models: $(ls -la models | wc -l) files"
echo "- utils: $(ls -la utils | wc -l) files"
echo "- prediction: $(ls -la prediction | wc -l) files"
echo "- dashboard: $(ls -la dashboard | wc -l) files"
```

## 7. Tự động đồng bộ code sử dụng systemd

Trên server, bạn có thể cấu hình một service systemd để tự động cập nhật code:

```bash
# Tạo file: /etc/systemd/system/ethusdt-sync.service
[Unit]
Description=ETHUSDT Dashboard Code Sync Service
After=network.target

[Service]
User=your_username
WorkingDirectory=/path/to/server/ethusdt_dashboard
ExecStart=/bin/bash -c 'git pull origin main && systemctl restart ethusdt-dashboard.service'
Type=oneshot

[Install]
WantedBy=multi-user.target
```

```bash
# Tạo file: /etc/systemd/system/ethusdt-sync.timer
[Unit]
Description=Run ETHUSDT Dashboard Code Sync Service daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

Sau đó kích hoạt timer:

```bash
sudo systemctl enable ethusdt-sync.timer
sudo systemctl start ethusdt-sync.timer
```

## 8. Khởi động lại ứng dụng sau khi đồng bộ

Sau khi đồng bộ code, nhớ khởi động lại ứng dụng:

```bash
# Sử dụng script stop.sh đã tạo trước đó để dừng ứng dụng
./stop.sh

# Sử dụng script run.sh để khởi động lại ứng dụng
./run.sh
```