# Hướng Dẫn Bảo Mật VPS Cho Hệ Thống Dự Đoán ETHUSDT

Tài liệu này cung cấp các biện pháp bảo mật cơ bản và nâng cao để bảo vệ VPS/máy chủ của bạn khi triển khai hệ thống dự đoán ETHUSDT.

## Bảo Mật Cơ Bản

### 1. Cập Nhật Hệ Thống Thường Xuyên

```bash
# Cập nhật danh sách gói
sudo apt update

# Cập nhật tất cả các gói
sudo apt upgrade -y

# Cập nhật hệ thống
sudo apt dist-upgrade -y

# Xóa các gói không cần thiết
sudo apt autoremove -y
```

Thiết lập cập nhật tự động:
```bash
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

### 2. Bảo Mật SSH

#### Thay Đổi Cổng SSH Mặc Định (22)

```bash
sudo nano /etc/ssh/sshd_config
```

Tìm dòng `#Port 22` và thay đổi thành:
```
Port 2222  # Chọn một cổng khác >1024
```

#### Tắt Đăng Nhập Root Qua SSH

Trong cùng file `/etc/ssh/sshd_config`, tìm và thay đổi:
```
PermitRootLogin no
```

#### Chỉ Cho Phép Đăng Nhập Bằng Khóa SSH

```bash
# Tạo cặp khóa SSH trên máy tính cá nhân của bạn (không phải VPS)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy khóa công khai sang VPS
ssh-copy-id -i ~/.ssh/id_ed25519.pub username@your-server-ip -p 22  # Hoặc port khác nếu đã thay đổi
```

Trong file `/etc/ssh/sshd_config`, thay đổi:
```
PasswordAuthentication no
ChallengeResponseAuthentication no
```

Khởi động lại dịch vụ SSH:
```bash
sudo systemctl restart sshd
```

### 3. Cấu Hình Tường Lửa (UFW)

```bash
# Cài đặt UFW nếu chưa có
sudo apt install -y ufw

# Thiết lập quy tắc mặc định
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Cho phép cổng SSH (với cổng đã thay đổi)
sudo ufw allow 2222/tcp  # Thay bằng cổng SSH bạn đã chọn

# Cho phép cổng web cho ứng dụng
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 5000/tcp  # Cổng Streamlit (nếu không sử dụng Nginx)

# Kích hoạt tường lửa
sudo ufw enable

# Kiểm tra trạng thái
sudo ufw status
```

### 4. Cấu Hình Fail2Ban Để Bảo Vệ Khỏi Tấn Công Brute Force

```bash
# Cài đặt Fail2Ban
sudo apt install -y fail2ban

# Tạo file cấu hình cục bộ
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local
```

Thêm cấu hình sau vào file:
```
[sshd]
enabled = true
port = 2222  # Thay bằng cổng SSH bạn đã chọn
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600  # Ban trong 1 giờ
```

Khởi động lại Fail2Ban:
```bash
sudo systemctl restart fail2ban
```

## Bảo Mật Nâng Cao

### 1. Bảo Vệ API Keys và Dữ Liệu Nhạy Cảm

Thay vì lưu API key trực tiếp trong môi trường, hãy sử dụng vault như HashiCorp Vault hoặc Google Cloud Secret Manager. Dưới đây là cách sử dụng cơ bản với tệp .env:

```bash
# Đảm bảo quyền truy cập hạn chế cho tệp .env
sudo chmod 600 /opt/ethusdt_predictor/.env
sudo chown root:root /opt/ethusdt_predictor/.env
```

### 2. Cài Đặt Và Cấu Hình ModSecurity (Web Application Firewall)

```bash
# Cài đặt ModSecurity với Nginx
sudo apt install -y nginx-plus-module-modsecurity

# Cấu hình ModSecurity
sudo nano /etc/nginx/modsecurity/modsecurity.conf
```

Thay đổi từ `DetectionOnly` sang `On`:
```
SecRuleEngine On
```

### 3. Giám Sát Hệ Thống Với Auditd

```bash
# Cài đặt auditd
sudo apt install -y auditd

# Cấu hình auditd để theo dõi các thay đổi hệ thống
sudo nano /etc/audit/rules.d/audit.rules
```

Thêm các quy tắc sau:
```
# Theo dõi thay đổi tệp cấu hình
-w /etc/ssh/sshd_config -p wa -k sshd_config
-w /etc/passwd -p wa -k passwd_changes
-w /etc/shadow -p wa -k shadow_changes

# Theo dõi thư mục ứng dụng
-w /opt/ethusdt_predictor -p wa -k app_changes
```

Khởi động lại auditd:
```bash
sudo systemctl restart auditd
```

### 4. Thiết Lập Logrotate Cho Tệp Nhật Ký Ứng Dụng

```bash
sudo nano /etc/logrotate.d/ethusdt_predictor
```

Thêm vào:
```
/var/log/ethusdt_predictor*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 root root
}
```

### 5. Cấu Hình Nginx Với Giới Hạn Tốc Độ Và Các Header Bảo Mật

```bash
sudo nano /etc/nginx/sites-available/ethusdt_predictor
```

Thêm các cấu hình sau:
```
# Giới hạn tốc độ
limit_req_zone $binary_remote_addr zone=app_limit:10m rate=10r/s;

server {
    # Các cấu hình khác...
    
    # Áp dụng giới hạn tốc độ
    limit_req zone=app_limit burst=20 nodelay;
    
    # Thêm các header bảo mật
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options SAMEORIGIN;
    add_header X-XSS-Protection "1; mode=block";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';";
    add_header Referrer-Policy no-referrer-when-downgrade;
    
    # Các cấu hình location...
}
```

### 6. Thường Xuyên Sao Lưu Cấu Hình Và Dữ Liệu

Tạo script sao lưu tự động:
```bash
nano /opt/backup_script.sh
```

Thêm vào:
```bash
#!/bin/bash
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
BACKUP_DIR="/opt/backups"
mkdir -p $BACKUP_DIR

# Sao lưu mã nguồn và cấu hình
tar -czf $BACKUP_DIR/ethusdt_app_$TIMESTAMP.tar.gz /opt/ethusdt_predictor --exclude=/opt/ethusdt_predictor/saved_models/data_cache

# Sao lưu dữ liệu mô hình
tar -czf $BACKUP_DIR/ethusdt_models_$TIMESTAMP.tar.gz /opt/ethusdt_predictor/saved_models --exclude=/opt/ethusdt_predictor/saved_models/data_cache

# Xóa các bản sao lưu cũ hơn 30 ngày
find $BACKUP_DIR -name "ethusdt_*" -type f -mtime +30 -delete
```

Đặt quyền thực thi và thiết lập cron job:
```bash
chmod +x /opt/backup_script.sh
sudo crontab -e
```

Thêm vào:
```
0 2 * * * /opt/backup_script.sh
```

## Hướng Dẫn Phục Hồi Sau Sự Cố

### 1. Phục Hồi Từ Bản Sao Lưu

```bash
# Dừng dịch vụ
sudo supervisorctl stop ethusdt_predictor

# Phục hồi từ bản sao lưu
tar -xzf /opt/backups/ethusdt_app_TIMESTAMP.tar.gz -C /
tar -xzf /opt/backups/ethusdt_models_TIMESTAMP.tar.gz -C /

# Khởi động lại dịch vụ
sudo supervisorctl start ethusdt_predictor
```

### 2. Khôi Phục Sau Tấn Công

```bash
# Kiểm tra nhật ký hệ thống để xác định phạm vi tấn công
sudo grep "Failed password" /var/log/auth.log

# Kiểm tra các quá trình đáng ngờ
ps auxf | grep -v grep | grep -i "ssh\|nc\|ncat\|netcat\|cryptominer\|xmrig"

# Kiểm tra các công việc cron đáng ngờ
sudo crontab -l
sudo cat /etc/crontab
sudo ls -la /etc/cron.*

# Khôi phục từ bản sao lưu sạch nếu cần thiết
```

## Kiểm Tra Bảo Mật Định Kỳ

### 1. Quét Lỗ Hổng Hệ Thống

```bash
# Cài đặt Lynis
sudo apt install lynis -y

# Chạy quét bảo mật
sudo lynis audit system
```

### 2. Kiểm Tra Cổng Mở

```bash
sudo apt install nmap -y
sudo nmap -sS -p- localhost
```

### 3. Kiểm Tra Tài Khoản và Quyền

```bash
# Kiểm tra tài khoản có thể đăng nhập
grep "sh$" /etc/passwd

# Kiểm tra quyền sudo
grep -Po '^sudo.+:\K.*$' /etc/group

# Kiểm tra các tệp setuid và setgid
sudo find / -type f \( -perm -4000 -o -perm -2000 \) -exec ls -l {} \;
```

## Tài Liệu Tham Khảo

1. [Bảo mật Ubuntu Server](https://ubuntu.com/server/docs/security-introduction)
2. [Hướng dẫn bảo mật Nginx](https://www.nginx.com/resources/wiki/start/topics/tutorials/security_headers/)
3. [Cấu hình ModSecurity với Nginx](https://www.nginx.com/blog/compiling-and-installing-modsecurity-for-open-source-nginx/)
4. [Triển khai Fail2Ban](https://www.digitalocean.com/community/tutorials/how-to-protect-ssh-with-fail2ban-on-ubuntu-20-04)