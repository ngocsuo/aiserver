#!/usr/bin/env python3
"""
Script chuẩn bị triển khai ETHUSDT Dashboard lên server mới.
Script này sẽ tạo một file zip có chứa tất cả các file cần thiết để chạy trên server mới hoàn toàn.
"""

import os
import shutil
import zipfile
import subprocess
import json
import datetime
import sys

def create_install_script():
    """Tạo script cài đặt cho server"""
    with open("server_install.sh", "w") as f:
        f.write("""#!/bin/bash
# Script cài đặt ETHUSDT Dashboard trên server mới

# Thông tin cài đặt
INSTALL_DIR="/root/ethusdt_dashboard"
BINANCE_API_KEY="${BINANCE_API_KEY:-""}"
BINANCE_API_SECRET="${BINANCE_API_SECRET:-""}"

echo "=== CÀI ĐẶT ETHUSDT DASHBOARD ==="
echo "Thời gian: $(date)"

# Kiểm tra API keys
if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
    echo "CẢNH BÁO: API keys chưa được cung cấp."
    echo "Bạn cần cập nhật .env sau khi cài đặt."
fi

# 1. Cập nhật hệ thống
echo "1. Cập nhật hệ thống..."
apt update && apt upgrade -y

# 2. Cài đặt các gói cần thiết
echo "2. Cài đặt các gói cần thiết..."
apt install -y python3 python3-pip python3-venv git rsync curl wget htop

# 3. Tạo và giải nén vào thư mục cài đặt
echo "3. Giải nén vào thư mục cài đặt..."
mkdir -p $INSTALL_DIR
unzip -o ethusdt_dashboard.zip -d $INSTALL_DIR

# 4. Thiết lập Python venv
echo "4. Thiết lập môi trường Python ảo..."
cd $INSTALL_DIR
python3 -m venv venv
source venv/bin/activate

# 5. Cài đặt các gói Python
echo "5. Cài đặt các gói Python..."
pip install --upgrade pip
pip install -r requirements.txt

# 6. Tạo file .env
echo "6. Thiết lập file .env..."
cat > $INSTALL_DIR/.env << EOF
# Binance API Keys
BINANCE_API_KEY=$BINANCE_API_KEY
BINANCE_API_SECRET=$BINANCE_API_SECRET
EOF

echo "File .env đã được tạo tại $INSTALL_DIR/.env"
if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
    echo "QUAN TRỌNG: Hãy cập nhật API key và secret trong file này!"
fi

# 7. Tạo service systemd
echo "7. Thiết lập systemd service..."
cat > /etc/systemd/system/ethusdt-dashboard.service << EOF
[Unit]
Description=ETHUSDT Dashboard Service
After=network.target

[Service]
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=$INSTALL_DIR"

[Install]
WantedBy=multi-user.target
EOF

# 8. Cấu hình firewall
echo "8. Cấu hình firewall..."
if command -v ufw &> /dev/null; then
    ufw allow 5000/tcp
    ufw status
else
    echo "UFW không được cài đặt. Bỏ qua cấu hình firewall."
fi

# 9. Reload systemd và enable service
echo "9. Cấu hình systemd..."
systemctl daemon-reload
systemctl enable ethusdt-dashboard
systemctl start ethusdt-dashboard

echo "=== CÀI ĐẶT HOÀN TẤT ==="
echo "Truy cập dashboard tại: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Các lệnh hữu ích:"
echo "- Xem logs: journalctl -fu ethusdt-dashboard"
echo "- Khởi động lại: systemctl restart ethusdt-dashboard"
echo "- Trạng thái: systemctl status ethusdt-dashboard"
""")
    
    # Chỉnh quyền
    os.chmod("server_install.sh", 0o755)

def create_requirements():
    """Tạo file requirements.txt từ dependencies hiện tại"""
    required_packages = [
        "streamlit>=1.22.0",
        "pandas>=1.5.3",
        "numpy>=1.24.3",
        "plotly>=5.14.1",
        "python-binance>=1.0.17",
        "scikit-learn>=1.2.2",
        "tensorflow>=2.12.0",
        "requests>=2.29.0",
        "psutil>=5.9.5",
        "pytz>=2023.3"
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(required_packages))

def create_readme():
    """Tạo README.md cho package"""
    with open("PACKAGE_README.md", "w") as f:
        f.write("""# ETHUSDT Dashboard - Package triển khai

## Hướng dẫn cài đặt

1. Upload file `ethusdt_dashboard.zip` lên server
2. Giải nén: `unzip ethusdt_dashboard.zip`
3. Chạy script cài đặt: `bash server_install.sh`

## Cấu hình API Keys

Sau khi cài đặt, cập nhật API keys trong file `.env`:
```
nano /root/ethusdt_dashboard/.env
```

Hoặc bạn có thể cung cấp API keys khi chạy script cài đặt:
```
BINANCE_API_KEY="your_api_key" BINANCE_API_SECRET="your_api_secret" bash server_install.sh
```

## Thông tin hệ thống
- Dashboard port: 5000
- Thư mục cài đặt: /root/ethusdt_dashboard
- Service: ethusdt-dashboard

## Các lệnh hữu ích
- Xem logs: `journalctl -fu ethusdt-dashboard`
- Khởi động lại: `systemctl restart ethusdt-dashboard`
- Trạng thái: `systemctl status ethusdt-dashboard`

## Nâng cấp phiên bản
Để nâng cấp lên phiên bản mới:
1. Upload package mới
2. Giải nén và chạy lại script cài đặt
""")

def generate_package(output_path=None):
    """
    Tạo package triển khai
    
    Args:
        output_path (str, optional): Đường dẫn đến file zip đầu ra
    """
    # Tạo timestamp để đặt tên file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = f"ethusdt_dashboard_{timestamp}.zip"
    
    # Tạo các file cần thiết
    create_install_script()
    create_requirements()
    create_readme()
    
    # Danh sách thư mục và file cần đóng gói
    include_dirs = [
        "utils",
        "models",
        "data",
        "dashboard",
        "prediction"
    ]
    
    include_files = [
        "app.py",
        "config.py",
        "server_install.sh",
        "requirements.txt",
        "PACKAGE_README.md"
    ]
    
    # Các file bổ sung nếu có
    extra_files = [
        "thread_safe_logging.py",
        "enhanced_data_collector.py",
        "fixed_train_models.py",
    ]
    
    for f in extra_files:
        if os.path.exists(f):
            include_files.append(f)
    
    # Tạo zip file
    print(f"Tạo package tại: {output_path}")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Thêm các file
        for file in include_files:
            if os.path.exists(file):
                print(f"  + Thêm file: {file}")
                zipf.write(file)
            else:
                print(f"  - Bỏ qua file không tồn tại: {file}")
        
        # Thêm các thư mục
        for folder in include_dirs:
            if os.path.exists(folder):
                print(f"  + Thêm thư mục: {folder}")
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        # Bỏ qua các file cache và temp
                        if file.endswith(('.pyc', '.pyo')) or file.startswith(('__pycache__', '.', '#')):
                            continue
                        file_path = os.path.join(root, file)
                        print(f"    + {file_path}")
                        zipf.write(file_path)
            else:
                print(f"  - Bỏ qua thư mục không tồn tại: {folder}")
                # Tạo thư mục trống để đảm bảo cấu trúc
                zipf.writestr(f"{folder}/.keep", "")
    
    # Xóa các file tạm thời
    temp_files = ["server_install.sh", "PACKAGE_README.md"]
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    
    # Hiển thị thông tin
    zip_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
    print(f"\nPackage đã được tạo: {output_path}")
    print(f"Kích thước: {zip_size:.2f} MB")
    print("\nHướng dẫn triển khai:")
    print("1. Upload file zip lên server")
    print("2. Giải nén: unzip", output_path)
    print("3. Chạy script cài đặt: bash server_install.sh")
    
    return output_path

def main():
    """Hàm chính"""
    # Parse arguments
    output_path = "ethusdt_dashboard.zip"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    generate_package(output_path)

if __name__ == "__main__":
    main()