#!/usr/bin/env python3
"""
Script chuẩn bị triển khai ETHUSDT Dashboard lên server.
Script này sẽ tạo một file zip có chứa tất cả các file cần thiết để chạy trên server.
"""

import os
import sys
import shutil
import zipfile
import tempfile
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Danh sách các thư mục và file cần thiết
ESSENTIAL_DIRECTORIES = [
    "deployment",
    "utils",
    "models",
    "prediction",
    "dashboard"
]

ESSENTIAL_FILES = [
    "app.py",
    "config.py",
    "run_clean.py",
    "run_with_monitoring.py",
    "feature_engineering_fix.py",
    "README.md",
    "deployment/README.md",
    "deployment/startup.sh",
    "deployment/deploy_service.py",
    "utils/log_filter.py",
    "utils/data_fix.py",
    "utils/thread_safe_logging.py",
]

# Danh sách các file và thư mục không nên đưa vào package
EXCLUDED_ITEMS = [
    "__pycache__",
    ".git",
    ".streamlit/cache",
    "logs",
    "venv",
    "*.pyc",
    "*.log",
    "data/*.csv",
    "*.pkl"
]

def create_install_script():
    """Tạo script cài đặt cho server"""
    install_script = """#!/bin/bash
# Script cài đặt ETHUSDT Dashboard trên server

# Cài đặt các gói phụ thuộc
echo "Đang cài đặt các gói phụ thuộc..."
pip install -r requirements.txt

# Tạo các thư mục cần thiết
echo "Tạo các thư mục cần thiết..."
mkdir -p logs
mkdir -p data
mkdir -p saved_models
mkdir -p deployment/logs

# Đặt quyền thực thi cho các script
echo "Đặt quyền thực thi cho các script..."
chmod +x run_clean.py
chmod +x deployment/startup.sh

# Cấu hình systemd service (nếu người dùng có quyền root)
if [ "$(id -u)" = "0" ]; then
    echo "Phát hiện quyền root, đang cấu hình systemd service..."
    cat > /etc/systemd/system/ethusdt-dashboard.service << EOL
[Unit]
Description=ETHUSDT Dashboard Service
After=network.target

[Service]
ExecStart=/bin/bash -c "cd $(pwd) && python run_clean.py --mode service"
WorkingDirectory=$(pwd)
Restart=always
RestartSec=10
User=$(whoami)
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOL

    # Khởi động service
    systemctl daemon-reload
    systemctl enable ethusdt-dashboard.service
    systemctl start ethusdt-dashboard.service
    echo "Đã cấu hình và khởi động systemd service."
else
    echo "Không có quyền root, bỏ qua cấu hình systemd service."
    echo "Để chạy ứng dụng, sử dụng lệnh: python run_clean.py --mode service"
fi

echo "Cài đặt hoàn tất. Ứng dụng sẽ chạy tại http://localhost:5000"
"""
    return install_script

def create_requirements():
    """Tạo file requirements.txt từ dependencies hiện tại"""
    requirements = """streamlit==1.31.1
pandas==2.1.4
numpy==1.26.3
plotly==5.18.0
python-binance==1.0.19
requests==2.31.0
psutil==5.9.8
scikit-learn==1.3.2
tensorflow==2.15.0
"""
    return requirements

def generate_package(output_path=None, include_data=False):
    """
    Tạo package triển khai
    
    Args:
        output_path (str, optional): Đường dẫn đến file zip đầu ra
        include_data (bool): Có bao gồm dữ liệu hiện có không
    """
    # Xác định tên file đầu ra
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"ethusdt_dashboard_deploy_{timestamp}.zip"
    
    # Tạo thư mục tạm thời
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        deploy_dir = temp_path / "ethusdt_dashboard"
        deploy_dir.mkdir()
        
        # Sao chép các thư mục cần thiết
        for directory in ESSENTIAL_DIRECTORIES:
            dir_path = Path(directory)
            if dir_path.exists():
                dest_dir = deploy_dir / directory
                
                # Tạo thư mục đích
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Sao chép tất cả file trong thư mục
                for item in dir_path.glob("**/*"):
                    # Kiểm tra xem có nên bỏ qua không
                    skip = False
                    for exclude in EXCLUDED_ITEMS:
                        if "*" in exclude:
                            # So khớp wildcard
                            pattern = exclude.replace("*", "")
                            if pattern in str(item):
                                skip = True
                                break
                        elif exclude in str(item):
                            skip = True
                            break
                    
                    if skip:
                        continue
                    
                    # Nếu là thư mục, tạo thư mục tương ứng
                    if item.is_dir():
                        (dest_dir / item.relative_to(dir_path)).mkdir(parents=True, exist_ok=True)
                    # Nếu là file, sao chép
                    elif item.is_file():
                        dest_file = dest_dir / item.relative_to(dir_path)
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest_file)
        
        # Sao chép các file cần thiết
        for file_path in ESSENTIAL_FILES:
            source = Path(file_path)
            if source.exists():
                dest = deploy_dir / file_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
        
        # Sao chép dữ liệu nếu cần
        if include_data:
            data_dir = Path("data")
            if data_dir.exists():
                dest_data_dir = deploy_dir / "data"
                dest_data_dir.mkdir(parents=True, exist_ok=True)
                
                # Sao chép tất cả file trong thư mục data (trừ các file bị loại trừ)
                for item in data_dir.glob("*"):
                    # Kiểm tra xem có nên bỏ qua không
                    skip = False
                    for exclude in EXCLUDED_ITEMS:
                        if "*" in exclude:
                            # So khớp wildcard
                            pattern = exclude.replace("*", "")
                            if pattern in str(item):
                                skip = True
                                break
                        elif exclude in str(item):
                            skip = True
                            break
                    
                    if skip:
                        continue
                    
                    # Nếu là thư mục, tạo thư mục tương ứng
                    if item.is_dir():
                        (dest_data_dir / item.name).mkdir(parents=True, exist_ok=True)
                    # Nếu là file, sao chép
                    elif item.is_file():
                        shutil.copy2(item, dest_data_dir / item.name)
        
        # Tạo script cài đặt
        with open(deploy_dir / "install.sh", "w") as f:
            f.write(create_install_script())
        
        # Đặt quyền thực thi cho script cài đặt
        os.chmod(deploy_dir / "install.sh", 0o755)
        
        # Tạo file requirements.txt
        with open(deploy_dir / "requirements.txt", "w") as f:
            f.write(create_requirements())
        
        # Tạo file README.md với hướng dẫn cài đặt
        readme_content = """# ETHUSDT Dashboard - Hướng dẫn triển khai

## Cài đặt

1. Giải nén file zip
2. Di chuyển vào thư mục giải nén:
   ```
   cd ethusdt_dashboard
   ```
3. Chạy script cài đặt:
   ```
   ./install.sh
   ```

## Chạy thủ công

Nếu bạn không muốn cài đặt systemd service, bạn có thể chạy ứng dụng thủ công:

```
python run_clean.py --mode service
```

## Cấu hình

Bạn có thể chỉnh sửa file `config.py` để thay đổi cấu hình ứng dụng.

## Kiểm tra trạng thái

Kiểm tra trạng thái của service:

```
systemctl status ethusdt-dashboard.service
```

## Xem log

Xem log của ứng dụng:

```
tail -f logs/app.log
```

## Truy cập ứng dụng

Ứng dụng sẽ chạy tại http://localhost:5000
"""
        with open(deploy_dir / "INSTALL.md", "w") as f:
            f.write(readme_content)
        
        # Tạo file zip
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(deploy_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_path)
                    zipf.write(file_path, arcname)
    
    print(f"Đã tạo package triển khai tại: {output_path}")
    return output_path

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description="Tạo package triển khai ETHUSDT Dashboard")
    parser.add_argument("--output", "-o", help="Đường dẫn đến file zip đầu ra")
    parser.add_argument("--include-data", "-d", action="store_true", help="Bao gồm dữ liệu hiện có")
    
    args = parser.parse_args()
    
    print("Đang tạo package triển khai ETHUSDT Dashboard...")
    output_path = generate_package(args.output, args.include_data)
    
    # In thông tin bổ sung
    print("\nThông tin triển khai:")
    print("---------------------")
    print("1. Sao chép file zip lên server của bạn")
    print("2. Giải nén file zip")
    print("3. Di chuyển vào thư mục giải nén")
    print("4. Chạy script cài đặt: ./install.sh")
    print("\nỨng dụng sẽ chạy tại http://server-ip:5000")

if __name__ == "__main__":
    main()