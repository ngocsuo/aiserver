#!/usr/bin/env python3
"""
Script để dọn dẹp mã nguồn ETHUSDT Dashboard
- Xóa các file tạm và cache
- Sắp xếp lại cấu trúc thư mục
- Đảm bảo tính nhất quán của mã nguồn
- Kiểm tra lỗi cơ bản
"""

import os
import sys
import shutil
import glob
import re
import subprocess
from pathlib import Path
from datetime import datetime

# Danh sách các thư mục cốt lõi cần giữ lại
CORE_DIRECTORIES = [
    "dashboard",
    "models",
    "prediction",
    "utils",
    "data",
    "saved_models",
    "logs",
    ".streamlit"
]

# Danh sách các file cốt lõi cần giữ lại
CORE_FILES = [
    "app.py",
    "config.py",
    "feature_engineering_fix.py",
    "run_clean.py",
    "run_with_monitoring.py",
    "prepare_for_server.py",
    "thread_safe_logging.py",
    "single_command_run.sh",
    "README.md",
]

# Danh sách các file và thư mục tạm thời có thể xóa
TEMP_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.log",
    "*__pycache__*",
    "*.csv",
    "*.pkl",
    ".ipynb_checkpoints",
    ".DS_Store",
    "*.bak",
    "*.tmp",
    "*~",
    "*.swp",
]

def clean_temp_files():
    """Xóa các file tạm thời"""
    print("Đang xóa các file tạm thời...")
    
    for pattern in TEMP_PATTERNS:
        for file_path in glob.glob(f"**/{pattern}", recursive=True):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"  Đã xóa file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"  Đã xóa thư mục: {file_path}")
            except Exception as e:
                print(f"  Lỗi khi xóa {file_path}: {e}")

def create_core_directories():
    """Tạo các thư mục cốt lõi nếu chưa tồn tại"""
    print("Đang tạo các thư mục cốt lõi...")
    
    for directory in CORE_DIRECTORIES:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  Đã tạo thư mục: {directory}")

def organize_python_files():
    """Tổ chức lại các file Python"""
    print("Đang tổ chức lại các file Python...")
    
    # Danh sách các file đã xử lý
    processed_files = []
    
    # Xử lý các file cốt lõi trước
    for file in CORE_FILES:
        if os.path.exists(file):
            processed_files.append(file)
    
    # Tạo thư mục backup
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Đã tạo thư mục backup: {backup_dir}")
    
    # Di chuyển các file thừa vào thư mục backup
    for py_file in glob.glob("*.py"):
        if py_file not in processed_files and py_file != os.path.basename(__file__):
            shutil.copy2(py_file, os.path.join(backup_dir, py_file))
            print(f"  Đã sao lưu file: {py_file}")

def check_code_quality():
    """Kiểm tra chất lượng mã nguồn cơ bản"""
    print("Đang kiểm tra chất lượng mã nguồn...")
    
    # Kiểm tra cú pháp Python
    try:
        for py_file in glob.glob("**/*.py", recursive=True):
            if "__pycache__" not in py_file and "backup_" not in py_file:
                try:
                    subprocess.check_output([sys.executable, "-m", "py_compile", py_file])
                    print(f"  ✅ Cú pháp hợp lệ: {py_file}")
                except subprocess.CalledProcessError:
                    print(f"  ❌ Lỗi cú pháp: {py_file}")
    except Exception as e:
        print(f"Lỗi khi kiểm tra cú pháp: {e}")

def check_binance_imports():
    """Kiểm tra cách import Binance trong các file Python"""
    print("Đang kiểm tra imports Binance...")
    
    binance_import_files = []
    
    for py_file in glob.glob("**/*.py", recursive=True):
        if "__pycache__" not in py_file and "backup_" not in py_file:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "import binance" in content or "from binance" in content:
                        binance_import_files.append(py_file)
            except Exception as e:
                print(f"  Lỗi khi đọc file {py_file}: {e}")
    
    if binance_import_files:
        print("\nCác file có import Binance:")
        for file in binance_import_files:
            print(f"  - {file}")
    else:
        print("  Không tìm thấy file nào import Binance trực tiếp.")

def check_streamlit_config():
    """Kiểm tra cấu hình Streamlit"""
    print("Đang kiểm tra cấu hình Streamlit...")
    
    config_dir = ".streamlit"
    config_file = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f"  Đã tạo thư mục: {config_dir}")
    
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            f.write('[server]\nheadless = true\naddress = "0.0.0.0"\nport = 5000\n')
        print(f"  Đã tạo file cấu hình: {config_file}")
    else:
        # Kiểm tra nội dung file cấu hình
        with open(config_file, 'r') as f:
            content = f.read()
        
        missing_configs = []
        if "headless = true" not in content:
            missing_configs.append("headless = true")
        if 'address = "0.0.0.0"' not in content:
            missing_configs.append('address = "0.0.0.0"')
        if "port = 5000" not in content:
            missing_configs.append("port = 5000")
        
        if missing_configs:
            print("  ⚠️ Cấu hình Streamlit thiếu các tham số sau:")
            for config in missing_configs:
                print(f"    - {config}")
            
            # Cập nhật file cấu hình
            with open(config_file, 'w') as f:
                f.write('[server]\nheadless = true\naddress = "0.0.0.0"\nport = 5000\n')
            print("  ✅ Đã cập nhật file cấu hình Streamlit")
        else:
            print("  ✅ Cấu hình Streamlit đã đúng")

def main():
    """Hàm chính"""
    print("=== BẮT ĐẦU DỌN DẸP MÃ NGUỒN ===")
    
    # Tạo các thư mục cốt lõi
    create_core_directories()
    
    # Dọn dẹp các file tạm thời
    clean_temp_files()
    
    # Tổ chức lại các file Python
    organize_python_files()
    
    # Kiểm tra chất lượng mã nguồn
    check_code_quality()
    
    # Kiểm tra imports Binance
    check_binance_imports()
    
    # Kiểm tra cấu hình Streamlit
    check_streamlit_config()
    
    print("\n=== HOÀN THÀNH DỌN DẸP MÃ NGUỒN ===")
    print("Bạn có thể chạy lệnh sau để kiểm tra ứng dụng:")
    print("  streamlit run app.py")
    print("\nHoặc sử dụng script:")
    print("  ./single_command_run.sh")

if __name__ == "__main__":
    main()