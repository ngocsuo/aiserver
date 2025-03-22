#!/usr/bin/env python3
"""
Script chạy sạch - khởi động ETHUSDT Dashboard sau khi tối ưu hóa log và sửa lỗi.

Script này thực hiện các bước sau:
1. Tối ưu hóa log hệ thống để giảm kích thước và lọc bỏ các thông tin không cần thiết
2. Áp dụng các bản vá lỗi đã biết
3. Khởi động ứng dụng với giám sát liên tục
"""

import os
import sys
import time
import logging
import subprocess
import argparse
from pathlib import Path

def setup_logging():
    """Thiết lập logging cơ bản"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("run_clean.log"),
            logging.StreamHandler()
        ]
    )

def run_command(command):
    """
    Chạy lệnh shell và hiển thị output
    
    Args:
        command (str): Lệnh cần chạy
        
    Returns:
        int: Mã thoát của lệnh
    """
    logging.info(f"Chạy lệnh: {command}")
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Hiển thị output realtime
    while True:
        # Đọc từ stdout
        stdout_line = process.stdout.readline()
        if stdout_line:
            logging.info(stdout_line.strip())
            
        # Đọc từ stderr
        stderr_line = process.stderr.readline()
        if stderr_line:
            logging.error(stderr_line.strip())
        
        # Kiểm tra xem process đã kết thúc chưa
        if process.poll() is not None:
            # Đọc hết output còn lại
            for line in process.stdout:
                if line:
                    logging.info(line.strip())
            for line in process.stderr:
                if line:
                    logging.error(line.strip())
            break
    
    # Lấy mã thoát
    returncode = process.poll()
    
    if returncode == 0:
        logging.info(f"Lệnh đã hoàn thành thành công")
    else:
        logging.error(f"Lệnh đã kết thúc với mã lỗi: {returncode}")
        
    return returncode

def ensure_directories():
    """Đảm bảo các thư mục cần thiết tồn tại"""
    dirs = ["logs", "deployment/logs", "data", "saved_models"]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Đảm bảo thư mục {dir_path} tồn tại")

def optimize_logs():
    """Tối ưu hóa và lọc log"""
    logging.info("Bắt đầu tối ưu hóa log...")
    
    # Chạy script tối ưu hóa log
    command = "python deployment/optimize_logs.py --all"
    return run_command(command) == 0

def apply_fixes():
    """Áp dụng các bản vá lỗi"""
    logging.info("Áp dụng các bản vá lỗi...")
    
    # Áp dụng bản vá cho feature_engineering
    result1 = run_command("python feature_engineering_fix.py") == 0
    
    # Áp dụng bản vá cho dữ liệu
    result2 = run_command("python -c \"from utils.data_fix import run_data_fix; run_data_fix()\"") == 0
    
    return result1 and result2

def run_with_monitoring(mode):
    """
    Chạy ứng dụng với giám sát
    
    Args:
        mode (str): Chế độ chạy
    """
    logging.info(f"Chạy ứng dụng với giám sát (chế độ: {mode})...")
    
    command = f"python run_with_monitoring.py --mode {mode}"
    return run_command(command)

def main():
    """Hàm chính"""
    # Khởi tạo parser
    parser = argparse.ArgumentParser(description="Chạy sạch ETHUSDT Dashboard")
    parser.add_argument("--skip-optimize", action="store_true", help="Bỏ qua bước tối ưu hóa log")
    parser.add_argument("--skip-fixes", action="store_true", help="Bỏ qua bước áp dụng bản vá lỗi")
    parser.add_argument("--mode", choices=["service", "direct", "script"], default="service", 
                       help="Chế độ chạy: service (dịch vụ triển khai), direct (chạy trực tiếp), script (chạy startup script)")
    
    # Parse tham số
    args = parser.parse_args()
    
    # Thiết lập logging
    setup_logging()
    
    # Hiển thị thông tin
    logging.info("=== KHỞI ĐỘNG SẠCH ETHUSDT DASHBOARD ===")
    
    # Đảm bảo các thư mục cần thiết
    ensure_directories()
    
    # Tối ưu hóa log
    if not args.skip_optimize:
        if not optimize_logs():
            logging.warning("Có vấn đề khi tối ưu hóa log, nhưng vẫn tiếp tục...")
    else:
        logging.info("Bỏ qua bước tối ưu hóa log theo yêu cầu")
    
    # Áp dụng các bản vá lỗi
    if not args.skip_fixes:
        if not apply_fixes():
            logging.warning("Có vấn đề khi áp dụng bản vá lỗi, nhưng vẫn tiếp tục...")
    else:
        logging.info("Bỏ qua bước áp dụng bản vá lỗi theo yêu cầu")
    
    # Chạy ứng dụng với giám sát
    run_with_monitoring(args.mode)

if __name__ == "__main__":
    main()