"""
Script tối ưu hóa và lọc log cho hệ thống ETHUSDT Dashboard.

Script này giúp:
1. Giảm kích thước của các file log hiện có
2. Thiết lập bộ lọc thông minh cho log mới
3. Xóa các log trùng lặp và không cần thiết
"""

import os
import re
import sys
import time
import logging
import argparse
from pathlib import Path

# Thêm thư mục gốc vào path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Import bộ lọc log
from utils.log_filter import apply_log_filter, LogManager

def optimize_existing_logs(log_dir="logs", backup_dir=None):
    """
    Tối ưu hóa các file log hiện có
    
    Args:
        log_dir (str): Thư mục chứa log
        backup_dir (str, optional): Thư mục sao lưu
    """
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Thư mục log {log_dir} không tồn tại. Bỏ qua.")
        return
        
    # Tạo thư mục sao lưu nếu cần
    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(exist_ok=True)
    
    # Pattern cần lọc bỏ
    patterns = [
        r"Kiểm tra kết nối proxy",
        r"Kết nối proxy thành công",
        r"Đã cấu hình socket proxy",
        r"Connecting to Binance API using proxy",
        r"Binance API connection successful",
        r"Binance system status: Normal",
        r"Binance Futures API accessible",
        r"Binance data collector initialized successfully",
        r"Successfully initialized Binance data collector",
        r"Binance server time:",
        r"cache data for period",
        r"đã tồn tại trong cache",
        r"Skipping chunk.*data already exists",
        r"Loaded compressed cached data"
    ]
    
    # Compile regex pattern
    compiled_patterns = [re.compile(pattern) for pattern in patterns]
    
    # Tìm tất cả file log
    log_files = list(log_dir.glob("*.log"))
    log_files.extend(log_dir.glob("*.log.*"))
    
    print(f"Tìm thấy {len(log_files)} file log.")
    
    # Xử lý từng file
    for log_file in log_files:
        # Sao lưu nếu cần
        if backup_dir:
            import shutil
            backup_file = backup_dir / log_file.name
            shutil.copy2(log_file, backup_file)
            print(f"Đã sao lưu {log_file} sang {backup_file}")
        
        # Đọc toàn bộ file
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Đếm số dòng trước khi lọc
        before_count = len(lines)
        
        # Lọc các dòng
        filtered_lines = []
        for line in lines:
            # Giữ lại các log ERROR và WARNING
            if " ERROR " in line or " WARNING " in line:
                filtered_lines.append(line)
                continue
                
            # Kiểm tra xem dòng có khớp với bất kỳ pattern nào không
            skip_line = False
            for pattern in compiled_patterns:
                if pattern.search(line):
                    skip_line = True
                    break
                    
            # Nếu không cần bỏ qua, thêm vào danh sách đã lọc
            if not skip_line:
                filtered_lines.append(line)
        
        # Đếm số dòng sau khi lọc
        after_count = len(filtered_lines)
        
        # Ghi lại file đã lọc
        with open(log_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
            
        # Thông báo kết quả
        print(f"Đã tối ưu file {log_file}: {before_count - after_count} dòng đã bị loại bỏ ({before_count} -> {after_count})")

def apply_log_rotation(log_dir="logs", max_files=5, max_size_mb=5):
    """
    Áp dụng log rotation cho các file log
    
    Args:
        log_dir (str): Thư mục chứa log
        max_files (int): Số file tối đa cho mỗi loại log
        max_size_mb (int): Kích thước tối đa của mỗi file log (MB)
    """
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Thư mục log {log_dir} không tồn tại. Bỏ qua.")
        return
        
    # Kích thước tối đa
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Tìm tất cả file log
    log_files = list(log_dir.glob("*.log"))
    
    # Nhóm các file log theo tên cơ sở
    log_groups = {}
    for log_file in log_files:
        base_name = log_file.name
        if base_name not in log_groups:
            log_groups[base_name] = []
        log_groups[base_name].append(log_file)
    
    # Xử lý từng nhóm
    for base_name, files in log_groups.items():
        # Kiểm tra kích thước của file chính
        main_file = log_dir / base_name
        if main_file.exists() and main_file.stat().st_size > max_size_bytes:
            # Cần thực hiện rotation
            # Tìm các file rotation hiện có
            rotation_files = list(log_dir.glob(f"{base_name}.*"))
            rotation_files.sort()
            
            # Xóa file cũ nhất nếu vượt quá số lượng tối đa
            while len(rotation_files) >= max_files:
                oldest_file = rotation_files.pop(0)
                try:
                    oldest_file.unlink()
                    print(f"Đã xóa file log cũ: {oldest_file}")
                except Exception as e:
                    print(f"Lỗi khi xóa file log {oldest_file}: {e}")
            
            # Thực hiện rotation
            # Tìm số thứ tự cao nhất hiện có
            max_index = 0
            for file in rotation_files:
                try:
                    index = int(file.name.split('.')[-1])
                    max_index = max(max_index, index)
                except:
                    pass
            
            # Thực hiện rotation từ cao xuống thấp
            for i in range(max_index, 0, -1):
                old_file = log_dir / f"{base_name}.{i}"
                new_file = log_dir / f"{base_name}.{i+1}"
                if old_file.exists():
                    try:
                        old_file.rename(new_file)
                    except Exception as e:
                        print(f"Lỗi khi đổi tên {old_file} thành {new_file}: {e}")
            
            # Đổi tên file chính
            try:
                main_file.rename(log_dir / f"{base_name}.1")
                print(f"Đã thực hiện rotation cho {main_file}")
            except Exception as e:
                print(f"Lỗi khi đổi tên {main_file}: {e}")
            
            # Tạo file mới
            try:
                with open(main_file, 'w') as f:
                    f.write(f"# Log file tạo mới vào {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            except Exception as e:
                print(f"Lỗi khi tạo file mới {main_file}: {e}")

def cleanup_old_logs(log_dir="logs", days=7):
    """
    Dọn dẹp file log cũ
    
    Args:
        log_dir (str): Thư mục chứa log
        days (int): Số ngày tối đa giữ log
    """
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Thư mục log {log_dir} không tồn tại. Bỏ qua.")
        return
        
    # Lấy thời gian hiện tại
    import datetime
    now = datetime.datetime.now()
    
    # Tìm tất cả file log
    log_files = list(log_dir.glob("*.log.*"))
    
    # Xóa file log cũ
    count = 0
    for log_file in log_files:
        # Lấy thời gian sửa đổi
        modified_time = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
        
        # Tính số ngày
        days_old = (now - modified_time).days
        
        # Xóa nếu quá cũ
        if days_old > days:
            try:
                log_file.unlink()
                count += 1
            except Exception as e:
                print(f"Lỗi khi xóa file log {log_file}: {e}")
    
    print(f"Đã xóa {count} file log cũ hơn {days} ngày.")

def main():
    # Khởi tạo parser
    parser = argparse.ArgumentParser(description="Tối ưu hóa và lọc log cho hệ thống ETHUSDT Dashboard")
    parser.add_argument("--optimize", action="store_true", help="Tối ưu hóa các file log hiện có")
    parser.add_argument("--rotate", action="store_true", help="Áp dụng log rotation")
    parser.add_argument("--cleanup", action="store_true", help="Dọn dẹp file log cũ")
    parser.add_argument("--filter", action="store_true", help="Áp dụng bộ lọc thông minh cho log mới")
    parser.add_argument("--all", action="store_true", help="Thực hiện tất cả các tác vụ")
    parser.add_argument("--log-dir", default="logs", help="Thư mục chứa log (mặc định: 'logs')")
    parser.add_argument("--backup-dir", default=None, help="Thư mục sao lưu (nếu cần)")
    parser.add_argument("--max-files", type=int, default=5, help="Số file tối đa cho mỗi loại log (mặc định: 5)")
    parser.add_argument("--max-size-mb", type=int, default=5, help="Kích thước tối đa của mỗi file log (MB) (mặc định: 5)")
    parser.add_argument("--days", type=int, default=7, help="Số ngày tối đa giữ log (mặc định: 7)")
    
    # Parse tham số
    args = parser.parse_args()
    
    # Thực hiện tối ưu hóa
    if args.optimize or args.all:
        print("\n--- TỐI ƯU HÓA CÁC FILE LOG HIỆN CÓ ---")
        optimize_existing_logs(args.log_dir, args.backup_dir)
    
    # Thực hiện log rotation
    if args.rotate or args.all:
        print("\n--- ÁP DỤNG LOG ROTATION ---")
        apply_log_rotation(args.log_dir, args.max_files, args.max_size_mb)
    
    # Dọn dẹp file log cũ
    if args.cleanup or args.all:
        print("\n--- DỌN DẸP FILE LOG CŨ ---")
        cleanup_old_logs(args.log_dir, args.days)
    
    # Áp dụng bộ lọc thông minh
    if args.filter or args.all:
        print("\n--- ÁP DỤNG BỘ LỌC THÔNG MINH ---")
        log_manager = apply_log_filter()
        print("Đã áp dụng bộ lọc thông minh cho log mới.")
    
    print("\nHoàn tất tối ưu hóa log.")

if __name__ == "__main__":
    main()