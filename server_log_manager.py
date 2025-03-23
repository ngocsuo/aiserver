#!/usr/bin/env python3
"""
Công cụ quản lý log và trạng thái hệ thống ETHUSDT Dashboard
- Hiển thị log theo thời gian thực
- Kiểm tra và hiển thị trạng thái hệ thống
- Hỗ trợ chẩn đoán và sửa lỗi từ xa
"""
import os
import sys
import time
import argparse
import subprocess
import json
import datetime

def get_datetime():
    """Lấy thời gian hiện tại theo định dạng đẹp"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def print_color(text, color="white"):
    """In text với màu sắc"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")

def print_header(text):
    """In tiêu đề đẹp"""
    print("\n" + "="*80)
    print_color(f"  {text}", "cyan")
    print("="*80)

def tail_log(log_file, lines=20, follow=False):
    """Hiển thị log theo thời gian thực"""
    if not os.path.exists(log_file):
        print_color(f"File log không tồn tại: {log_file}", "red")
        return

    print_header(f"Xem log từ file: {log_file}")
    
    if follow:
        print_color("Đang theo dõi log trong thời gian thực. Nhấn Ctrl+C để thoát.", "yellow")
        try:
            subprocess.run(["tail", "-f", "-n", str(lines), log_file])
        except KeyboardInterrupt:
            print_color("\nĐã dừng theo dõi log.", "yellow")
    else:
        subprocess.run(["tail", "-n", str(lines), log_file])

def grep_log(log_file, pattern, context=2):
    """Tìm kiếm trong log với từ khóa"""
    if not os.path.exists(log_file):
        print_color(f"File log không tồn tại: {log_file}", "red")
        return

    print_header(f"Tìm kiếm '{pattern}' trong {log_file}")
    
    try:
        subprocess.run(["grep", "-A", str(context), "-B", str(context), "--color=auto", pattern, log_file])
    except subprocess.CalledProcessError:
        print_color(f"Không tìm thấy kết quả cho '{pattern}'", "yellow")

def check_system_status():
    """Kiểm tra trạng thái hệ thống"""
    print_header("Trạng thái hệ thống")
    
    # Kiểm tra CPU và RAM
    print_color("Thông tin CPU:", "green")
    subprocess.run(["top", "-bn1", "|", "grep", "load"], shell=True)
    
    print_color("\nThông tin bộ nhớ:", "green")
    subprocess.run(["free", "-h"])
    
    print_color("\nKhông gian đĩa:", "green")
    subprocess.run(["df", "-h", "/"])
    
    print_color("\nCác tiến trình quan trọng:", "green")
    subprocess.run(["ps", "-aux", "|", "grep", "python", "|", "grep", "-v", "grep"], shell=True)

def check_app_status():
    """Kiểm tra trạng thái ứng dụng ETHUSDT Dashboard"""
    print_header("Trạng thái ETHUSDT Dashboard")
    
    # Kiểm tra xem ứng dụng có đang chạy
    try:
        result = subprocess.run(["pgrep", "-f", "streamlit run app.py"], 
                                capture_output=True, text=True)
        if result.stdout.strip():
            print_color("✅ ETHUSDT Dashboard đang chạy", "green")
            print_color(f"PID: {result.stdout.strip()}", "green")
        else:
            print_color("❌ ETHUSDT Dashboard không chạy", "red")
    except Exception as e:
        print_color(f"Lỗi khi kiểm tra trạng thái: {str(e)}", "red")

    # Kiểm tra port 5000
    try:
        result = subprocess.run(["lsof", "-i", ":5000"], 
                                capture_output=True, text=True)
        if "streamlit" in result.stdout:
            print_color("\n✅ Streamlit đang chạy trên port 5000", "green")
        else:
            print_color("\n❌ Không có dịch vụ nào đang chạy trên port 5000", "red")
    except Exception as e:
        print_color(f"Lỗi khi kiểm tra port: {str(e)}", "red")

def check_training_status():
    """Kiểm tra trạng thái huấn luyện mô hình"""
    print_header("Trạng thái huấn luyện mô hình")
    
    training_logs_file = "training_logs.txt"
    if os.path.exists(training_logs_file):
        print_color("Các dòng log huấn luyện gần đây:", "green")
        subprocess.run(["tail", "-n", "10", training_logs_file])
    else:
        print_color("File log huấn luyện không tồn tại", "yellow")
    
    training_status_file = "training_status.json"
    if os.path.exists(training_status_file):
        try:
            with open(training_status_file, 'r') as f:
                status = json.load(f)
                print_color("\nTrạng thái huấn luyện hiện tại:", "green")
                print(json.dumps(status, indent=2))
        except Exception as e:
            print_color(f"Lỗi khi đọc file trạng thái: {str(e)}", "red")
    else:
        print_color("File trạng thái huấn luyện không tồn tại", "yellow")
    
    # Kiểm tra các mô hình đã huấn luyện
    saved_models_dir = "saved_models"
    if os.path.exists(saved_models_dir) and os.path.isdir(saved_models_dir):
        print_color("\nDanh sách mô hình đã huấn luyện:", "green")
        subprocess.run(["ls", "-la", saved_models_dir])
    else:
        print_color("\nThư mục saved_models không tồn tại", "yellow")

def restart_app():
    """Khởi động lại ứng dụng ETHUSDT Dashboard"""
    print_header("Khởi động lại ETHUSDT Dashboard")
    
    # Kiểm tra xem ứng dụng có đang chạy
    try:
        # Tìm và kết thúc tiến trình hiện tại
        subprocess.run(["pkill", "-f", "streamlit run app.py"])
        print_color("✅ Đã dừng tiến trình cũ", "green")
    except Exception as e:
        print_color(f"Không thể dừng tiến trình cũ: {str(e)}", "yellow")
    
    # Khởi động lại ứng dụng
    try:
        print_color("Đang khởi động lại ứng dụng...", "yellow")
        subprocess.Popen([
            "nohup", "streamlit", "run", "app.py", 
            "--server.port=5000", 
            "--server.address=0.0.0.0", 
            "--server.headless=true"
        ], stdout=open("app.log", "a"), stderr=subprocess.STDOUT)
        
        print_color("✅ Ứng dụng đã được khởi động lại thành công", "green")
        print_color("Hãy đợi vài giây để ứng dụng khởi động hoàn toàn...", "yellow")
        
        # Đợi một chút và kiểm tra lại trạng thái
        time.sleep(5)
        check_app_status()
    except Exception as e:
        print_color(f"❌ Lỗi khi khởi động lại ứng dụng: {str(e)}", "red")

def create_debug_report():
    """Tạo báo cáo gỡ lỗi đầy đủ"""
    report_file = f"debug_report_{int(time.time())}.txt"
    print_header(f"Tạo báo cáo gỡ lỗi: {report_file}")
    
    with open(report_file, 'w') as f:
        f.write(f"===== BÁO CÁO GỠ LỖI ETHUSDT DASHBOARD =====\n")
        f.write(f"Thời gian: {get_datetime()}\n\n")
        
        # Thông tin hệ thống
        f.write("===== THÔNG TIN HỆ THỐNG =====\n")
        result = subprocess.run(["uname", "-a"], capture_output=True, text=True)
        f.write(f"OS: {result.stdout}")
        
        result = subprocess.run(["free", "-h"], capture_output=True, text=True)
        f.write(f"\nBộ nhớ:\n{result.stdout}")
        
        result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True)
        f.write(f"\nĐĩa cứng:\n{result.stdout}")
        
        # Trạng thái ứng dụng
        f.write("\n===== TRẠNG THÁI ỨNG DỤNG =====\n")
        result = subprocess.run(["ps", "-aux", "|", "grep", "python", "|", "grep", "-v", "grep"], 
                                shell=True, capture_output=True, text=True)
        f.write(f"Tiến trình Python:\n{result.stdout}")
        
        result = subprocess.run(["netstat", "-tulpn", "|", "grep", "5000"], 
                                shell=True, capture_output=True, text=True)
        f.write(f"\nCổng 5000:\n{result.stdout}")
        
        # Nhật ký ứng dụng
        f.write("\n===== NHẬT KÝ ỨNG DỤNG =====\n")
        if os.path.exists("app.log"):
            result = subprocess.run(["tail", "-n", "50", "app.log"], 
                                    capture_output=True, text=True)
            f.write(f"app.log (50 dòng cuối):\n{result.stdout}")
        
        if os.path.exists("training_logs.txt"):
            result = subprocess.run(["tail", "-n", "50", "training_logs.txt"], 
                                    capture_output=True, text=True)
            f.write(f"\ntraining_logs.txt (50 dòng cuối):\n{result.stdout}")
    
    print_color(f"✅ Đã tạo báo cáo gỡ lỗi: {report_file}", "green")
    print_color(f"Bạn có thể xem báo cáo bằng lệnh: cat {report_file}", "yellow")

def fix_common_issues():
    """Tự động sửa các lỗi phổ biến"""
    print_header("Sửa lỗi tự động")
    
    issues_fixed = 0
    
    # Kiểm tra và tạo các thư mục cần thiết
    for directory in ["data", "saved_models", "logs", "utils"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print_color(f"✅ Đã tạo thư mục thiếu: {directory}", "green")
            issues_fixed += 1
    
    # Kiểm tra và tạo file logging
    required_files = ["training_logs.txt", "app.log"]
    for file in required_files:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                f.write(f"# Log file created at {get_datetime()}\n")
            os.chmod(file, 0o666)  # Cho phép đọc và ghi
            print_color(f"✅ Đã tạo file log thiếu: {file}", "green")
            issues_fixed += 1
    
    # Kiểm tra và sửa thread_safe_logging.py
    utils_dir = "utils"
    thread_safe_logging = os.path.join(utils_dir, "thread_safe_logging.py")
    if not os.path.exists(thread_safe_logging):
        # Sao chép từ thread_safe_logging_module.py nếu có
        if os.path.exists("thread_safe_logging_module.py"):
            subprocess.run(["cp", "thread_safe_logging_module.py", thread_safe_logging])
            print_color(f"✅ Đã tạo file thread_safe_logging.py từ module có sẵn", "green")
            issues_fixed += 1
        else:
            # Tạo file mới nếu không có sẵn
            thread_safe_content = '''"""
Thread-safe logging functions for AI Trading System
"""
import os
import time
import threading

_log_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    """Thread-safe function to log messages to a file"""
    with _log_lock:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(log_file, "a") as f:
                f.write(f"{timestamp} - {message}\\n")
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")

def log_to_console(message):
    """Thread-safe function to log messages to console"""
    with _log_lock:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - {message}")

def thread_safe_log(message, log_file="training_logs.txt"):
    """Combined logging function that logs to both file and console"""
    log_to_file(message, log_file)
    log_to_console(message)

def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    """Read log entries from file with a maximum number of lines"""
    try:
        if not os.path.exists(log_file):
            return ["No log file found"]
            
        with open(log_file, "r") as f:
            lines = f.readlines()
            
        return lines[-max_lines:] if len(lines) > max_lines else lines
    except Exception as e:
        return [f"Error reading log file: {str(e)}"]
'''
            with open(thread_safe_logging, 'w') as f:
                f.write(thread_safe_content)
            print_color(f"✅ Đã tạo file thread_safe_logging.py mới", "green")
            issues_fixed += 1
    
    # Kiểm tra và sửa các lỗi phổ biến trong config.py
    config_file = "config.py"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        # Kiểm tra và sửa USE_PROXY
        if "USE_PROXY = True" in config_content:
            config_content = config_content.replace("USE_PROXY = True", "USE_PROXY = False")
            with open(config_file, 'w') as f:
                f.write(config_content)
            print_color("✅ Đã tắt proxy trong config.py", "green")
            issues_fixed += 1
        
        # Kiểm tra available_timeframes
        if "available_timeframes" not in config_content:
            # Tìm vị trí để thêm cấu hình
            if "TRADING_SETTINGS = DEFAULT_TRADING_CONFIG" in config_content:
                config_content = config_content.replace(
                    "TRADING_SETTINGS = DEFAULT_TRADING_CONFIG",
                    "TRADING_SETTINGS = DEFAULT_TRADING_CONFIG\n\n# Available timeframes for trading\nTRADING_SETTINGS[\"available_timeframes\"] = [\"1m\", \"5m\", \"15m\", \"1h\", \"4h\"]\nTRADING_SETTINGS[\"default_timeframe\"] = \"5m\""
                )
                with open(config_file, 'w') as f:
                    f.write(config_content)
                print_color("✅ Đã thêm available_timeframes vào config.py", "green")
                issues_fixed += 1
    
    if issues_fixed > 0:
        print_color(f"✅ Đã sửa tổng cộng {issues_fixed} vấn đề", "green")
    else:
        print_color("✅ Không phát hiện vấn đề nào cần sửa", "green")

def main():
    parser = argparse.ArgumentParser(description="Công cụ quản lý log và trạng thái ETHUSDT Dashboard")
    
    # Tạo các nhóm lệnh
    subparsers = parser.add_subparsers(dest="command", help="Lệnh cần thực hiện")
    
    # Lệnh xem log
    log_parser = subparsers.add_parser("log", help="Xem log")
    log_parser.add_argument("-f", "--file", default="app.log", help="File log cần xem (mặc định: app.log)")
    log_parser.add_argument("-n", "--lines", type=int, default=20, help="Số dòng cần hiển thị (mặc định: 20)")
    log_parser.add_argument("--follow", action="store_true", help="Theo dõi log theo thời gian thực")
    
    # Lệnh tìm kiếm trong log
    grep_parser = subparsers.add_parser("grep", help="Tìm kiếm trong log")
    grep_parser.add_argument("pattern", help="Từ khóa cần tìm")
    grep_parser.add_argument("-f", "--file", default="app.log", help="File log cần tìm (mặc định: app.log)")
    grep_parser.add_argument("-c", "--context", type=int, default=2, help="Số dòng context trước và sau (mặc định: 2)")
    
    # Lệnh kiểm tra trạng thái
    status_parser = subparsers.add_parser("status", help="Kiểm tra trạng thái")
    status_group = status_parser.add_mutually_exclusive_group()
    status_group.add_argument("--system", action="store_true", help="Kiểm tra trạng thái hệ thống")
    status_group.add_argument("--app", action="store_true", help="Kiểm tra trạng thái ứng dụng")
    status_group.add_argument("--training", action="store_true", help="Kiểm tra trạng thái huấn luyện")
    
    # Lệnh khởi động lại
    restart_parser = subparsers.add_parser("restart", help="Khởi động lại ứng dụng")
    
    # Lệnh tạo báo cáo gỡ lỗi
    debug_parser = subparsers.add_parser("debug", help="Tạo báo cáo gỡ lỗi")
    
    # Lệnh sửa lỗi tự động
    fix_parser = subparsers.add_parser("fix", help="Sửa lỗi tự động")
    
    # Lệnh show hướng dẫn
    help_parser = subparsers.add_parser("help", help="Hiển thị hướng dẫn sử dụng")
    
    args = parser.parse_args()
    
    # Xử lý các lệnh
    if args.command == "log":
        tail_log(args.file, args.lines, args.follow)
    elif args.command == "grep":
        grep_log(args.file, args.pattern, args.context)
    elif args.command == "status":
        if args.system:
            check_system_status()
        elif args.training:
            check_training_status()
        elif args.app:
            check_app_status()
        else:
            # Mặc định kiểm tra tất cả
            check_app_status()
            check_training_status()
    elif args.command == "restart":
        restart_app()
    elif args.command == "debug":
        create_debug_report()
    elif args.command == "fix":
        fix_common_issues()
    elif args.command == "help" or args.command is None:
        # Hiển thị hướng dẫn sử dụng
        print_header("HƯỚNG DẪN SỬ DỤNG SERVER_LOG_MANAGER.PY")
        print_color("Công cụ quản lý log và trạng thái ETHUSDT Dashboard", "yellow")
        print_color("\nCác lệnh chính:", "green")
        print("  ./server_log_manager.py log     : Xem log")
        print("  ./server_log_manager.py grep    : Tìm kiếm trong log")
        print("  ./server_log_manager.py status  : Kiểm tra trạng thái")
        print("  ./server_log_manager.py restart : Khởi động lại ứng dụng")
        print("  ./server_log_manager.py debug   : Tạo báo cáo gỡ lỗi")
        print("  ./server_log_manager.py fix     : Sửa lỗi tự động")
        print("  ./server_log_manager.py help    : Hiển thị hướng dẫn")
        
        print_color("\nVí dụ sử dụng:", "green")
        print("  ./server_log_manager.py log -f app.log -n 50 --follow")
        print("  ./server_log_manager.py grep ERROR -f training_logs.txt")
        print("  ./server_log_manager.py status --app")

if __name__ == "__main__":
    main()