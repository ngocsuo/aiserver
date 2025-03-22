"""
Thread-safe logging functions for AI Trading System
"""
import os
import threading
import logging
from datetime import datetime

# Thiết lập cơ bản cho logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Khóa thread để đảm bảo an toàn khi ghi file
file_lock = threading.Lock()
console_lock = threading.Lock()

def log_to_file(message, log_file="training_logs.txt"):
    """Thread-safe function to log messages to a file"""
    with file_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp} - {message}\n")

def log_to_console(message):
    """Thread-safe function to log messages to console"""
    with console_lock:
        logger = logging.getLogger("training")
        logger.info(message)
        
def thread_safe_log(message, log_file="training_logs.txt"):
    """Combined logging function that logs to both file and console"""
    log_to_file(message, log_file)
    log_to_console(message)
    
def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
    """Read log entries from file with a maximum number of lines"""
    try:
        with file_lock:
            # Đảm bảo tệp tin tồn tại
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    pass
                return []
            
            # Đọc tất cả các dòng
            with open(log_file, "r") as f:
                lines = f.readlines()
            
            # Lấy các dòng cuối cùng
            return lines[-max_lines:] if lines else []
    except Exception as e:
        logging.error(f"Error reading logs: {e}")
        return []