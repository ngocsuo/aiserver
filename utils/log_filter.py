"""
Module lọc và tối ưu log cho hệ thống ETHUSDT Dashboard.

Công cụ này giúp giảm thiểu lượng log được tạo ra và chỉ lưu trữ những thông tin cần thiết,
tập trung vào các lỗi, cảnh báo và thông tin quan trọng. Điều này giúp cải thiện hiệu suất
và giảm dung lượng file log.
"""

import os
import re
import time
import logging
import logging.handlers
import datetime
from pathlib import Path

# Cài đặt thư mục logs
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_DIR.mkdir(exist_ok=True)

# Định nghĩa các pattern cần lọc
CONNECTION_PATTERNS = [
    r"Kiểm tra kết nối proxy",
    r"Kết nối proxy thành công",
    r"Đã cấu hình socket proxy",
    r"Connecting to Binance API using proxy",
    r"Binance API connection successful",
    r"Binance system status: Normal",
    r"Binance Futures API accessible",
    r"Binance data collector initialized successfully",
    r"Successfully initialized Binance data collector",
    r"Binance server time:"
]

CACHE_PATTERNS = [
    r"cache data for period",
    r"đã tồn tại trong cache",
    r"Skipping chunk.*data already exists",
    r"Loaded compressed cached data"
]

# Logger chính
logger = logging.getLogger("LogFilter")

class SmartLogFilter(logging.Filter):
    """Bộ lọc thông minh cho log"""
    
    def __init__(self, name=""):
        super().__init__(name)
        self.last_connection_log = 0
        self.connection_log_count = 0
        self.connection_summary_interval = 60  # Tóm tắt thông tin kết nối mỗi 60 giây
        
        self.last_cache_log = 0
        self.cache_log_count = 0
        self.cache_summary_interval = 300  # Tóm tắt thông tin cache mỗi 5 phút
        
    def filter(self, record):
        """
        Lọc các bản ghi log theo quy tắc
        
        Args:
            record: Bản ghi log
            
        Returns:
            bool: True nếu bản ghi nên được giữ lại, False nếu nên loại bỏ
        """
        # Luôn giữ lại các log ERROR và WARNING
        if record.levelno >= logging.WARNING:
            return True
            
        # Lọc log kết nối
        if any(re.search(pattern, record.getMessage()) for pattern in CONNECTION_PATTERNS):
            current_time = time.time()
            
            # Đếm số lượng log kết nối
            self.connection_log_count += 1
            
            # Chỉ giữ log kết nối đầu tiên và tóm tắt định kỳ
            if current_time - self.last_connection_log > self.connection_summary_interval:
                # Tạo bản ghi tóm tắt
                if self.connection_log_count > 1:
                    logger.info(f"Tổng hợp: {self.connection_log_count} log kết nối trong {self.connection_summary_interval} giây qua")
                
                # Reset bộ đếm
                self.last_connection_log = current_time
                self.connection_log_count = 0
                return True
            else:
                # Thêm vào bộ đếm nhưng không hiển thị
                return False
                
        # Lọc log cache
        if any(re.search(pattern, record.getMessage()) for pattern in CACHE_PATTERNS):
            current_time = time.time()
            
            # Đếm số lượng log cache
            self.cache_log_count += 1
            
            # Chỉ giữ log cache đầu tiên và tóm tắt định kỳ
            if current_time - self.last_cache_log > self.cache_summary_interval:
                # Tạo bản ghi tóm tắt
                if self.cache_log_count > 1:
                    logger.info(f"Tổng hợp: {self.cache_log_count} log cache trong {self.cache_summary_interval} giây qua")
                
                # Reset bộ đếm
                self.last_cache_log = current_time
                self.cache_log_count = 0
                return True
            else:
                # Thêm vào bộ đếm nhưng không hiển thị
                return False
        
        # Giữ lại tất cả các log khác
        return True

class LogManager:
    """Quản lý log cho toàn bộ hệ thống"""
    
    def __init__(self, app_name="ETHUSDT_Dashboard", log_dir=DEFAULT_LOG_DIR):
        """
        Khởi tạo quản lý log
        
        Args:
            app_name (str): Tên ứng dụng
            log_dir (Path): Thư mục chứa log
        """
        self.app_name = app_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # File log chính
        self.main_log_file = self.log_dir / f"{app_name}.log"
        
        # File log lỗi
        self.error_log_file = self.log_dir / f"{app_name}_errors.log"
        
        # Cấu hình root logger
        self.setup_root_logger()
        
    def setup_root_logger(self):
        """Cấu hình root logger"""
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Xóa tất cả handler hiện có
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Tạo handler cho log file chính
        file_handler = logging.handlers.RotatingFileHandler(
            self.main_log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=10
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Thêm bộ lọc thông minh
        smart_filter = SmartLogFilter()
        file_handler.addFilter(smart_filter)
        
        # Tạo handler cho log lỗi
        error_handler = logging.handlers.RotatingFileHandler(
            self.error_log_file,
            maxBytes=2 * 1024 * 1024,  # 2 MB
            backupCount=5
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Tạo handler cho console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        console_handler.addFilter(smart_filter)
        
        # Thêm các handler vào root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(console_handler)
        
        # Thông báo
        logging.info(f"Đã thiết lập hệ thống log với bộ lọc thông minh")
        
    def get_logger(self, name):
        """
        Lấy logger con
        
        Args:
            name (str): Tên logger
            
        Returns:
            logging.Logger: Logger con
        """
        return logging.getLogger(name)
    
    def cleanup_old_logs(self, days=7):
        """
        Dọn dẹp file log cũ
        
        Args:
            days (int): Số ngày tối đa giữ log
        """
        # Lấy thời gian hiện tại
        now = datetime.datetime.now()
        
        # Tìm tất cả file log
        log_files = list(self.log_dir.glob("*.log.*"))
        
        # Xóa file log cũ
        for log_file in log_files:
            # Lấy thời gian sửa đổi
            modified_time = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
            
            # Tính số ngày
            days_old = (now - modified_time).days
            
            # Xóa nếu quá cũ
            if days_old > days:
                try:
                    log_file.unlink()
                    logging.info(f"Đã xóa file log cũ: {log_file}")
                except Exception as e:
                    logging.error(f"Lỗi khi xóa file log {log_file}: {e}")

def setup_logging():
    """
    Thiết lập hệ thống log cho ứng dụng
    
    Returns:
        LogManager: Quản lý log
    """
    # Tạo quản lý log
    log_manager = LogManager()
    
    # Dọn dẹp log cũ
    log_manager.cleanup_old_logs()
    
    return log_manager

def apply_log_filter():
    """
    Áp dụng bộ lọc log cho toàn bộ hệ thống
    """
    # Thiết lập hệ thống log
    log_manager = setup_logging()
    
    # Thông báo
    logger = log_manager.get_logger("LogFilter")
    logger.info("Đã áp dụng bộ lọc log cho toàn bộ hệ thống")
    
    return log_manager

if __name__ == "__main__":
    # Áp dụng bộ lọc log
    log_manager = apply_log_filter()
    
    # Thử nghiệm
    test_logger = log_manager.get_logger("Test")
    test_logger.info("Thử nghiệm bộ lọc log")