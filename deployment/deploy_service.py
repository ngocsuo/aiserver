"""
Dịch vụ triển khai và giám sát cho ứng dụng ETHUSDT Dashboard.

Module này thực hiện và quản lý việc triển khai ứng dụng, ghi nhật ký và khởi động lại ứng dụng khi cần.
"""

import os
import time
import signal
import logging
import threading
import subprocess
import sys
import psutil
import datetime
import json
from pathlib import Path

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deployment/deploy.log'
)

logger = logging.getLogger("DeploymentService")

class DeploymentService:
    """Dịch vụ triển khai và giám sát ứng dụng"""
    
    def __init__(self):
        """Khởi tạo dịch vụ triển khai"""
        # Lấy thư mục gốc của dự án
        self.root_dir = Path(os.path.abspath(os.path.dirname(__file__))).parent
        
        # Đường dẫn đến script khởi động
        self.startup_script = self.root_dir / "deployment" / "startup.sh"
        
        # Đường dẫn đến file logs
        self.logs_dir = self.root_dir / "logs"
        self.deploy_logs_dir = self.root_dir / "deployment" / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.deploy_logs_dir.mkdir(exist_ok=True, parents=True)
        
        # Đường dẫn đến file log ứng dụng
        self.app_log_file = self.logs_dir / "app.log"
        
        # Đường dẫn đến file theo dõi tài nguyên
        self.resource_log_file = self.deploy_logs_dir / "resources.log"
        
        # Đường dẫn đến file trạng thái hệ thống
        self.system_status_file = self.root_dir / "system_stats.json"
        
        # Biến theo dõi trạng thái
        self.running = False
        self.restart_count = 0
        self.app_process = None
        self.monitor_thread = None
        self.stats_thread = None
        
    def start(self):
        """Bắt đầu dịch vụ triển khai"""
        if self.running:
            logger.warning("Dịch vụ triển khai đã đang chạy!")
            return
            
        self.running = True
        logger.info("Bắt đầu dịch vụ triển khai...")
        
        # Khởi động thread giám sát
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Khởi động thread theo dõi tài nguyên
        self.stats_thread = threading.Thread(target=self._resource_monitor)
        self.stats_thread.daemon = True
        self.stats_thread.start()
        
        # Khởi động ứng dụng
        self._start_app()
        
        logger.info("Dịch vụ triển khai đã được khởi động thành công")
        
    def stop(self):
        """Dừng dịch vụ triển khai"""
        if not self.running:
            logger.warning("Dịch vụ triển khai không chạy!")
            return
            
        self.running = False
        
        # Dừng ứng dụng
        if self.app_process:
            logger.info("Dừng ứng dụng...")
            try:
                # Gửi tín hiệu TERM để dừng ứng dụng
                os.killpg(os.getpgid(self.app_process.pid), signal.SIGTERM)
                # Đợi ứng dụng dừng
                self.app_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Nếu không dừng sau 30 giây, buộc dừng
                logger.warning("Ứng dụng không dừng lại sau 30 giây, buộc dừng...")
                os.killpg(os.getpgid(self.app_process.pid), signal.SIGKILL)
            except Exception as e:
                logger.error(f"Lỗi khi dừng ứng dụng: {str(e)}")
                
        # Đợi các thread kết thúc
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        if self.stats_thread:
            self.stats_thread.join(timeout=5)
            
        logger.info("Dịch vụ triển khai đã dừng")
        
    def _start_app(self):
        """Khởi động ứng dụng bằng script startup.sh"""
        try:
            logger.info("Khởi động ứng dụng...")
            
            # Mở file log để ghi output
            log_file = open(self.app_log_file, 'a')
            
            # Chạy script khởi động
            self.app_process = subprocess.Popen(
                [str(self.startup_script)],
                stdout=log_file,
                stderr=log_file,
                shell=True,
                preexec_fn=os.setsid  # Tạo process group mới
            )
            
            logger.info(f"Ứng dụng đã khởi động với PID: {self.app_process.pid}")
            self.restart_count += 1
            
            # Cập nhật trạng thái
            self._update_status({
                "app_status": "running",
                "pid": self.app_process.pid,
                "restart_count": self.restart_count,
                "last_start": datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Lỗi khi khởi động ứng dụng: {str(e)}")
            
            # Cập nhật trạng thái
            self._update_status({
                "app_status": "error",
                "error": str(e),
                "last_error": datetime.datetime.now().isoformat()
            })
            
    def _monitor_loop(self):
        """Vòng lặp giám sát ứng dụng"""
        while self.running:
            try:
                # Kiểm tra xem ứng dụng có đang chạy không
                if self.app_process and self.app_process.poll() is not None:
                    # Ứng dụng đã dừng
                    returncode = self.app_process.returncode
                    logger.warning(f"Ứng dụng đã dừng với mã thoát: {returncode}")
                    
                    # Cập nhật trạng thái
                    self._update_status({
                        "app_status": "stopped",
                        "exit_code": returncode,
                        "last_stop": datetime.datetime.now().isoformat()
                    })
                    
                    # Khởi động lại ứng dụng
                    logger.info("Khởi động lại ứng dụng...")
                    self._start_app()
                
                # Kiểm tra kết nối đến ứng dụng
                if not self._check_app_connection() and self.app_process and self.app_process.poll() is None:
                    # Ứng dụng không phản hồi nhưng vẫn chạy
                    logger.warning("Ứng dụng không phản hồi. Khởi động lại...")
                    
                    # Cập nhật trạng thái
                    self._update_status({
                        "app_status": "unresponsive",
                        "last_unresponsive": datetime.datetime.now().isoformat()
                    })
                    
                    # Dừng ứng dụng hiện tại
                    try:
                        os.killpg(os.getpgid(self.app_process.pid), signal.SIGTERM)
                        self.app_process.wait(timeout=5)
                    except:
                        try:
                            os.killpg(os.getpgid(self.app_process.pid), signal.SIGKILL)
                        except:
                            pass
                    
                    # Khởi động lại ứng dụng
                    self._start_app()
                
                # Kiểm tra sử dụng tài nguyên
                self._check_resource_usage()
                
            except Exception as e:
                logger.error(f"Lỗi trong vòng lặp giám sát: {str(e)}")
                
                # Cập nhật trạng thái
                self._update_status({
                    "monitor_error": str(e),
                    "last_monitor_error": datetime.datetime.now().isoformat()
                })
            
            # Chờ 60 giây trước khi kiểm tra lại
            time.sleep(60)
            
    def _check_app_connection(self):
        """Kiểm tra kết nối đến ứng dụng"""
        try:
            # Sử dụng curl để kiểm tra kết nối
            result = subprocess.run(
                "curl -s -o /dev/null -w '%{http_code}' http://localhost:5000",
                shell=True,
                capture_output=True,
                text=True
            )
            
            # Lấy mã trạng thái
            status_code = result.stdout.strip()
            
            # Nếu mã trạng thái là 200, ứng dụng đang hoạt động
            is_running = status_code == "200"
            
            # Cập nhật trạng thái
            self._update_status({
                "connection_check": {
                    "status_code": status_code,
                    "is_running": is_running,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            })
            
            return is_running
            
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra kết nối: {str(e)}")
            
            # Cập nhật trạng thái
            self._update_status({
                "connection_check": {
                    "error": str(e),
                    "is_running": False,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            })
            
            return False
            
    def _check_resource_usage(self):
        """Kiểm tra sử dụng tài nguyên hệ thống"""
        try:
            # Lấy thông tin CPU và RAM
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Ghi vào log
            with open(self.resource_log_file, 'a') as f:
                timestamp = datetime.datetime.now().isoformat()
                f.write(f"{timestamp},{cpu_percent},{memory_percent},{disk_percent}\n")
            
            # Kiểm tra nếu tài nguyên quá cao
            if memory_percent > 90:
                logger.warning(f"Sử dụng RAM cao: {memory_percent}%")
                
                # Nếu RAM quá cao, khởi động lại ứng dụng
                if memory_percent > 95:
                    logger.warning("RAM quá cao (>95%), khởi động lại ứng dụng...")
                    
                    # Cập nhật trạng thái
                    self._update_status({
                        "resource_warning": "high_memory",
                        "memory_percent": memory_percent,
                        "action": "restart_app",
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                    # Dừng ứng dụng hiện tại
                    if self.app_process and self.app_process.poll() is None:
                        try:
                            os.killpg(os.getpgid(self.app_process.pid), signal.SIGTERM)
                            self.app_process.wait(timeout=5)
                        except:
                            try:
                                os.killpg(os.getpgid(self.app_process.pid), signal.SIGKILL)
                            except:
                                pass
                    
                    # Khởi động lại ứng dụng
                    self._start_app()
            
            # Cập nhật trạng thái
            self._update_status({
                "resources": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                    "timestamp": datetime.datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra tài nguyên: {str(e)}")
            
    def _resource_monitor(self):
        """Thread theo dõi tài nguyên hệ thống"""
        while self.running:
            try:
                # Lấy thông tin tài nguyên
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                # Chuẩn bị thông tin
                stats = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "cpu": {
                        "percent": cpu_percent,
                        "count": psutil.cpu_count()
                    },
                    "memory": {
                        "total_mb": memory.total / (1024 * 1024),
                        "available_mb": memory.available / (1024 * 1024),
                        "percent": memory_percent
                    },
                    "disk": {
                        "total_gb": disk.total / (1024 * 1024 * 1024),
                        "free_gb": disk.free / (1024 * 1024 * 1024),
                        "percent": disk_percent
                    },
                    "app": {
                        "running": self.app_process is not None and self.app_process.poll() is None,
                        "pid": self.app_process.pid if self.app_process and self.app_process.poll() is None else None,
                        "restart_count": self.restart_count
                    }
                }
                
                # Lưu thông tin
                with open(self.system_status_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
            except Exception as e:
                logger.error(f"Lỗi khi thu thập thông tin tài nguyên: {str(e)}")
                
            # Chờ 5 giây
            time.sleep(5)
            
    def _update_status(self, status_update):
        """Cập nhật trạng thái hệ thống"""
        try:
            # Đọc trạng thái hiện tại nếu có
            current_status = {}
            if self.system_status_file.exists():
                with open(self.system_status_file, 'r') as f:
                    current_status = json.load(f)
            
            # Cập nhật với thông tin mới
            current_status.update(status_update)
            
            # Lưu trạng thái
            with open(self.system_status_file, 'w') as f:
                json.dump(current_status, f, indent=2)
                
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật trạng thái: {str(e)}")
            
def main():
    """Hàm chính"""
    # Khởi tạo dịch vụ triển khai
    service = DeploymentService()
    
    # Xử lý tín hiệu
    def signal_handler(sig, frame):
        print(f"Nhận tín hiệu {sig}, dừng dịch vụ...")
        service.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Bắt đầu dịch vụ
        service.start()
        
        # Giữ chương trình chạy
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Dừng dịch vụ khi nhận Ctrl+C
        print("Nhận Ctrl+C, dừng dịch vụ...")
        service.stop()
        
if __name__ == "__main__":
    main()