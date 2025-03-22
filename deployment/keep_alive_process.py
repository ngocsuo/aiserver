"""
Giám sát và duy trì hoạt động liên tục của hệ thống ETHUSDT Dashboard.

Mô-đun này tạo ra một tiến trình độc lập để theo dõi trạng thái hệ thống và khởi động lại
các thành phần bị tắt. Nó cũng ghi lại thông tin tài nguyên và lỗi để phân tích sau này.
"""

import os
import time
import signal
import logging
import threading
import subprocess
import psutil
import datetime
import json
from pathlib import Path

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deployment/deploy.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DeploymentMonitor")

class DeploymentMonitor:
    """Lớp giám sát và duy trì hoạt động liên tục của ứng dụng"""
    
    def __init__(self, app_command="streamlit run app.py --server.port 5000 --server.address=0.0.0.0 --server.headless=true", 
                 check_interval=60):
        """
        Khởi tạo monitor
        
        Args:
            app_command (str): Lệnh để khởi động ứng dụng
            check_interval (int): Khoảng thời gian giữa các lần kiểm tra (giây)
        """
        self.app_command = app_command
        self.check_interval = check_interval
        self.running = False
        self.app_process = None
        self.monitor_thread = None
        self.stats_thread = None
        self.deploy_dir = Path("deployment")
        self.deploy_dir.mkdir(exist_ok=True)
        
        # Tạo thư mục logs nếu chưa tồn tại
        self.logs_dir = self.deploy_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Tập tin chứa trạng thái hệ thống
        self.system_stats_file = self.deploy_dir / "system_stats.json"
        
        # Đặt đường dẫn đầy đủ cho tệp log
        self.app_log_file = self.logs_dir / "app.log"
        self.system_log_file = self.logs_dir / "system.log"
        
        # Vì tệp tin log chiếm nhiều dung lượng nên chỉ giữ lại log của 7 ngày gần nhất
        self.max_log_days = 7
        
    def start(self):
        """Bắt đầu giám sát và khởi động ứng dụng"""
        if self.running:
            logger.warning("Monitor đã đang chạy!")
            return
            
        self.running = True
        logger.info("Bắt đầu giám sát hệ thống...")
        
        # Dọn dẹp các log cũ
        self._cleanup_old_logs()
        
        # Khởi động thread giám sát
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Khởi động thread thu thập thống kê
        self.stats_thread = threading.Thread(target=self._collect_stats)
        self.stats_thread.daemon = True
        self.stats_thread.start()
        
        # Khởi động ứng dụng
        self._start_app()
        
        logger.info("Hệ thống giám sát đã được khởi động thành công")
        
    def stop(self):
        """Dừng giám sát và ứng dụng"""
        if not self.running:
            logger.warning("Monitor không chạy!")
            return
            
        self.running = False
        
        # Dừng ứng dụng
        if self.app_process:
            try:
                logger.info("Dừng ứng dụng...")
                os.killpg(os.getpgid(self.app_process.pid), signal.SIGTERM)
                self.app_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Ứng dụng không dừng lại, buộc dừng...")
                os.killpg(os.getpgid(self.app_process.pid), signal.SIGKILL)
            except Exception as e:
                logger.error(f"Lỗi khi dừng ứng dụng: {str(e)}")
                
        # Đợi các thread kết thúc
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        if self.stats_thread:
            self.stats_thread.join(timeout=5)
            
        logger.info("Hệ thống giám sát đã dừng")
        
    def _start_app(self):
        """Khởi động ứng dụng với xử lý lỗi"""
        try:
            logger.info(f"Khởi động ứng dụng với lệnh: {self.app_command}")
            
            # Mở tệp log để ghi output
            app_log = open(self.app_log_file, 'a')
            
            # Khởi động ứng dụng và tạo process group mới
            self.app_process = subprocess.Popen(
                self.app_command,
                stdout=app_log,
                stderr=app_log,
                shell=True,
                preexec_fn=os.setsid
            )
            
            logger.info(f"Ứng dụng đã khởi động với PID: {self.app_process.pid}")
            
        except Exception as e:
            logger.error(f"Lỗi khi khởi động ứng dụng: {str(e)}")
            
    def _monitor_loop(self):
        """Vòng lặp giám sát ứng dụng"""
        while self.running:
            try:
                # Kiểm tra xem ứng dụng có đang chạy không
                if self.app_process and self.app_process.poll() is not None:
                    logger.warning(f"Ứng dụng đã dừng với mã thoát: {self.app_process.returncode}")
                    
                    # Khởi động lại ứng dụng
                    self._start_app()
                
                # Kiểm tra kết nối đến ứng dụng
                if not self._check_app_connection():
                    logger.warning("Không thể kết nối đến ứng dụng. Khởi động lại...")
                    
                    # Dừng ứng dụng hiện tại
                    if self.app_process:
                        try:
                            os.killpg(os.getpgid(self.app_process.pid), signal.SIGTERM)
                            self.app_process.wait(timeout=5)
                        except:
                            # Nếu không thể dừng, buộc dừng
                            try:
                                os.killpg(os.getpgid(self.app_process.pid), signal.SIGKILL)
                            except:
                                pass
                    
                    # Khởi động lại ứng dụng
                    self._start_app()
                
                # Kiểm tra tài nguyên hệ thống
                self._check_system_resources()
                
            except Exception as e:
                logger.error(f"Lỗi trong vòng lặp giám sát: {str(e)}")
            
            # Chờ đến khoảng thời gian kiểm tra tiếp theo
            time.sleep(self.check_interval)
            
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
            
            # Nếu mã trạng thái là 200, ứng dụng đang chạy
            return result.stdout.strip() == "200"
            
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra kết nối: {str(e)}")
            return False
            
    def _check_system_resources(self):
        """Kiểm tra tài nguyên hệ thống và thực hiện các hành động cần thiết"""
        try:
            # Lấy thông tin CPU và RAM
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Ghi thông tin vào log
            with open(self.system_log_file, 'a') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} - CPU: {cpu_percent}% | RAM: {memory_percent}%\n")
            
            # Nếu tài nguyên quá cao, ghi cảnh báo
            if cpu_percent > 90 or memory_percent > 90:
                logger.warning(f"Tài nguyên hệ thống cao: CPU {cpu_percent}% | RAM {memory_percent}%")
                
                # Nếu RAM quá cao, cân nhắc khởi động lại ứng dụng
                if memory_percent > 95:
                    logger.warning("RAM quá cao (>95%), khởi động lại ứng dụng để giải phóng bộ nhớ")
                    
                    # Dừng ứng dụng hiện tại
                    if self.app_process:
                        try:
                            os.killpg(os.getpgid(self.app_process.pid), signal.SIGTERM)
                            self.app_process.wait(timeout=5)
                        except:
                            # Nếu không thể dừng, buộc dừng
                            try:
                                os.killpg(os.getpgid(self.app_process.pid), signal.SIGKILL)
                            except:
                                pass
                    
                    # Khởi động lại ứng dụng
                    self._start_app()
            
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra tài nguyên hệ thống: {str(e)}")
            
    def _collect_stats(self):
        """Thu thập thống kê hệ thống định kỳ"""
        while self.running:
            try:
                # Thu thập thông tin hệ thống
                stats = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'cpu': {
                        'percent': psutil.cpu_percent(interval=1),
                        'count': psutil.cpu_count()
                    },
                    'memory': {
                        'total': psutil.virtual_memory().total,
                        'available': psutil.virtual_memory().available,
                        'percent': psutil.virtual_memory().percent
                    },
                    'disk': {
                        'total': psutil.disk_usage('/').total,
                        'free': psutil.disk_usage('/').free,
                        'percent': psutil.disk_usage('/').percent
                    },
                    'process': {
                        'running': self.app_process is not None and self.app_process.poll() is None,
                        'pid': self.app_process.pid if self.app_process else None,
                        'uptime': self._get_process_uptime()
                    }
                }
                
                # Lưu thông tin vào tệp
                with open(self.system_stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
            except Exception as e:
                logger.error(f"Lỗi khi thu thập thống kê: {str(e)}")
                
            # Cập nhật mỗi 5 giây
            time.sleep(5)
            
    def _get_process_uptime(self):
        """Lấy thời gian chạy của tiến trình ứng dụng (nếu đang chạy)"""
        if not self.app_process or self.app_process.poll() is not None:
            return None
            
        try:
            # Lấy tiến trình và thời gian bắt đầu
            process = psutil.Process(self.app_process.pid)
            start_time = process.create_time()
            
            # Tính thời gian chạy theo giây
            uptime_seconds = time.time() - start_time
            
            return uptime_seconds
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy thời gian chạy: {str(e)}")
            return None
            
    def _cleanup_old_logs(self):
        """Dọn dẹp các tệp log cũ"""
        try:
            # Lấy danh sách các tệp log
            log_files = list(self.logs_dir.glob("*.log.*"))
            
            # Lấy ngày hiện tại
            today = datetime.datetime.now()
            
            # Xóa các tệp log cũ hơn max_log_days ngày
            for log_file in log_files:
                try:
                    # Lấy thời gian sửa đổi của tệp
                    mtime = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
                    
                    # Nếu tệp cũ hơn max_log_days ngày, xóa nó
                    if (today - mtime).days > self.max_log_days:
                        logger.info(f"Xóa tệp log cũ: {log_file}")
                        log_file.unlink()
                except Exception as e:
                    logger.error(f"Lỗi khi xóa tệp log cũ {log_file}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Lỗi khi dọn dẹp các tệp log cũ: {str(e)}")
            
def main():
    """Hàm chính để khởi động hệ thống giám sát"""
    # Khởi tạo đối tượng giám sát
    monitor = DeploymentMonitor()
    
    # Đăng ký xử lý tín hiệu để dừng giám sát khi nhận SIGTERM hoặc SIGINT
    def signal_handler(sig, frame):
        logger.info(f"Nhận tín hiệu {sig}, dừng giám sát...")
        monitor.stop()
        
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Bắt đầu giám sát
        monitor.start()
        
        # Giữ cho chương trình chạy
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Dừng giám sát khi nhận Ctrl+C
        logger.info("Nhận Ctrl+C, dừng giám sát...")
        monitor.stop()
        
if __name__ == "__main__":
    main()