"""
Module giám sát tài nguyên hệ thống (CPU, RAM, Disk)
"""
import os
import psutil
import time
import threading
import json
from datetime import datetime

class SystemMonitor:
    """
    Lớp giám sát tài nguyên hệ thống và lưu thông tin định kỳ
    """
    def __init__(self, update_interval=5):
        """
        Khởi tạo monitor với khoảng thời gian cập nhật
        
        Args:
            update_interval (int): Khoảng thời gian cập nhật (giây)
        """
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.system_stats = {
            'cpu': {
                'percent': 0,
                'count': psutil.cpu_count(),
                'stats': []  # Lịch sử CPU usage
            },
            'memory': {
                'percent': 0,
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used,
                'stats': []  # Lịch sử memory usage
            },
            'disk': {
                'percent': 0,
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free,
                'used': psutil.disk_usage('/').used
            },
            'last_update': datetime.now().isoformat()
        }
        
        # Giới hạn số lượng điểm dữ liệu lưu trong lịch sử
        self.max_history_points = 60  # 5 phút lịch sử với update 5s
        
    def start(self):
        """Bắt đầu giám sát trong thread riêng"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop)
            self.thread.daemon = True  # Thread sẽ tự động kết thúc khi chương trình chính thoát
            self.thread.start()
            
    def stop(self):
        """Dừng giám sát"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Vòng lặp giám sát tài nguyên"""
        while self.running:
            try:
                self._update_stats()
                self._save_stats()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Lỗi khi giám sát tài nguyên: {str(e)}")
                time.sleep(self.update_interval)
                
    def _update_stats(self):
        """Cập nhật thông tin tài nguyên hệ thống"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_stats['cpu']['percent'] = cpu_percent
        self.system_stats['cpu']['stats'].append({
            'time': datetime.now().isoformat(), 
            'value': cpu_percent
        })
        
        # Giới hạn lịch sử
        if len(self.system_stats['cpu']['stats']) > self.max_history_points:
            self.system_stats['cpu']['stats'] = self.system_stats['cpu']['stats'][-self.max_history_points:]
            
        # Memory
        memory = psutil.virtual_memory()
        self.system_stats['memory']['percent'] = memory.percent
        self.system_stats['memory']['available'] = memory.available
        self.system_stats['memory']['used'] = memory.used
        self.system_stats['memory']['stats'].append({
            'time': datetime.now().isoformat(), 
            'value': memory.percent
        })
        
        # Giới hạn lịch sử
        if len(self.system_stats['memory']['stats']) > self.max_history_points:
            self.system_stats['memory']['stats'] = self.system_stats['memory']['stats'][-self.max_history_points:]
            
        # Disk
        disk = psutil.disk_usage('/')
        self.system_stats['disk']['percent'] = disk.percent
        self.system_stats['disk']['free'] = disk.free
        self.system_stats['disk']['used'] = disk.used
        
        # Thời gian cập nhật
        self.system_stats['last_update'] = datetime.now().isoformat()
        
    def _save_stats(self):
        """Lưu thông tin tài nguyên vào file"""
        try:
            # Chỉ lưu thông tin cơ bản, không bao gồm lịch sử chi tiết để tránh file quá lớn
            basic_stats = {
                'cpu': {
                    'percent': self.system_stats['cpu']['percent'],
                    'count': self.system_stats['cpu']['count']
                },
                'memory': {
                    'percent': self.system_stats['memory']['percent'],
                    'total': self.system_stats['memory']['total'],
                    'available': self.system_stats['memory']['available'],
                    'used': self.system_stats['memory']['used']
                },
                'disk': self.system_stats['disk'],
                'last_update': self.system_stats['last_update']
            }
            
            with open('system_stats.json', 'w') as f:
                json.dump(basic_stats, f)
        except Exception as e:
            print(f"Lỗi khi lưu thông tin tài nguyên: {str(e)}")
            
    def get_stats(self):
        """
        Lấy thông tin tài nguyên hệ thống hiện tại
        
        Returns:
            dict: Thông tin tài nguyên
        """
        return self.system_stats

    def get_formatted_stats(self):
        """
        Lấy thông tin tài nguyên đã được định dạng để hiển thị
        
        Returns:
            dict: Thông tin tài nguyên đã định dạng
        """
        stats = self.system_stats
        
        # Chuyển đổi byte sang MB hoặc GB
        memory_total_gb = stats['memory']['total'] / (1024 ** 3)
        memory_used_gb = stats['memory']['used'] / (1024 ** 3)
        memory_available_gb = stats['memory']['available'] / (1024 ** 3)
        
        disk_total_gb = stats['disk']['total'] / (1024 ** 3)
        disk_used_gb = stats['disk']['used'] / (1024 ** 3)
        disk_free_gb = stats['disk']['free'] / (1024 ** 3)
        
        # Thời gian cập nhật
        last_update = datetime.fromisoformat(stats['last_update'])
        formatted_time = last_update.strftime("%H:%M:%S")
        
        return {
            'cpu': {
                'percent': stats['cpu']['percent'],
                'count': stats['cpu']['count'],
                'text': f"{stats['cpu']['percent']}% (of {stats['cpu']['count']} cores)",
                'stats': stats['cpu']['stats']
            },
            'memory': {
                'percent': stats['memory']['percent'],
                'total': memory_total_gb,
                'used': memory_used_gb,
                'available': memory_available_gb,
                'text': f"{stats['memory']['percent']}% ({memory_used_gb:.1f}/{memory_total_gb:.1f} GB)",
                'stats': stats['memory']['stats']
            },
            'disk': {
                'percent': stats['disk']['percent'],
                'total': disk_total_gb,
                'used': disk_used_gb,
                'free': disk_free_gb,
                'text': f"{stats['disk']['percent']}% ({disk_used_gb:.1f}/{disk_total_gb:.1f} GB)"
            },
            'last_update': formatted_time
        }

# Singleton instance
_system_monitor = None

def get_system_monitor():
    """
    Lấy instance của SystemMonitor (singleton pattern)
    
    Returns:
        SystemMonitor: Instance của SystemMonitor
    """
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor

def start_monitoring():
    """
    Bắt đầu giám sát tài nguyên hệ thống
    """
    monitor = get_system_monitor()
    monitor.start()
    return monitor

def stop_monitoring():
    """
    Dừng giám sát tài nguyên hệ thống
    """
    global _system_monitor
    if _system_monitor:
        _system_monitor.stop()
        _system_monitor = None

def get_system_stats():
    """
    Lấy thông tin tài nguyên hệ thống hiện tại
    
    Returns:
        dict: Thông tin tài nguyên đã định dạng
    """
    monitor = get_system_monitor()
    return monitor.get_formatted_stats()