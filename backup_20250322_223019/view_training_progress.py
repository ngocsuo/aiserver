"""
Đoạn mã kiểm tra tiến trình huấn luyện mô hình
Sử dụng để theo dõi trạng thái huấn luyện và xác định có đang bị treo hay không
"""
import os
import time
import pandas as pd

def check_training_progress():
    """
    Kiểm tra tiến trình huấn luyện dựa trên file log
    """
    log_file = "training_logs.txt"
    
    if not os.path.exists(log_file):
        print(f"Không tìm thấy file log {log_file}")
        return
    
    # Đọc file log
    with open(log_file, "r") as f:
        logs = f.readlines()
    
    # Hiển thị 20 dòng log cuối cùng
    print(f"=== 20 dòng log huấn luyện mới nhất ===")
    for log in logs[-20:]:
        print(log.strip())
    
    # Kiểm tra nếu file đã được cập nhật trong 5 phút gần đây
    last_modified = os.path.getmtime(log_file)
    current_time = time.time()
    time_diff = current_time - last_modified
    
    if time_diff < 300:  # 5 phút
        print(f"\nFile log được cập nhật cách đây {int(time_diff)} giây")
        print("⚠ Quá trình huấn luyện ĐANG TIẾP TỤC")
    else:
        print(f"\nFile log chưa được cập nhật trong {int(time_diff/60)} phút")
        print("❌ Quá trình huấn luyện có thể bị TREO hoặc đã DỪNG")
    
    # Kiểm tra thư mục models để xem có mô hình nào đã được lưu chưa
    model_dir = os.path.join("saved_models")  # hoặc config.MODELS_DIR nếu có
    if os.path.exists(model_dir):
        models = os.listdir(model_dir)
        if models:
            print(f"\nĐã tìm thấy {len(models)} mô hình đã lưu:")
            for model in models:
                model_time = os.path.getmtime(os.path.join(model_dir, model))
                model_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_time))
                print(f"- {model} (cập nhật: {model_time_str})")

def check_memory_usage():
    """
    Kiểm tra việc sử dụng bộ nhớ của quá trình huấn luyện (chỉ hoạt động trên Linux)
    """
    try:
        import psutil
        
        # Lấy thông tin sử dụng bộ nhớ
        memory = psutil.virtual_memory()
        
        print(f"\n=== Thông tin sử dụng bộ nhớ ===")
        print(f"Tổng bộ nhớ: {memory.total / (1024**3):.2f} GB")
        print(f"Bộ nhớ đã sử dụng: {memory.used / (1024**3):.2f} GB ({memory.percent}%)")
        print(f"Bộ nhớ khả dụng: {memory.available / (1024**3):.2f} GB")
        
        # Tìm kiếm các tiến trình Python đang chạy
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
            if 'python' in proc.info['name'].lower():
                try:
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else '',
                        'memory': proc.info['memory_info'].rss / (1024**2)  # MB
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        if python_processes:
            print(f"\n=== Các tiến trình Python đang chạy ===")
            for proc in sorted(python_processes, key=lambda x: x['memory'], reverse=True):
                print(f"PID: {proc['pid']}, Bộ nhớ: {proc['memory']:.2f} MB")
                print(f"Command: {proc['cmdline'][:100]}..." if len(proc['cmdline']) > 100 else f"Command: {proc['cmdline']}")
                print("-" * 50)
    
    except ImportError:
        print("Không thể kiểm tra bộ nhớ - psutil không được cài đặt")
        print("Cài đặt psutil: pip install psutil")

if __name__ == "__main__":
    check_training_progress()
    check_memory_usage()