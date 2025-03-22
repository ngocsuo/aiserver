#!/usr/bin/env python3
"""
Script đặc biệt để theo dõi quá trình huấn luyện mô hình AI trong ETHUSDT Dashboard.
"""
import os
import time
import glob
import threading
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainingMonitor")

def tail_file(filename, n=10):
    """Đọc n dòng cuối cùng của file"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) > n else lines
    except Exception as e:
        return [f"Lỗi khi đọc file {filename}: {str(e)}"]

def check_model_files():
    """Kiểm tra các tệp mô hình đã được lưu"""
    model_dirs = ['saved_models', 'models']
    model_files = []
    
    for directory in model_dirs:
        if os.path.exists(directory):
            files = glob.glob(f"{directory}/**/*.h5", recursive=True)
            if files:
                model_files.extend(files)
                
    # Lọc các tệp mới nhất
    if model_files:
        # Sắp xếp theo thời gian tạo
        sorted_files = sorted(model_files, key=os.path.getmtime, reverse=True)
        print(f"\n=== MÔ HÌNH ĐÃ LƯU MỚI NHẤT ({len(sorted_files)} files) ===")
        for i, file in enumerate(sorted_files[:5]):  # Chỉ hiển thị 5 mô hình mới nhất
            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
            file_size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
            print(f"{i+1}. {file} - {mod_time.strftime('%H:%M:%S')} ({file_size:.2f} MB)")
    else:
        print("\n=== CHƯA TÌM THẤY TẬP TIN MÔ HÌNH ĐÃ LƯU ===")

def tail_log_file(filename, keywords=None, n=10):
    """Hiển thị n dòng cuối cùng của file log với từ khóa tìm kiếm tùy chọn"""
    lines = tail_file(filename, n=100)  # Đọc nhiều dòng hơn để lọc
    
    if keywords:
        filtered_lines = []
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in keywords):
                filtered_lines.append(line)
        lines = filtered_lines[-n:] if len(filtered_lines) > n else filtered_lines
    else:
        lines = lines[-n:]
    
    return lines

def display_logs():
    """Hiển thị logs huấn luyện"""
    # Danh sách các từ khóa liên quan đến huấn luyện
    training_keywords = [
        "train", "model", "lstm", "transformer", "cnn", "huấn luyện", 
        "accuracy", "loss", "epoch", "batch", "neural", "predict"
    ]
    
    # Hiển thị các logs huấn luyện
    training_logs = tail_log_file("training_logs.txt", training_keywords, n=15)
    print("\n=== LOGS HUẤN LUYỆN GẦN NHẤT ===")
    for line in training_logs:
        print(line.strip())
    
    # Kiểm tra các file logs khác
    log_files = ["app.log"]
    for log_file in log_files:
        if os.path.exists(log_file):
            app_logs = tail_log_file(log_file, training_keywords, n=10)
            if app_logs:
                print(f"\n=== LOGS TỪ {log_file} ===")
                for line in app_logs:
                    print(line.strip())

def check_running_processes():
    """Kiểm tra các tiến trình đang chạy liên quan đến huấn luyện"""
    try:
        import psutil
        
        python_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'python' in proc.info['name'].lower():
                cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else ""
                if any(kw in cmdline.lower() for kw in ['train', 'model', 'app.py']):
                    python_procs.append((proc.info['pid'], cmdline))
        
        if python_procs:
            print("\n=== TIẾN TRÌNH PYTHON ĐANG CHẠY ===")
            for pid, cmdline in python_procs:
                print(f"PID {pid}: {cmdline[:100]}...")
        else:
            print("\n=== KHÔNG TÌM THẤY TIẾN TRÌNH PYTHON LIÊN QUAN ===")
    except ImportError:
        print("\n=== KHÔNG THỂ KIỂM TRA TIẾN TRÌNH (psutil không được cài đặt) ===")

def monitor_loop():
    """Vòng lặp giám sát chính"""
    try:
        print("\n" + "="*60)
        print(f"BẮT ĐẦU GIÁM SÁT QUÁ TRÌNH HUẤN LUYỆN: {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        # Kiểm tra tệp tin mô hình
        check_model_files()
        
        # Hiển thị logs
        display_logs()
        
        # Kiểm tra tiến trình
        check_running_processes()
        
        print("\n" + "="*60)
        print(f"KẾT THÚC GIÁM SÁT: {datetime.now().strftime('%H:%M:%S')}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình giám sát: {str(e)}")

def main():
    """Chức năng chính của script"""
    # Gửi tín hiệu bắt đầu huấn luyện đến app.py
    try:
        with open("force_train.signal", "w") as f:
            f.write(f"Tạo tín hiệu vào {datetime.now().isoformat()}")
            print("Đã tạo tín hiệu yêu cầu huấn luyện")
    except Exception as e:
        print(f"Không thể tạo tín hiệu yêu cầu huấn luyện: {str(e)}")

    # Giám sát liên tục
    while True:
        monitor_loop()
        print("\nĐợi 30 giây trước khi kiểm tra lại...")
        time.sleep(30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng giám sát.")
