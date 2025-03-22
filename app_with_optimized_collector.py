"""
File chuyển hướng cho mục đích triển khai (deployment)
Chỉ đơn giản là nạp module app.py để chạy ứng dụng chính
"""

# Đoạn mã này chỉ đơn giản là import app.py chính
import sys
import os

print("INFO: Chuyển hướng từ app_with_optimized_collector.py sang app.py")

# Đặt đường dẫn để import app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import toàn bộ mã từ app.py
try:
    from app import *
    
    # Nếu app.py không có entry point rõ ràng, ta cần gọi hàm main từ app.py
    if __name__ == "__main__":
        print("INFO: Chạy ứng dụng từ app.py thay vì app_with_optimized_collector.py")
        # Nếu app.py có một hàm main, hãy gọi nó ở đây
        # app_main()
except Exception as e:
    print(f"Lỗi khi import app.py: {e}")
    sys.exit(1)