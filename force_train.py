#!/usr/bin/env python3
"""
Script đặc biệt để kích hoạt huấn luyện mô hình trong ETHUSDT Dashboard.
"""
import os
import time
import json
import logging
from datetime import datetime

# Tạo tín hiệu yêu cầu huấn luyện
with open("force_training.signal", "w") as f:
    json.dump({"timestamp": datetime.now().isoformat(), "force": True}, f)
print("✅ Đã tạo tín hiệu yêu cầu huấn luyện")

# Ghi nhật ký yêu cầu huấn luyện
with open("training_logs.txt", "a") as f:
    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 🚨 YÊU CẦU HUẤN LUYỆN THỦ CÔNG - Kích hoạt trực tiếp\n")
print("✅ Đã ghi nhật ký yêu cầu huấn luyện")

print(f"⏱️  {datetime.now().strftime('%H:%M:%S')} - Đang chờ hệ thống phản hồi...")
