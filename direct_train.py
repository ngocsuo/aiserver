#!/usr/bin/env python3
"""
Script trực tiếp để kích hoạt huấn luyện mô hình không thông qua Streamlit
"""
import os
import sys
import time
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('direct_training.log')
    ]
)
logger = logging.getLogger("DirectTrainer")

# Import các module cần thiết
import pandas as pd
import numpy as np

# Thêm trực tiếp hàm truy xuất thành phần huấn luyện
def train_directly():
    """Huấn luyện trực tiếp các mô hình mà không thông qua Streamlit"""
    logger.info("Bắt đầu huấn luyện trực tiếp...")
    
    # Import các module cần thiết
    from utils.data_collector import create_data_collector
    from utils.data_processor import DataProcessor
    from models.model_trainer import ModelTrainer
    import config
    
    # Ghi nhật ký huấn luyện
    def log_message(message):
        """Ghi nhật ký huấn luyện vào file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("training_logs.txt", "a") as f:
            f.write(f"{timestamp} - {message}\n")
        logger.info(message)
    
    try:
        log_message("🚀 BẮT ĐẦU HUẤN LUYỆN TRỰC TIẾP")
        
        # Tạo các đối tượng cần thiết
        log_message("Tạo data collector, processor và model trainer...")
        data_collector = create_data_collector()
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        
        # Thu thập dữ liệu
        log_message("Thu thập dữ liệu lịch sử...")
        if hasattr(config, 'HISTORICAL_START_DATE') and config.HISTORICAL_START_DATE:
            log_message(f"Sử dụng ngày bắt đầu: {config.HISTORICAL_START_DATE}")
            data = data_collector.collect_historical_data(
                timeframe=config.PRIMARY_TIMEFRAME,
                start_date=config.HISTORICAL_START_DATE
            )
        else:
            log_message(f"Sử dụng {config.LOOKBACK_PERIODS} nến gần nhất")
            data = data_collector.collect_historical_data(
                timeframe=config.PRIMARY_TIMEFRAME,
                limit=config.LOOKBACK_PERIODS
            )
        
        if data is None or len(data) == 0:
            log_message("❌ KHÔNG THỂ thu thập dữ liệu cho huấn luyện")
            return False
            
        log_message(f"✅ Đã thu thập {len(data)} nến dữ liệu")
        
        # Xử lý dữ liệu
        log_message("Xử lý dữ liệu...")
        processed_data = data_processor.process_data(data)
        log_message(f"✅ Đã xử lý {len(processed_data)} mẫu dữ liệu")
        
        # Chuẩn bị dữ liệu cho các mô hình khác nhau
        log_message("Chuẩn bị dữ liệu huấn luyện...")
        sequence_data = data_processor.prepare_sequence_data(processed_data)
        image_data = data_processor.prepare_image_data(processed_data)
        log_message("✅ Đã chuẩn bị dữ liệu cho các mô hình")
        
        # Huấn luyện tất cả các mô hình
        log_message("🔄 Bắt đầu huấn luyện các mô hình...")
        
        # Đếm số đặc trưng
        feature_count = processed_data.shape[1] - 1  # Trừ cột target
        
        # Huấn luyện mô hình LSTM
        log_message("Huấn luyện mô hình LSTM...")
        lstm_model = model_trainer.train_lstm(sequence_data)
        log_message("✅ Đã huấn luyện mô hình LSTM")
        
        # Huấn luyện mô hình Transformer
        log_message("Huấn luyện mô hình Transformer...")
        transformer_model = model_trainer.train_transformer(sequence_data)
        log_message("✅ Đã huấn luyện mô hình Transformer")
        
        # Huấn luyện mô hình CNN
        log_message("Huấn luyện mô hình CNN...")
        cnn_model = model_trainer.train_cnn(image_data)
        log_message("✅ Đã huấn luyện mô hình CNN")
        
        # Huấn luyện mô hình Historical Similarity
        log_message("Huấn luyện mô hình Historical Similarity...")
        historical_model = model_trainer.train_historical_similarity(sequence_data)
        log_message("✅ Đã huấn luyện mô hình Historical Similarity")
        
        # Huấn luyện Meta-Learner
        log_message("Huấn luyện mô hình Meta-Learner...")
        meta_model = model_trainer.train_meta_learner(sequence_data, image_data)
        log_message("✅ Đã huấn luyện mô hình Meta-Learner")
        
        log_message("✅ Huấn luyện thành công tất cả các mô hình!")
        
        # Lưu models vào file
        models = {
            'lstm': lstm_model,
            'transformer': transformer_model,
            'cnn': cnn_model,
            'historical_similarity': historical_model,
            'meta_learner': meta_model
        }
        
        # Lưu models vào thư mục
        import os
        import pickle
        import json
        
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
            
        with open("saved_models/models.pkl", "wb") as f:
            pickle.dump(models, f)
            
        # Lưu metadata về quá trình huấn luyện
        training_status = {
            'last_training_time': datetime.now().isoformat(),
            'data_points': len(data),
            'feature_count': feature_count,
            'training_samples': len(processed_data),
            'model_version': config.MODEL_VERSION if hasattr(config, 'MODEL_VERSION') else "1.0.0",
            'training_complete': True
        }
        
        with open("saved_models/training_status.json", "w") as f:
            json.dump(training_status, f)
            
        log_message("✅ Đã lưu tất cả mô hình vào saved_models/models.pkl")
        log_message("✅ ĐÃ HOÀN THÀNH QUÁ TRÌNH HUẤN LUYỆN TRỰC TIẾP")
        
        return True
            
    except Exception as e:
        log_message(f"❌ Lỗi khi huấn luyện: {str(e)}")
        import traceback
        log_message(f"Chi tiết lỗi: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("=========================================")
    print("KHỞI ĐỘNG HUẤN LUYỆN MÔ HÌNH TRỰC TIẾP")
    print("=========================================")
    
    # Ghi nhật ký khởi động
    with open("training_logs.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 🚀 BẮT ĐẦU HUẤN LUYỆN TRỰC TIẾP TỪ SCRIPT\n")
    
    # Huấn luyện
    result = train_directly()
    
    # Hiển thị kết quả
    if result:
        print("✅ HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    else:
        print("❌ HUẤN LUYỆN THẤT BẠI!")
        
    print("=========================================")
