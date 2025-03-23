"""
Mã sửa lỗi cho train_models_background
"""

import os
import time
import threading
import traceback
import json
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import logging

# Thêm các thư viện cần thiết
from utils.thread_safe_logging import thread_safe_log, read_logs_from_file

def train_models_background():
    """Hàm huấn luyện chạy trong thread riêng biệt"""
    try:
        thread_safe_log("Bắt đầu huấn luyện mô hình...")
        
        # Đảm bảo mọi thư mục cần thiết đều tồn tại
        os.makedirs("saved_models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Thiết lập trạng thái huấn luyện
        training_status = {
            "is_training": True,
            "progress": 0,
            "message": "Đang chuẩn bị dữ liệu...",
            "start_time": time.time(),
            "last_update": time.time(),
            "errors": []
        }
        
        # Lưu trạng thái vào file
        with open("training_status.json", "w") as f:
            json.dump(training_status, f)
            
        thread_safe_log("Đang tải dữ liệu...")
        
        # Cập nhật trạng thái
        training_status["message"] = "Đang tải dữ liệu..."
        training_status["progress"] = 10
        with open("training_status.json", "w") as f:
            json.dump(training_status, f)
            
        # Thử kết nối đến Binance API và lấy dữ liệu
        try:
            thread_safe_log("Kết nối đến Binance API...")
            # Import các module cần thiết ở đây để tránh lỗi import cycle
            import config
            from utils.data_collector_factory import create_data_collector
            
            data_collector = create_data_collector()
            if data_collector is None:
                raise Exception("Không thể tạo data collector")
                
            # Lấy dữ liệu lịch sử
            thread_safe_log("Lấy dữ liệu lịch sử cho mô hình...")
            historical_data = data_collector.collect_historical_data(
                symbol=config.SYMBOL,
                timeframe=config.PRIMARY_TIMEFRAME,
                limit=config.LOOKBACK_PERIODS
            )
            
            if historical_data is None or len(historical_data) < 100:
                raise Exception(f"Không đủ dữ liệu lịch sử: {len(historical_data) if historical_data is not None else 0} mẫu")
            
            thread_safe_log(f"Đã tải {len(historical_data)} dòng dữ liệu")
            
            # Cập nhật trạng thái
            training_status["message"] = f"Đã tải {len(historical_data)} dòng dữ liệu"
            training_status["progress"] = 20
            with open("training_status.json", "w") as f:
                json.dump(training_status, f)
            
            # Xử lý đặc trưng
            thread_safe_log("Xử lý đặc trưng...")
            from utils.feature_engineering import FeatureEngineer
            
            feature_engineer = FeatureEngineer()
            try:
                processed_data = feature_engineer.process_features(
                    historical_data,
                    add_basic=True,
                    add_technical=True,
                    add_pattern=False,
                    add_labels=True,
                    normalize=True
                )
                
                if processed_data is None or len(processed_data) < 100:
                    raise Exception(f"Xử lý đặc trưng thất bại: {len(processed_data) if processed_data is not None else 0} mẫu")
                
                thread_safe_log(f"Đã xử lý đặc trưng: {len(processed_data)} mẫu với {len(processed_data.columns)} đặc trưng")
            except Exception as fe_error:
                thread_safe_log(f"Lỗi xử lý đặc trưng: {str(fe_error)}")
                # Thử sửa lỗi
                thread_safe_log("Thử phương pháp xử lý đặc trưng đơn giản hơn...")
                
                # Chỉ tính toán các đặc trưng cơ bản
                processed_data = historical_data.copy()
                processed_data['returns'] = processed_data['close'].pct_change()
                processed_data['target'] = processed_data['returns'].shift(-1) > 0
                processed_data = processed_data.dropna()
            
            # Cập nhật trạng thái
            training_status["message"] = "Đang chuẩn bị dữ liệu huấn luyện..."
            training_status["progress"] = 30
            with open("training_status.json", "w") as f:
                json.dump(training_status, f)
            
            # Chuẩn bị dữ liệu cho huấn luyện
            thread_safe_log("Chuẩn bị dữ liệu huấn luyện...")
            from utils.data_processor import DataProcessor
            
            data_processor = DataProcessor()
            sequence_data, image_data = data_processor.prepare_training_data(processed_data)
            
            if sequence_data is None or 'x_train' not in sequence_data or len(sequence_data['x_train']) < 10:
                raise Exception("Chuẩn bị dữ liệu huấn luyện thất bại")
            
            thread_safe_log(f"Đã chuẩn bị {len(sequence_data['x_train'])} mẫu huấn luyện")
            
            # Cập nhật trạng thái
            training_status["message"] = "Đang huấn luyện mô hình..."
            training_status["progress"] = 40
            with open("training_status.json", "w") as f:
                json.dump(training_status, f)
            
            # Huấn luyện mô hình
            thread_safe_log("Bắt đầu huấn luyện mô hình...")
            from models.model_trainer import ModelTrainer
            
            model_trainer = ModelTrainer()
            
            # Lưu tạm dữ liệu để debug nếu cần
            sequence_data_file = "data/sequence_data.json"
            try:
                with open(sequence_data_file, "w") as f:
                    # Chỉ lưu metadata vì dữ liệu có thể rất lớn
                    metadata = {
                        "x_train_shape": list(sequence_data["x_train"].shape),
                        "y_train_shape": list(sequence_data["y_train"].shape),
                        "x_val_shape": list(sequence_data["x_val"].shape),
                        "y_val_shape": list(sequence_data["y_val"].shape),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    json.dump(metadata, f)
            except Exception as e:
                thread_safe_log(f"Không thể lưu metadata dữ liệu: {str(e)}")
            
            # Sử dụng try/except để xử lý lỗi phiên bản model_trainer khác nhau
            try:
                # Phiên bản model_trainer mới trả về tuple (model, history)
                # Kiểm tra xem model trainer trả về gì
                # Thử huấn luyện một mô hình nhỏ để kiểm tra kiểu dữ liệu trả về
                test_result = model_trainer.train_lstm(sequence_data)
                
                # Nếu đây là tuple, chúng ta đang sử dụng phiên bản mới
                if isinstance(test_result, tuple):
                    thread_safe_log("Phát hiện model_trainer trả về tuple, đang điều chỉnh...")
                    models = {}
                    # Huấn luyện từng mô hình và chỉ lấy mô hình, bỏ qua history
                    lstm_model, _ = model_trainer.train_lstm(sequence_data)
                    models["lstm"] = lstm_model
                    
                    # Cập nhật trạng thái
                    training_status["message"] = "Đang huấn luyện mô hình Transformer..."
                    training_status["progress"] = 60
                    with open("training_status.json", "w") as f:
                        json.dump(training_status, f)
                    
                    transformer_model, _ = model_trainer.train_transformer(sequence_data)
                    models["transformer"] = transformer_model
                    
                    # Cập nhật trạng thái
                    training_status["message"] = "Đang huấn luyện mô hình CNN..."
                    training_status["progress"] = 70
                    with open("training_status.json", "w") as f:
                        json.dump(training_status, f)
                    
                    cnn_model, _ = model_trainer.train_cnn(image_data)
                    models["cnn"] = cnn_model
                    
                    # Cập nhật trạng thái
                    training_status["message"] = "Đang huấn luyện mô hình Meta..."
                    training_status["progress"] = 80
                    with open("training_status.json", "w") as f:
                        json.dump(training_status, f)
                    
                    meta_model, _ = model_trainer.train_meta_learner(sequence_data, image_data)
                    models["meta"] = meta_model
                else:
                    # Phiên bản cũ trả về model trực tiếp
                    thread_safe_log("Sử dụng model_trainer trả về model trực tiếp...")
                    models = model_trainer.train_all_models(sequence_data, image_data)
            except Exception as model_error:
                thread_safe_log(f"Lỗi khi huấn luyện mô hình: {str(model_error)}")
                # Thử cách khác - sử dụng train_all_models trực tiếp
                thread_safe_log("Thử huấn luyện bằng phương thức train_all_models...")
                try:
                    models = model_trainer.train_all_models(sequence_data, image_data)
                except Exception as all_models_error:
                    thread_safe_log(f"Lỗi khi gọi train_all_models: {str(all_models_error)}")
                    raise Exception(f"Không thể huấn luyện mô hình: {str(model_error)}\nLỗi thứ hai: {str(all_models_error)}")
            
            # Cập nhật trạng thái
            training_status["message"] = "Đang lưu mô hình..."
            training_status["progress"] = 90
            with open("training_status.json", "w") as f:
                json.dump(training_status, f)
            
            # Lưu thông tin về mô hình
            model_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "data_points": len(processed_data),
                "features": list(processed_data.columns),
                "models": list(models.keys()),
                "training_samples": len(sequence_data["x_train"]),
                "validation_samples": len(sequence_data["x_val"]),
                "test_samples": len(sequence_data["x_test"]) if "x_test" in sequence_data else 0
            }
            
            with open("saved_models/model_info.json", "w") as f:
                json.dump(model_info, f)
            
            thread_safe_log("Huấn luyện mô hình hoàn tất!")
            
            # Cập nhật trạng thái hoàn tất
            training_status["is_training"] = False
            training_status["progress"] = 100
            training_status["message"] = "Huấn luyện mô hình hoàn tất!"
            training_status["end_time"] = time.time()
            training_status["duration"] = training_status["end_time"] - training_status["start_time"]
            with open("training_status.json", "w") as f:
                json.dump(training_status, f)
                
        except Exception as e:
            thread_safe_log(f"Lỗi khi huấn luyện: {str(e)}")
            thread_safe_log(traceback.format_exc())
            
            # Cập nhật trạng thái lỗi
            training_status["is_training"] = False
            training_status["progress"] = -1
            training_status["message"] = f"Lỗi huấn luyện: {str(e)}"
            training_status["errors"].append({
                "timestamp": time.time(),
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            training_status["end_time"] = time.time()
            training_status["duration"] = training_status["end_time"] - training_status["start_time"]
            with open("training_status.json", "w") as f:
                json.dump(training_status, f)
                
    except Exception as outer_e:
        thread_safe_log(f"Lỗi ngoài cùng khi huấn luyện: {str(outer_e)}")
        thread_safe_log(traceback.format_exc())
        
        # Đảm bảo luôn cập nhật trạng thái thất bại
        try:
            with open("training_status.json", "w") as f:
                json.dump({
                    "is_training": False,
                    "progress": -1,
                    "message": f"Lỗi nghiêm trọng: {str(outer_e)}",
                    "errors": [{
                        "timestamp": time.time(),
                        "error": str(outer_e),
                        "traceback": traceback.format_exc()
                    }],
                    "end_time": time.time()
                }, f)
        except:
            pass


"""
Để sử dụng mã này:

1. Đảm bảo thư mục utils có file thread_safe_logging.py
2. Thay thế TOÀN BỘ hàm train_models_background() cũ
3. Tạo file training_logs.txt trống: 
   touch training_logs.txt && chmod 666 training_logs.txt
4. Khởi động lại ứng dụng:
   streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
"""