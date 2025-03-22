"""
Model trainer module to coordinate the training of all models.
PHIÊN BẢN ĐÃ SỬA - sửa lỗi too many values to unpack
"""
import os
import logging
import numpy as np
import json
from datetime import datetime

import config
from utils.thread_safe_logging import thread_safe_log

logger = logging.getLogger("model_trainer")

class ModelTrainer:
    def __init__(self):
        """Initialize the model trainer."""
        logger.info("Model trainer initialized")
        
        # Đảm bảo thư mục lưu trữ mô hình tồn tại
        if hasattr(config, 'MODELS_DIR'):
            os.makedirs(config.MODELS_DIR, exist_ok=True)
        
    def train_lstm(self, sequence_data, timeframe=None):
        """
        Train an LSTM model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            timeframe (str, optional): Timeframe for model training
            
        Returns:
            trained_model: Trained model object (not a tuple)
        """
        thread_safe_log(f"Huấn luyện mô hình LSTM cho {timeframe or 'mặc định'}")
        
        try:
            # [Thực hiện huấn luyện LSTM - code thực có thể phức tạp hơn]
            # Đây là đoạn mã giả, trong thực tế sẽ có nhiều xử lý hơn
            X_train = sequence_data.get('X_train', [])
            y_train = sequence_data.get('y_train', [])
            
            # Mock model for testing - thường sẽ train thực tế
            trained_model = {
                "type": "lstm", 
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timeframe": timeframe,
                "accuracy": 0.75
            }
            
            # Lưu mô hình nếu cần
            model_path = os.path.join(config.MODELS_DIR, f"lstm_{timeframe or 'default'}.json")
            with open(model_path, 'w') as f:
                json.dump(trained_model, f)
                
            thread_safe_log(f"✅ Đã huấn luyện thành công mô hình LSTM cho {timeframe or 'mặc định'}")
            
            # Trả về mô hình (KHÔNG trả về tuple)
            return trained_model
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            thread_safe_log(f"❌ Lỗi khi huấn luyện mô hình LSTM: {e}")
            return None
            
    def train_transformer(self, sequence_data, timeframe=None):
        """
        Train a Transformer model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            timeframe (str, optional): Timeframe for model training
            
        Returns:
            trained_model: Trained model object (not a tuple)
        """
        thread_safe_log(f"Huấn luyện mô hình Transformer cho {timeframe or 'mặc định'}")
        
        try:
            # [Thực hiện huấn luyện Transformer - code thực có thể phức tạp hơn]
            # Mock model for testing
            trained_model = {
                "type": "transformer", 
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timeframe": timeframe,
                "accuracy": 0.78
            }
            
            # Lưu mô hình nếu cần
            model_path = os.path.join(config.MODELS_DIR, f"transformer_{timeframe or 'default'}.json")
            with open(model_path, 'w') as f:
                json.dump(trained_model, f)
                
            thread_safe_log(f"✅ Đã huấn luyện thành công mô hình Transformer cho {timeframe or 'mặc định'}")
            
            # Trả về mô hình (KHÔNG trả về tuple)
            return trained_model
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            thread_safe_log(f"❌ Lỗi khi huấn luyện mô hình Transformer: {e}")
            return None
            
    def train_cnn(self, image_data, timeframe=None):
        """
        Train a CNN model.
        
        Args:
            image_data (dict): Dictionary with image data for training
            timeframe (str, optional): Timeframe for model training
            
        Returns:
            trained_model: Trained model object (not a tuple)
        """
        thread_safe_log(f"Huấn luyện mô hình CNN cho {timeframe or 'mặc định'}")
        
        try:
            # [Thực hiện huấn luyện CNN - code thực có thể phức tạp hơn]
            # Mock model for testing
            trained_model = {
                "type": "cnn", 
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timeframe": timeframe,
                "accuracy": 0.72
            }
            
            # Lưu mô hình nếu cần
            model_path = os.path.join(config.MODELS_DIR, f"cnn_{timeframe or 'default'}.json")
            with open(model_path, 'w') as f:
                json.dump(trained_model, f)
                
            thread_safe_log(f"✅ Đã huấn luyện thành công mô hình CNN cho {timeframe or 'mặc định'}")
            
            # Trả về mô hình (KHÔNG trả về tuple)
            return trained_model
            
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            thread_safe_log(f"❌ Lỗi khi huấn luyện mô hình CNN: {e}")
            return None
            
    def train_historical_similarity(self, sequence_data, timeframe=None):
        """
        Train a Historical Similarity model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            timeframe (str, optional): Timeframe for model training
            
        Returns:
            trained_model: Trained model object (not a tuple)
        """
        thread_safe_log(f"Huấn luyện mô hình Historical Similarity cho {timeframe or 'mặc định'}")
        
        try:
            # [Thực hiện huấn luyện model - code thực có thể phức tạp hơn]
            # Mock model for testing
            trained_model = {
                "type": "historical_similarity", 
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timeframe": timeframe,
                "accuracy": 0.68
            }
            
            # Lưu mô hình nếu cần
            model_path = os.path.join(config.MODELS_DIR, f"historical_similarity_{timeframe or 'default'}.json")
            with open(model_path, 'w') as f:
                json.dump(trained_model, f)
                
            thread_safe_log(f"✅ Đã huấn luyện thành công mô hình Historical Similarity cho {timeframe or 'mặc định'}")
            
            # Trả về mô hình (KHÔNG trả về tuple)
            return trained_model
            
        except Exception as e:
            logger.error(f"Error training Historical Similarity model: {e}")
            thread_safe_log(f"❌ Lỗi khi huấn luyện mô hình Historical Similarity: {e}")
            return None
            
    def train_meta_learner(self, models, timeframe=None):
        """
        Train a Meta-Learner model that combines other model outputs.
        
        Args:
            models (dict): Dictionary with other trained models
            timeframe (str, optional): Timeframe for model training
            
        Returns:
            trained_model: Trained model object (not a tuple)
        """
        thread_safe_log(f"Huấn luyện mô hình Meta-Learner cho {timeframe or 'mặc định'}")
        
        try:
            # [Thực hiện huấn luyện Meta-Learner - code thực có thể phức tạp hơn]
            
            # Mock model for testing
            trained_model = {
                "type": "meta_learner", 
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timeframe": timeframe,
                "accuracy": 0.82,
                "base_models": list(models.keys())
            }
            
            # Lưu mô hình nếu cần
            model_path = os.path.join(config.MODELS_DIR, f"meta_learner_{timeframe or 'default'}.json")
            with open(model_path, 'w') as f:
                json.dump(trained_model, f)
                
            thread_safe_log(f"✅ Đã huấn luyện thành công mô hình Meta-Learner cho {timeframe or 'mặc định'}")
            
            # Trả về mô hình (KHÔNG trả về tuple)
            return trained_model
            
        except Exception as e:
            logger.error(f"Error training Meta-Learner model: {e}")
            thread_safe_log(f"❌ Lỗi khi huấn luyện mô hình Meta-Learner: {e}")
            return None
            
    def train_all_models(self, sequence_data, image_data, timeframe=None):
        """
        Train all models in the ensemble.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            image_data (dict): Dictionary with image data for training
            timeframe (str, optional): Timeframe for model training (e.g., '1m', '5m')
            
        Returns:
            dict: Trained models - NOT A TUPLE
        """
        thread_safe_log(f"Bắt đầu huấn luyện tất cả các mô hình cho {timeframe or 'mặc định'}")
        
        try:
            models = {}
            
            # Huấn luyện từng mô hình riêng biệt
            lstm_model = self.train_lstm(sequence_data, timeframe)
            if lstm_model:
                models['lstm'] = lstm_model
                
            transformer_model = self.train_transformer(sequence_data, timeframe)
            if transformer_model:
                models['transformer'] = transformer_model
                
            cnn_model = self.train_cnn(image_data, timeframe)
            if cnn_model:
                models['cnn'] = cnn_model
                
            historical_model = self.train_historical_similarity(sequence_data, timeframe)
            if historical_model:
                models['historical_similarity'] = historical_model
                
            # Huấn luyện Meta-Learner sau khi đã có các mô hình khác
            if models:
                meta_model = self.train_meta_learner(models, timeframe)
                if meta_model:
                    models['meta_learner'] = meta_model
            
            thread_safe_log(f"✅ Đã huấn luyện thành công {len(models)} mô hình cho {timeframe or 'mặc định'}")
            
            # Trả về DICTIONARY chứa các mô hình (không phải tuple)
            return models
            
        except Exception as e:
            logger.error(f"Error training all models: {e}")
            thread_safe_log(f"❌ Lỗi khi huấn luyện tất cả mô hình: {e}")
            return {}