# Sửa lỗi "too many values to unpack (expected 2)" trong hàm train_all_models

## Vấn đề:

Hiện tại, trong file `models/continuous_trainer.py` có dòng gọi:
```python
models = self.model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
```

Nhưng trong file `models/model_trainer.py`, hàm train_all_models được định nghĩa chỉ nhận 2 tham số:
```python
def train_all_models(self, sequence_data, image_data):
```

## Giải pháp (Cách 2):

Sửa file `models/model_trainer.py` để hàm train_all_models nhận thêm tham số timeframe:

```python
def train_all_models(self, sequence_data, image_data, timeframe=None):
    """
    Train all models in the ensemble.
    
    Args:
        sequence_data (dict): Dictionary with sequence data for training
        image_data (dict): Dictionary with image data for training
        timeframe (str, optional): Timeframe for model training (e.g., '1m', '5m')
        
    Returns:
        dict: Trained models
    """
    try:
        if timeframe:
            logger.info(f"Training all models for timeframe: {timeframe}")
        else:
            logger.info("Training all models")
        
        # Train each model type
        lstm_model, lstm_history = self.train_lstm(sequence_data)
        transformer_model, transformer_history = self.train_transformer(sequence_data)
        cnn_model, cnn_history = self.train_cnn(image_data)
        historical_model, _ = self.train_historical_similarity(sequence_data)
        meta_model, _ = self.train_meta_learner(sequence_data, image_data)
        
        # Store models and training histories
        self.models = {
            'lstm': lstm_model,
            'transformer': transformer_model,
            'cnn': cnn_model,
            'historical_similarity': historical_model,
            'meta_learner': meta_model
        }
        
        self.histories = {
            'lstm': lstm_history,
            'transformer': transformer_history,
            'cnn': cnn_history
        }
        
        # Lưu thông tin về timeframe
        if timeframe:
            self.timeframe = timeframe
        
        logger.info("All models trained successfully")
        
        return self.models
        
    except Exception as e:
        logger.error(f"Error training all models: {e}")
        return None
```

Những thay đổi chính:
1. Thêm tham số `timeframe=None` vào định nghĩa hàm
2. Thêm tham số vào phần docstring
3. Log thông tin timeframe nếu có
4. Lưu thông tin timeframe vào self.timeframe

## Các bước triển khai:

1. Tìm file `models/model_trainer.py` trên server của bạn
2. Tìm hàm `train_all_models` (khoảng dòng 200-245)
3. Thay thế toàn bộ hàm bằng đoạn code đã sửa ở trên
4. Lưu file và khởi động lại ứng dụng

## Kiểm tra:

Sau khi triển khai, hãy kiểm tra logs để đảm bảo lỗi "too many values to unpack (expected 2)" không còn xuất hiện nữa.

## Tương thích ngược:

Giải pháp này vẫn hoạt động tốt với các phần code hiện tại không truyền tham số timeframe (như trong _train_with_all_data) vì timeframe được đặt là tham số tùy chọn và mặc định là None.