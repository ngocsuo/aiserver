# Sửa lỗi tham số cho hàm train_all_models

## Vấn đề:
Trong file `models/continuous_trainer.py`, có đoạn:
```python
models = self.model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
```

Nhưng trong `models/model_trainer.py`, hàm được định nghĩa chỉ nhận 2 tham số:
```python
def train_all_models(self, sequence_data, image_data):
```

## Lỗi hiện tại:
```
Error during training execution: too many values to unpack (expected 2)
```

## Cách sửa:

### Phương án 1: Cập nhật hàm train_all_models để nhận thêm tham số timeframe
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
        logger.info(f"Training all models for timeframe: {timeframe}")
        
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
        
        logger.info("All models trained successfully")
        
        return self.models
        
    except Exception as e:
        logger.error(f"Error training all models: {e}")
        return None
```

### Phương án 2: Sửa đoạn gọi hàm trong continuous_trainer.py
```python
# Thay dòng này:
models = self.model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)

# Bằng dòng này:
models = self.model_trainer.train_all_models(sequence_data, image_data)
```

## Hướng dẫn triển khai:

Phương án 1 là tốt nhất vì nó duy trì tính năng truyền thông tin timeframe vào hàm train_all_models. Bạn nên sửa cả hai file để đảm bảo tính nhất quán.

1. Sửa file `models/model_trainer.py` để hàm train_all_models nhận thêm tham số timeframe
2. Nếu không thể sửa file model_trainer.py vì lý do nào đó, hãy sửa file continuous_trainer.py để bỏ tham số timeframe

## Lưu ý khi triển khai:

Đảm bảo gỡ bỏ hoặc thay thế mọi lệnh gọi hàm train_all_models không phù hợp với định nghĩa hàm mới. Tìm kiếm thông qua lệnh:

```bash
grep -r "train_all_models" --include="*.py" .
```

để xác định tất cả các lệnh gọi hàm này trong dự án.