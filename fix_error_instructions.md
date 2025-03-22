# Hướng dẫn sửa lỗi "too many values to unpack (expected 2)" trong continuous_trainer.py

## Vấn đề

Lỗi xuất hiện là do trong phương thức `_train_for_timeframe()` của file `continuous_trainer.py` đang cố gắng unpack kết quả từ hàm `train_all_models()` thành hai biến (`models` và `histories`), trong khi hàm `train_all_models()` trong `model_trainer.py` chỉ trả về một giá trị (chỉ trả về `models`).

Chi tiết dòng code gây lỗi:

```python
# Đoạn code gây lỗi
models, histories = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
```

## Cách sửa lỗi

### Bước 1: Tìm phương thức `_train_for_timeframe` trong file continuous_trainer.py

Mở file `continuous_trainer.py` và tìm phương thức `_train_for_timeframe`.

### Bước 2: Thực hiện thay đổi sau

Thay đổi dòng:

```python
models, histories = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
```

Thành:

```python
models = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
```

### Bước 3: Lưu file và khởi động lại ứng dụng

Sau khi lưu file, khởi động lại ứng dụng bằng lệnh:

```bash
systemctl restart ai_trading_system
```

hoặc phương thức khởi động lại tương ứng trên hệ thống của bạn.

## Giải thích kỹ thuật

Khi xem xét mã nguồn của hàm `train_all_models()` trong file `model_trainer.py`, chúng ta thấy hàm này chỉ trả về một giá trị, đó là từ điển `self.models`. Nó không trả về hai giá trị như cách gọi trong `continuous_trainer.py` đang mong đợi.

Điều này là nguyên nhân gây ra lỗi `too many values to unpack (expected 2)`. Khi một hàm chỉ trả về một giá trị nhưng code đang cố gắng unpack thành hai giá trị, Python sẽ báo lỗi này.

Lưu ý rằng trong `model_trainer.py`, các thông tin về histories đã được lưu vào thuộc tính `self.histories` của đối tượng `ModelTrainer`, nhưng không được trả về từ hàm `train_all_models()`.

## Kiểm tra sau khi sửa lỗi

Sau khi sửa lỗi, bạn nên kiểm tra logs để đảm bảo rằng lỗi đã được khắc phục:

```bash
tail -f /var/log/ai_trading/continuous_trainer.log
```

Nếu không còn thấy lỗi `too many values to unpack (expected 2)` nữa, tức là sửa lỗi đã thành công.