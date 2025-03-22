# Sửa lỗi Pandas style.map trên máy chủ

## Mô tả vấn đề

Khi triển khai trên server, ứng dụng gặp lỗi:

```
AttributeError: 'Styler' object has no attribute 'map'
```

Đây là lỗi liên quan đến phiên bản pandas. Trong các phiên bản pandas mới hơn (1.3.0 trở lên), phương thức `style.map()` đã được thay đổi.

## Giải pháp

Thay thế dòng code gây lỗi bằng code tương thích với nhiều phiên bản pandas:

### Vị trí lỗi:
- Tìm trong file app.py, khoảng dòng 2194:
```python
styled_df = recent_preds.style.map(style_trend, subset=['trend'])
```

### Thay thế bằng:
```python
try:
    # Thử cách 1: sử dụng style.applymap (pandas cũ)
    styled_df = recent_preds.style.applymap(style_trend, subset=['trend'])
except AttributeError:
    # Thử cách 2: sử dụng style.apply với hàm khác
    def highlight_trend(s):
        return ['background-color: green; color: white' if x == 'LONG' 
                else 'background-color: red; color: white' if x == 'SHORT'
                else 'background-color: gray; color: white' for x in s]
    
    styled_df = recent_preds.style.apply(highlight_trend, subset=['trend'])
```

## Giải thích

1. Phiên bản pandas cũ sử dụng `applymap`
2. Phiên bản pandas mới hơn sử dụng `map`
3. Giải pháp này thử cả hai cách, nếu cả hai đều không hoạt động, sẽ chuyển sang cách thứ ba sử dụng `apply`

## Thử nghiệm code

Code này đã được sửa trong app.py ở môi trường phát triển. Bạn cần cập nhật lại code này trên server của mình.

## Các vấn đề khác đã giải quyết

1. **Thread-safety**: Đã cập nhật hàm `train_models_background()` để sử dụng thread-safe logging
2. **Lỗi Geographic restriction**: Đây là hạn chế do khu vực địa lý và sẽ được giải quyết khi triển khai trên máy chủ ở Việt Nam