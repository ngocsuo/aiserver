# ETHUSDT Dashboard - Hệ thống dự đoán thông minh

Hệ thống dự đoán thông minh cho cặp tiền ETHUSDT trên sàn Binance, sử dụng nhiều mô hình AI (LSTM, Transformer, CNN, và các mô hình khác).

## Tính năng

- **Thu thập dữ liệu thời gian thực** từ Binance API với xử lý lỗi nâng cao
- **Huấn luyện mô hình tự động** mỗi 30 phút để đảm bảo dự đoán cập nhật
- **Giao diện Dashboard** trực quan với Streamlit
- **Nhiều khung thời gian**: Phân tích đồng thời dữ liệu 1 phút và 5 phút
- **Đa dạng mô hình AI**: Kết hợp nhiều mô hình để dự đoán chính xác hơn
- **Hỗ trợ proxy**: Kết nối đến Binance API từ mọi khu vực địa lý
- **Giám sát hệ thống**: Theo dõi tài nguyên và tự động khắc phục sự cố

## Cài đặt nhanh

1. Đảm bảo các biến môi trường đã được thiết lập:
   - `BINANCE_API_KEY` và `BINANCE_API_SECRET`

2. Khởi động ứng dụng sạch với script tự động:

```bash
python run_clean.py
```

Script này sẽ:
- Tối ưu hóa log hiện có
- Áp dụng các bản vá lỗi
- Khởi động hệ thống với giám sát liên tục

## Cài đặt thủ công

1. Cài đặt các phụ thuộc:

```bash
pip install -r requirements.txt
```

2. Chạy ứng dụng:

```bash
streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
```

## Sử dụng nâng cao

### Chạy với giám sát hệ thống

```bash
python run_with_monitoring.py --mode service
```

### Tối ưu hóa log

```bash
python deployment/optimize_logs.py --all
```

### Huấn luyện mô hình trực tiếp

```bash
python direct_train.py
```

## Cấu trúc dự án

```
├── app.py                        # Ứng dụng Streamlit chính
├── config.py                     # Cấu hình hệ thống
├── data/                         # Dữ liệu lịch sử và cache
├── deployment/                   # Công cụ triển khai và giám sát
├── logs/                         # File log
├── models/                       # Mô hình AI
├── prediction/                   # Module dự đoán
├── saved_models/                 # Mô hình đã lưu
├── run_clean.py                  # Script chạy sạch
├── run_with_monitoring.py        # Script giám sát
└── utils/                        # Tiện ích
```

## Xử lý sự cố

Nếu gặp phải vấn đề, kiểm tra:

1. **Lỗi kết nối**:
   ```bash
   python test_binance_connection.py
   ```

2. **Lỗi mô hình**:
   ```bash
   python view_training_progress.py
   ```

3. **Lỗi dữ liệu**:
   ```bash
   python -c "from utils.data_fix import run_data_fix; run_data_fix()"
   ```

## Phát triển

Khi phát triển thêm tính năng mới, vui lòng tuân thủ:

1. Viết docstring cho tất cả các hàm và lớp mới
2. Thêm xử lý lỗi đầy đủ
3. Giữ code trong các module riêng biệt 
4. Cập nhật tài liệu nếu cần

## Giấy phép

© 2025. Bản quyền thuộc về tác giả.