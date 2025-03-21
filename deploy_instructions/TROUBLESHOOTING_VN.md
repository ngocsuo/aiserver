# Hướng Dẫn Khắc Phục Sự Cố

Tài liệu này cung cấp các hướng dẫn để giải quyết các vấn đề phổ biến khi triển khai và vận hành hệ thống dự đoán ETHUSDT.

## Vấn Đề Kết Nối Binance API

### Lỗi "Service unavailable from a restricted location"

**Vấn đề:** Binance chặn truy cập từ một số địa điểm địa lý nhất định.

**Giải pháp:**
1. Đảm bảo VPS/máy chủ của bạn được đặt tại quốc gia không bị Binance chặn.
2. Nếu đang test trên máy chủ bị hạn chế, xem xét chuyển sang nhà cung cấp VPS khác:
   - Vultr, DigitalOcean, Linode, OVH, Hetzner - chọn khu vực châu Á như Singapore, Nhật Bản, hoặc Hàn Quốc
   - Hoặc các nhà cung cấp VPS Việt Nam cho độ trễ thấp nhất

### Lỗi "Invalid API Key" hoặc "No API Keys"

**Vấn đề:** API key không được cấu hình đúng hoặc không tồn tại.

**Giải pháp:**
1. Kiểm tra API key và secret trong file `.env` hoặc biến môi trường:
   ```bash
   cat /opt/ethusdt_predictor/.env
   ```
   
2. Đảm bảo key được tạo từ tài khoản Binance của bạn:
   - Đăng nhập vào Binance
   - Vào API Management
   - Tạo API key mới với quyền đọc dữ liệu (và quyền giao dịch nếu sử dụng chức năng giao dịch tự động)

3. Đảm bảo định dạng đúng trong file `.env`:
   ```
   BINANCE_API_KEY=your_key_here_without_quotes
   BINANCE_API_SECRET=your_secret_here_without_quotes
   ```

### Lỗi "IP Restriction/Whitelist"

**Vấn đề:** API key chỉ được phép sử dụng từ các IP cụ thể.

**Giải pháp:**
1. Thêm IP của VPS/máy chủ vào danh sách IP được phép trong cài đặt API Binance.
2. Hoặc tạo API key mới mà không giới hạn IP (cẩn thận với bảo mật).

## Vấn Đề Khi Khởi Động Ứng Dụng

### ModuleNotFoundError: Không tìm thấy module

**Vấn đề:** Thiếu thư viện Python cần thiết.

**Giải pháp:**
1. Cài đặt thư viện bị thiếu:
   ```bash
   source /opt/ethusdt_predictor/venv/bin/activate
   pip install tên-thư-viện
   ```

2. Cài đặt tất cả thư viện từ file requirements:
   ```bash
   pip install -r /opt/ethusdt_predictor/deploy_instructions/requirements.txt
   ```

### Lỗi "Address already in use" (Địa chỉ đã được sử dụng)

**Vấn đề:** Cổng 5000 đã được sử dụng bởi ứng dụng khác.

**Giải pháp:**
1. Tìm và dừng quy trình đang sử dụng cổng:
   ```bash
   sudo lsof -i :5000
   sudo kill -9 PID  # Thay PID bằng ID quy trình tìm thấy
   ```

2. Thay đổi cổng trong cấu hình Streamlit:
   ```bash
   nano /opt/ethusdt_predictor/.streamlit/config.toml
   ```
   Thay đổi `port = 5000` thành một cổng khác như `port = 8501`

### Lỗi Vượt Quá Giới Hạn Bộ Nhớ

**Vấn đề:** Ứng dụng sử dụng quá nhiều RAM và bị kill bởi hệ thống.

**Giải pháp:**
1. Tăng RAM cho VPS/máy chủ nếu có thể.
2. Điều chỉnh tham số trong `config.py` để giảm tiêu thụ bộ nhớ:
   ```bash
   nano /opt/ethusdt_predictor/config.py
   ```
   - Giảm `LOOKBACK_PERIODS` xuống (ví dụ: 1000 thay vì 5000)
   - Tăng `UPDATE_INTERVAL` lên (ví dụ: 60 thay vì 20)
   - Giảm `TRAINING_BATCH_SIZE` xuống

3. Tăng swapfile nếu RAM giới hạn:
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

## Vấn Đề Về Hiệu Suất

### Ứng Dụng Chạy Chậm

**Vấn đề:** Hệ thống phản hồi chậm hoặc xử lý dữ liệu mất nhiều thời gian.

**Giải pháp:**
1. Tối ưu hóa tham số trong `config.py`:
   - Giảm số lượng timeframes phụ nếu không cần thiết
   - Giảm số lượng indicators phức tạp

2. Tăng tài nguyên hệ thống:
   - Sử dụng CPU có hiệu suất cao hơn
   - Tăng RAM
   - Sử dụng ổ SSD thay vì HDD

3. Tối ưu hóa việc huấn luyện mô hình:
   - Điều chỉnh `TRAINING_FREQUENCY` để giảm tần suất huấn luyện lại
   - Giảm `EPOCHS` trong quá trình huấn luyện

### Sử Dụng CPU Cao

**Vấn đề:** Ứng dụng sử dụng CPU quá nhiều.

**Giải pháp:**
1. Giới hạn số lượng worker trong cấu hình Streamlit:
   ```bash
   nano /opt/ethusdt_predictor/.streamlit/config.toml
   ```
   Thêm:
   ```
   [server]
   maxUploadSize = 10
   maxMessageSize = 50
   workers = 1
   ```

2. Giảm tần suất cập nhật dữ liệu và tính toán:
   - Tăng `UPDATE_INTERVAL` trong `config.py`
   - Giảm độ phức tạp của các chỉ báo kỹ thuật

## Vấn Đề Lưu Trữ Và Dữ Liệu

### Lỗi "No space left on device" (Hết không gian lưu trữ)

**Vấn đề:** Ổ đĩa đã đầy, thường do dữ liệu lịch sử hoặc log quá lớn.

**Giải pháp:**
1. Xóa dữ liệu cache cũ:
   ```bash
   rm -rf /opt/ethusdt_predictor/saved_models/data_cache/*
   ```

2. Xóa các nhật ký cũ:
   ```bash
   sudo find /var/log -type f -name "*.gz" -delete
   sudo find /var/log -type f -name "*.1" -delete
   ```

3. Cấu hình tự động xóa dữ liệu cũ:
   Thêm vào crontab:
   ```
   0 1 * * * find /opt/ethusdt_predictor/saved_models/data_cache -type f -mtime +30 -delete
   ```

### Lỗi "Permission denied" (Từ chối quyền truy cập)

**Vấn đề:** Thiếu quyền truy cập vào file hoặc thư mục.

**Giải pháp:**
1. Đặt quyền sở hữu cho toàn bộ thư mục ứng dụng:
   ```bash
   sudo chown -R your-username:your-username /opt/ethusdt_predictor
   ```

2. Đặt quyền thích hợp:
   ```bash
   sudo chmod -R 755 /opt/ethusdt_predictor
   sudo chmod 600 /opt/ethusdt_predictor/.env  # Bảo vệ API keys
   ```

## Vấn Đề Về Mô Hình Và Dự Đoán

### Lỗi "Model File Not Found" (Không tìm thấy file mô hình)

**Vấn đề:** Các file mô hình đã huấn luyện không tồn tại hoặc bị hỏng.

**Giải pháp:**
1. Khởi động ứng dụng và để nó tự huấn luyện mô hình mới:
   ```bash
   cd /opt/ethusdt_predictor
   source venv/bin/activate
   streamlit run app.py
   ```

2. Kiểm tra và đảm bảo các thư mục cần thiết tồn tại:
   ```bash
   mkdir -p /opt/ethusdt_predictor/saved_models
   ```

### Lỗi "Failed to Train Models" (Huấn luyện mô hình thất bại)

**Vấn đề:** Quá trình huấn luyện mô hình bị lỗi.

**Giải pháp:**
1. Kiểm tra logs để xác định nguyên nhân cụ thể:
   ```bash
   sudo tail -f /var/log/ethusdt_predictor.err.log
   ```

2. Cấu hình giảm độ phức tạp của mô hình trong `config.py`:
   - Giảm `SEQUENCE_LENGTH`
   - Giảm số lượng layer trong mô hình nếu có thể

## Vấn Đề Về Mạng

### Lỗi Kết Nối Không Ổn Định

**Vấn đề:** Kết nối đến API Binance không ổn định, gây ra các vấn đề về dữ liệu.

**Giải pháp:**
1. Cải thiện cơ chế retry trong code:
   - Đã được tích hợp nhưng có thể cần điều chỉnh số lần thử lại trong class BinanceDataCollector

2. Kiểm tra độ ổn định mạng:
   ```bash
   ping api.binance.com
   mtr api.binance.com
   ```

3. Thay đổi nhà cung cấp DNS:
   ```bash
   sudo nano /etc/resolv.conf
   ```
   Thêm:
   ```
   nameserver 1.1.1.1
   nameserver 8.8.8.8
   ```

### Lỗi "Network Timeout" (Hết thời gian chờ mạng)

**Vấn đề:** Yêu cầu API mất quá nhiều thời gian.

**Giải pháp:**
1. Tăng timeout trong class BinanceDataCollector nếu cần.
2. Sử dụng VPS có kết nối mạng tốt hơn/gần hơn với các máy chủ Binance.

## Vấn Đề Giao Dịch Tự Động

### Lỗi "Insufficient Balance" (Số dư không đủ)

**Vấn đề:** Không đủ tiền trong tài khoản Binance Futures.

**Giải pháp:**
1. Nạp thêm USDT vào tài khoản Binance Futures.
2. Giảm kích thước giao dịch trong cấu hình giao dịch.

### Lỗi "Order Failed" (Đặt lệnh thất bại)

**Vấn đề:** Không thể đặt lệnh trên Binance vì các lý do khác nhau.

**Giải pháp:**
1. Kiểm tra logs để xác định nguyên nhân cụ thể.
2. Đảm bảo đòn bẩy được đặt đúng cho cặp giao dịch:
   ```python
   trading_manager.set_leverage("ETHUSDT", 10)  # Đặt đòn bẩy 10x
   ```
3. Kiểm tra quyền API (API key cần có quyền giao dịch).

## Liên Hệ Hỗ Trợ

Nếu bạn không thể giải quyết vấn đề sau khi thử các phương pháp trên, vui lòng liên hệ:

- Email: your-email@example.com
- Telegram: @your_telegram_username

Khi yêu cầu hỗ trợ, vui lòng cung cấp:
1. Thông báo lỗi đầy đủ
2. Nhật ký (logs) liên quan
3. Phiên bản hệ thống và môi trường (hệ điều hành, RAM, CPU, v.v.)
4. Các bước bạn đã thử để khắc phục vấn đề