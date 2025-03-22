#!/bin/bash
# Script khởi động ETHUSDT Dashboard Server
# Tác giả: AI Assistant
# Ngày: $(date +%Y-%m-%d)

# Đường dẫn đến thư mục dự án
PROJECT_DIR=$(dirname $(readlink -f $0))/..
cd $PROJECT_DIR

# Tạo các thư mục cần thiết nếu chưa tồn tại
mkdir -p logs data saved_models

# Tạo file cấu hình Streamlit
mkdir -p .streamlit
echo '[server]
headless = true
address = "0.0.0.0"
port = 5000' > .streamlit/config.toml

# Kiểm tra xem đã cài đặt các thư viện cần thiết chưa
echo "Kiểm tra các thư viện cần thiết..."
pip install -q streamlit pandas numpy python-binance plotly requests psutil scikit-learn tensorflow twilio

# Kiểm tra kết nối Binance API
echo "Kiểm tra kết nối Binance API..."
python -c "
import os
import sys
from binance.client import Client

api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print('Lỗi: BINANCE_API_KEY hoặc BINANCE_API_SECRET không được thiết lập')
    print('Vui lòng thiết lập các biến môi trường trước khi chạy script')
    print('Ví dụ:')
    print('export BINANCE_API_KEY=your_api_key')
    print('export BINANCE_API_SECRET=your_api_secret')
    sys.exit(1)

try:
    client = Client(api_key, api_secret)
    status = client.get_system_status()
    if status['status'] == 0:
        print(f'Kết nối Binance API thành công! Trạng thái: {status[\"msg\"]}')
    else:
        print(f'Lỗi: Binance API không khả dụng. Trạng thái: {status[\"msg\"]}')
        sys.exit(1)
except Exception as e:
    print(f'Lỗi kết nối Binance API: {e}')
    sys.exit(1)
"

# Nếu không có lỗi kết nối, tiếp tục khởi động ứng dụng
if [ $? -ne 0 ]; then
    echo "Lỗi: Không thể kết nối đến Binance API. Vui lòng kiểm tra lại cấu hình API key và secret."
    exit 1
fi

# Khởi động ứng dụng
echo "Khởi động ETHUSDT Dashboard..."

# Kiểm tra xem có file run_with_monitoring.py không
if [ -f "run_with_monitoring.py" ]; then
    echo "Khởi động với monitoring..."
    nohup python run_with_monitoring.py > logs/app.log 2>&1 &
    PID=$!
    echo "Ứng dụng đã được khởi động với PID: $PID"
    echo "Bạn có thể xem log tại: logs/app.log"
    echo "Truy cập dashboard tại: http://localhost:5000"
else
    # Nếu không có file run_with_monitoring.py, chạy trực tiếp bằng streamlit
    echo "Khởi động trực tiếp bằng Streamlit..."
    nohup streamlit run app.py > logs/app.log 2>&1 &
    PID=$!
    echo "Ứng dụng đã được khởi động với PID: $PID"
    echo "Bạn có thể xem log tại: logs/app.log"
    echo "Truy cập dashboard tại: http://localhost:5000"
fi

# Ghi thông tin PID vào file để có thể dừng ứng dụng sau này
echo $PID > .pid

# Tạo script dừng ứng dụng
cat > stop.sh << 'EOF'
#!/bin/bash
if [ -f .pid ]; then
    PID=$(cat .pid)
    echo "Dừng ứng dụng với PID: $PID"
    kill -9 $PID 2>/dev/null
    rm .pid
    echo "Đã dừng ứng dụng."
else
    echo "Không tìm thấy file .pid. Có thể ứng dụng chưa được khởi động."
    # Tìm kiếm và dừng tất cả các tiến trình liên quan
    echo "Tìm và dừng tất cả các tiến trình Streamlit..."
    pkill -f "streamlit run app.py" 2>/dev/null
    pkill -f "python run_with_monitoring.py" 2>/dev/null
    echo "Đã dừng tất cả các tiến trình liên quan."
fi
EOF

chmod +x stop.sh

echo "====================================================="
echo "ETHUSDT Dashboard đã khởi động thành công!"
echo "Truy cập dashboard tại: http://server-ip:5000"
echo "Để dừng ứng dụng, chạy lệnh: ./stop.sh"
echo "====================================================="