#!/bin/bash
# Script khởi động ETHUSDT Dashboard chỉ với một lệnh
# Tác giả: AI Assistant
# Ngày: $(date +%Y-%m-%d)

# Thiết lập biến môi trường
export PYTHONPATH=$(pwd)
export BINANCE_API_KEY="${BINANCE_API_KEY:-your_api_key_here}"
export BINANCE_API_SECRET="${BINANCE_API_SECRET:-your_api_secret_here}"

# Tạo các thư mục cần thiết
mkdir -p logs data saved_models

# Tạo file cấu hình Streamlit
mkdir -p .streamlit
cat > .streamlit/config.toml << EOF
[server]
headless = true
address = "0.0.0.0"
port = 5000
EOF

# Cài đặt các thư viện cần thiết
pip install -q streamlit pandas numpy python-binance plotly requests psutil scikit-learn tensorflow twilio

# Chạy ứng dụng
echo "Khởi động ETHUSDT Dashboard..."
if [ -f "run_clean.py" ]; then
    # Nếu có file run_clean.py, sử dụng nó để chạy ứng dụng với giám sát
    python run_clean.py --mode service
else
    # Nếu không, chạy trực tiếp bằng Streamlit
    streamlit run app.py
fi