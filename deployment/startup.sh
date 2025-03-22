#!/bin/bash

# Script khởi động và duy trì ứng dụng ETHUSDT Dashboard
# Thực hiện sửa lỗi và áp dụng các bản vá trước khi khởi động

# Đặt thư mục làm việc là thư mục gốc của dự án
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

# Hàm ghi log
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ROOT_DIR/deployment/deploy.log"
}

# Kiểm tra và tạo các thư mục cần thiết
mkdir -p "$ROOT_DIR/data"
mkdir -p "$ROOT_DIR/logs"
mkdir -p "$ROOT_DIR/deployment/logs"
mkdir -p "$ROOT_DIR/saved_models"

# Đặt biến môi trường
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# Áp dụng các bản vá
log "Áp dụng bản vá sửa lỗi feature_engineering..."
python feature_engineering_fix.py

# Kiểm tra xem có lỗi dữ liệu không
log "Kiểm tra và sửa lỗi dữ liệu..."
python -c "from utils.data_fix import run_data_fix; run_data_fix()"

# Khởi động ứng dụng chính
log "Khởi động ứng dụng chính..."
streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true 