#!/bin/bash
# Script cài đặt môi trường cho ETHUSDT Dashboard trên server mới
# Sử dụng: sudo ./prepare_server_env.sh

# Kiểm tra quyền root
if [ "$(id -u)" != "0" ]; then
   echo "Script này phải được chạy với quyền root" 1>&2
   exit 1
fi

# Thông tin cấu hình
APP_DIR="/opt/ethusdt-dashboard"
APP_USER="ethusdt"
APP_PORT=5000
LOG_DIR="$APP_DIR/logs"
DATA_DIR="$APP_DIR/data"
MODELS_DIR="$APP_DIR/saved_models"

# Thiết lập màu
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Cài đặt ETHUSDT Dashboard ===${NC}"
echo -e "${YELLOW}Bắt đầu cài đặt môi trường...${NC}"

# 1. Cập nhật hệ thống
echo -e "${GREEN}[1/9] Đang cập nhật hệ thống...${NC}"
apt update && apt upgrade -y

# 2. Cài đặt các gói phụ thuộc
echo -e "${GREEN}[2/9] Đang cài đặt các gói phụ thuộc...${NC}"
apt install -y python3 python3-pip python3-venv git build-essential libssl-dev libffi-dev python3-dev curl

# 3. Tạo thư mục ứng dụng
echo -e "${GREEN}[3/9] Đang tạo thư mục ứng dụng...${NC}"
mkdir -p $APP_DIR
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR
mkdir -p $MODELS_DIR

# 4. Tạo người dùng hệ thống
echo -e "${GREEN}[4/9] Đang tạo người dùng hệ thống...${NC}"
id -u $APP_USER &>/dev/null || useradd -r -d $APP_DIR -s /bin/bash $APP_USER

# 5. Thiết lập môi trường Python
echo -e "${GREEN}[5/9] Đang thiết lập môi trường Python...${NC}"
cd $APP_DIR
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 6. Tạo service systemd
echo -e "${GREEN}[6/9] Đang tạo service systemd...${NC}"
cat > /etc/systemd/system/ethusdt-dashboard.service << EOF
[Unit]
Description=ETHUSDT Dashboard
After=network.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/streamlit run app.py --server.port=$APP_PORT --server.address=0.0.0.0 --server.headless=true
Restart=always
StandardOutput=append:$LOG_DIR/service.log
StandardError=append:$LOG_DIR/service_error.log
Environment="PYTHONPATH=$APP_DIR"
Environment="PATH=$APP_DIR/venv/bin:$PATH"
Environment="LOG_DIR=$LOG_DIR"
Environment="DATA_DIR=$DATA_DIR"
Environment="MODELS_DIR=$MODELS_DIR"

[Install]
WantedBy=multi-user.target
EOF

# 7. Cấu hình thư mục Streamlit
echo -e "${GREEN}[7/9] Đang cấu hình Streamlit...${NC}"
mkdir -p $APP_DIR/.streamlit
cat > $APP_DIR/.streamlit/config.toml << EOF
[server]
headless = true
address = "0.0.0.0"
port = $APP_PORT
maxUploadSize = 5
maxMessageSize = 500

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans-serif"
EOF

# 8. Thiết lập quyền
echo -e "${GREEN}[8/9] Đang thiết lập quyền...${NC}"
chown -R $APP_USER:$APP_USER $APP_DIR
chmod -R 755 $APP_DIR
chmod +x $APP_DIR/venv/bin/streamlit

# 9. Tạo tệp đánh dấu hoàn thành
echo -e "${GREEN}[9/9] Hoàn thành cài đặt môi trường!${NC}"
touch $APP_DIR/.env_setup_complete

# Thông báo hoàn thành
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}Cài đặt môi trường đã hoàn tất!${NC}"
echo -e "${YELLOW}Tiếp theo:${NC}"
echo -e "1. Sao chép mã nguồn vào $APP_DIR"
echo -e "2. Cài đặt các gói phụ thuộc: cd $APP_DIR && source venv/bin/activate && pip install -r requirements_server.txt"
echo -e "3. Cấu hình API key Binance: chỉnh sửa file $APP_DIR/.env"
echo -e "4. Khởi động dịch vụ: sudo systemctl enable ethusdt-dashboard && sudo systemctl start ethusdt-dashboard"
echo -e "5. Kiểm tra trạng thái: sudo systemctl status ethusdt-dashboard"
echo -e "${GREEN}============================================${NC}"

exit 0