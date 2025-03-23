#!/bin/bash
# Script chuẩn bị môi trường máy chủ cho ETHUSDT Dashboard
# Phiên bản: 1.0.0

# Định nghĩa màu sắc
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Kiểm tra quyền root
if [ "$(id -u)" -ne 0 ]; then
    echo -e "${RED}Lỗi: Script này cần chạy với quyền root.${NC}"
    echo "Vui lòng chạy: sudo $0"
    exit 1
fi

# Định nghĩa biến
APP_NAME="ethusdt-dashboard"
APP_DIR="/opt/$APP_NAME"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"
PYTHON_VERSION="3.10"
PYTHON_ALTERNATIVE="python3.10"
GIT_REPO="https://github.com/yourusername/$APP_NAME.git"  # Thay thế bằng repo thực tế

# Kiểm tra distro
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
    echo -e "${BLUE}Phát hiện hệ điều hành: $PRETTY_NAME${NC}"
else
    echo -e "${RED}Không thể xác định hệ điều hành.${NC}"
    exit 1
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}     Cài đặt ETHUSDT Dashboard Server      ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Cập nhật hệ thống
echo -e "${BLUE}1. Cập nhật hệ thống...${NC}"
apt update && apt upgrade -y

# Cài đặt các gói phụ thuộc
echo -e "${BLUE}2. Cài đặt các gói phụ thuộc...${NC}"
apt install -y build-essential git curl wget unzip software-properties-common \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python3-pip \
    supervisor nginx libpq-dev

# Cài đặt Python phiên bản mới nhất
echo -e "${BLUE}3. Cài đặt Python...${NC}"
if [ "$OS" = "ubuntu" ]; then
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update
    apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
    
    # Đặt Python mặc định
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}
elif [ "$OS" = "debian" ]; then
    apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
fi

# Kiểm tra phiên bản Python
python3 --version

# Tạo thư mục ứng dụng
echo -e "${BLUE}4. Tạo thư mục ứng dụng...${NC}"
mkdir -p $APP_DIR
mkdir -p $APP_DIR/data
mkdir -p $APP_DIR/logs
mkdir -p $APP_DIR/saved_models

# Tạo môi trường ảo
echo -e "${BLUE}5. Tạo môi trường ảo Python...${NC}"
python3 -m venv $APP_DIR/venv

# Tạo file systemd service
echo -e "${BLUE}6. Tạo file systemd service...${NC}"
cat > $SERVICE_FILE << EOL
[Unit]
Description=ETHUSDT Dashboard Service
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$APP_DIR/venv/bin/python -m streamlit run $APP_DIR/app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=5
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=$APP_NAME

[Install]
WantedBy=multi-user.target
EOL

# Tạo script khởi động
echo -e "${BLUE}7. Tạo script khởi động...${NC}"
cat > $APP_DIR/start.sh << EOL
#!/bin/bash
cd $APP_DIR
source venv/bin/activate
streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
EOL
chmod +x $APP_DIR/start.sh

# Tạo script cập nhật
echo -e "${BLUE}8. Tạo script cập nhật...${NC}"
cat > $APP_DIR/update.sh << EOL
#!/bin/bash
cd $APP_DIR
git pull
source venv/bin/activate
pip install -r requirements.txt
systemctl restart $APP_NAME
EOL
chmod +x $APP_DIR/update.sh

# Bật systemd service
echo -e "${BLUE}9. Bật systemd service...${NC}"
systemctl daemon-reload
systemctl enable $APP_NAME

# Thông tin hoàn tất
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Cài đặt máy chủ hoàn tất!              ${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "${BLUE}Môi trường đã được chuẩn bị xong.${NC}"
echo -e "${BLUE}Bây giờ bạn cần:${NC}"
echo -e "1. Sao chép mã nguồn vào $APP_DIR"
echo -e "   git clone $GIT_REPO $APP_DIR"
echo -e "2. Cài đặt các thư viện Python"
echo -e "   cd $APP_DIR && source venv/bin/activate && pip install -r requirements.txt"
echo -e "3. Khởi động dịch vụ"
echo -e "   systemctl start $APP_NAME"
echo -e "4. Kiểm tra trạng thái"
echo -e "   systemctl status $APP_NAME"
echo -e "5. Kiểm tra log"
echo -e "   journalctl -u $APP_NAME -f"

echo -e "${YELLOW}Đường dẫn tới ứng dụng: http://YOUR_SERVER_IP:5000${NC}"
echo -e "${BLUE}============================================${NC}"