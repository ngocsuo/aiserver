#!/bin/bash
# Script cài đặt môi trường trên server

# Màu sắc
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="/root/ethusdt_dashboard"
PYTHON_VERSION="3.10"
VENV_DIR="$SCRIPT_DIR/venv"

echo -e "${YELLOW}=== CÀI ĐẶT MÔI TRƯỜNG CHO ETHUSDT DASHBOARD ===${NC}"

# Kiểm tra xem script có được chạy với quyền root không
if [ "$(id -u)" != "0" ]; then
   echo -e "${RED}Script này cần được chạy với quyền root${NC}" 
   exit 1
fi

# Cập nhật hệ thống
echo -e "${YELLOW}Cập nhật hệ thống...${NC}"
apt update && apt upgrade -y

# Cài đặt các gói cần thiết
echo -e "${YELLOW}Cài đặt các gói cần thiết...${NC}"
apt install -y python$PYTHON_VERSION python3-pip python3-venv python3-dev build-essential libssl-dev libffi-dev git curl wget unzip

# Tạo các thư mục cần thiết
echo -e "${YELLOW}Tạo cấu trúc thư mục...${NC}"
mkdir -p $SCRIPT_DIR/data
mkdir -p $SCRIPT_DIR/logs
mkdir -p $SCRIPT_DIR/saved_models
mkdir -p $SCRIPT_DIR/.streamlit

# Tạo môi trường ảo Python
echo -e "${YELLOW}Tạo môi trường ảo Python...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    python$PYTHON_VERSION -m venv $VENV_DIR
    echo -e "${GREEN}Môi trường ảo đã được tạo tại $VENV_DIR${NC}"
else
    echo -e "${GREEN}Môi trường ảo đã tồn tại tại $VENV_DIR${NC}"
fi

# Nâng cấp pip
echo -e "${YELLOW}Nâng cấp pip...${NC}"
$VENV_DIR/bin/pip install --upgrade pip

# Cài đặt các thư viện Python từ requirements
echo -e "${YELLOW}Cài đặt các thư viện Python cần thiết...${NC}"
$VENV_DIR/bin/pip install -r $SCRIPT_DIR/requirements_server.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Cài đặt thư viện thành công!${NC}"
else
    echo -e "${RED}Cài đặt thư viện thất bại!${NC}"
    exit 1
fi

# Cấu hình Streamlit
echo -e "${YELLOW}Cấu hình Streamlit...${NC}"
cat > $SCRIPT_DIR/.streamlit/config.toml << EOF
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
port = 5000
address = "0.0.0.0"

[browser]
serverAddress = "localhost"
serverPort = 5000
EOF

# Tạo file service cho systemd
echo -e "${YELLOW}Tạo service systemd...${NC}"
cat > /etc/systemd/system/ethusdt-dashboard.service << EOF
[Unit]
Description=ETHUSDT Dashboard Service
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$VENV_DIR/bin/python -m streamlit run $SCRIPT_DIR/app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Tạo script restart.sh
echo -e "${YELLOW}Tạo script khởi động lại...${NC}"
cat > $SCRIPT_DIR/restart.sh << 'EOF'
#!/bin/bash
# Script khởi động lại ứng dụng

# Dừng service
systemctl stop ethusdt-dashboard

# Đợi 2 giây
sleep 2

# Khởi động lại service
systemctl start ethusdt-dashboard

# Hiển thị trạng thái
systemctl status ethusdt-dashboard --no-pager
EOF

chmod +x $SCRIPT_DIR/restart.sh

# Kích hoạt và khởi động service
echo -e "${YELLOW}Kích hoạt và khởi động service...${NC}"
systemctl daemon-reload
systemctl enable ethusdt-dashboard
systemctl restart ethusdt-dashboard

# Kiểm tra trạng thái
echo -e "${YELLOW}Kiểm tra trạng thái service...${NC}"
systemctl status ethusdt-dashboard --no-pager

echo -e "${GREEN}=== CÀI ĐẶT HOÀN TẤT ===${NC}"
echo -e "${GREEN}ETHUSDT Dashboard đã được cài đặt và khởi động.${NC}"
echo -e "${GREEN}Bạn có thể truy cập ứng dụng tại: http://SERVER_IP:5000${NC}"
echo -e "${GREEN}Sử dụng lệnh 'systemctl status ethusdt-dashboard' để kiểm tra trạng thái.${NC}"
echo -e "${GREEN}Sử dụng lệnh '$SCRIPT_DIR/restart.sh' để khởi động lại ứng dụng.${NC}"