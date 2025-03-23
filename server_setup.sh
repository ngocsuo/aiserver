#!/bin/bash
# Script thiết lập môi trường server cho ETHUSDT Dashboard

# Màu sắc cho đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== CÀI ĐẶT MÔI TRƯỜNG SERVER ETHUSDT DASHBOARD ===${NC}"
echo "Thời gian: $(date)"

# Kiểm tra quyền root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Vui lòng chạy script với quyền root (sudo).${NC}"
    exit 1
fi

# Cập nhật hệ thống
echo -e "${BLUE}Cập nhật hệ thống...${NC}"
apt update && apt upgrade -y

# Cài đặt các gói phụ thuộc cần thiết
echo -e "${BLUE}Cài đặt các gói phụ thuộc...${NC}"
apt install -y python3 python3-pip python3-venv git wget curl unzip build-essential

# Tạo symlink python3 -> python
echo -e "${BLUE}Tạo symlink python3 -> python...${NC}"
which python3 || (echo -e "${RED}Python3 không được tìm thấy. Đang cài đặt...${NC}" && apt-get install -y python3)
ln -sf $(which python3) /usr/bin/python

# Kiểm tra Python
echo -e "${BLUE}Kiểm tra phiên bản Python...${NC}"
python --version

# Thư mục triển khai
DEPLOY_DIR="/root/ethusdt_dashboard"
echo -e "${BLUE}Thiết lập thư mục triển khai: $DEPLOY_DIR${NC}"

# Tạo thư mục triển khai nếu chưa tồn tại
if [ ! -d "$DEPLOY_DIR" ]; then
    echo -e "${BLUE}Tạo thư mục triển khai...${NC}"
    mkdir -p "$DEPLOY_DIR"
fi

# Di chuyển vào thư mục triển khai
cd "$DEPLOY_DIR"

# Tạo và kích hoạt môi trường ảo Python
echo -e "${BLUE}Tạo môi trường ảo Python...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv || (echo -e "${RED}Không thể tạo môi trường ảo. Đang cài đặt python3-venv...${NC}" && apt-get install -y python3-venv && python3 -m venv venv)
fi

# Kích hoạt môi trường ảo
echo -e "${BLUE}Kích hoạt môi trường ảo...${NC}"
source venv/bin/activate

# Cài đặt các gói phụ thuộc Python
echo -e "${BLUE}Cài đặt các gói phụ thuộc Python...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Nếu không có requirements.txt, cài đặt các gói cơ bản
    pip install streamlit pandas numpy plotly python-binance scikit-learn tensorflow
fi

# Kiểm tra xem Streamlit có hoạt động không
echo -e "${BLUE}Kiểm tra streamlit...${NC}"
streamlit --version

# Cấu hình dịch vụ systemd
echo -e "${BLUE}Cấu hình dịch vụ systemd...${NC}"
cat > /etc/systemd/system/ethusdt-dashboard.service << EOF
[Unit]
Description=ETHUSDT Dashboard Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$DEPLOY_DIR
ExecStart=/bin/bash -c "source $DEPLOY_DIR/venv/bin/activate && streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true"
Restart=on-failure
RestartSec=10s
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$DEPLOY_DIR/venv/bin"

[Install]
WantedBy=multi-user.target
EOF

# Tải API keys từ biến môi trường hoặc thiết lập mặc định
echo -e "${BLUE}Thiết lập biến môi trường...${NC}"
cat > "$DEPLOY_DIR/.env" << EOF
BINANCE_API_KEY=${BINANCE_API_KEY:-""}
BINANCE_API_SECRET=${BINANCE_API_SECRET:-""}
EOF

# Tạo thư mục cho dữ liệu và logs
echo -e "${BLUE}Tạo thư mục cho dữ liệu và logs...${NC}"
mkdir -p "$DEPLOY_DIR/data" "$DEPLOY_DIR/logs"

# Đảm bảo quyền truy cập
echo -e "${BLUE}Thiết lập quyền truy cập...${NC}"
chmod -R 755 "$DEPLOY_DIR"
touch "$DEPLOY_DIR/training_logs.txt"
chmod 666 "$DEPLOY_DIR/training_logs.txt"

# Khởi động lại dịch vụ systemd
echo -e "${BLUE}Khởi động lại dịch vụ systemd...${NC}"
systemctl daemon-reload
systemctl enable ethusdt-dashboard
systemctl restart ethusdt-dashboard

# Kiểm tra trạng thái dịch vụ
echo -e "${BLUE}Kiểm tra trạng thái dịch vụ...${NC}"
systemctl status ethusdt-dashboard

echo -e "${GREEN}=== THIẾT LẬP HOÀN TẤT ===${NC}"
echo "ETHUSDT Dashboard có thể truy cập tại http://$(hostname -I | awk '{print $1}'):5000"
echo "Logs có thể xem với lệnh: journalctl -fu ethusdt-dashboard"