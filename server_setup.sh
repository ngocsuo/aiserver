#!/bin/bash
# Script cài đặt tự động cho ETHUSDT Dashboard trên server Ubuntu 22.04

# Đặt màu sắc cho output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== Bắt đầu cài đặt ETHUSDT Dashboard =====${NC}"

# Thư mục cài đặt
INSTALL_DIR="/root/ethusdt_dashboard"

# Tạo thư mục cài đặt nếu chưa tồn tại
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Cập nhật hệ thống
echo -e "${YELLOW}Cập nhật hệ thống...${NC}"
apt update && apt upgrade -y

# Cài đặt các gói cần thiết
echo -e "${YELLOW}Cài đặt các gói cần thiết...${NC}"
apt install -y python3 python3-pip python3-venv git curl wget unzip
apt install -y build-essential libssl-dev libffi-dev python3-dev
apt install -y rsync netstat net-tools
apt install -y gcc g++ make cmake pkg-config

# Cài đặt TA-Lib (thư viện phân tích kỹ thuật)
echo -e "${YELLOW}Cài đặt TA-Lib...${NC}"
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
make install
cd $INSTALL_DIR

# Tạo môi trường ảo Python
echo -e "${YELLOW}Tạo môi trường ảo Python...${NC}"
python3 -m venv venv
source venv/bin/activate

# Cài đặt wheel trước để tránh lỗi khi cài đặt các gói khác
pip install --upgrade pip
pip install wheel setuptools

# Copy file requirements_server.txt sang server
echo -e "${YELLOW}Tạo các thư mục cần thiết...${NC}"
mkdir -p $INSTALL_DIR/data
mkdir -p $INSTALL_DIR/logs
mkdir -p $INSTALL_DIR/saved_models
mkdir -p $INSTALL_DIR/utils
mkdir -p $INSTALL_DIR/models
mkdir -p $INSTALL_DIR/prediction
mkdir -p $INSTALL_DIR/dashboard
mkdir -p $INSTALL_DIR/deployment

# Cài đặt các thư viện Python từ requirements.txt
echo -e "${YELLOW}Cài đặt các thư viện Python...${NC}"
if [ -f "requirements_server.txt" ]; then
    pip install -r requirements_server.txt
else
    echo -e "${RED}Không tìm thấy file requirements_server.txt!${NC}"
    exit 1
fi

# Tạo script restart
echo -e "${YELLOW}Tạo script khởi động...${NC}"
cat > $INSTALL_DIR/restart.sh << 'EOF'
#!/bin/bash
# Script khởi động lại ứng dụng

# Dừng tất cả các tiến trình Streamlit đang chạy
pkill -f streamlit

# Kích hoạt môi trường ảo
source /root/ethusdt_dashboard/venv/bin/activate

# Cài đặt biến môi trường
export BINANCE_API_KEY=""
export BINANCE_API_SECRET=""

# Khởi động ứng dụng
cd /root/ethusdt_dashboard
nohup streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true > /root/ethusdt_dashboard/logs/streamlit.log 2>&1 &

echo "Ứng dụng đã được khởi động lại!"
EOF

chmod +x $INSTALL_DIR/restart.sh

# Tạo file .streamlit/config.toml
mkdir -p $INSTALL_DIR/.streamlit
cat > $INSTALL_DIR/.streamlit/config.toml << 'EOF'
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
port = 5000

[browser]
gatherUsageStats = false
EOF

# Tạo systemd service để tự động khởi động
echo -e "${YELLOW}Tạo systemd service...${NC}"
cat > /etc/systemd/system/ethusdt-dashboard.service << EOF
[Unit]
Description=ETHUSDT Dashboard Service
After=network.target

[Service]
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=/bin/bash -c "source $INSTALL_DIR/venv/bin/activate && streamlit run $INSTALL_DIR/app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true"
Restart=always
Environment="BINANCE_API_KEY="
Environment="BINANCE_API_SECRET="

[Install]
WantedBy=multi-user.target
EOF

# Enable và khởi động service
systemctl daemon-reload
systemctl enable ethusdt-dashboard.service

echo -e "${GREEN}===== Cài đặt hoàn tất! =====${NC}"
echo -e "${YELLOW}Các bước tiếp theo:${NC}"
echo "1. Đồng bộ code từ Replit sử dụng sync_to_server.sh"
echo "2. Khởi động ứng dụng bằng lệnh: systemctl start ethusdt-dashboard"
echo "3. Kiểm tra trạng thái: systemctl status ethusdt-dashboard"
echo -e "${GREEN}Sau khi hoàn tất, bạn có thể truy cập ứng dụng tại: http://45.76.196.13:5000${NC}"