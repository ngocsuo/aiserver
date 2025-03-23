#!/bin/bash
# Script cài đặt dự án ETHUSDT Dashboard trên server mới hoàn toàn

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== CÀI ĐẶT ETHUSDT DASHBOARD TRÊN SERVER ===${NC}"
echo "Thời gian: $(date)"

# Thư mục cài đặt
SERVER_DIR="/root/ethusdt_dashboard"
mkdir -p "$SERVER_DIR"
cd "$SERVER_DIR"

# Tạo cấu trúc thư mục
echo -e "${BLUE}Tạo cấu trúc thư mục dự án...${NC}"
mkdir -p data logs models/saved utils dashboard prediction

# Cài đặt Python venv
echo -e "${BLUE}Cài đặt môi trường ảo Python...${NC}"
python3 -m venv venv
source venv/bin/activate

# Cài đặt các gói cần thiết
echo -e "${BLUE}Cài đặt các gói Python cần thiết...${NC}"
pip install --upgrade pip
if [ -f "requirements_server.txt" ]; then
    pip install -r requirements_server.txt
else
    echo -e "${YELLOW}File requirements_server.txt không tồn tại. Cài đặt các gói mặc định...${NC}"
    pip install streamlit pandas numpy plotly python-binance scikit-learn tensorflow requests psutil
fi

# Tạo .env file để lưu API key
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Tạo file .env...${NC}"
    cat > .env << EOF
# Binance API Keys
BINANCE_API_KEY=${BINANCE_API_KEY}
BINANCE_API_SECRET=${BINANCE_API_SECRET}
# Proxy Settings (nếu cần)
HTTP_PROXY=64.176.51.107:3128
HTTPS_PROXY=64.176.51.107:3128
EOF
    echo -e "${GREEN}File .env đã được tạo. Hãy cập nhật API key thực tế sau.${NC}"
fi

# Tạo file service để chạy tự động
echo -e "${BLUE}Tạo systemd service...${NC}"
cat > /etc/systemd/system/ethusdt-dashboard.service << EOF
[Unit]
Description=ETHUSDT Dashboard Service
After=network.target

[Service]
User=root
WorkingDirectory=${SERVER_DIR}
ExecStart=${SERVER_DIR}/venv/bin/streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment="PATH=${SERVER_DIR}/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=${SERVER_DIR}"

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd và enable service
systemctl daemon-reload
systemctl enable ethusdt-dashboard

# Tạo script để xem log
echo -e "${BLUE}Tạo script tiện ích...${NC}"
cat > view_logs.sh << 'EOF'
#!/bin/bash
journalctl -fu ethusdt-dashboard
EOF

cat > restart.sh << 'EOF'
#!/bin/bash
systemctl restart ethusdt-dashboard
echo "Service đã được khởi động lại"
systemctl status ethusdt-dashboard
EOF

chmod +x view_logs.sh restart.sh

# Tạo file .env trong thư mục gốc
echo -e "${BLUE}Tạo file .env...${NC}"
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Binance API Keys 
BINANCE_API_KEY=
BINANCE_API_SECRET=
# Proxy Settings
HTTP_PROXY=64.176.51.107:3128
HTTPS_PROXY=64.176.51.107:3128
EOF
fi

# Kiểm tra và mở port 5000 trong firewall
echo -e "${BLUE}Cấu hình firewall...${NC}"
if command -v ufw &> /dev/null; then
    ufw allow 5000/tcp
    ufw status
fi

echo -e "${GREEN}=== CÀI ĐẶT HOÀN TẤT ===${NC}"
echo "Bạn có thể khởi động dịch vụ với: systemctl start ethusdt-dashboard"
echo "Xem log với: journalctl -fu ethusdt-dashboard"
echo "Truy cập dashboard tại: http://$(hostname -I | awk '{print $1}'):5000"