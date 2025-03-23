#!/bin/bash
# Script cài đặt ETHUSDT Dashboard trên server trắng mới hoàn toàn

# Màu sắc cho đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thông tin server
INSTALL_DIR="/root/ethusdt_dashboard"
BINANCE_API_KEY=""
BINANCE_API_SECRET=""
PROXY_SERVER="64.176.51.107:3128"

echo -e "${YELLOW}=== CÀI ĐẶT MỚI ETHUSDT DASHBOARD ===${NC}"
echo "Thời gian: $(date)"

# Kiểm tra quyền root
if [ "$(id -u)" != "0" ]; then
   echo -e "${RED}Lỗi: Script này cần quyền root.${NC}"
   echo "Vui lòng chạy với sudo hoặc bằng tài khoản root."
   exit 1
fi

# 1. Cập nhật hệ thống
echo -e "${BLUE}1. Cập nhật hệ thống...${NC}"
apt update && apt upgrade -y

# 2. Cài đặt các gói cần thiết
echo -e "${BLUE}2. Cài đặt các gói cần thiết...${NC}"
apt install -y python3 python3-pip python3-venv git rsync curl wget htop nano unzip

# 3. Tạo thư mục cài đặt
echo -e "${BLUE}3. Tạo thư mục cài đặt...${NC}"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# 4. Tạo cấu trúc thư mục
echo -e "${BLUE}4. Tạo cấu trúc thư mục...${NC}"
mkdir -p data logs models/saved utils dashboard prediction

# 5. Thiết lập Python venv
echo -e "${BLUE}5. Thiết lập môi trường Python ảo...${NC}"
python3 -m venv venv
source venv/bin/activate

# 6. Cài đặt các gói Python
echo -e "${BLUE}6. Cài đặt các gói Python...${NC}"
pip install --upgrade pip
pip install streamlit pandas numpy plotly python-binance scikit-learn tensorflow requests psutil

# 7. Tạo file .env
echo -e "${BLUE}7. Thiết lập file .env...${NC}"
cat > $INSTALL_DIR/.env << EOF
# Binance API Keys
BINANCE_API_KEY=$BINANCE_API_KEY
BINANCE_API_SECRET=$BINANCE_API_SECRET
# Proxy Settings
HTTP_PROXY=$PROXY_SERVER
HTTPS_PROXY=$PROXY_SERVER
EOF

echo -e "${YELLOW}File .env đã được tạo tại $INSTALL_DIR/.env${NC}"
echo -e "${YELLOW}QUAN TRỌNG: Hãy cập nhật API key và secret trong file này!${NC}"

# 8. Tạo service systemd
echo -e "${BLUE}8. Thiết lập systemd service...${NC}"
cat > /etc/systemd/system/ethusdt-dashboard.service << EOF
[Unit]
Description=ETHUSDT Dashboard Service
After=network.target

[Service]
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=$INSTALL_DIR"

[Install]
WantedBy=multi-user.target
EOF

# 9. Tạo các script tiện ích
echo -e "${BLUE}9. Tạo các script tiện ích...${NC}"

cat > $INSTALL_DIR/view_logs.sh << 'EOF'
#!/bin/bash
journalctl -fu ethusdt-dashboard
EOF

cat > $INSTALL_DIR/restart.sh << 'EOF'
#!/bin/bash
systemctl restart ethusdt-dashboard
echo "Dịch vụ đã được khởi động lại"
systemctl status ethusdt-dashboard
EOF

chmod +x $INSTALL_DIR/view_logs.sh $INSTALL_DIR/restart.sh

# 10. Cấu hình firewall
echo -e "${BLUE}10. Cấu hình firewall...${NC}"
if command -v ufw &> /dev/null; then
    ufw allow 5000/tcp
    ufw status
else
    echo -e "${YELLOW}UFW không được cài đặt. Bỏ qua cấu hình firewall.${NC}"
fi

# 11. Tạo tệp hướng dẫn
echo -e "${BLUE}11. Tạo tệp hướng dẫn...${NC}"
cat > $INSTALL_DIR/README.md << 'EOF'
# ETHUSDT Dashboard

## Thông tin hệ thống
- Dashboard port: 5000
- Thư mục cài đặt: /root/ethusdt_dashboard
- Service: ethusdt-dashboard

## Các lệnh hữu ích
- Xem logs: ./view_logs.sh hoặc journalctl -fu ethusdt-dashboard
- Khởi động lại: ./restart.sh hoặc systemctl restart ethusdt-dashboard
- Trạng thái: systemctl status ethusdt-dashboard

## API Keys
API keys được lưu trong file .env
Để cập nhật API keys, chỉnh sửa file này: nano .env

## Cấu hình proxy
Proxy được cấu hình trong file .env
Nếu cần thay đổi, hãy chỉnh sửa HTTP_PROXY và HTTPS_PROXY trong file .env
EOF

# 12. Reload systemd và enable service
echo -e "${BLUE}12. Cấu hình systemd...${NC}"
systemctl daemon-reload
systemctl enable ethusdt-dashboard

echo -e "${GREEN}=== CÀI ĐẶT HOÀN TẤT ===${NC}"
echo -e "${YELLOW}Bây giờ bạn cần đồng bộ mã nguồn từ Replit lên server.${NC}"
echo "Từ Replit, hãy chạy:"
echo -e "${BLUE}./sync_to_server.sh${NC}"
echo -e "(Đảm bảo cập nhật IP server trong sync_to_server.sh trước)"
echo 
echo "Sau khi đồng bộ xong, khởi động dịch vụ với:"
echo -e "${BLUE}systemctl start ethusdt-dashboard${NC}"
echo 
echo "Truy cập dashboard tại: http://$(hostname -I | awk '{print $1}'):5000"