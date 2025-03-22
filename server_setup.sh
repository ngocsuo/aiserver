#!/bin/bash
# Script cài đặt ETHUSDT Dashboard lên server

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

INSTALL_DIR="/root/ethusdt_dashboard"
LOG_FILE="$INSTALL_DIR/setup_log.txt"

echo -e "${YELLOW}=== CÀI ĐẶT ETHUSDT DASHBOARD LÊN SERVER ===${NC}" | tee -a $LOG_FILE
echo "$(date): Bắt đầu cài đặt" | tee -a $LOG_FILE

# Tạo thư mục cài đặt nếu chưa tồn tại
if [ ! -d "$INSTALL_DIR" ]; then
    echo -e "${BLUE}Tạo thư mục cài đặt $INSTALL_DIR...${NC}" | tee -a $LOG_FILE
    mkdir -p $INSTALL_DIR
fi

# Kiểm tra và cài đặt Python 3.10
echo -e "${BLUE}Kiểm tra phiên bản Python...${NC}" | tee -a $LOG_FILE
if command -v python3.10 &> /dev/null; then
    PYTHON_VERSION=$(python3.10 --version)
    echo -e "${GREEN}Python đã được cài đặt: $PYTHON_VERSION${NC}" | tee -a $LOG_FILE
else
    echo -e "${YELLOW}Python 3.10 chưa được cài đặt. Đang cài đặt...${NC}" | tee -a $LOG_FILE
    apt-get update | tee -a $LOG_FILE
    apt-get install -y software-properties-common | tee -a $LOG_FILE
    add-apt-repository -y ppa:deadsnakes/ppa | tee -a $LOG_FILE
    apt-get update | tee -a $LOG_FILE
    apt-get install -y python3.10 python3.10-venv python3.10-dev | tee -a $LOG_FILE
    
    if command -v python3.10 &> /dev/null; then
        echo -e "${GREEN}Python 3.10 đã được cài đặt thành công.${NC}" | tee -a $LOG_FILE
    else
        echo -e "${RED}Không thể cài đặt Python 3.10. Vui lòng kiểm tra lại.${NC}" | tee -a $LOG_FILE
        exit 1
    fi
fi

# Cài đặt pip nếu chưa có
echo -e "${BLUE}Kiểm tra pip...${NC}" | tee -a $LOG_FILE
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}pip chưa được cài đặt. Đang cài đặt...${NC}" | tee -a $LOG_FILE
    apt-get install -y python3-pip | tee -a $LOG_FILE
fi

# Cài đặt virtualenv
echo -e "${BLUE}Cài đặt virtualenv...${NC}" | tee -a $LOG_FILE
pip3 install virtualenv | tee -a $LOG_FILE

# Tạo và kích hoạt môi trường ảo
if [ ! -d "$INSTALL_DIR/venv" ]; then
    echo -e "${BLUE}Tạo môi trường ảo...${NC}" | tee -a $LOG_FILE
    cd $INSTALL_DIR
    python3.10 -m venv venv | tee -a $LOG_FILE
fi

echo -e "${BLUE}Kích hoạt môi trường ảo...${NC}" | tee -a $LOG_FILE
source $INSTALL_DIR/venv/bin/activate | tee -a $LOG_FILE

# Cài đặt các gói phụ thuộc
echo -e "${BLUE}Cài đặt các gói phụ thuộc...${NC}" | tee -a $LOG_FILE
if [ -f "$INSTALL_DIR/requirements_server.txt" ]; then
    pip install -r $INSTALL_DIR/requirements_server.txt | tee -a $LOG_FILE
else
    echo -e "${YELLOW}File requirements_server.txt không tồn tại. Cài đặt các gói mặc định...${NC}" | tee -a $LOG_FILE
    pip install streamlit pandas numpy plotly python-binance scikit-learn tensorflow requests | tee -a $LOG_FILE
fi

# Tạo thư mục dữ liệu và logs nếu chưa tồn tại
echo -e "${BLUE}Tạo thư mục dữ liệu và logs...${NC}" | tee -a $LOG_FILE
mkdir -p $INSTALL_DIR/data $INSTALL_DIR/logs | tee -a $LOG_FILE

# Tạo thư mục cho các model đã huấn luyện
echo -e "${BLUE}Tạo thư mục lưu trữ model...${NC}" | tee -a $LOG_FILE
mkdir -p $INSTALL_DIR/models/saved | tee -a $LOG_FILE

# Cài đặt và cấu hình systemd service
echo -e "${BLUE}Cài đặt systemd service...${NC}" | tee -a $LOG_FILE
cat > /etc/systemd/system/ethusdt-dashboard.service << EOF
[Unit]
Description=ETHUSDT Dashboard Service
After=network.target

[Service]
User=root
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$INSTALL_DIR/venv/bin/streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Khởi động dịch vụ
echo -e "${BLUE}Khởi động dịch vụ...${NC}" | tee -a $LOG_FILE
systemctl daemon-reload | tee -a $LOG_FILE
systemctl enable ethusdt-dashboard | tee -a $LOG_FILE
systemctl restart ethusdt-dashboard | tee -a $LOG_FILE

# Kiểm tra trạng thái service
echo -e "${BLUE}Kiểm tra trạng thái service...${NC}" | tee -a $LOG_FILE
systemctl status ethusdt-dashboard | tee -a $LOG_FILE

# Thiết lập tường lửa
echo -e "${BLUE}Thiết lập tường lửa...${NC}" | tee -a $LOG_FILE
if command -v ufw &> /dev/null; then
    ufw allow 5000/tcp | tee -a $LOG_FILE
    echo -e "${GREEN}Đã mở cổng 5000 trên tường lửa.${NC}" | tee -a $LOG_FILE
else
    echo -e "${YELLOW}UFW không được cài đặt. Bỏ qua thiết lập tường lửa.${NC}" | tee -a $LOG_FILE
fi

# Kiểm tra trạng thái ứng dụng
echo -e "${BLUE}Chờ ứng dụng khởi động (10 giây)...${NC}" | tee -a $LOG_FILE
sleep 10

echo -e "${BLUE}Kiểm tra ứng dụng...${NC}" | tee -a $LOG_FILE
if curl -s http://localhost:5000 > /dev/null; then
    echo -e "${GREEN}Ứng dụng đang chạy tại http://localhost:5000${NC}" | tee -a $LOG_FILE
    
    # Lấy địa chỉ IP công khai
    PUBLIC_IP=$(curl -s ifconfig.me)
    echo -e "${GREEN}Truy cập ứng dụng tại: http://$PUBLIC_IP:5000${NC}" | tee -a $LOG_FILE
else
    echo -e "${RED}Không thể kết nối đến ứng dụng. Vui lòng kiểm tra logs.${NC}" | tee -a $LOG_FILE
    journalctl -u ethusdt-dashboard -n 50 | tee -a $LOG_FILE
fi

# Tạo tập lệnh khởi động lại dịch vụ
echo -e "${BLUE}Tạo tập lệnh khởi động lại dịch vụ...${NC}" | tee -a $LOG_FILE
cat > $INSTALL_DIR/restart.sh << EOF
#!/bin/bash
systemctl restart ethusdt-dashboard
echo "Đã khởi động lại dịch vụ ETHUSDT Dashboard!"
EOF
chmod +x $INSTALL_DIR/restart.sh | tee -a $LOG_FILE

# Tạo tập lệnh debug
echo -e "${BLUE}Tạo tập lệnh debug...${NC}" | tee -a $LOG_FILE
cat > $INSTALL_DIR/debug.sh << EOF
#!/bin/bash
echo "===== ETHUSDT Dashboard Debug Tool ====="
echo "1. Hiển thị logs"
echo "2. Khởi động lại dịch vụ"
echo "3. Kiểm tra trạng thái dịch vụ"
echo "4. Kiểm tra các quá trình Python đang chạy"
echo "5. Kiểm tra kết nối mạng"
echo "6. Kiểm tra tài nguyên hệ thống"
echo "7. Thoát"

read -p "Chọn một tùy chọn (1-7): " choice

case \$choice in
    1)
        echo "===== Logs gần đây ====="
        journalctl -u ethusdt-dashboard -n 100 --no-pager
        ;;
    2)
        echo "===== Khởi động lại dịch vụ ====="
        systemctl restart ethusdt-dashboard
        sleep 2
        systemctl status ethusdt-dashboard
        ;;
    3)
        echo "===== Trạng thái dịch vụ ====="
        systemctl status ethusdt-dashboard
        ;;
    4)
        echo "===== Các quá trình Python đang chạy ====="
        ps aux | grep python | grep -v grep
        ;;
    5)
        echo "===== Kiểm tra kết nối mạng ====="
        echo "Kiểm tra kết nối đến Binance:"
        curl -s https://api.binance.com/api/v3/ping
        echo -e "\nKiểm tra kết nối đến Internet:"
        ping -c 4 google.com
        ;;
    6)
        echo "===== Tài nguyên hệ thống ====="
        echo "CPU:"
        top -bn1 | head -20
        echo "RAM:"
        free -m
        echo "Disk:"
        df -h
        ;;
    7)
        echo "Thoát"
        exit 0
        ;;
    *)
        echo "Lựa chọn không hợp lệ!"
        ;;
esac
EOF
chmod +x $INSTALL_DIR/debug.sh | tee -a $LOG_FILE

# Tạo thư mục backup
echo -e "${BLUE}Tạo thư mục backup...${NC}" | tee -a $LOG_FILE
mkdir -p $INSTALL_DIR/backup | tee -a $LOG_FILE

# Tạo tập lệnh backup
echo -e "${BLUE}Tạo tập lệnh backup...${NC}" | tee -a $LOG_FILE
cat > $INSTALL_DIR/backup_app.sh << EOF
#!/bin/bash
BACKUP_DIR="$INSTALL_DIR/backup"
TIMESTAMP=\$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="\$BACKUP_DIR/ethusdt_dashboard_\$TIMESTAMP.tar.gz"

echo "Tạo backup vào \$BACKUP_FILE..."
tar -czf \$BACKUP_FILE -C $INSTALL_DIR app.py config.py models utils dashboard data
echo "Backup hoàn tất!"
ls -lh \$BACKUP_FILE
EOF
chmod +x $INSTALL_DIR/backup_app.sh | tee -a $LOG_FILE

echo -e "${GREEN}=== CÀI ĐẶT HOÀN TẤT ===${NC}" | tee -a $LOG_FILE
echo "$(date): Hoàn tất cài đặt" | tee -a $LOG_FILE
echo -e "${GREEN}Ứng dụng ETHUSDT Dashboard đã được cài đặt và khởi động thành công!${NC}" | tee -a $LOG_FILE
echo -e "Truy cập ứng dụng tại: http://$(curl -s ifconfig.me):5000" | tee -a $LOG_FILE
echo -e "Các tập lệnh hữu ích:" | tee -a $LOG_FILE
echo -e "  - $INSTALL_DIR/restart.sh: Khởi động lại dịch vụ" | tee -a $LOG_FILE
echo -e "  - $INSTALL_DIR/debug.sh: Công cụ debug" | tee -a $LOG_FILE
echo -e "  - $INSTALL_DIR/backup_app.sh: Tạo backup" | tee -a $LOG_FILE