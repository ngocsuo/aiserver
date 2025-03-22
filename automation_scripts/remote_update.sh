#!/bin/bash
# Script cập nhật và khởi động lại ứng dụng từ xa

# Cấu hình
SERVER="45.76.196.13"
USER="root"
REMOTE_DIR="/root/ethusdt_dashboard"

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Kiểm tra kết nối server
echo -e "${YELLOW}Kiểm tra kết nối đến server $SERVER...${NC}"
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$SERVER "echo 'Kết nối thành công!'" || {
    echo -e "${RED}Không thể kết nối đến server. Kiểm tra lại kết nối mạng hoặc thông tin đăng nhập.${NC}"
    exit 1
}

# Menu lựa chọn hành động
echo -e "${GREEN}================ MENU QUẢN LÝ ETHUSDT DASHBOARD =================${NC}"
echo "1) Khởi động lại ứng dụng"
echo "2) Kiểm tra logs"
echo "3) Kiểm tra trạng thái ứng dụng"
echo "4) Khởi động lại server"
echo "5) Cập nhật API keys"
echo "6) Cập nhật code từ Replit"
echo "7) Thoát"

read -p "Nhập lựa chọn của bạn: " choice

case $choice in
    1)
        echo -e "${YELLOW}Đang khởi động lại ứng dụng...${NC}"
        ssh -o StrictHostKeyChecking=no $USER@$SERVER "systemctl restart ethusdt-dashboard"
        echo -e "${GREEN}Ứng dụng đã được khởi động lại!${NC}"
        ;;
    2)
        echo -e "${YELLOW}Đang tải logs...${NC}"
        ssh -o StrictHostKeyChecking=no $USER@$SERVER "tail -n 100 $REMOTE_DIR/logs/streamlit.log"
        ;;
    3)
        echo -e "${YELLOW}Đang kiểm tra trạng thái ứng dụng...${NC}"
        ssh -o StrictHostKeyChecking=no $USER@$SERVER "systemctl status ethusdt-dashboard"
        ;;
    4)
        read -p "Bạn có chắc chắn muốn khởi động lại server? (y/n): " confirm
        if [[ $confirm == "y" || $confirm == "Y" ]]; then
            echo -e "${YELLOW}Đang khởi động lại server...${NC}"
            ssh -o StrictHostKeyChecking=no $USER@$SERVER "reboot"
            echo -e "${GREEN}Lệnh khởi động lại đã được gửi đến server. Server sẽ khởi động lại trong vài giây.${NC}"
        else
            echo -e "${YELLOW}Hủy bỏ khởi động lại server.${NC}"
        fi
        ;;
    5)
        echo -e "${YELLOW}Đang cập nhật API keys...${NC}"
        read -p "Nhập BINANCE_API_KEY mới: " api_key
        read -p "Nhập BINANCE_API_SECRET mới: " api_secret
        
        ssh -o StrictHostKeyChecking=no $USER@$SERVER "
        sed -i 's|export BINANCE_API_KEY=.*|export BINANCE_API_KEY=\"$api_key\"|g' $REMOTE_DIR/restart.sh
        sed -i 's|export BINANCE_API_SECRET=.*|export BINANCE_API_SECRET=\"$api_secret\"|g' $REMOTE_DIR/restart.sh
        
        sed -i 's|Environment=\"BINANCE_API_KEY=.*\"|Environment=\"BINANCE_API_KEY=$api_key\"|g' /etc/systemd/system/ethusdt-dashboard.service
        sed -i 's|Environment=\"BINANCE_API_SECRET=.*\"|Environment=\"BINANCE_API_SECRET=$api_secret\"|g' /etc/systemd/system/ethusdt-dashboard.service
        
        systemctl daemon-reload
        systemctl restart ethusdt-dashboard
        "
        
        echo -e "${GREEN}API keys đã được cập nhật và ứng dụng đã được khởi động lại!${NC}"
        ;;
    6)
        echo -e "${YELLOW}Sử dụng script sync_to_server.sh để cập nhật code.${NC}"
        echo -e "${YELLOW}Bạn có muốn chạy script đó ngay bây giờ không? (y/n)${NC}"
        read -p "> " run_sync
        if [[ $run_sync == "y" || $run_sync == "Y" ]]; then
            cd ..
            ./sync_to_server.sh
        else
            echo -e "${YELLOW}Hủy bỏ cập nhật code.${NC}"
        fi
        ;;
    7)
        echo -e "${GREEN}Tạm biệt!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Lựa chọn không hợp lệ. Vui lòng chọn từ 1-7.${NC}"
        ;;
esac