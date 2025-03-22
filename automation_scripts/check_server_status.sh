#!/bin/bash
# Script để kiểm tra trạng thái ứng dụng trên server

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SERVER="45.76.196.13"
USER="root"
REMOTE_DIR="/root/ethusdt_dashboard"
PORT=5000

echo -e "${YELLOW}=== KIỂM TRA TRẠNG THÁI SERVER ===${NC}"

# Kiểm tra kết nối đến server
echo -e "${BLUE}Kiểm tra kết nối đến server...${NC}"
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$SERVER "echo 'Kết nối thành công!'" &> /dev/null; then
    echo -e "${RED}Không thể kết nối đến server. Kiểm tra lại kết nối mạng hoặc thông tin đăng nhập.${NC}"
    exit 1
fi
echo -e "${GREEN}Kết nối đến server thành công.${NC}"

# Kiểm tra trạng thái systemd service
echo -e "${BLUE}Kiểm tra trạng thái service...${NC}"
SSH_OUTPUT=$(ssh -o StrictHostKeyChecking=no $USER@$SERVER "systemctl status ethusdt-dashboard 2>&1")
if echo "$SSH_OUTPUT" | grep -q "Active: active (running)"; then
    echo -e "${GREEN}Service đang chạy.${NC}"
    echo -e "${BLUE}Chi tiết service:${NC}"
    echo -e "$SSH_OUTPUT" | grep -E "Active:|Main PID:|Status:|Tasks:|Memory:|CPU:"
else
    echo -e "${RED}Service không hoạt động hoặc không tồn tại.${NC}"
    echo -e "${BLUE}Chi tiết lỗi:${NC}"
    echo "$SSH_OUTPUT"
fi

# Kiểm tra trạng thái web app
echo -e "${BLUE}Kiểm tra trạng thái web app...${NC}"
WEB_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://$SERVER:$PORT)
if [ "$WEB_STATUS" == "200" ]; then
    echo -e "${GREEN}Web app đang chạy và trả về status code 200.${NC}"
else
    echo -e "${RED}Web app không phản hồi hoặc trả về lỗi (status code: $WEB_STATUS).${NC}"
fi

# Kiểm tra tài nguyên hệ thống
echo -e "${BLUE}Kiểm tra tài nguyên hệ thống...${NC}"
SYSTEM_INFO=$(ssh -o StrictHostKeyChecking=no $USER@$SERVER "
    echo 'Thông tin CPU:';
    mpstat 1 1 | grep 'Average' | awk '{print \"CPU Usage: \" (100-\$NF) \"%\"}';
    echo '';
    echo 'Thông tin bộ nhớ:';
    free -h | grep -v '+' | grep -E 'Mem|Swap';
    echo '';
    echo 'Thông tin ổ đĩa:';
    df -h | grep -v 'tmpfs' | grep -v 'udev' | grep -E '/$|/root';
    echo '';
    echo 'Các tiến trình Python đang chạy:';
    ps aux | grep python | grep -v grep | awk '{print \$2, \$11, \$12}';
    echo '';
    echo 'Các cổng đang lắng nghe:';
    netstat -tulpn 2>/dev/null | grep -E ':(5000|8000|8080)' | grep LISTEN;
    echo '';
    echo 'Thời gian hoạt động của hệ thống:';
    uptime;
")

echo -e "${BLUE}Thông tin hệ thống:${NC}"
echo "$SYSTEM_INFO"

# Kiểm tra log của ứng dụng
echo -e "${BLUE}Log ứng dụng (10 dòng cuối):${NC}"
LOG_OUTPUT=$(ssh -o StrictHostKeyChecking=no $USER@$SERVER "
    if [ -f $REMOTE_DIR/logs/streamlit.log ]; then
        tail -n 10 $REMOTE_DIR/logs/streamlit.log
    else
        echo 'Không tìm thấy file log ứng dụng.'
    fi
")
echo "$LOG_OUTPUT"

echo -e "${YELLOW}=== KIỂM TRA HOÀN TẤT ===${NC}"