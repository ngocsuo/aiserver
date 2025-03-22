#!/bin/bash
# Script kiểm tra trạng thái server và ứng dụng

# Cấu hình
SERVER="45.76.196.13"
USER="root"
REMOTE_DIR="/root/ethusdt_dashboard"

# Màu sắc
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== KIỂM TRA TRẠNG THÁI SERVER ETHUSDT DASHBOARD ===${NC}"

# Kiểm tra server có online không
echo -e "${YELLOW}Kiểm tra kết nối đến server...${NC}"
if ping -c 1 $SERVER &> /dev/null; then
    echo -e "${GREEN}Server online!${NC}"
else
    echo -e "${RED}Server không phản hồi!${NC}"
    exit 1
fi

# Kiểm tra SSH
echo -e "${YELLOW}Kiểm tra kết nối SSH...${NC}"
if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$SERVER "echo 'SSH OK'" &> /dev/null; then
    echo -e "${GREEN}Kết nối SSH thành công!${NC}"
else
    echo -e "${RED}Không thể kết nối SSH đến server!${NC}"
    exit 1
fi

# Kiểm tra dịch vụ
echo -e "${YELLOW}Kiểm tra trạng thái dịch vụ ETHUSDT Dashboard...${NC}"
service_status=$(ssh -o StrictHostKeyChecking=no $USER@$SERVER "systemctl is-active ethusdt-dashboard")
if [ "$service_status" == "active" ]; then
    echo -e "${GREEN}Dịch vụ đang chạy!${NC}"
else
    echo -e "${RED}Dịch vụ không hoạt động (trạng thái: $service_status)${NC}"
fi

# Kiểm tra tài nguyên hệ thống
echo -e "${YELLOW}Kiểm tra tài nguyên hệ thống...${NC}"
ssh -o StrictHostKeyChecking=no $USER@$SERVER "echo -e '${BLUE}CPU:${NC}'; top -bn1 | grep 'Cpu(s)' | awk '{print \$2 \"%\"}'; echo -e '${BLUE}Memory:${NC}'; free -m | grep Mem | awk '{printf(\"Used: %s MB / Total: %s MB (%.2f%%)\", \$3, \$2, \$3*100/\$2)}'; echo ''; echo -e '${BLUE}Disk:${NC}'; df -h / | grep -v Filesystem | awk '{print \$5 \" used (\" \$3 \"/\" \$2 \")\" }'"

# Kiểm tra port 5000
echo -e "${YELLOW}Kiểm tra port 5000 (Streamlit)...${NC}"
port_status=$(ssh -o StrictHostKeyChecking=no $USER@$SERVER "netstat -tuln | grep ':5000 ' | wc -l")
if [ "$port_status" -gt 0 ]; then
    echo -e "${GREEN}Port 5000 đang mở và ứng dụng đang chạy!${NC}"
else
    echo -e "${RED}Port 5000 không mở hoặc ứng dụng không chạy!${NC}"
fi

# Kiểm tra logs
echo -e "${YELLOW}Kiểm tra 5 dòng log cuối cùng:${NC}"
ssh -o StrictHostKeyChecking=no $USER@$SERVER "tail -n 5 $REMOTE_DIR/logs/streamlit.log"

echo -e "${BLUE}=== KẾT THÚC KIỂM TRA ===${NC}"