#!/bin/bash
# Script đồng bộ mã nguồn từ Replit lên server

# Thông tin server - thay thế bằng thông tin server thực tế của bạn
SERVER="your_actual_server_ip"  # Thay your_actual_server_ip bằng IP thực tế của server mới của bạn
SSH_PORT="22"  # Cổng SSH mặc định, thay đổi nếu cần
USER="root"
REMOTE_DIR="/root/ethusdt_dashboard"

# Màu sắc cho đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== ĐỒNG BỘ ETHUSDT DASHBOARD LÊN SERVER ===${NC}"
echo "Thời gian: $(date)"

# Tạo tệp cấu hình SSH tạm thời nếu chưa có
if [ ! -d ~/.ssh ]; then
    echo -e "${BLUE}Tạo thư mục SSH...${NC}"
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
fi

if [ ! -f ~/.ssh/config ]; then
    echo -e "${BLUE}Tạo tệp cấu hình SSH...${NC}"
    cat > ~/.ssh/config << EOF
Host $SERVER
    User $USER
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
    chmod 600 ~/.ssh/config
fi

# Kiểm tra kết nối đến server
echo -e "${BLUE}Kiểm tra kết nối đến server...${NC}"
ssh -p $SSH_PORT -o ConnectTimeout=5 $USER@$SERVER "echo 'Kết nối thành công'" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}Không thể kết nối đến server.${NC}"
    echo "Vui lòng kiểm tra lại thông tin kết nối và đảm bảo server đang chạy."
    exit 1
fi

# Kiểm tra thư mục từ xa
echo -e "${BLUE}Kiểm tra thư mục từ xa...${NC}"
ssh -p $SSH_PORT $USER@$SERVER "if [ ! -d '$REMOTE_DIR' ]; then mkdir -p '$REMOTE_DIR'; fi"

# Đồng bộ tệp
echo -e "${BLUE}Đồng bộ các tệp lên server...${NC}"
rsync -avz -e "ssh -p $SSH_PORT" --exclude=".*" --exclude="venv" --exclude="__pycache__" \
    --exclude="*.pyc" --exclude="*.pyo" --exclude="*.log" \
    --exclude="node_modules" --exclude="*.zip" \
    --delete \
    ./* $USER@$SERVER:$REMOTE_DIR/

# Kiểm tra kết quả
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Đồng bộ thành công!${NC}"
else
    echo -e "${RED}Đồng bộ thất bại!${NC}"
    exit 1
fi

# Thiết lập quyền trên server
echo -e "${BLUE}Thiết lập quyền thực thi cho các script...${NC}"
ssh -p $SSH_PORT $USER@$SERVER "chmod +x $REMOTE_DIR/*.sh $REMOTE_DIR/automation_scripts/*.sh $REMOTE_DIR/deployment/*.sh 2>/dev/null || true"

# Khởi động lại dịch vụ trên server
echo -e "${BLUE}Khởi động lại dịch vụ ETHUSDT Dashboard...${NC}"
ssh -p $SSH_PORT $USER@$SERVER "cd $REMOTE_DIR && (chmod +x server_setup.sh && ./server_setup.sh || systemctl restart ethusdt-dashboard)"

# Kiểm tra trạng thái dịch vụ
echo -e "${BLUE}Kiểm tra trạng thái dịch vụ...${NC}"
ssh -p $SSH_PORT $USER@$SERVER "systemctl status ethusdt-dashboard | grep Active || echo 'Dịch vụ không tìm thấy. Đảm bảo server_setup.sh đã được chạy.'"

echo -e "${GREEN}=== ĐỒNG BỘ HOÀN TẤT ===${NC}"
echo "Bạn có thể truy cập dashboard tại: http://$SERVER:5000"