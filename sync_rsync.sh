#!/bin/bash
# Script đồng bộ hóa dự án ETHUSDT Dashboard sử dụng rsync
# Phiên bản 2.0 - Sử dụng rsync và SSH key

# Màu sắc cho đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Cấu hình server
SERVER_IP=""
SERVER_USER="root"
SSH_PORT="22" 
REMOTE_DIR="/root/ethusdt_dashboard"

# Danh sách file và thư mục cần loại trừ (không đồng bộ)
EXCLUDE_LIST=(
    ".git"
    "__pycache__"
    "*.pyc"
    "venv"
    ".env"
    "node_modules"
    "*.log"
    "*.db"
    "data/*.cache"
    "data/*.zip"
    "data/*.tmp"
    "saved_models/*.h5"
    "logs"
    ".ipynb_checkpoints"
)

# Danh sách file và thư mục quan trọng phải đồng bộ
IMPORTANT_FILES=(
    "app.py"
    "config.py"
    "server_setup.sh"
    "utils"
    "models"
    "prediction"
    "dashboard"
)

# Hiển thị thông tin đồng bộ
echo -e "${YELLOW}=== ĐỒNG BỘ DỰ ÁN ETHUSDT DASHBOARD ===${NC}"
echo "Thời gian: $(date)"

# Yêu cầu thông tin server nếu chưa cấu hình
if [ -z "$SERVER_IP" ]; then
    read -p "Nhập địa chỉ IP server: " SERVER_IP
fi

if [ -z "$SSH_PORT" ]; then
    read -p "Nhập cổng SSH (mặc định: 22): " SSH_PORT
    SSH_PORT=${SSH_PORT:-22}
fi

echo -e "${BLUE}Thông tin kết nối:${NC}"
echo "Server: $SERVER_USER@$SERVER_IP:$SSH_PORT"
echo "Thư mục từ xa: $REMOTE_DIR"

# Tạo các tham số loại trừ cho rsync
EXCLUDE_PARAMS=""
for item in "${EXCLUDE_LIST[@]}"; do
    EXCLUDE_PARAMS="$EXCLUDE_PARAMS --exclude='$item'"
done

# Tạo danh sách file và thư mục quan trọng
SYNC_ITEMS=""
for item in "${IMPORTANT_FILES[@]}"; do
    SYNC_ITEMS="$SYNC_ITEMS $item"
done

# Kiểm tra kết nối SSH
echo -e "${BLUE}Kiểm tra kết nối đến server...${NC}"
if ! ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "echo 'Kết nối thành công'" > /dev/null 2>&1; then
    echo -e "${RED}Không thể kết nối đến server. Vui lòng kiểm tra thông tin và quyền truy cập.${NC}"
    exit 1
fi

# Tạo thư mục từ xa nếu chưa tồn tại
echo -e "${BLUE}Kiểm tra và tạo thư mục từ xa nếu cần...${NC}"
ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "mkdir -p $REMOTE_DIR"

# Sao lưu dữ liệu quan trọng trên server trước khi đồng bộ
echo -e "${BLUE}Sao lưu dữ liệu quan trọng trên server...${NC}"
BACKUP_DIR="$REMOTE_DIR/backup_$(date +%Y%m%d_%H%M%S)"
ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "mkdir -p $BACKUP_DIR && cp -r $REMOTE_DIR/data $BACKUP_DIR/ 2>/dev/null || true && cp -r $REMOTE_DIR/saved_models $BACKUP_DIR/ 2>/dev/null || true && cp $REMOTE_DIR/.env $BACKUP_DIR/ 2>/dev/null || true"

# Thực hiện đồng bộ với rsync
echo -e "${BLUE}Đồng bộ hóa dữ liệu...${NC}"
eval "rsync -avz --progress -e 'ssh -p $SSH_PORT' $EXCLUDE_PARAMS $SYNC_ITEMS $SERVER_USER@$SERVER_IP:$REMOTE_DIR/"

# Cập nhật quyền thực thi cho các file script
echo -e "${BLUE}Cập nhật quyền thực thi cho các file script...${NC}"
ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "chmod +x $REMOTE_DIR/*.sh $REMOTE_DIR/automation_scripts/*.sh 2>/dev/null || true"

# Chạy thiết lập server
echo -e "${BLUE}Chạy script thiết lập server...${NC}"
ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && (chmod +x server_setup.sh && ./server_setup.sh || systemctl restart ethusdt-dashboard)"

# Kiểm tra trạng thái dịch vụ
echo -e "${BLUE}Kiểm tra trạng thái dịch vụ...${NC}"
ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "systemctl status ethusdt-dashboard | grep Active || echo 'Dịch vụ không tìm thấy. Đảm bảo server_setup.sh đã được chạy.'"

echo -e "${GREEN}=== ĐỒNG BỘ HOÀN TẤT ===${NC}"
echo "Bạn có thể truy cập dashboard tại: http://$SERVER_IP:5000"