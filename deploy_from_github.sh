#!/bin/bash
# Script triển khai ETHUSDT Dashboard từ GitHub đến máy chủ

# Màu sắc cho đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Cấu hình
SERVER_IP=""
SERVER_USER="root"
SSH_PORT="22"
REMOTE_DIR="/root/ethusdt_dashboard"
GITHUB_REPO=""
BRANCH="main"

echo -e "${YELLOW}=== TRIỂN KHAI ETHUSDT DASHBOARD TỪ GITHUB ===${NC}"
echo "Thời gian: $(date)"

# Yêu cầu thông tin nếu chưa cấu hình
if [ -z "$SERVER_IP" ]; then
    read -p "Nhập địa chỉ IP máy chủ: " SERVER_IP
fi

if [ -z "$SSH_PORT" ]; then
    read -p "Nhập cổng SSH (mặc định: 22): " SSH_PORT
    SSH_PORT=${SSH_PORT:-22}
fi

if [ -z "$GITHUB_REPO" ]; then
    read -p "Nhập URL repository GitHub (ví dụ: https://github.com/username/ethusdt-dashboard.git): " GITHUB_REPO
fi

echo -e "${BLUE}Thông tin kết nối:${NC}"
echo "Server: $SERVER_USER@$SERVER_IP:$SSH_PORT"
echo "Thư mục từ xa: $REMOTE_DIR"
echo "GitHub Repository: $GITHUB_REPO"
echo "Nhánh: $BRANCH"

# Kiểm tra kết nối SSH
echo -e "${BLUE}Kiểm tra kết nối đến máy chủ...${NC}"
if ! ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "echo 'Kết nối thành công'" > /dev/null 2>&1; then
    echo -e "${RED}Không thể kết nối đến máy chủ. Vui lòng kiểm tra thông tin và quyền truy cập.${NC}"
    exit 1
fi

# Kiểm tra Git trên máy chủ
echo -e "${BLUE}Kiểm tra Git trên máy chủ...${NC}"
if ! ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "command -v git" > /dev/null 2>&1; then
    echo -e "${BLUE}Cài đặt Git trên máy chủ...${NC}"
    ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "apt-get update && apt-get install -y git"
fi

# Sao lưu dữ liệu và cấu hình quan trọng trên máy chủ
echo -e "${BLUE}Sao lưu dữ liệu quan trọng trên máy chủ...${NC}"
BACKUP_DIR="$REMOTE_DIR/backup_$(date +%Y%m%d_%H%M%S)"
ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "mkdir -p $BACKUP_DIR && \
    if [ -d \"$REMOTE_DIR\" ]; then \
        cp -r $REMOTE_DIR/data $BACKUP_DIR/ 2>/dev/null || true && \
        cp -r $REMOTE_DIR/saved_models $BACKUP_DIR/ 2>/dev/null || true && \
        cp $REMOTE_DIR/.env $BACKUP_DIR/ 2>/dev/null || true && \
        cp $REMOTE_DIR/binance_time.json $BACKUP_DIR/ 2>/dev/null || true && \
        cp -r $REMOTE_DIR/logs $BACKUP_DIR/ 2>/dev/null || true; \
    fi"

# Triển khai từ GitHub
echo -e "${BLUE}Triển khai mã nguồn từ GitHub...${NC}"
ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "if [ -d \"$REMOTE_DIR/.git\" ]; then \
    cd $REMOTE_DIR && git fetch --all && git reset --hard origin/$BRANCH && git pull; \
else \
    rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR && \
    git clone --depth 1 -b $BRANCH $GITHUB_REPO $REMOTE_DIR; \
fi"

# Khôi phục dữ liệu và cấu hình quan trọng
echo -e "${BLUE}Khôi phục dữ liệu và cấu hình quan trọng...${NC}"
ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "if [ -d \"$BACKUP_DIR\" ]; then \
    mkdir -p $REMOTE_DIR/data $REMOTE_DIR/saved_models $REMOTE_DIR/logs && \
    cp -r $BACKUP_DIR/data/* $REMOTE_DIR/data/ 2>/dev/null || true && \
    cp -r $BACKUP_DIR/saved_models/* $REMOTE_DIR/saved_models/ 2>/dev/null || true && \
    cp $BACKUP_DIR/.env $REMOTE_DIR/ 2>/dev/null || true && \
    cp $BACKUP_DIR/binance_time.json $REMOTE_DIR/ 2>/dev/null || true && \
    cp -r $BACKUP_DIR/logs/* $REMOTE_DIR/logs/ 2>/dev/null || true; \
fi"

# Chạy thiết lập server
echo -e "${BLUE}Chạy script thiết lập server...${NC}"
ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "cd $REMOTE_DIR && (chmod +x server_setup.sh && ./server_setup.sh || systemctl restart ethusdt-dashboard)"

# Kiểm tra trạng thái dịch vụ
echo -e "${BLUE}Kiểm tra trạng thái dịch vụ...${NC}"
ssh -p $SSH_PORT $SERVER_USER@$SERVER_IP "systemctl status ethusdt-dashboard | grep Active || echo 'Dịch vụ không tìm thấy. Đảm bảo server_setup.sh đã được chạy.'"

echo -e "${GREEN}=== TRIỂN KHAI HOÀN TẤT ===${NC}"
echo "Bạn có thể truy cập dashboard tại: http://$SERVER_IP:5000"
echo "Các bước tiếp theo:"
echo "1. Kiểm tra ứng dụng hoạt động"
echo "2. Kiểm tra logs bằng lệnh: ssh $SERVER_USER@$SERVER_IP -p $SSH_PORT 'journalctl -fu ethusdt-dashboard'"