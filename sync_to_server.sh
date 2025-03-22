#!/bin/bash
# Script đồng bộ ETHUSDT Dashboard từ Replit lên server

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Thông tin server
SERVER="45.76.196.13"
USER="root"
REMOTE_DIR="/root/ethusdt_dashboard"

echo -e "${YELLOW}=== ĐỒNG BỘ ETHUSDT DASHBOARD LÊN SERVER ===${NC}"

# Kiểm tra kết nối đến server
echo -e "${YELLOW}Kiểm tra kết nối đến server...${NC}"
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$SERVER "echo 'Kết nối thành công!'" &> /dev/null; then
    echo -e "${RED}Không thể kết nối đến server. Kiểm tra lại kết nối mạng hoặc thông tin đăng nhập.${NC}"
    exit 1
fi

# Tạo thư mục trên server nếu chưa tồn tại
echo -e "${YELLOW}Tạo thư mục trên server nếu chưa tồn tại...${NC}"
ssh -o StrictHostKeyChecking=no $USER@$SERVER "mkdir -p $REMOTE_DIR"

# Tạo danh sách file cần đồng bộ (loại trừ các file và thư mục không cần thiết)
echo -e "${YELLOW}Chuẩn bị danh sách file cần đồng bộ...${NC}"
cat > .rsync-exclude.txt << EOF
.git/
.gitignore
.env
*.log
*.md
__pycache__/
*.pyc
*.pyo
venv/
node_modules/
.pytest_cache/
.coverage
.rsync-exclude.txt
EOF

# Đồng bộ mã nguồn
echo -e "${YELLOW}Đồng bộ mã nguồn lên server...${NC}"
rsync -avz --exclude-from=.rsync-exclude.txt \
    --delete \
    -e "ssh -o StrictHostKeyChecking=no" \
    ./ $USER@$SERVER:$REMOTE_DIR/

# Chuyển Binance API keys từ biến môi trường của Replit sang server
echo -e "${YELLOW}Chuyển API keys lên server...${NC}"
if [ -n "$BINANCE_API_KEY" ] && [ -n "$BINANCE_API_SECRET" ]; then
    ssh -o StrictHostKeyChecking=no $USER@$SERVER "cat > $REMOTE_DIR/.env << EOF
BINANCE_API_KEY=$BINANCE_API_KEY
BINANCE_API_SECRET=$BINANCE_API_SECRET
EOF"
    echo -e "${GREEN}API keys đã được đồng bộ lên server.${NC}"
else
    echo -e "${RED}Không tìm thấy BINANCE_API_KEY hoặc BINANCE_API_SECRET trong biến môi trường.${NC}"
    echo -e "${YELLOW}Truy cập vào server và tạo file .env với BINANCE_API_KEY và BINANCE_API_SECRET.${NC}"
fi

# Thiết lập quyền thực thi cho các script
echo -e "${YELLOW}Thiết lập quyền thực thi cho các script...${NC}"
ssh -o StrictHostKeyChecking=no $USER@$SERVER "chmod +x $REMOTE_DIR/*.sh $REMOTE_DIR/automation_scripts/*.sh"

# Kiểm tra requirements_server.txt và cài đặt nếu cần
echo -e "${YELLOW}Kiểm tra requirements_server.txt để cài đặt thư viện mới...${NC}"
ssh -o StrictHostKeyChecking=no $USER@$SERVER "if [ -f \"$REMOTE_DIR/venv/bin/pip\" ] && [ -f \"$REMOTE_DIR/requirements_server.txt\" ]; then 
    $REMOTE_DIR/venv/bin/pip install -r $REMOTE_DIR/requirements_server.txt
fi"

# Khởi động lại ứng dụng trên server
echo -e "${YELLOW}Khởi động lại ứng dụng trên server...${NC}"
ssh -o StrictHostKeyChecking=no $USER@$SERVER "$REMOTE_DIR/restart.sh" || {
    echo -e "${RED}Không thể khởi động lại ứng dụng. Có thể cần thiết lập môi trường lần đầu.${NC}"
    echo -e "${YELLOW}Kết nối SSH vào server và chạy: $REMOTE_DIR/server_setup.sh${NC}"
}

# Xóa file exclude tạm thời
rm -f .rsync-exclude.txt

echo -e "${GREEN}=== ĐỒNG BỘ HOÀN TẤT ===${NC}"
echo -e "${GREEN}Truy cập ứng dụng tại: http://$SERVER:5000${NC}"