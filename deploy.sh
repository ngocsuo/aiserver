#!/bin/bash
# Script triển khai nhanh lên server

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== TRIỂN KHAI NHANH ETHUSDT DASHBOARD ===${NC}"
echo "Thời gian: $(date)"

# Chạy script post_update_hook.sh nếu tồn tại
if [ -f "automation_scripts/post_update_hook.sh" ]; then
    echo -e "${BLUE}Chạy script đồng bộ tự động...${NC}"
    ./automation_scripts/post_update_hook.sh
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Đồng bộ thành công!${NC}"
    else
        echo -e "${RED}Đồng bộ thất bại! Mã lỗi: $exit_code${NC}"
        echo "Đang thử phương pháp dự phòng..."
        if [ -f "sync_to_server.sh" ]; then
            echo -e "${BLUE}Chạy script đồng bộ thay thế...${NC}"
            ./sync_to_server.sh
        else
            echo -e "${RED}Không tìm thấy script đồng bộ dự phòng. Kết thúc!${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}Không tìm thấy script đồng bộ tự động!${NC}"
    echo "Đang tìm kiếm các phương pháp dự phòng..."
    
    if [ -f "sync_to_server.sh" ]; then
        echo -e "${BLUE}Tìm thấy script sync_to_server.sh. Đang chạy...${NC}"
        ./sync_to_server.sh
    else
        echo -e "${YELLOW}Không tìm thấy script đồng bộ. Tạo script tạm thời...${NC}"
        
        # Tạo script tạm thời
        cat > temp_deploy.sh << 'EOF'
#!/bin/bash
SERVER="45.76.196.13"
USER="root"
REMOTE_DIR="/root/ethusdt_dashboard"

echo "Đồng bộ code lên server..."
rsync -avz --exclude=".*" --exclude="venv" --exclude="__pycache__" \
    --exclude="*.pyc" --exclude="*.pyo" --exclude="*.log" \
    ./* $USER@$SERVER:$REMOTE_DIR/

echo "Khởi động lại service..."
ssh $USER@$SERVER "cd $REMOTE_DIR && chmod +x server_setup.sh && ./server_setup.sh || systemctl restart ethusdt-dashboard"
EOF
        
        chmod +x temp_deploy.sh
        ./temp_deploy.sh
        rm temp_deploy.sh
    fi
fi

echo -e "${GREEN}=== TRIỂN KHAI HOÀN TẤT ===${NC}"