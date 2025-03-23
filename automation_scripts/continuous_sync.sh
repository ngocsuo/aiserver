#!/bin/bash
# Script tự động đồng bộ khi có thay đổi

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Cấu hình
SERVER="45.76.196.13"
USER="root"
REMOTE_DIR="/root/ethusdt_dashboard"
LOCAL_DIR="."
SYNC_INTERVAL=60  # Kiểm tra mỗi 60 giây
LOG_FILE="sync_log.txt"
CHECKSUM_FILE=".last_checksum"

# Hiển thị trạng thái
echo -e "${YELLOW}=== ĐỒNG BỘ LIÊN TỤC ETHUSDT DASHBOARD ===${NC}"
echo "Thời gian: $(date)"
echo "Kiểm tra mỗi ${SYNC_INTERVAL} giây"
echo "Log file: ${LOG_FILE}"

# Kiểm tra kết nối SSH
echo -e "${BLUE}Kiểm tra kết nối tới server...${NC}"
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$SERVER "echo 'Kết nối thành công!'" &> /dev/null; then
    echo -e "${RED}Không thể kết nối đến server. Kiểm tra lại thông tin đăng nhập hoặc kết nối mạng.${NC}"
    exit 1
fi
echo -e "${GREEN}Kết nối thành công!${NC}"

# Tạo thư mục đích nếu chưa tồn tại
ssh $USER@$SERVER "mkdir -p $REMOTE_DIR"

# Hàm tính toán checksum của thư mục
calculate_checksum() {
    find $LOCAL_DIR -type f -not -path "*/\.*" -not -path "*/venv/*" -not -path "*/__pycache__/*" \
        -not -name "*.pyc" -not -name "*.pyo" -not -name "*.log" | \
        sort | xargs stat -c "%Y %s %n" 2>/dev/null | md5sum | cut -d' ' -f1
}

# Hàm đồng bộ
sync_to_server() {
    echo -e "${BLUE}Đồng bộ code lên server...${NC}" | tee -a $LOG_FILE
    echo "$(date): Bắt đầu đồng bộ hóa" >> $LOG_FILE
    
    # Đồng bộ bằng rsync hoặc scp
    rsync -avz --exclude=".*" --exclude="venv" --exclude="__pycache__" \
        --exclude="*.pyc" --exclude="*.pyo" --exclude="*.log" \
        "$LOCAL_DIR/" "$USER@$SERVER:$REMOTE_DIR/" | tee -a $LOG_FILE
    
    # Khởi động lại service
    if ssh $USER@$SERVER "systemctl is-active ethusdt-dashboard &> /dev/null"; then
        echo -e "${BLUE}Khởi động lại service...${NC}" | tee -a $LOG_FILE
        ssh $USER@$SERVER "systemctl restart ethusdt-dashboard" | tee -a $LOG_FILE
        
        # Kiểm tra trạng thái
        SERVICE_STATUS=$(ssh $USER@$SERVER "systemctl is-active ethusdt-dashboard")
        if [ "$SERVICE_STATUS" = "active" ]; then
            echo -e "${GREEN}Service khởi động lại thành công!${NC}" | tee -a $LOG_FILE
        else
            echo -e "${RED}Có lỗi khi khởi động lại service.${NC}" | tee -a $LOG_FILE
        fi
    else
        echo -e "${YELLOW}Service chưa tồn tại, chạy setup...${NC}" | tee -a $LOG_FILE
        ssh $USER@$SERVER "cd $REMOTE_DIR && [ -f server_setup.sh ] && chmod +x server_setup.sh && ./server_setup.sh" | tee -a $LOG_FILE
    fi
    
    echo "$(date): Đồng bộ hóa hoàn tất" >> $LOG_FILE
    echo -e "${GREEN}Đồng bộ hoàn tất!${NC}"
}

# Vòng lặp đồng bộ
echo -e "${BLUE}Bắt đầu vòng lặp đồng bộ. Nhấn Ctrl+C để dừng.${NC}"
current_checksum=""
saved_checksum=""

# Kiểm tra nếu có checksum cũ
if [ -f "$CHECKSUM_FILE" ]; then
    saved_checksum=$(cat "$CHECKSUM_FILE")
fi

# Đồng bộ lần đầu tiên
current_checksum=$(calculate_checksum)
echo "$current_checksum" > "$CHECKSUM_FILE"
sync_to_server

while true; do
    sleep $SYNC_INTERVAL
    
    # Tính toán checksum hiện tại
    current_checksum=$(calculate_checksum)
    
    # Kiểm tra xem có thay đổi không
    if [ "$current_checksum" != "$saved_checksum" ]; then
        echo -e "${YELLOW}Phát hiện thay đổi, tiến hành đồng bộ...${NC}"
        sync_to_server
        saved_checksum="$current_checksum"
        echo "$current_checksum" > "$CHECKSUM_FILE"
    else
        echo -e "${GREEN}$(date): Không có thay đổi.${NC}"
    fi
done