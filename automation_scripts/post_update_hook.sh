#!/bin/bash
# Hook chạy sau khi cập nhật code

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
LOG_FILE="deploy_log.txt"

echo -e "${YELLOW}=== ĐỒNG BỘ SAU KHI CẬP NHẬT ===${NC}" | tee -a $LOG_FILE
echo "$(date): Bắt đầu đồng bộ tự động" | tee -a $LOG_FILE

# Danh sách các thư mục cần đồng bộ
DIRS_TO_SYNC=(
    "dashboard"
    "models"
    "prediction"
    "utils"
    "automation_scripts"
)

# Danh sách các file quan trọng cần đồng bộ
FILES_TO_SYNC=(
    "app.py"
    "config.py"
    "requirements_server.txt"
    "*.md"
    "api.py"
    "thread_safe_logging.py"
    "continuous_trainer_fix.py"
    "continuous_trainer_fixed.py"
    "enhanced_data_collector.py"
    "enhanced_proxy_config.py"
)

# Kiểm tra kết nối SSH
echo -e "${BLUE}Kiểm tra kết nối tới server...${NC}" | tee -a $LOG_FILE
if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$SERVER "echo 'Kết nối thành công!'" &> /dev/null; then
    echo -e "${GREEN}Kết nối thành công!${NC}" | tee -a $LOG_FILE
else
    echo -e "${RED}Không thể kết nối đến server. Kiểm tra lại thông tin đăng nhập hoặc kết nối mạng.${NC}" | tee -a $LOG_FILE
    exit 1
fi

# Đảm bảo thư mục đích tồn tại
echo -e "${BLUE}Tạo thư mục đích trên server...${NC}" | tee -a $LOG_FILE
ssh $USER@$SERVER "mkdir -p $REMOTE_DIR" | tee -a $LOG_FILE
ssh $USER@$SERVER "mkdir -p $REMOTE_DIR/data" | tee -a $LOG_FILE
ssh $USER@$SERVER "mkdir -p $REMOTE_DIR/logs" | tee -a $LOG_FILE
ssh $USER@$SERVER "mkdir -p $REMOTE_DIR/models/saved" | tee -a $LOG_FILE

# Xây dựng câu lệnh rsync loại trừ
EXCLUDE_ARGS="--exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='venv' --exclude='.env' --exclude='node_modules' --exclude='.DS_Store' --exclude='data/*.csv' --exclude='data/*.json' --exclude='logs/*.log'"

# Đồng bộ các thư mục
echo -e "${BLUE}Đồng bộ các thư mục...${NC}" | tee -a $LOG_FILE
for DIR in "${DIRS_TO_SYNC[@]}"; do
    if [ -d "$DIR" ]; then
        echo "Đồng bộ thư mục: $DIR" | tee -a $LOG_FILE
        eval rsync -avz $EXCLUDE_ARGS "$DIR/" "$USER@$SERVER:$REMOTE_DIR/$DIR/" | tee -a $LOG_FILE
    else
        echo -e "${YELLOW}Thư mục $DIR không tồn tại, bỏ qua.${NC}" | tee -a $LOG_FILE
    fi
done

# Đồng bộ các file
echo -e "${BLUE}Đồng bộ các file...${NC}" | tee -a $LOG_FILE
for FILE_PATTERN in "${FILES_TO_SYNC[@]}"; do
    FILE_COUNT=$(ls -1 $FILE_PATTERN 2>/dev/null | wc -l)
    if [ $FILE_COUNT -gt 0 ]; then
        echo "Đồng bộ: $FILE_PATTERN" | tee -a $LOG_FILE
        for FILE in $FILE_PATTERN; do
            if [ -f "$FILE" ]; then
                BASENAME=$(basename "$FILE")
                rsync -avz "$FILE" "$USER@$SERVER:$REMOTE_DIR/$BASENAME" | tee -a $LOG_FILE
            fi
        done
    else
        echo -e "${YELLOW}Không tìm thấy file nào khớp với mẫu $FILE_PATTERN, bỏ qua.${NC}" | tee -a $LOG_FILE
    fi
done

# Khởi động lại service
echo -e "${BLUE}Khởi động lại service trên server...${NC}" | tee -a $LOG_FILE
if ssh $USER@$SERVER "systemctl is-active ethusdt-dashboard &> /dev/null"; then
    echo "Service đã tồn tại, chỉ khởi động lại..." | tee -a $LOG_FILE
    ssh $USER@$SERVER "systemctl restart ethusdt-dashboard" | tee -a $LOG_FILE
    
    # Kiểm tra trạng thái service
    SERVICE_STATUS=$(ssh $USER@$SERVER "systemctl is-active ethusdt-dashboard")
    if [ "$SERVICE_STATUS" = "active" ]; then
        echo -e "${GREEN}Service đã khởi động lại thành công.${NC}" | tee -a $LOG_FILE
    else
        echo -e "${RED}Có lỗi khi khởi động lại service.${NC}" | tee -a $LOG_FILE
    fi
else
    echo "Service chưa tồn tại, chạy setup script..." | tee -a $LOG_FILE
    ssh $USER@$SERVER "cd $REMOTE_DIR && chmod +x server_setup.sh && ./server_setup.sh" | tee -a $LOG_FILE
fi

echo -e "${GREEN}=== ĐỒNG BỘ TỰ ĐỘNG HOÀN TẤT ===${NC}" | tee -a $LOG_FILE
echo "$(date): Kết thúc đồng bộ tự động" | tee -a $LOG_FILE