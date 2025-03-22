#!/bin/bash
# Script đồng bộ hóa ETHUSDT Dashboard lên server

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
LOG_FILE="sync_log.txt"

# Danh sách các thư mục cần đồng bộ
DIRS_TO_SYNC=(
    "dashboard"
    "models"
    "prediction"
    "utils"
)

# Danh sách các file cần đồng bộ
FILES_TO_SYNC=(
    "app.py"
    "config.py"
    "requirements_server.txt"
    "*.md"
    "api.py"
)

# Danh sách các file và thư mục loại trừ
EXCLUDES=(
    ".git"
    ".gitignore"
    "__pycache__"
    "*.pyc"
    "venv"
    ".env"
    "node_modules"
    ".DS_Store"
    "data/*.csv"
    "data/*.json"
    "logs/*.log"
)

echo -e "${YELLOW}=== ĐỒNG BỘ ETHUSDT DASHBOARD LÊN SERVER ===${NC}" | tee -a $LOG_FILE
echo "$(date): Bắt đầu đồng bộ hóa" | tee -a $LOG_FILE

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
EXCLUDE_ARGS=""
for EXCLUDE in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude='$EXCLUDE'"
done

# Đồng bộ các thư mục
echo -e "${BLUE}Đồng bộ các thư mục...${NC}" | tee -a $LOG_FILE
for DIR in "${DIRS_TO_SYNC[@]}"; do
    if [ -d "$LOCAL_DIR/$DIR" ]; then
        echo "Đồng bộ thư mục: $DIR" | tee -a $LOG_FILE
        eval rsync -avz --progress $EXCLUDE_ARGS "$LOCAL_DIR/$DIR/" "$USER@$SERVER:$REMOTE_DIR/$DIR/" | tee -a $LOG_FILE
    else
        echo -e "${YELLOW}Thư mục $DIR không tồn tại, bỏ qua.${NC}" | tee -a $LOG_FILE
    fi
done

# Đồng bộ các file
echo -e "${BLUE}Đồng bộ các file...${NC}" | tee -a $LOG_FILE
for FILE_PATTERN in "${FILES_TO_SYNC[@]}"; do
    FILE_COUNT=$(ls -1 $LOCAL_DIR/$FILE_PATTERN 2>/dev/null | wc -l)
    if [ $FILE_COUNT -gt 0 ]; then
        echo "Đồng bộ: $FILE_PATTERN" | tee -a $LOG_FILE
        for FILE in $LOCAL_DIR/$FILE_PATTERN; do
            if [ -f "$FILE" ]; then
                BASENAME=$(basename "$FILE")
                rsync -avz --progress "$FILE" "$USER@$SERVER:$REMOTE_DIR/$BASENAME" | tee -a $LOG_FILE
            fi
        done
    else
        echo -e "${YELLOW}Không tìm thấy file nào khớp với mẫu $FILE_PATTERN, bỏ qua.${NC}" | tee -a $LOG_FILE
    fi
done

# Đồng bộ scripts và đặt quyền thực thi
echo -e "${BLUE}Đồng bộ các script...${NC}" | tee -a $LOG_FILE
if [ -f "$LOCAL_DIR/server_setup.sh" ]; then
    rsync -avz --progress "$LOCAL_DIR/server_setup.sh" "$USER@$SERVER:$REMOTE_DIR/server_setup.sh" | tee -a $LOG_FILE
    ssh $USER@$SERVER "chmod +x $REMOTE_DIR/server_setup.sh" | tee -a $LOG_FILE
fi

# Kiểm tra và đồng bộ requirements_server.txt
if [ -f "$LOCAL_DIR/requirements_server.txt" ]; then
    echo -e "${BLUE}Đồng bộ requirements_server.txt...${NC}" | tee -a $LOG_FILE
    rsync -avz --progress "$LOCAL_DIR/requirements_server.txt" "$USER@$SERVER:$REMOTE_DIR/requirements_server.txt" | tee -a $LOG_FILE
else
    echo -e "${YELLOW}File requirements_server.txt không tồn tại, tạo file tạm thời...${NC}" | tee -a $LOG_FILE
    cat > "$LOCAL_DIR/requirements_server.txt.tmp" << EOF
streamlit>=1.31.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.14.0
python-binance>=1.0.16
scikit-learn>=1.2.0
tensorflow>=2.12.0
requests>=2.28.0
websocket-client>=1.5.0
python-dotenv>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
psutil>=5.9.0
EOF
    rsync -avz --progress "$LOCAL_DIR/requirements_server.txt.tmp" "$USER@$SERVER:$REMOTE_DIR/requirements_server.txt" | tee -a $LOG_FILE
    rm "$LOCAL_DIR/requirements_server.txt.tmp"
fi

# Khởi động lại service
echo -e "${BLUE}Khởi động lại service trên server...${NC}" | tee -a $LOG_FILE
# Kiểm tra xem service đã tồn tại chưa
if ssh $USER@$SERVER "systemctl is-active ethusdt-dashboard &> /dev/null"; then
    echo "Service đã tồn tại, chỉ khởi động lại..." | tee -a $LOG_FILE
    ssh $USER@$SERVER "systemctl restart ethusdt-dashboard" | tee -a $LOG_FILE
    # Kiểm tra trạng thái service
    SSH_OUTPUT=$(ssh $USER@$SERVER "systemctl status ethusdt-dashboard")
    if echo "$SSH_OUTPUT" | grep -q "Active: active (running)"; then
        echo -e "${GREEN}Service đã khởi động lại thành công.${NC}" | tee -a $LOG_FILE
    else
        echo -e "${RED}Có lỗi khi khởi động lại service.${NC}" | tee -a $LOG_FILE
        echo -e "${BLUE}Chi tiết:${NC}\n$SSH_OUTPUT" | tee -a $LOG_FILE
    fi
else
    echo "Service chưa tồn tại, chạy server_setup.sh..." | tee -a $LOG_FILE
    ssh $USER@$SERVER "cd $REMOTE_DIR && chmod +x server_setup.sh && ./server_setup.sh" | tee -a $LOG_FILE
fi

# Hiển thị log
echo -e "${BLUE}10 dòng log gần nhất từ server:${NC}" | tee -a $LOG_FILE
ssh $USER@$SERVER "journalctl -u ethusdt-dashboard -n 10 --no-pager" | tee -a $LOG_FILE

echo -e "${GREEN}=== ĐỒNG BỘ HOÀN TẤT ===${NC}" | tee -a $LOG_FILE
echo "$(date): Kết thúc đồng bộ hóa" | tee -a $LOG_FILE
echo -e "Xem log chi tiết tại: ${LOG_FILE}"