#!/bin/bash
# Script kiểm tra trạng thái server ETHUSDT Dashboard

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

echo -e "${YELLOW}=== KIỂM TRA TRẠNG THÁI SERVER ETHUSDT DASHBOARD ===${NC}"
echo "Thời gian: $(date)"

# Kiểm tra kết nối SSH
echo -e "${BLUE}Kiểm tra kết nối SSH đến server...${NC}"
if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$SERVER "echo 'Kết nối thành công!'" &> /dev/null; then
    echo -e "${GREEN}Kết nối SSH thành công!${NC}"
else
    echo -e "${RED}Không thể kết nối đến server. Kiểm tra lại thông tin đăng nhập hoặc kết nối mạng.${NC}"
    exit 1
fi

# Kiểm tra dung lượng ổ đĩa
echo -e "${BLUE}Kiểm tra dung lượng ổ đĩa...${NC}"
DISK_USAGE=$(ssh $USER@$SERVER "df -h / | grep -v Filesystem")
DISK_USED=$(echo $DISK_USAGE | awk '{print $5}' | sed 's/%//')
DISK_AVAIL=$(echo $DISK_USAGE | awk '{print $4}')

if [ $DISK_USED -gt 90 ]; then
    echo -e "${RED}Cảnh báo: Ổ đĩa sử dụng $DISK_USED% (Còn trống: $DISK_AVAIL)${NC}"
elif [ $DISK_USED -gt 80 ]; then
    echo -e "${YELLOW}Chú ý: Ổ đĩa sử dụng $DISK_USED% (Còn trống: $DISK_AVAIL)${NC}"
else
    echo -e "${GREEN}Ổ đĩa sử dụng $DISK_USED% (Còn trống: $DISK_AVAIL)${NC}"
fi

# Kiểm tra RAM
echo -e "${BLUE}Kiểm tra RAM...${NC}"
MEM_INFO=$(ssh $USER@$SERVER "free -m | grep Mem")
MEM_TOTAL=$(echo $MEM_INFO | awk '{print $2}')
MEM_USED=$(echo $MEM_INFO | awk '{print $3}')
MEM_AVAIL=$(echo $MEM_INFO | awk '{print $7}')
MEM_PERCENT=$((MEM_USED * 100 / MEM_TOTAL))

if [ $MEM_PERCENT -gt 90 ]; then
    echo -e "${RED}Cảnh báo: RAM sử dụng $MEM_PERCENT% ($MEM_USED MB / $MEM_TOTAL MB, Còn trống: $MEM_AVAIL MB)${NC}"
elif [ $MEM_PERCENT -gt 80 ]; then
    echo -e "${YELLOW}Chú ý: RAM sử dụng $MEM_PERCENT% ($MEM_USED MB / $MEM_TOTAL MB, Còn trống: $MEM_AVAIL MB)${NC}"
else
    echo -e "${GREEN}RAM sử dụng $MEM_PERCENT% ($MEM_USED MB / $MEM_TOTAL MB, Còn trống: $MEM_AVAIL MB)${NC}"
fi

# Kiểm tra CPU
echo -e "${BLUE}Kiểm tra CPU...${NC}"
CPU_LOAD=$(ssh $USER@$SERVER "cat /proc/loadavg")
LOAD_1M=$(echo $CPU_LOAD | awk '{print $1}')
LOAD_5M=$(echo $CPU_LOAD | awk '{print $2}')
LOAD_15M=$(echo $CPU_LOAD | awk '{print $3}')
CPU_CORES=$(ssh $USER@$SERVER "nproc")
LOAD_PERCENT=$(awk -v load=$LOAD_1M -v cores=$CPU_CORES 'BEGIN {printf "%.0f", load * 100 / cores}')

if [ $LOAD_PERCENT -gt 90 ]; then
    echo -e "${RED}Cảnh báo: CPU load ${LOAD_PERCENT}% (1m: ${LOAD_1M}, 5m: ${LOAD_5M}, 15m: ${LOAD_15M}, Số lõi: ${CPU_CORES})${NC}"
elif [ $LOAD_PERCENT -gt 70 ]; then
    echo -e "${YELLOW}Chú ý: CPU load ${LOAD_PERCENT}% (1m: ${LOAD_1M}, 5m: ${LOAD_5M}, 15m: ${LOAD_15M}, Số lõi: ${CPU_CORES})${NC}"
else
    echo -e "${GREEN}CPU load ${LOAD_PERCENT}% (1m: ${LOAD_1M}, 5m: ${LOAD_5M}, 15m: ${LOAD_15M}, Số lõi: ${CPU_CORES})${NC}"
fi

# Kiểm tra trạng thái service
echo -e "${BLUE}Kiểm tra trạng thái service...${NC}"
SERVICE_STATUS=$(ssh $USER@$SERVER "systemctl is-active ethusdt-dashboard")
if [ "$SERVICE_STATUS" == "active" ]; then
    echo -e "${GREEN}Service đang chạy (active)${NC}"
    
    # Kiểm tra thời gian chạy liên tục
    UPTIME=$(ssh $USER@$SERVER "systemctl show ethusdt-dashboard -p ActiveEnterTimestamp | sed 's/ActiveEnterTimestamp=//'")
    if [ ! -z "$UPTIME" ]; then
        CURRENT_TIME=$(ssh $USER@$SERVER "date +%s")
        UPTIME_SECONDS=$((CURRENT_TIME - $(date -d "$UPTIME" +%s)))
        UPTIME_DAYS=$((UPTIME_SECONDS / 86400))
        UPTIME_HOURS=$(((UPTIME_SECONDS % 86400) / 3600))
        UPTIME_MINUTES=$(((UPTIME_SECONDS % 3600) / 60))
        
        echo -e "${GREEN}Thời gian chạy: ${UPTIME_DAYS} ngày, ${UPTIME_HOURS} giờ, ${UPTIME_MINUTES} phút${NC}"
    fi
else
    echo -e "${RED}Service không chạy ($SERVICE_STATUS)${NC}"
    
    # Kiểm tra log để xem lý do lỗi
    echo -e "${BLUE}Kiểm tra log lỗi...${NC}"
    ERROR_LOG=$(ssh $USER@$SERVER "journalctl -u ethusdt-dashboard -n 20 --no-pager | grep -i error")
    if [ ! -z "$ERROR_LOG" ]; then
        echo -e "${RED}Lỗi được phát hiện trong log:${NC}"
        echo "$ERROR_LOG"
    fi
fi

# Kiểm tra cổng dịch vụ
echo -e "${BLUE}Kiểm tra cổng dịch vụ...${NC}"
if ssh $USER@$SERVER "netstat -tuln | grep -q ':5000'"; then
    echo -e "${GREEN}Cổng 5000 đang mở và lắng nghe kết nối${NC}"
else
    echo -e "${RED}Cổng 5000 không mở! Dịch vụ có thể không chạy chính xác.${NC}"
fi

# Kiểm tra các quá trình Python đang chạy
echo -e "${BLUE}Kiểm tra các quá trình Python đang chạy...${NC}"
PYTHON_PROCESSES=$(ssh $USER@$SERVER "ps aux | grep python | grep -v grep")
if [ ! -z "$PYTHON_PROCESSES" ]; then
    PROCESS_COUNT=$(echo "$PYTHON_PROCESSES" | wc -l)
    echo -e "${GREEN}Có ${PROCESS_COUNT} quá trình Python đang chạy:${NC}"
    ssh $USER@$SERVER "ps aux | grep python | grep -v grep | head -5"
    if [ $PROCESS_COUNT -gt 5 ]; then
        echo -e "${YELLOW}...và ${PROCESS_COUNT-5} quá trình khác${NC}"
    fi
else
    echo -e "${RED}Không có quá trình Python nào đang chạy!${NC}"
fi

# Kiểm tra log ứng dụng
echo -e "${BLUE}Kiểm tra log ứng dụng mới nhất...${NC}"
if ssh $USER@$SERVER "[ -f $REMOTE_DIR/app.log ] && echo 'exists'"; then
    APP_LOG=$(ssh $USER@$SERVER "tail -n 10 $REMOTE_DIR/app.log")
    echo -e "${GREEN}10 dòng log mới nhất:${NC}"
    echo "$APP_LOG"
else
    echo -e "${YELLOW}Không tìm thấy file log ứng dụng ($REMOTE_DIR/app.log)${NC}"
fi

# Kiểm tra các model đã huấn luyện
echo -e "${BLUE}Kiểm tra các model đã huấn luyện...${NC}"
MODELS_COUNT=$(ssh $USER@$SERVER "find $REMOTE_DIR/models -name \"*.h5\" | wc -l")
if [ $MODELS_COUNT -gt 0 ]; then
    echo -e "${GREEN}Có ${MODELS_COUNT} model đã được huấn luyện:${NC}"
    ssh $USER@$SERVER "find $REMOTE_DIR/models -name \"*.h5\" -printf \"%f\n\" | head -5"
    if [ $MODELS_COUNT -gt 5 ]; then
        echo -e "${YELLOW}...và ${MODELS_COUNT-5} model khác${NC}"
    fi
else
    echo -e "${YELLOW}Không tìm thấy model nào đã được huấn luyện.${NC}"
fi

# Kiểm tra log training
echo -e "${BLUE}Kiểm tra log huấn luyện...${NC}"
if ssh $USER@$SERVER "[ -f $REMOTE_DIR/training_logs.txt ] && echo 'exists'"; then
    TRAINING_LOG=$(ssh $USER@$SERVER "tail -n 10 $REMOTE_DIR/training_logs.txt")
    echo -e "${GREEN}10 dòng log huấn luyện mới nhất:${NC}"
    echo "$TRAINING_LOG"
else
    echo -e "${YELLOW}Không tìm thấy file log huấn luyện ($REMOTE_DIR/training_logs.txt)${NC}"
fi

# Kiểm tra dữ liệu đã tải
echo -e "${BLUE}Kiểm tra dữ liệu đã tải...${NC}"
DATA_COUNT=$(ssh $USER@$SERVER "find $REMOTE_DIR/data -type f | wc -l")
if [ $DATA_COUNT -gt 0 ]; then
    DATA_SIZE=$(ssh $USER@$SERVER "du -sh $REMOTE_DIR/data | cut -f1")
    echo -e "${GREEN}Có ${DATA_COUNT} file dữ liệu (Kích thước: ${DATA_SIZE})${NC}"
else
    echo -e "${YELLOW}Không tìm thấy dữ liệu đã tải.${NC}"
fi

echo -e "${YELLOW}=== KIỂM TRA TRẠNG THÁI HOÀN TẤT ===${NC}"