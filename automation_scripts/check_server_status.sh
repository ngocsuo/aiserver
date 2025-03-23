#!/bin/bash
# Kiểm tra trạng thái server ETHUSDT Dashboard

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
DASHBOARD_PORT="5000"

echo -e "${YELLOW}=== KIỂM TRA TRẠNG THÁI SERVER ETHUSDT DASHBOARD ===${NC}"
echo "Thời gian: $(date)"

# Kiểm tra kết nối SSH
echo -e "${BLUE}Kiểm tra kết nối SSH...${NC}"
if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$SERVER "echo 'Kết nối thành công!'" &> /dev/null; then
    echo -e "${GREEN}✓ Kết nối SSH thành công!${NC}"
else
    echo -e "${RED}✗ Không thể kết nối qua SSH. Kiểm tra lại thông tin đăng nhập hoặc kết nối mạng.${NC}"
    exit 1
fi

# Kiểm tra uptime và tải CPU
echo -e "${BLUE}Kiểm tra uptime và tải CPU...${NC}"
UPTIME=$(ssh $USER@$SERVER "uptime")
echo "Uptime: $UPTIME"

LOAD=$(ssh $USER@$SERVER "cat /proc/loadavg | awk '{print \$1,\$2,\$3}'")
echo -e "Tải CPU: ${LOAD}"

# Kiểm tra bộ nhớ RAM
echo -e "${BLUE}Kiểm tra bộ nhớ RAM...${NC}"
RAM_INFO=$(ssh $USER@$SERVER "free -h | grep -i 'mem'")
echo "$RAM_INFO"

# Kiểm tra dung lượng ổ cứng
echo -e "${BLUE}Kiểm tra dung lượng ổ cứng...${NC}"
DISK_INFO=$(ssh $USER@$SERVER "df -h / | grep -v Filesystem")
echo "$DISK_INFO"

# Kiểm tra trạng thái dịch vụ
echo -e "${BLUE}Kiểm tra trạng thái dịch vụ...${NC}"
SERVICE_STATUS=$(ssh $USER@$SERVER "systemctl is-active ethusdt-dashboard")

if [ "$SERVICE_STATUS" = "active" ]; then
    echo -e "${GREEN}✓ Dịch vụ ethusdt-dashboard đang hoạt động (active)${NC}"
    
    # Lấy thêm chi tiết về dịch vụ
    SERVICE_DETAILS=$(ssh $USER@$SERVER "systemctl status ethusdt-dashboard | grep -E 'Active:|Main PID:'")
    echo "$SERVICE_DETAILS"
else
    echo -e "${RED}✗ Dịch vụ ethusdt-dashboard không hoạt động (inactive)${NC}"
    
    # Kiểm tra log gần nhất
    echo "Log gần nhất:"
    ssh $USER@$SERVER "journalctl -u ethusdt-dashboard -n 5 --no-pager"
fi

# Kiểm tra port
echo -e "${BLUE}Kiểm tra port của ứng dụng...${NC}"
PORT_STATUS=$(ssh $USER@$SERVER "netstat -tuln | grep :$DASHBOARD_PORT")

if [ -n "$PORT_STATUS" ]; then
    echo -e "${GREEN}✓ Port $DASHBOARD_PORT đang mở và lắng nghe kết nối${NC}"
    echo "$PORT_STATUS"
else
    echo -e "${RED}✗ Port $DASHBOARD_PORT không mở hoặc không lắng nghe kết nối${NC}"
fi

# Kiểm tra thư mục ứng dụng
echo -e "${BLUE}Kiểm tra thư mục ứng dụng...${NC}"
DIR_STATUS=$(ssh $USER@$SERVER "[ -d $REMOTE_DIR ] && echo 'Tồn tại' || echo 'Không tồn tại'")

if [ "$DIR_STATUS" = "Tồn tại" ]; then
    echo -e "${GREEN}✓ Thư mục $REMOTE_DIR tồn tại${NC}"
    
    # Liệt kê các file chính
    echo "Các file và thư mục chính:"
    ssh $USER@$SERVER "ls -la $REMOTE_DIR | head -10"
    
    # Kiểm tra kích thước thư mục
    DIR_SIZE=$(ssh $USER@$SERVER "du -sh $REMOTE_DIR | cut -f1")
    echo "Kích thước thư mục: $DIR_SIZE"
else
    echo -e "${RED}✗ Thư mục $REMOTE_DIR không tồn tại${NC}"
fi

# Kiểm tra tiến trình Python/Streamlit
echo -e "${BLUE}Kiểm tra tiến trình Python/Streamlit...${NC}"
PYTHON_PROCESSES=$(ssh $USER@$SERVER "ps aux | grep -E 'python|streamlit' | grep -v grep")

if [ -n "$PYTHON_PROCESSES" ]; then
    echo -e "${GREEN}✓ Tiến trình Python/Streamlit đang chạy:${NC}"
    echo "$PYTHON_PROCESSES" | head -5
    
    # Số lượng tiến trình Python
    PROCESS_COUNT=$(echo "$PYTHON_PROCESSES" | wc -l)
    echo "Tổng số tiến trình Python: $PROCESS_COUNT"
else
    echo -e "${RED}✗ Không có tiến trình Python/Streamlit nào đang chạy${NC}"
fi

# Kiểm tra logs gần đây
echo -e "${BLUE}Kiểm tra logs của ứng dụng...${NC}"
LOG_FILES=$(ssh $USER@$SERVER "find $REMOTE_DIR/logs -type f -name '*.log' 2>/dev/null")

if [ -n "$LOG_FILES" ]; then
    echo -e "${GREEN}✓ Tìm thấy các file log:${NC}"
    echo "$LOG_FILES"
    
    # Hiển thị log gần nhất
    LATEST_LOG=$(ssh $USER@$SERVER "find $REMOTE_DIR/logs -type f -name '*.log' -exec ls -lt {} \; | head -1 | awk '{print \$9}'")
    if [ -n "$LATEST_LOG" ]; then
        echo -e "\nNội dung log gần nhất ($LATEST_LOG):"
        ssh $USER@$SERVER "tail -n 10 $LATEST_LOG"
    fi
else
    echo -e "${YELLOW}! Không tìm thấy file log nào trong thư mục $REMOTE_DIR/logs${NC}"
    
    # Kiểm tra log của systemd
    echo "Kiểm tra systemd logs:"
    ssh $USER@$SERVER "journalctl -u ethusdt-dashboard -n 10 --no-pager"
fi

# Kiểm tra kết nối đến Binance API
echo -e "${BLUE}Kiểm tra kết nối đến Binance API...${NC}"
API_STATUS=$(ssh $USER@$SERVER "curl -s -o /dev/null -w '%{http_code}' https://api.binance.com/api/v3/ping")

if [ "$API_STATUS" = "200" ]; then
    echo -e "${GREEN}✓ Kết nối đến Binance API thành công (HTTP Status: $API_STATUS)${NC}"
else
    echo -e "${RED}✗ Không thể kết nối đến Binance API (HTTP Status: $API_STATUS)${NC}"
fi

# Hiển thị endpoints
echo -e "${BLUE}Thông tin về endpoints...${NC}"
echo "Dashboard URL: http://$SERVER:$DASHBOARD_PORT"
echo "API URL: http://$SERVER:$DASHBOARD_PORT/api"

echo -e "${YELLOW}=== KIỂM TRA HOÀN TẤT ===${NC}"