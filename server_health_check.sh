#!/bin/bash
# Script kiểm tra sức khỏe hệ thống ETHUSDT Dashboard trên server
# Sử dụng: ./server_health_check.sh [--detailed]

# Thiết lập màu
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thông tin cấu hình
APP_DIR="/opt/ethusdt-dashboard"
APP_USER="ethusdt"
APP_PORT=5000
LOG_DIR="$APP_DIR/logs"
DATA_DIR="$APP_DIR/data"
MODELS_DIR="$APP_DIR/saved_models"
SERVICE_NAME="ethusdt-dashboard"

# Định dạng tiêu đề và kết quả
function print_header() {
    echo -e "\n${BLUE}===== $1 =====${NC}"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Kiểm tra chi tiết hay không
DETAILED=false
if [ "$1" == "--detailed" ]; then
    DETAILED=true
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     ETHUSDT Dashboard Health Check     ${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Thời gian: $(date)"
echo -e "Server: $(hostname)"

# 1. Kiểm tra hệ thống
print_header "Kiểm tra tài nguyên hệ thống"

# Kiểm tra CPU
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
echo -e "CPU Usage: ${CPU_USAGE}%"
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    print_error "CPU đang quá tải (${CPU_USAGE}%)"
elif (( $(echo "$CPU_USAGE > 50" | bc -l) )); then
    print_warning "CPU đang cao (${CPU_USAGE}%)"
else
    print_success "CPU bình thường (${CPU_USAGE}%)"
fi

# Kiểm tra RAM
MEM_TOTAL=$(free -m | grep Mem | awk '{print $2}')
MEM_USED=$(free -m | grep Mem | awk '{print $3}')
MEM_PERCENT=$(echo "scale=2; $MEM_USED*100/$MEM_TOTAL" | bc)
echo -e "Memory Usage: ${MEM_USED}MB / ${MEM_TOTAL}MB (${MEM_PERCENT}%)"
if (( $(echo "$MEM_PERCENT > 90" | bc -l) )); then
    print_error "RAM đang quá tải (${MEM_PERCENT}%)"
elif (( $(echo "$MEM_PERCENT > 70" | bc -l) )); then
    print_warning "RAM đang cao (${MEM_PERCENT}%)"
else
    print_success "RAM bình thường (${MEM_PERCENT}%)"
fi

# Kiểm tra disk
DISK_TOTAL=$(df -h / | awk 'NR==2 {print $2}')
DISK_USED=$(df -h / | awk 'NR==2 {print $3}')
DISK_PERCENT=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
echo -e "Disk Usage: ${DISK_USED} / ${DISK_TOTAL} (${DISK_PERCENT}%)"
if (( DISK_PERCENT > 90 )); then
    print_error "Disk gần hết (${DISK_PERCENT}%)"
elif (( DISK_PERCENT > 75 )); then
    print_warning "Disk đang cao (${DISK_PERCENT}%)"
else
    print_success "Disk bình thường (${DISK_PERCENT}%)"
fi

# 2. Kiểm tra dịch vụ
print_header "Kiểm tra dịch vụ"

# Kiểm tra dịch vụ systemd
if systemctl is-active --quiet $SERVICE_NAME; then
    print_success "Dịch vụ $SERVICE_NAME đang chạy"
else
    print_error "Dịch vụ $SERVICE_NAME không chạy"
    echo -e "Khởi động dịch vụ: sudo systemctl start $SERVICE_NAME"
fi

# Kiểm tra port
if netstat -tuln | grep -q ":$APP_PORT"; then
    print_success "Port $APP_PORT đang mở và có dịch vụ đang lắng nghe"
else
    print_error "Port $APP_PORT không có dịch vụ nào đang lắng nghe"
    echo -e "Kiểm tra trạng thái dịch vụ: sudo systemctl status $SERVICE_NAME"
fi

# 3. Kiểm tra kết nối mạng
print_header "Kiểm tra kết nối mạng"

# Kiểm tra kết nối internet
if ping -c 1 google.com &> /dev/null; then
    print_success "Kết nối internet hoạt động"
else
    print_error "Không thể kết nối internet"
fi

# Kiểm tra kết nối Binance API
if curl -s "https://api.binance.com/api/v3/ping" &> /dev/null; then
    print_success "Kết nối Binance API hoạt động"
else
    print_error "Không thể kết nối Binance API"
    echo -e "Kiểm tra proxy hoặc cấu hình mạng"
fi

# 4. Kiểm tra dữ liệu và logs
print_header "Kiểm tra dữ liệu và logs"

# Kiểm tra thư mục dữ liệu
if [ -d "$DATA_DIR" ]; then
    DATA_FILES=$(find $DATA_DIR -type f | wc -l)
    print_success "Thư mục dữ liệu tồn tại ($DATA_FILES files)"
else
    print_error "Thư mục dữ liệu không tồn tại"
fi

# Kiểm tra thư mục mô hình
if [ -d "$MODELS_DIR" ]; then
    MODEL_FILES=$(find $MODELS_DIR -type f | wc -l)
    print_success "Thư mục mô hình tồn tại ($MODEL_FILES files)"
    
    # Kiểm tra thời gian cập nhật mô hình
    if [ $MODEL_FILES -gt 0 ]; then
        LATEST_MODEL=$(find $MODELS_DIR -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        LATEST_MODEL_TIME=$(stat -c %y "$LATEST_MODEL" | cut -d'.' -f1)
        echo -e "Mô hình mới nhất: $(basename "$LATEST_MODEL") (${LATEST_MODEL_TIME})"
    fi
else
    print_error "Thư mục mô hình không tồn tại"
fi

# Kiểm tra logs
if [ -d "$LOG_DIR" ]; then
    LOG_FILES=$(find $LOG_DIR -type f | wc -l)
    print_success "Thư mục logs tồn tại ($LOG_FILES files)"
    
    # Kiểm tra log lỗi gần đây
    if [ -f "$LOG_DIR/service_error.log" ]; then
        ERROR_COUNT=$(grep -i "error\|exception\|fail" $LOG_DIR/service_error.log | wc -l)
        if [ $ERROR_COUNT -gt 0 ]; then
            print_warning "Tìm thấy $ERROR_COUNT lỗi trong log"
            if [ "$DETAILED" = true ]; then
                echo -e "\nLỗi gần đây:"
                grep -i "error\|exception\|fail" $LOG_DIR/service_error.log | tail -5
            fi
        else
            print_success "Không tìm thấy lỗi trong log"
        fi
    fi
else
    print_error "Thư mục logs không tồn tại"
fi

# 5. Kiểm tra ứng dụng web
print_header "Kiểm tra ứng dụng web"

# Kiểm tra response của ứng dụng web
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$APP_PORT)
if [ "$HTTP_CODE" = "200" ]; then
    print_success "Ứng dụng web trả về HTTP 200 OK"
else
    print_error "Ứng dụng web trả về HTTP $HTTP_CODE"
    echo -e "Kiểm tra log lỗi: sudo tail -f $LOG_DIR/service_error.log"
fi

# Xuất thông tin chi tiết nếu có yêu cầu
if [ "$DETAILED" = true ]; then
    print_header "Thông tin chi tiết"
    
    echo -e "\n${YELLOW}Tiến trình Streamlit:${NC}"
    ps -ef | grep streamlit | grep -v grep
    
    echo -e "\n${YELLOW}Logs gần đây:${NC}"
    if [ -f "$LOG_DIR/service.log" ]; then
        tail -n 10 $LOG_DIR/service.log
    else
        echo "Không tìm thấy file log"
    fi
    
    echo -e "\n${YELLOW}Thông tin phiên bản:${NC}"
    if [ -f "$APP_DIR/venv/bin/python" ]; then
        echo -e "Python: $($APP_DIR/venv/bin/python --version 2>&1)"
        echo -e "Streamlit: $($APP_DIR/venv/bin/pip show streamlit | grep Version | awk '{print $2}')"
        echo -e "TensorFlow: $($APP_DIR/venv/bin/pip show tensorflow | grep Version | awk '{print $2}')"
    fi
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}     Kết thúc kiểm tra sức khỏe         ${NC}"
echo -e "${BLUE}========================================${NC}"

# Hướng dẫn cách khắc phục
print_header "Hướng dẫn khắc phục"
echo -e "- Khởi động lại dịch vụ: ${YELLOW}sudo systemctl restart $SERVICE_NAME${NC}"
echo -e "- Xem log ứng dụng: ${YELLOW}sudo tail -f $LOG_DIR/service.log${NC}"
echo -e "- Xem log lỗi: ${YELLOW}sudo tail -f $LOG_DIR/service_error.log${NC}"
echo -e "- Kiểm tra trạng thái dịch vụ: ${YELLOW}sudo systemctl status $SERVICE_NAME${NC}"
echo -e "- Kiểm tra chi tiết sức khỏe: ${YELLOW}$0 --detailed${NC}"

exit 0