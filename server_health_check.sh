#!/bin/bash
# Script kiểm tra sức khỏe hệ thống chạy ETHUSDT Dashboard
# Phiên bản: 1.0.0

# Định nghĩa màu sắc
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Kiểm tra quyền root
if [ "$(id -u)" -ne 0 ]; then
    echo -e "${YELLOW}Cảnh báo: Script nên chạy với quyền root để có đầy đủ chức năng kiểm tra${NC}"
    echo "Thử chạy: sudo $0"
    echo ""
fi

# Hàm kiểm tra và in thông tin
check_service() {
    local service_name="$1"
    if systemctl is-active --quiet "$service_name"; then
        echo -e "  ${GREEN}✅ Dịch vụ $service_name đang chạy${NC}"
    else
        echo -e "  ${RED}❌ Dịch vụ $service_name không chạy${NC}"
        echo -e "  ${BLUE}   Thử khởi động lại: sudo systemctl restart $service_name${NC}"
    fi
}

check_port() {
    local port="$1"
    local service_name="$2"
    
    if command -v netstat >/dev/null 2>&1; then
        if netstat -tuln | grep -q ":$port "; then
            echo -e "  ${GREEN}✅ Cổng $port đang mở (dịch vụ $service_name)${NC}"
        else
            echo -e "  ${RED}❌ Cổng $port không mở${NC}"
            if [ -n "$service_name" ]; then
                echo -e "  ${BLUE}   Kiểm tra dịch vụ $service_name: sudo systemctl status $service_name${NC}"
            fi
        fi
    else
        # Sử dụng /dev/tcp nếu netstat không có sẵn
        if timeout 1 bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
            echo -e "  ${GREEN}✅ Cổng $port đang mở (dịch vụ $service_name)${NC}"
        else
            echo -e "  ${RED}❌ Cổng $port không mở${NC}"
            if [ -n "$service_name" ]; then
                echo -e "  ${BLUE}   Kiểm tra dịch vụ $service_name: sudo systemctl status $service_name${NC}"
            fi
        fi
    fi
}

check_disk_space() {
    local path="$1"
    local min_free_pct=10
    
    local disk_usage=$(df -h "$path" | awk 'NR==2 {print $5}' | sed 's/%//')
    local free_pct=$((100 - disk_usage))
    
    if [ "$free_pct" -lt "$min_free_pct" ]; then
        echo -e "  ${RED}❌ Ổ đĩa tại $path chỉ còn $free_pct% trống (${disk_usage}% đã sử dụng)${NC}"
        echo -e "  ${BLUE}   Cân nhắc xóa bớt log cũ hoặc dữ liệu tạm${NC}"
    else
        echo -e "  ${GREEN}✅ Ổ đĩa tại $path còn $free_pct% trống (${disk_usage}% đã sử dụng)${NC}"
    fi
}

check_memory() {
    local free_mem=$(free -m | awk 'NR==2 {print $7}')
    local total_mem=$(free -m | awk 'NR==2 {print $2}')
    local free_pct=$((free_mem * 100 / total_mem))
    
    if [ "$free_pct" -lt 20 ]; then
        echo -e "  ${YELLOW}⚠️ Bộ nhớ còn trống: $free_mem MB ($free_pct%)${NC}"
        echo -e "  ${BLUE}   Có thể cần thêm bộ nhớ hoặc tối ưu ứng dụng${NC}"
    else
        echo -e "  ${GREEN}✅ Bộ nhớ còn trống: $free_mem MB ($free_pct%)${NC}"
    fi
}

check_cpu_load() {
    local cpu_load=$(uptime | awk -F'[a-z]:' '{ print $2}' | awk -F',' '{ print $1}' | tr -d ' ')
    local cpu_cores=$(grep -c ^processor /proc/cpuinfo)
    
    local load_per_core=$(echo "$cpu_load $cpu_cores" | awk '{printf "%.2f", $1 / $2}')
    if (( $(echo "$load_per_core > 0.8" | bc -l) )); then
        echo -e "  ${YELLOW}⚠️ Tải CPU: $cpu_load (${load_per_core}/core với $cpu_cores cores)${NC}"
        echo -e "  ${BLUE}   Tải CPU cao, có thể ảnh hưởng đến hiệu suất${NC}"
    else
        echo -e "  ${GREEN}✅ Tải CPU: $cpu_load (${load_per_core}/core với $cpu_cores cores)${NC}"
    fi
}

check_python_version() {
    local app_dir="$1"
    cd "$app_dir" || { echo -e "  ${RED}❌ Không thể truy cập thư mục $app_dir${NC}"; return; }
    
    if command -v python3 >/dev/null 2>&1; then
        local python_ver=$(python3 --version 2>&1)
        echo -e "  ${GREEN}✅ Python đã cài đặt: $python_ver${NC}"
        
        if [ -f "requirements.txt" ]; then
            echo -e "  ${BLUE}ℹ️ Kiểm tra thư viện cần thiết từ requirements.txt${NC}"
            local missing_deps=0
            while IFS= read -r line || [[ -n "$line" ]]; do
                if [[ ! -z "$line" && ! "$line" =~ ^# ]]; then
                    local pkg=$(echo "$line" | cut -d'=' -f1 | tr -d ' ')
                    if ! python3 -c "import $pkg" 2>/dev/null; then
                        echo -e "    ${RED}❌ Thiếu thư viện: $pkg${NC}"
                        missing_deps=1
                    fi
                fi
            done < requirements.txt
            
            if [ $missing_deps -eq 0 ]; then
                echo -e "  ${GREEN}✅ Đã cài đặt đầy đủ thư viện Python${NC}"
            else
                echo -e "  ${BLUE}   Cài đặt thư viện thiếu: pip install -r requirements.txt${NC}"
            fi
        fi
    else
        echo -e "  ${RED}❌ Python chưa được cài đặt${NC}"
    fi
}

check_log_errors() {
    local service_name="$1"
    local log_lines=50
    
    echo -e "  ${BLUE}ℹ️ Kiểm tra lỗi trong log của $service_name (${log_lines} dòng gần nhất)${NC}"
    
    if systemctl status "$service_name" >/dev/null 2>&1; then
        local error_count=$(journalctl -u "$service_name" -n "$log_lines" | grep -i "error\|failed\|exception" | wc -l)
        
        if [ "$error_count" -gt 0 ]; then
            echo -e "  ${YELLOW}⚠️ Phát hiện $error_count lỗi tiềm ẩn trong log${NC}"
            echo -e "  ${BLUE}   Kiểm tra chi tiết: journalctl -u $service_name -n $log_lines | grep -i 'error\\|failed\\|exception'${NC}"
        else
            echo -e "  ${GREEN}✅ Không phát hiện lỗi trong log gần đây${NC}"
        fi
    else
        echo -e "  ${RED}❌ Không thể kiểm tra log vì dịch vụ $service_name không tồn tại${NC}"
    fi
}

# In tiêu đề
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    ETHUSDT Dashboard System Health Check   ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Thông tin hệ thống
echo -e "${BLUE}Thông tin hệ thống:${NC}"
echo -e "  Host: $(hostname)"
echo -e "  OS: $(cat /etc/os-release | grep "PRETTY_NAME" | cut -d'"' -f2)"
echo -e "  Kernel: $(uname -r)"
echo -e "  Uptime: $(uptime -p)"
echo ""

# Kiểm tra tài nguyên hệ thống
echo -e "${BLUE}Tài nguyên hệ thống:${NC}"
check_cpu_load
check_memory
check_disk_space "/"
echo ""

# Kiểm tra dịch vụ
echo -e "${BLUE}Dịch vụ:${NC}"
check_service "ethusdt-dashboard"
echo ""

# Kiểm tra cổng
echo -e "${BLUE}Cổng mạng:${NC}"
check_port "5000" "ethusdt-dashboard"
echo ""

# Kiểm tra lỗi trong log
echo -e "${BLUE}Log:${NC}"
check_log_errors "ethusdt-dashboard"
echo ""

# Kiểm tra Python/thư viện
APP_DIR="/opt/ethusdt-dashboard"
if [ ! -d "$APP_DIR" ]; then
    APP_DIR="$PWD"
fi

echo -e "${BLUE}Python và thư viện:${NC}"
check_python_version "$APP_DIR"
echo ""

# Các bước khuyến nghị và khắc phục
echo -e "${BLUE}Khuyến nghị:${NC}"
echo -e "  1. Nếu dịch vụ không chạy: sudo systemctl restart ethusdt-dashboard"
echo -e "  2. Kiểm tra logs chi tiết: sudo journalctl -u ethusdt-dashboard -f"
echo -e "  3. Cập nhật thư viện: cd $APP_DIR && pip install -r requirements.txt"
echo -e "  4. Kiểm tra kết nối Binance API: curl -s https://api.binance.com/api/v3/ping"
echo ""

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}      Health Check Completed                ${NC}"
echo -e "${BLUE}============================================${NC}"