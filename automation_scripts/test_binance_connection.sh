#!/bin/bash
# Kiểm tra kết nối đến Binance API

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== KIỂM TRA KẾT NỐI BINANCE API ===${NC}"
echo "Thời gian: $(date)"

# Kiểm tra kết nối mạng cơ bản
echo -e "${BLUE}Kiểm tra kết nối Internet...${NC}"
if ping -c 1 google.com &> /dev/null; then
    echo -e "${GREEN}Kết nối Internet hoạt động bình thường.${NC}"
else
    echo -e "${RED}Không có kết nối Internet!${NC}"
    echo "Vui lòng kiểm tra kết nối mạng và thử lại."
    exit 1
fi

# Kiểm tra kết nối đến Binance
echo -e "${BLUE}Kiểm tra kết nối đến Binance API...${NC}"
BINANCE_API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://api.binance.com/api/v3/ping)
if [ "$BINANCE_API_STATUS" -eq 200 ]; then
    echo -e "${GREEN}Kết nối đến Binance API thành công.${NC}"
else
    echo -e "${RED}Không thể kết nối đến Binance API! (HTTP Code: $BINANCE_API_STATUS)${NC}"
    echo "Kiểm tra xem Binance có bị chặn ở khu vực của bạn không."
fi

# Kiểm tra kết nối thông qua proxy
if [ -n "$http_proxy" ] || [ -n "$https_proxy" ]; then
    echo -e "${BLUE}Kiểm tra kết nối thông qua proxy...${NC}"
    echo "Proxy đang sử dụng: $http_proxy"
    
    PROXY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --proxy "$http_proxy" https://api.binance.com/api/v3/ping)
    if [ "$PROXY_STATUS" -eq 200 ]; then
        echo -e "${GREEN}Kết nối proxy đến Binance API thành công.${NC}"
    else
        echo -e "${RED}Không thể kết nối qua proxy đến Binance API! (HTTP Code: $PROXY_STATUS)${NC}"
    fi
else
    echo -e "${YELLOW}Không phát hiện cấu hình proxy.${NC}"
    
    # Kiểm tra xem có file cấu hình proxy không
    if [ -f "../config.py" ]; then
        echo -e "${BLUE}Kiểm tra cấu hình proxy trong config.py...${NC}"
        PROXY_CONFIG=$(grep -E "PROXY|proxy" ../config.py)
        if [ -n "$PROXY_CONFIG" ]; then
            echo "Cấu hình proxy được tìm thấy trong config.py:"
            echo "$PROXY_CONFIG"
        else
            echo "Không tìm thấy cấu hình proxy trong config.py."
        fi
    fi
fi

# Kiểm tra truy cập API yêu cầu khóa API
echo -e "${BLUE}Kiểm tra API Binance yêu cầu xác thực...${NC}"

if [ -f "../.env" ]; then
    echo "File .env được tìm thấy, kiểm tra thông tin API key..."
    API_KEY_EXISTS=$(grep -E "BINANCE_API_KEY" ../env)
    API_SECRET_EXISTS=$(grep -E "BINANCE_API_SECRET" ../env)
    
    if [ -n "$API_KEY_EXISTS" ] && [ -n "$API_SECRET_EXISTS" ]; then
        echo -e "${GREEN}Tìm thấy thông tin API key trong file .env.${NC}"
    else
        echo -e "${YELLOW}Không tìm thấy thông tin API key đầy đủ trong file .env.${NC}"
    fi
else
    echo -e "${YELLOW}Không tìm thấy file .env${NC}"
    
    # Kiểm tra các biến môi trường
    if [ -n "$BINANCE_API_KEY" ] && [ -n "$BINANCE_API_SECRET" ]; then
        echo -e "${GREEN}Tìm thấy biến môi trường BINANCE_API_KEY và BINANCE_API_SECRET.${NC}"
    else
        echo -e "${YELLOW}Không tìm thấy biến môi trường BINANCE_API_KEY và BINANCE_API_SECRET.${NC}"
    fi
fi

# Kiểm tra thời gian Binance
echo -e "${BLUE}Kiểm tra thời gian Binance server...${NC}"
BINANCE_TIME=$(curl -s https://api.binance.com/api/v3/time | grep -o '"serverTime":[0-9]*' | cut -d':' -f2)
if [ -n "$BINANCE_TIME" ]; then
    BINANCE_TIME_HUMAN=$(date -d @$((BINANCE_TIME/1000)) "+%Y-%m-%d %H:%M:%S")
    LOCAL_TIME=$(date "+%Y-%m-%d %H:%M:%S")
    echo "Thời gian Binance server: $BINANCE_TIME_HUMAN"
    echo "Thời gian máy chủ local: $LOCAL_TIME"
    
    # Tính chênh lệch thời gian
    BINANCE_TIME_SECONDS=$((BINANCE_TIME/1000))
    LOCAL_TIME_SECONDS=$(date +%s)
    TIME_DIFF=$((LOCAL_TIME_SECONDS - BINANCE_TIME_SECONDS))
    TIME_DIFF_ABS=${TIME_DIFF#-}  # Lấy giá trị tuyệt đối
    
    if [ $TIME_DIFF_ABS -lt 60 ]; then
        echo -e "${GREEN}Chênh lệch thời gian: $TIME_DIFF giây (chấp nhận được).${NC}"
    else
        echo -e "${RED}Chênh lệch thời gian: $TIME_DIFF giây (quá lớn).${NC}"
        echo "Cân nhắc đồng bộ hóa thời gian máy chủ."
    fi
else
    echo -e "${RED}Không thể lấy thời gian từ Binance server!${NC}"
fi

# Kiểm tra trạng thái hệ thống Binance
echo -e "${BLUE}Kiểm tra trạng thái hệ thống Binance...${NC}"
SYSTEM_STATUS=$(curl -s https://api.binance.com/sapi/v1/system/status)
if echo "$SYSTEM_STATUS" | grep -q '"status":0'; then
    echo -e "${GREEN}Hệ thống Binance đang hoạt động bình thường.${NC}"
else
    echo -e "${RED}Hệ thống Binance có thể đang bảo trì!${NC}"
    echo "Trạng thái: $SYSTEM_STATUS"
fi

echo -e "${YELLOW}=== KIỂM TRA HOÀN TẤT ===${NC}"