#!/bin/bash
# Script để kiểm tra kết nối đến Binance API

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Kiểm tra yêu cầu
if [ ! -f .env ] && [ ! -f .replit.env ]; then
    echo -e "${YELLOW}Không tìm thấy file .env hoặc .replit.env để lấy thông tin API key.${NC}"
    echo -e "${BLUE}Sử dụng API không xác thực cho kiểm tra cơ bản.${NC}"
    USE_AUTH=false
else
    if [ -f .env ]; then
        source .env
    elif [ -f .replit.env ]; then
        source .replit.env
    fi
    
    if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
        echo -e "${YELLOW}Không tìm thấy BINANCE_API_KEY hoặc BINANCE_API_SECRET trong file .env${NC}"
        echo -e "${BLUE}Sử dụng API không xác thực cho kiểm tra cơ bản.${NC}"
        USE_AUTH=false
    else
        USE_AUTH=true
        echo -e "${BLUE}Đã tìm thấy thông tin xác thực API Binance.${NC}"
    fi
fi

echo -e "${YELLOW}=== KIỂM TRA KẾT NỐI ĐẾN BINANCE API ===${NC}"

# Kiểm tra kết nối cơ bản đến Binance
echo -e "${BLUE}Kiểm tra kết nối cơ bản (ping)...${NC}"
PING_RESPONSE=$(curl -s https://api.binance.com/api/v3/ping)
if [ "$PING_RESPONSE" == "{}" ]; then
    echo -e "${GREEN}Kết nối cơ bản thành công!${NC}"
else
    echo -e "${RED}Không thể kết nối đến Binance API. Kiểm tra kết nối mạng.${NC}"
    echo "Response: $PING_RESPONSE"
    exit 1
fi

# Kiểm tra thời gian máy chủ
echo -e "${BLUE}Kiểm tra thời gian máy chủ Binance...${NC}"
SERVER_TIME_RESPONSE=$(curl -s https://api.binance.com/api/v3/time)
SERVER_TIME=$(echo $SERVER_TIME_RESPONSE | grep -o '"serverTime":[0-9]*' | cut -d':' -f2)
if [ ! -z "$SERVER_TIME" ]; then
    SERVER_TIME_HUMAN=$(date -d @$(($SERVER_TIME/1000)) "+%Y-%m-%d %H:%M:%S")
    echo -e "${GREEN}Thời gian máy chủ Binance: $SERVER_TIME_HUMAN${NC}"
    
    # Kiểm tra độ lệch thời gian
    LOCAL_TIME=$(date +%s)
    SERVER_TIME_SEC=$(($SERVER_TIME/1000))
    TIME_DIFF=$((LOCAL_TIME - SERVER_TIME_SEC))
    TIME_DIFF_ABS=${TIME_DIFF#-}
    
    if [ $TIME_DIFF_ABS -gt 1000 ]; then
        echo -e "${RED}Cảnh báo: Thời gian máy của bạn và thời gian máy chủ Binance lệch nhau $TIME_DIFF_ABS giây!${NC}"
        echo -e "${RED}Điều này có thể gây ra lỗi khi gọi API yêu cầu chữ ký.${NC}"
    else
        echo -e "${GREEN}Thời gian máy của bạn đồng bộ tốt với máy chủ Binance.${NC}"
    fi
else
    echo -e "${RED}Không thể lấy thời gian máy chủ Binance.${NC}"
    echo "Response: $SERVER_TIME_RESPONSE"
fi

# Kiểm tra thông tin hệ thống
echo -e "${BLUE}Kiểm tra trạng thái hệ thống Binance...${NC}"
SYSTEM_STATUS=$(curl -s https://api.binance.com/sapi/v1/system/status)
MAINTENANCE_STATUS=$(echo $SYSTEM_STATUS | grep -o '"status":[0-9]*' | cut -d':' -f2)

if [ "$MAINTENANCE_STATUS" == "0" ]; then
    echo -e "${GREEN}Hệ thống Binance đang hoạt động bình thường.${NC}"
elif [ "$MAINTENANCE_STATUS" == "1" ]; then
    echo -e "${RED}Hệ thống Binance đang trong chế độ bảo trì!${NC}"
    exit 1
else
    echo -e "${YELLOW}Không thể xác định trạng thái hệ thống Binance.${NC}"
    echo "Response: $SYSTEM_STATUS"
fi

# Kiểm tra thông tin tài khoản nếu có API key
if [ "$USE_AUTH" = true ]; then
    echo -e "${BLUE}Kiểm tra thông tin tài khoản...${NC}"
    
    # Tạo chữ ký cho API call
    TIMESTAMP=$(date +%s)000
    QUERY_STRING="timestamp=$TIMESTAMP"
    SIGNATURE=$(echo -n "$QUERY_STRING" | openssl dgst -sha256 -hmac "$BINANCE_API_SECRET" | cut -d ' ' -f2)
    
    # Gọi API kiểm tra thông tin tài khoản
    ACCOUNT_INFO=$(curl -s -H "X-MBX-APIKEY: $BINANCE_API_KEY" -X GET "https://api.binance.com/api/v3/account?$QUERY_STRING&signature=$SIGNATURE")
    
    # Kiểm tra phản hồi
    if echo "$ACCOUNT_INFO" | grep -q "makerCommission"; then
        echo -e "${GREEN}Xác thực API thành công!${NC}"
        
        # Hiển thị số dư một số đồng chính
        BTC_BALANCE=$(echo $ACCOUNT_INFO | grep -o '"BTC","free":"[0-9.]*"' | cut -d'"' -f6)
        ETH_BALANCE=$(echo $ACCOUNT_INFO | grep -o '"ETH","free":"[0-9.]*"' | cut -d'"' -f6)
        USDT_BALANCE=$(echo $ACCOUNT_INFO | grep -o '"USDT","free":"[0-9.]*"' | cut -d'"' -f6)
        
        echo -e "${BLUE}Số dư tài khoản:${NC}"
        [ ! -z "$BTC_BALANCE" ] && echo -e "BTC: $BTC_BALANCE"
        [ ! -z "$ETH_BALANCE" ] && echo -e "ETH: $ETH_BALANCE"
        [ ! -z "$USDT_BALANCE" ] && echo -e "USDT: $USDT_BALANCE"
        
        # Kiểm tra Futures API
        echo -e "${BLUE}Kiểm tra Binance Futures API...${NC}"
        TIMESTAMP=$(date +%s)000
        QUERY_STRING="timestamp=$TIMESTAMP"
        SIGNATURE=$(echo -n "$QUERY_STRING" | openssl dgst -sha256 -hmac "$BINANCE_API_SECRET" | cut -d ' ' -f2)
        
        FUTURES_ACCOUNT=$(curl -s -H "X-MBX-APIKEY: $BINANCE_API_KEY" -X GET "https://fapi.binance.com/fapi/v2/account?$QUERY_STRING&signature=$SIGNATURE")
        
        if echo "$FUTURES_ACCOUNT" | grep -q "totalMarginBalance"; then
            echo -e "${GREEN}Kết nối Binance Futures API thành công!${NC}"
            MARGIN_BALANCE=$(echo $FUTURES_ACCOUNT | grep -o '"totalMarginBalance":"[0-9.]*"' | cut -d'"' -f4)
            echo -e "Số dư margin: $MARGIN_BALANCE USDT"
        else
            echo -e "${RED}Không thể kết nối đến Binance Futures API hoặc tài khoản không hỗ trợ Futures.${NC}"
            echo "Response: $FUTURES_ACCOUNT"
        fi
    else
        echo -e "${RED}Xác thực API thất bại!${NC}"
        echo "Response: $ACCOUNT_INFO"
    fi
else
    # Kiểm tra API công khai
    echo -e "${BLUE}Kiểm tra API dữ liệu thị trường...${NC}"
    
    # Lấy dữ liệu ETHUSDT
    ETHUSDT_PRICE=$(curl -s "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT")
    if echo "$ETHUSDT_PRICE" | grep -q "price"; then
        PRICE=$(echo $ETHUSDT_PRICE | grep -o '"price":"[0-9.]*"' | cut -d'"' -f4)
        echo -e "${GREEN}Kết nối API thị trường thành công!${NC}"
        echo -e "Giá ETHUSDT hiện tại: $PRICE"
    else
        echo -e "${RED}Không thể lấy dữ liệu giá ETHUSDT.${NC}"
        echo "Response: $ETHUSDT_PRICE"
    fi
    
    # Kiểm tra dữ liệu K-line
    echo -e "${BLUE}Kiểm tra dữ liệu K-line...${NC}"
    KLINES=$(curl -s "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1m&limit=1")
    if echo "$KLINES" | grep -q "\[\["; then
        echo -e "${GREEN}Lấy dữ liệu K-line thành công!${NC}"
    else
        echo -e "${RED}Không thể lấy dữ liệu K-line.${NC}"
        echo "Response: $KLINES"
    fi
    
    # Kiểm tra Futures API
    echo -e "${BLUE}Kiểm tra API Futures công khai...${NC}"
    FUTURES_PRICE=$(curl -s "https://fapi.binance.com/fapi/v1/ticker/price?symbol=ETHUSDT")
    if echo "$FUTURES_PRICE" | grep -q "price"; then
        FUTURES_PRICE_VALUE=$(echo $FUTURES_PRICE | grep -o '"price":"[0-9.]*"' | cut -d'"' -f4)
        echo -e "${GREEN}Kết nối API Futures thành công!${NC}"
        echo -e "Giá ETHUSDT Futures hiện tại: $FUTURES_PRICE_VALUE"
    else
        echo -e "${RED}Không thể kết nối đến Binance Futures API.${NC}"
        echo "Response: $FUTURES_PRICE"
    fi
fi

# Kiểm tra xem package python-binance có được cài đặt không
echo -e "${BLUE}Kiểm tra package python-binance...${NC}"
if pip list | grep -q python-binance; then
    BINANCE_VERSION=$(pip show python-binance | grep Version | cut -d ' ' -f2)
    echo -e "${GREEN}Đã cài đặt python-binance phiên bản $BINANCE_VERSION${NC}"
    
    # Kiểm tra kết nối thông qua Python
    echo -e "${BLUE}Thử kết nối thông qua Python...${NC}"
    PYTHON_TEST=$(python3 -c "
import sys
try:
    from binance.client import Client
    client = Client()
    info = client.get_exchange_info()
    print('OK: Kết nối thành công thông qua Python')
except Exception as e:
    print(f'ERROR: {str(e)}')
    sys.exit(1)
")
    
    if echo "$PYTHON_TEST" | grep -q "OK:"; then
        echo -e "${GREEN}$PYTHON_TEST${NC}"
    else
        ERROR_MSG=$(echo "$PYTHON_TEST" | grep "ERROR:")
        echo -e "${RED}$ERROR_MSG${NC}"
    fi
else
    echo -e "${YELLOW}Package python-binance chưa được cài đặt.${NC}"
    echo -e "Cài đặt bằng lệnh: pip install python-binance"
fi

echo -e "${YELLOW}=== KIỂM TRA HOÀN TẤT ===${NC}"