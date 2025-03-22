#!/bin/bash
# Script kiểm tra kết nối Binance API trên server

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Thông tin server
SERVER="45.76.196.13"
USER="root"
REMOTE_DIR="/root/ethusdt_dashboard"

echo -e "${YELLOW}=== KIỂM TRA KẾT NỐI BINANCE API TRÊN SERVER ===${NC}"

# Kiểm tra kết nối đến server
echo -e "${YELLOW}Kiểm tra kết nối đến server...${NC}"
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$SERVER "echo 'Kết nối thành công!'" &> /dev/null; then
    echo -e "${RED}Không thể kết nối đến server. Kiểm tra lại kết nối mạng hoặc thông tin đăng nhập.${NC}"
    exit 1
fi

# Tạo script kiểm tra API tạm thời trên server
echo -e "${YELLOW}Tạo script kiểm tra API trên server...${NC}"
ssh -o StrictHostKeyChecking=no $USER@$SERVER "cat > $REMOTE_DIR/test_api.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import requests
import time

# Màu sắc terminal
GREEN = '\\033[92m'
YELLOW = '\\033[93m'
RED = '\\033[91m'
NC = '\\033[0m'  # No Color

def test_binance_api():
    print(f'{YELLOW}Kiểm tra kết nối tới Binance API...{NC}')
    
    # Lấy API keys từ biến môi trường
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        print(f'{RED}Không tìm thấy BINANCE_API_KEY hoặc BINANCE_API_SECRET trong biến môi trường.{NC}')
        return False
    
    # Kiểm tra độ dài API key
    if len(api_key) < 10 or len(api_secret) < 10:
        print(f'{RED}API key hoặc API secret có độ dài không hợp lệ.{NC}')
        return False
    
    # Thử ping API
    try:
        response = requests.get('https://api.binance.com/api/v3/ping')
        if response.status_code == 200:
            print(f'{GREEN}Ping API thành công: {response.status_code}{NC}')
        else:
            print(f'{RED}Ping API thất bại: {response.status_code}{NC}')
            return False
    except Exception as e:
        print(f'{RED}Không thể kết nối tới Binance API: {e}{NC}')
        return False
    
    # Kiểm tra server time
    try:
        response = requests.get('https://api.binance.com/api/v3/time')
        if response.status_code == 200:
            server_time = response.json()['serverTime']
            current_time = int(time.time() * 1000)
            time_diff = abs(current_time - server_time)
            
            print(f'{GREEN}Server time: {server_time}{NC}')
            print(f'{GREEN}Local time: {current_time}{NC}')
            print(f'{GREEN}Time difference: {time_diff} ms{NC}')
            
            if time_diff > 10000:  # 10 seconds
                print(f'{YELLOW}Cảnh báo: Độ lệch thời gian quá lớn ({time_diff} ms){NC}')
        else:
            print(f'{RED}Không thể lấy server time: {response.status_code}{NC}')
            return False
    except Exception as e:
        print(f'{RED}Lỗi khi lấy server time: {e}{NC}')
        return False
    
    # Kiểm tra thông tin tài khoản - cần API auth
    try:
        headers = {
            'X-MBX-APIKEY': api_key
        }
        response = requests.get('https://api.binance.com/api/v3/account', headers=headers)
        
        if response.status_code == 200:
            print(f'{GREEN}Xác thực API thành công!{NC}')
            return True
        else:
            error_message = response.json().get('msg', 'Unknown error')
            print(f'{RED}Xác thực API thất bại: {error_message} (Mã: {response.status_code}){NC}')
            return False
    except Exception as e:
        print(f'{RED}Lỗi khi xác thực API: {e}{NC}')
        return False

if __name__ == '__main__':
    success = test_binance_api()
    if success:
        print(f'{GREEN}Kết nối Binance API thành công!{NC}')
        sys.exit(0)
    else:
        print(f'{RED}Kết nối Binance API thất bại!{NC}')
        sys.exit(1)
EOF

chmod +x $REMOTE_DIR/test_api.py
"

# Chạy script kiểm tra
echo -e "${YELLOW}Chạy kiểm tra API keys...${NC}"
ssh -o StrictHostKeyChecking=no $USER@$SERVER "cd $REMOTE_DIR && python3 test_api.py"

status=$?
if [ $status -eq 0 ]; then
    echo -e "${GREEN}Kết nối Binance API thành công trên server!${NC}"
else
    echo -e "${RED}Kết nối Binance API thất bại trên server! Kiểm tra lại API keys.${NC}"
fi

# Xóa script tạm thời
ssh -o StrictHostKeyChecking=no $USER@$SERVER "rm $REMOTE_DIR/test_api.py"

exit $status