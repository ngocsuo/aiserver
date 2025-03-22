#!/bin/bash
# Script để kiểm tra kết nối Binance API trên server

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

SERVER="45.76.196.13"
USER="root"
REMOTE_DIR="/root/ethusdt_dashboard"
TIMEOUT=10

echo -e "${YELLOW}=== KIỂM TRA KẾT NỐI BINANCE API TRÊN SERVER ===${NC}"

# Tạo script test tạm thời
cat > /tmp/test_binance.py << 'EOF'
#!/usr/bin/env python3
"""
Script kiểm tra kết nối Binance API.
"""
import os
import sys
import time
import json
import requests
from datetime import datetime

try:
    # Thử sử dụng thư viện python-binance
    from binance.client import Client
    BINANCE_LIB = "python-binance"
except ImportError:
    BINANCE_LIB = "requests"

def test_direct_connection():
    """Kiểm tra kết nối trực tiếp đến API Binance"""
    try:
        response = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
        if response.status_code == 200:
            print("✅ Kết nối trực tiếp đến Binance API thành công")
            return True
        else:
            print(f"❌ Kết nối trực tiếp thất bại, mã trạng thái: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Lỗi kết nối trực tiếp: {str(e)}")
        return False

def test_time_sync():
    """Kiểm tra đồng bộ thời gian với máy chủ Binance"""
    try:
        response = requests.get('https://api.binance.com/api/v3/time', timeout=5)
        if response.status_code == 200:
            server_time = response.json()['serverTime']
            local_time = int(time.time() * 1000)
            diff = abs(server_time - local_time)
            
            if diff < 1000:  # Chênh lệch dưới 1 giây
                print(f"✅ Đồng bộ thời gian OK (chênh lệch {diff}ms)")
                return True
            else:
                print(f"⚠️ Chênh lệch thời gian lớn: {diff}ms")
                return True
        else:
            print(f"❌ Kiểm tra thời gian thất bại, mã trạng thái: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Lỗi kiểm tra thời gian: {str(e)}")
        return False

def test_api_keys():
    """Kiểm tra các API key đã cấu hình"""
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Không tìm thấy BINANCE_API_KEY hoặc BINANCE_API_SECRET trong biến môi trường")
        return False
    
    if BINANCE_LIB == "python-binance":
        try:
            client = Client(api_key, api_secret)
            account = client.get_account()
            print(f"✅ API keys hợp lệ - Tài khoản có {len(account['balances'])} loại tài sản")
            return True
        except Exception as e:
            print(f"❌ Lỗi API keys: {str(e)}")
            return False
    else:
        try:
            timestamp = int(time.time() * 1000)
            query_string = f'timestamp={timestamp}'
            
            # Import thư viện cho tính toán HMAC-SHA256
            import hmac
            import hashlib
            
            # Tạo chữ ký
            signature = hmac.new(
                api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Tạo URL cho yêu cầu
            url = f'https://api.binance.com/api/v3/account?{query_string}&signature={signature}'
            
            # Thực hiện yêu cầu GET với API key trong header
            headers = {
                'X-MBX-APIKEY': api_key
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                account = response.json()
                print(f"✅ API keys hợp lệ - Tài khoản có {len(account['balances'])} loại tài sản")
                return True
            else:
                print(f"❌ Lỗi API keys: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Lỗi kiểm tra API keys: {str(e)}")
            return False

def test_market_data():
    """Kiểm tra truy cập dữ liệu thị trường"""
    try:
        response = requests.get('https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1m&limit=1', timeout=5)
        if response.status_code == 200:
            klines = response.json()
            if len(klines) > 0:
                print(f"✅ Truy cập dữ liệu thị trường OK - Dữ liệu ETHUSDT mới nhất: {datetime.fromtimestamp(klines[0][0]/1000).strftime('%Y-%m-%d %H:%M:%S')}")
                return True
            else:
                print("⚠️ Truy cập dữ liệu thị trường OK nhưng không có dữ liệu")
                return True
        else:
            print(f"❌ Truy cập dữ liệu thị trường thất bại, mã trạng thái: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Lỗi truy cập dữ liệu thị trường: {str(e)}")
        return False

def test_proxy_settings():
    """Kiểm tra cài đặt proxy"""
    proxies = {}
    http_proxy = os.environ.get('HTTP_PROXY')
    https_proxy = os.environ.get('HTTPS_PROXY')
    
    if http_proxy:
        proxies['http'] = http_proxy
        print(f"ℹ️ HTTP proxy được cấu hình: {http_proxy}")
    
    if https_proxy:
        proxies['https'] = https_proxy
        print(f"ℹ️ HTTPS proxy được cấu hình: {https_proxy}")
    
    if not proxies:
        print("ℹ️ Không có proxy được cấu hình")
    
    try:
        if proxies:
            response = requests.get('https://api.binance.com/api/v3/ping', proxies=proxies, timeout=5)
            if response.status_code == 200:
                print("✅ Kết nối qua proxy thành công")
                return True
            else:
                print(f"❌ Kết nối qua proxy thất bại, mã trạng thái: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Lỗi kết nối qua proxy: {str(e)}")
        return False
    
    return True

def main():
    """Hàm chính"""
    print(f"Thời gian kiểm tra: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hệ thống: {os.uname().sysname} {os.uname().release}")
    print(f"Thư viện Binance: {BINANCE_LIB}")
    print("-" * 50)
    
    success = 0
    tests = 0
    
    # Chạy các bài kiểm tra
    tests += 1
    if test_direct_connection():
        success += 1
    
    tests += 1
    if test_time_sync():
        success += 1
    
    tests += 1
    if test_api_keys():
        success += 1
    
    tests += 1
    if test_market_data():
        success += 1
    
    tests += 1
    if test_proxy_settings():
        success += 1
    
    print("-" * 50)
    print(f"Kết quả: {success}/{tests} kiểm tra thành công")
    
    if success == tests:
        print("✅ Tất cả các kiểm tra đều thành công!")
        sys.exit(0)
    elif success >= tests - 1:
        print("⚠️ Hầu hết các kiểm tra đều thành công, có thể sử dụng được")
        sys.exit(0)
    else:
        print("❌ Kiểm tra thất bại, cần xem xét lại cấu hình")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

echo -e "${YELLOW}Đang sao chép script kiểm tra lên server...${NC}"
# Sao chép script kiểm tra lên server
scp -o StrictHostKeyChecking=no /tmp/test_binance.py $USER@$SERVER:$REMOTE_DIR/test_binance.py

echo -e "${YELLOW}Đang chạy kiểm tra kết nối trên server...${NC}"
# Thực thi script kiểm tra trên server
ssh -o StrictHostKeyChecking=no $USER@$SERVER "cd $REMOTE_DIR && chmod +x test_binance.py && python3 test_binance.py"

# Kiểm tra kết quả
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Kiểm tra kết nối Binance API trên server thành công!${NC}"
else
    echo -e "${RED}Kiểm tra kết nối Binance API trên server thất bại!${NC}"
    echo -e "${YELLOW}Vui lòng kiểm tra lại cấu hình API keys và proxy (nếu có) trên server.${NC}"
fi

# Xóa script tạm thời trên server
ssh -o StrictHostKeyChecking=no $USER@$SERVER "rm -f $REMOTE_DIR/test_binance.py"
rm -f /tmp/test_binance.py

echo -e "${YELLOW}=== KIỂM TRA HOÀN TẤT ===${NC}"