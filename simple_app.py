"""
Ứng dụng Streamlit đơn giản để kiểm tra kết nối Binance API
"""
import streamlit as st
import os
import requests
import pandas as pd
from datetime import datetime

# Thiết lập tiêu đề
st.set_page_config(page_title="Kiểm tra kết nối Binance API", layout="wide")
st.title("Kiểm tra kết nối Binance API")

# Hiển thị thời gian hiện tại
st.write(f"Thời gian hiện tại: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Kiểm tra API Keys
api_key = os.environ.get("BINANCE_API_KEY", "")
api_secret = os.environ.get("BINANCE_API_SECRET", "")

if api_key and api_secret:
    st.success("✅ API Keys đã được cấu hình")
    
    # Hiển thị một phần của key để xác nhận
    masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
    st.write(f"API Key: {masked_key}")
else:
    st.error("❌ API Keys chưa được cấu hình")

# Kiểm tra kết nối trực tiếp đến Binance
st.subheader("Kiểm tra kết nối trực tiếp")

if st.button("Kiểm tra kết nối đến Binance"):
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
        if response.status_code == 200:
            st.success(f"✅ Kết nối trực tiếp đến Binance thành công: {response.status_code}")
        else:
            st.error(f"❌ Lỗi kết nối trực tiếp đến Binance: {response.status_code}")
            st.text(response.text)
    except Exception as e:
        st.error(f"❌ Lỗi kết nối trực tiếp đến Binance: {str(e)}")

# Kiểm tra kết nối qua proxy
st.subheader("Kiểm tra kết nối qua proxy")

# Kiểm tra proxy có trong cấu hình không
proxy_url = os.environ.get("PROXY_URL", "64.176.51.107:3128:hvnteam:matkhau123")

if proxy_url:
    st.write(f"Proxy: {proxy_url}")
    
    # Parse proxy URL
    proxy_parts = proxy_url.split(":")
    if len(proxy_parts) >= 2:
        proxy_host = proxy_parts[0]
        proxy_port = proxy_parts[1]
        
        # Build proxy dict for requests
        proxies = {
            "http": f"http://{proxy_host}:{proxy_port}",
            "https": f"http://{proxy_host}:{proxy_port}"
        }
        
        # Add auth if provided
        if len(proxy_parts) >= 4:
            proxy_user = proxy_parts[2]
            proxy_pass = proxy_parts[3]
            proxies = {
                "http": f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}",
                "https": f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
            }
            
        # Test with proxy
        if st.button("Kiểm tra kết nối qua proxy"):
            try:
                response = requests.get(
                    "https://api.binance.com/api/v3/ping", 
                    proxies=proxies,
                    timeout=10
                )
                if response.status_code == 200:
                    st.success(f"✅ Kết nối qua proxy đến Binance thành công: {response.status_code}")
                else:
                    st.error(f"❌ Lỗi kết nối qua proxy đến Binance: {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"❌ Lỗi kết nối qua proxy đến Binance: {str(e)}")
else:
    st.warning("⚠️ Không tìm thấy cấu hình proxy")

# Lấy thông tin thị trường
st.subheader("Thông tin thị trường ETH/USDT")

if st.button("Lấy giá ETH/USDT hiện tại"):
    try:
        # Thử kết nối trực tiếp trước
        response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT", timeout=10)
        
        # Nếu không được, thử qua proxy
        if response.status_code != 200 and proxy_url:
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT",
                proxies=proxies,
                timeout=10
            )
            
        if response.status_code == 200:
            data = response.json()
            price = float(data['price'])
            st.success(f"Giá ETH/USDT hiện tại: ${price:,.2f}")
        else:
            st.error(f"Không thể lấy giá ETH/USDT: {response.status_code}")
            st.text(response.text)
    except Exception as e:
        st.error(f"Lỗi khi lấy giá ETH/USDT: {str(e)}")

st.markdown("---")
st.write("Thông tin hệ thống:")
st.write(f"- Streamlit version: {st.__version__}")
st.write(f"- Python version: {__import__('sys').version.split(' ')[0]}")
st.write(f"- Requests version: {requests.__version__}")
st.write(f"- Pandas version: {pd.__version__}")

# Mã nguồn để theo dõi lỗi
with st.expander("Xem mã nguồn"):
    st.code("""
import streamlit as st
import os
import requests
import pandas as pd
from datetime import datetime

# Thiết lập tiêu đề
st.set_page_config(page_title="Kiểm tra kết nối Binance API", layout="wide")
st.title("Kiểm tra kết nối Binance API")

# Hiển thị thời gian hiện tại
st.write(f"Thời gian hiện tại: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Kiểm tra API Keys
api_key = os.environ.get("BINANCE_API_KEY", "")
api_secret = os.environ.get("BINANCE_API_SECRET", "")

if api_key and api_secret:
    st.success("✅ API Keys đã được cấu hình")
    
    # Hiển thị một phần của key để xác nhận
    masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
    st.write(f"API Key: {masked_key}")
else:
    st.error("❌ API Keys chưa được cấu hình")

# Kiểm tra kết nối trực tiếp đến Binance
st.subheader("Kiểm tra kết nối trực tiếp")

if st.button("Kiểm tra kết nối đến Binance"):
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
        if response.status_code == 200:
            st.success(f"✅ Kết nối trực tiếp đến Binance thành công: {response.status_code}")
        else:
            st.error(f"❌ Lỗi kết nối trực tiếp đến Binance: {response.status_code}")
            st.text(response.text)
    except Exception as e:
        st.error(f"❌ Lỗi kết nối trực tiếp đến Binance: {str(e)}")
""", language="python")