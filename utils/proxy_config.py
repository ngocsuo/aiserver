"""
Module cấu hình proxy cho Binance API
"""
import os
import requests
import socks
import socket
from urllib.parse import urlparse
import logging

# Thiết lập logging
logger = logging.getLogger("proxy_config")

# Thông tin proxy từ biến môi trường hoặc cấu hình cứng
PROXY_URL = "64.176.51.107:3128:hvnteam:matkhau123"  # format: host:port:username:password

def parse_proxy_url(proxy_url):
    """
    Phân tích URL proxy thành các thành phần
    
    Args:
        proxy_url (str): URL proxy theo format "host:port:username:password"
        
    Returns:
        dict: Các thành phần của proxy
    """
    parts = proxy_url.split(':')
    
    if len(parts) >= 2:
        proxy_config = {
            'host': parts[0],
            'port': int(parts[1]),
        }
        
        if len(parts) >= 4:
            proxy_config['username'] = parts[2]
            proxy_config['password'] = parts[3]
            
        return proxy_config
    
    return None

def get_proxy_url_format():
    """
    Trả về URL proxy theo định dạng cho requests
    
    Returns:
        str: URL proxy theo định dạng "http://username:password@host:port"
    """
    proxy_config = parse_proxy_url(PROXY_URL)
    
    if not proxy_config:
        return None
        
    if 'username' in proxy_config and 'password' in proxy_config:
        return f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['host']}:{proxy_config['port']}"
    else:
        return f"http://{proxy_config['host']}:{proxy_config['port']}"

def configure_proxy():
    """
    Cấu hình proxy cho toàn bộ ứng dụng
    
    Returns:
        dict: Cấu hình proxy cho requests
    """
    proxy_config = parse_proxy_url(PROXY_URL)
    
    if not proxy_config:
        logger.warning("Không tìm thấy cấu hình proxy hợp lệ")
        return {}
    
    try:
        # Cấu hình cho thư viện requests
        proxy_url = get_proxy_url_format()
        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        
        # Kiểm tra kết nối proxy
        try:
            logger.info(f"Kiểm tra kết nối proxy: {proxy_config['host']}:{proxy_config['port']}")
            test_url = "https://api.binance.com/api/v3/ping"
            response = requests.get(test_url, proxies=proxies, timeout=10)
            if response.ok:
                logger.info(f"Kết nối proxy thành công: {proxy_config['host']}:{proxy_config['port']}")
            else:
                logger.warning(f"Kết nối proxy không thành công: {response.status_code}")
        except Exception as e:
            logger.error(f"Lỗi kiểm tra kết nối proxy: {e}")
        
        return proxies
    except Exception as e:
        logger.error(f"Lỗi cấu hình proxy: {e}")
        return {}

def configure_socket_proxy():
    """
    Cấu hình proxy cho socket (PySocks)
    
    Returns:
        bool: Kết quả cấu hình
    """
    proxy_config = parse_proxy_url(PROXY_URL)
    
    if not proxy_config:
        logger.warning("Không tìm thấy cấu hình socket proxy hợp lệ")
        return False
    
    try:
        # Cấu hình PySocks (nếu có)
        import socks
        
        socks.set_default_proxy(
            socks.HTTP, 
            proxy_config['host'], 
            proxy_config['port'],
            username=proxy_config.get('username'),
            password=proxy_config.get('password')
        )
        socket.socket = socks.socksocket
        
        logger.info(f"Đã cấu hình socket proxy: {proxy_config['host']}:{proxy_config['port']}")
        return True
    except ImportError:
        logger.error("Không thể import thư viện socks, cài đặt bằng: pip install PySocks")
        return False
    except Exception as e:
        logger.error(f"Lỗi cấu hình socket proxy: {e}")
        return False