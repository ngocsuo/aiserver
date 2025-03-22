"""
Module cấu hình proxy cho Binance API
"""
import os
import logging
import socket
import requests
import socks

# Thiết lập logging
logger = logging.getLogger("proxy_config")

# Thông tin proxy mặc định (sẽ được ghi đè nếu đọc từ biến môi trường)
DEFAULT_PROXY = "64.176.51.107:3128:hvnteam:matkhau123"

def parse_proxy_url(proxy_url):
    """
    Phân tích URL proxy thành các thành phần
    
    Args:
        proxy_url (str): URL proxy theo format "host:port:username:password"
        
    Returns:
        dict: Các thành phần của proxy
    """
    if not proxy_url:
        return None
        
    parts = proxy_url.split(":")
    
    if len(parts) == 2:
        # Format host:port
        return {
            "host": parts[0],
            "port": int(parts[1]),
            "auth": False,
            "username": None,
            "password": None
        }
    elif len(parts) == 4:
        # Format host:port:username:password
        return {
            "host": parts[0],
            "port": int(parts[1]),
            "auth": True,
            "username": parts[2],
            "password": parts[3]
        }
    else:
        logger.error(f"Không thể phân tích định dạng proxy: {proxy_url}")
        return None

def get_proxy_url_format():
    """
    Trả về URL proxy theo định dạng cho requests
    
    Returns:
        str: URL proxy theo định dạng "http://username:password@host:port"
    """
    # Lấy thông tin proxy từ biến môi trường hoặc cấu hình mặc định
    proxy_str = os.environ.get("PROXY_URL", DEFAULT_PROXY)
    
    if not proxy_str:
        return None
        
    proxy_config = parse_proxy_url(proxy_str)
    
    if not proxy_config:
        return None
        
    if proxy_config["auth"]:
        return f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['host']}:{proxy_config['port']}"
    else:
        return f"http://{proxy_config['host']}:{proxy_config['port']}"

def configure_proxy():
    """
    Cấu hình proxy cho toàn bộ ứng dụng
    
    Returns:
        dict: Cấu hình proxy cho requests
    """
    # Lấy thông tin proxy từ biến môi trường hoặc cấu hình mặc định
    proxy_str = os.environ.get("PROXY_URL", DEFAULT_PROXY)
    
    if not proxy_str:
        logger.warning("Không tìm thấy cấu hình proxy")
        return None
        
    proxy_config = parse_proxy_url(proxy_str)
    
    if not proxy_config:
        return None
        
    logger.info(f"Kiểm tra kết nối proxy: {proxy_config['host']}:{proxy_config['port']}")
    
    # Kiểm tra xem proxy có thể kết nối được không
    try:
        if proxy_config["auth"]:
            proxies = {
                "http": f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['host']}:{proxy_config['port']}",
                "https": f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['host']}:{proxy_config['port']}"
            }
        else:
            proxies = {
                "http": f"http://{proxy_config['host']}:{proxy_config['port']}",
                "https": f"http://{proxy_config['host']}:{proxy_config['port']}"
            }
            
        # Thử kết nối đến Binance qua proxy
        response = requests.get("https://api.binance.com/api/v3/ping", 
                              proxies=proxies, 
                              timeout=5)
        
        if response.status_code == 200:
            logger.info(f"Kết nối proxy thành công: {proxy_config['host']}:{proxy_config['port']}")
            return proxies
        else:
            logger.warning(f"Kết nối proxy thất bại: HTTP {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi kết nối proxy: {str(e)}")
        return None

def configure_socket_proxy():
    """
    Cấu hình proxy cho socket (PySocks)
    
    Returns:
        bool: Kết quả cấu hình
    """
    # Lấy thông tin proxy từ biến môi trường hoặc cấu hình mặc định
    proxy_str = os.environ.get("PROXY_URL", DEFAULT_PROXY)
    
    if not proxy_str:
        return False
        
    proxy_config = parse_proxy_url(proxy_str)
    
    if not proxy_config:
        return False
    
    try:
        # Cấu hình proxy cho socket
        socks.set_default_proxy(
            socks.HTTP, 
            proxy_config["host"], 
            proxy_config["port"],
            username=proxy_config["username"] if proxy_config["auth"] else None,
            password=proxy_config["password"] if proxy_config["auth"] else None
        )
        
        # Không ghi đè socket.socket hoàn toàn để tránh ảnh hưởng đến các kết nối khác
        # Chỉ ghi đè khi cần thiết trong các module cụ thể
        
        logger.info(f"Đã cấu hình socket proxy: {proxy_config['host']}:{proxy_config['port']}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi cấu hình socket proxy: {str(e)}")
        return False