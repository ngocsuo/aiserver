"""
Cấu hình proxy nâng cao cho kết nối Binance API
"""
import os
import logging
import socket
import requests
import socks
import time
import random

# Thiết lập logging
logger = logging.getLogger("proxy_config")

# Danh sách proxy dự phòng - Thêm nhiều proxy để tăng khả năng kết nối thành công
BACKUP_PROXIES = [
    # Proxy mới - các proxy đáng tin cậy và chưa bị Binance phát hiện
    "146.56.175.38:80",        # Seoul, South Korea
    "161.82.252.35:4153",      # Dubai, UAE  
    "51.79.50.22:9300",        # Singapore
    "156.67.172.185:3128",     # Singapore 
    "20.81.62.32:3128",        # Tokyo, Japan
    "103.83.232.122:80",       # Hong Kong
    "178.18.245.74:8888",      # Tokyo, Japan
    "45.8.107.73:80",          # Singapore
    "13.112.197.90:80",        # Tokyo, Japan
    "169.57.1.85:8123",        # Asia Region
    "152.67.99.80:80",         # Singapore
    "142.93.113.81:80",        # Singapore
    
    # Proxy cũ (giữ lại phòng trường hợp vẫn dùng được)
    "64.176.51.107:3128:hvnteam:matkhau123",  # Proxy hiện tại
    "38.154.227.167:5868:hvnteam:matkhau123", # Proxy dự phòng 1 
    "45.155.68.129:8133:hvnteam:matkhau123",  # Proxy dự phòng 2
    "185.199.229.156:7492:hvnteam:matkhau123",# Proxy dự phòng 3
    "185.199.228.220:7300:hvnteam:matkhau123",# Proxy dự phòng 4
    "185.199.231.45:8382:hvnteam:matkhau123", # Proxy dự phòng 5
    "154.95.36.199:6893:hvnteam:matkhau123",  # Proxy dự phòng 6
]

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

def test_proxy(proxy_config, test_url="https://api.binance.com/api/v3/ping"):
    """
    Kiểm tra kết nối proxy
    
    Args:
        proxy_config (dict): Cấu hình proxy
        test_url (str): URL để kiểm tra kết nối
        
    Returns:
        bool: Kết quả kiểm tra
    """
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
            
        # Thử kết nối đến URL test qua proxy
        response = requests.get(test_url, proxies=proxies, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Kết nối proxy thành công: {proxy_config['host']}:{proxy_config['port']}")
            return True
        else:
            logger.warning(f"Kết nối proxy thất bại: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi kết nối proxy: {str(e)}")
        return False

def find_working_proxy(proxy_list=None):
    """
    Tìm proxy hoạt động từ danh sách
    
    Args:
        proxy_list (list): Danh sách proxy dạng chuỗi
        
    Returns:
        dict: Cấu hình proxy hoạt động
    """
    if proxy_list is None:
        # Sử dụng danh sách proxy dự phòng
        proxy_list = BACKUP_PROXIES
        
    # Thêm proxy từ biến môi trường nếu có
    env_proxy = os.environ.get("PROXY_URL")
    if env_proxy and env_proxy not in proxy_list:
        proxy_list.insert(0, env_proxy)  # Ưu tiên proxy từ biến môi trường
        
    # Xáo trộn danh sách proxy để thử ngẫu nhiên
    random.shuffle(proxy_list)
    
    # Thử từng proxy trong danh sách
    for proxy_url in proxy_list:
        proxy_config = parse_proxy_url(proxy_url)
        if not proxy_config:
            continue
            
        logger.info(f"Kiểm tra proxy: {proxy_config['host']}:{proxy_config['port']}")
        if test_proxy(proxy_config):
            return proxy_config
            
    # Không tìm thấy proxy hoạt động
    logger.error("Không tìm thấy proxy hoạt động!")
    return None

def get_proxy_format(proxy_config):
    """
    Tạo định dạng proxy cho requests
    
    Args:
        proxy_config (dict): Cấu hình proxy
        
    Returns:
        dict: Cấu hình proxy cho requests
    """
    if not proxy_config:
        return None
        
    if proxy_config["auth"]:
        return {
            "http": f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['host']}:{proxy_config['port']}",
            "https": f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['host']}:{proxy_config['port']}"
        }
    else:
        return {
            "http": f"http://{proxy_config['host']}:{proxy_config['port']}",
            "https": f"http://{proxy_config['host']}:{proxy_config['port']}"
        }

def get_proxy_url_format(proxy_config):
    """
    Trả về URL proxy theo định dạng cho requests
    
    Args:
        proxy_config (dict): Cấu hình proxy
        
    Returns:
        str: URL proxy theo định dạng "http://username:password@host:port"
    """
    if not proxy_config:
        return None
        
    if proxy_config["auth"]:
        return f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['host']}:{proxy_config['port']}"
    else:
        return f"http://{proxy_config['host']}:{proxy_config['port']}"

def configure_enhanced_proxy():
    """
    Cấu hình proxy nâng cao cho toàn bộ ứng dụng
    
    Returns:
        tuple: (proxies, proxy_config) - Cấu hình proxy cho requests và chi tiết cấu hình
    """
    # Tìm proxy hoạt động
    proxy_config = find_working_proxy()
    
    if not proxy_config:
        logger.warning("Không tìm thấy proxy hoạt động")
        return None, None
        
    # Cấu hình socket proxy
    try:
        socks.set_default_proxy(
            socks.HTTP, 
            proxy_config["host"], 
            proxy_config["port"],
            username=proxy_config["username"] if proxy_config["auth"] else None,
            password=proxy_config["password"] if proxy_config["auth"] else None
        )
        
        logger.info(f"Đã cấu hình socket proxy: {proxy_config['host']}:{proxy_config['port']}")
    except Exception as e:
        logger.error(f"Lỗi khi cấu hình socket proxy: {str(e)}")
    
    # Trả về định dạng proxy cho requests
    return get_proxy_format(proxy_config), proxy_config

# Hàm trợ giúp cấu hình kết nối Binance với proxy
def configure_binance_client(api_key, api_secret):
    """
    Cấu hình Binance client với proxy
    
    Args:
        api_key (str): Binance API key
        api_secret (str): Binance API secret
        
    Returns:
        tuple: (client, status) - Binance client và trạng thái kết nối
    """
    from binance.client import Client
    
    # Cấu hình proxy nâng cao
    proxies, proxy_config = configure_enhanced_proxy()
    
    status = {
        "connected": False,
        "error": None,
        "message": "Initializing connection to Binance API",
        "last_check": time.strftime("%Y-%m-%d %H:%M:%S"),
        "using_proxy": False
    }
    
    if proxies and proxy_config:
        try:
            logger.info(f"Connecting to Binance API using proxy: {proxy_config['host']}:{proxy_config['port']}")
            status["using_proxy"] = True
            
            # Tạo client với proxy
            client = Client(
                api_key, 
                api_secret,
                {"timeout": 30, "proxies": proxies}
            )
            
            # Test kết nối
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(20)  # 20 giây timeout
            
            try:
                # Ping server để kiểm tra kết nối
                client.ping()
                logger.info("Binance API connection successful")
                
                # Kiểm tra trạng thái hệ thống
                system_status = client.get_system_status()
                if system_status['status'] == 0:
                    logger.info("Binance system status: Normal")
                else:
                    logger.warning(f"Binance system status: Maintenance - {system_status['msg']}")
                
                # Kiểm tra xem API Futures có thể truy cập không
                test_klines = client.futures_klines(
                    symbol="ETHUSDT",
                    interval="1m",
                    limit=1
                )
                
                if test_klines and len(test_klines) > 0:
                    logger.info("Binance Futures API accessible")
                    
                    # Cập nhật trạng thái kết nối
                    status["connected"] = True
                    status["message"] = "Connected to Binance API"
                    status["error"] = None
                    
                    return client, status
                else:
                    logger.error("Error accessing Binance Futures API")
                    status["error"] = "Error accessing Binance Futures API"
            except Exception as e:
                logger.error(f"Error checking Binance connection: {str(e)}")
                status["error"] = str(e)
            finally:
                # Khôi phục timeout
                socket.setdefaulttimeout(original_timeout)
                
        except Exception as e:
            logger.error(f"Error creating Binance client: {str(e)}")
            status["error"] = str(e)
    else:
        logger.warning("No working proxy found for Binance connection")
        status["error"] = "No working proxy found"
        status["message"] = "No working proxy found for Binance connection"
    
    # Nếu proxy không hoạt động, thử kết nối trực tiếp
    try:
        logger.info("Attempting direct connection to Binance API")
        client = Client(api_key, api_secret, {"timeout": 30})
        
        # Test kết nối
        client.ping()
        logger.info("Direct Binance API connection successful")
        
        # Kiểm tra trạng thái hệ thống
        system_status = client.get_system_status()
        if system_status['status'] == 0:
            logger.info("Binance system status: Normal")
        else:
            logger.warning(f"Binance system status: Maintenance - {system_status['msg']}")
        
        # Kiểm tra xem API Futures có thể truy cập không
        test_klines = client.futures_klines(
            symbol="ETHUSDT",
            interval="1m",
            limit=1
        )
        
        if test_klines and len(test_klines) > 0:
            logger.info("Binance Futures API accessible")
            
            # Cập nhật trạng thái kết nối
            status["connected"] = True
            status["message"] = "Connected directly to Binance API"
            status["error"] = None
            status["using_proxy"] = False
            
            return client, status
    except Exception as e:
        logger.error(f"Error with direct Binance connection: {str(e)}")
        status["error"] = str(e)
        
    return None, status