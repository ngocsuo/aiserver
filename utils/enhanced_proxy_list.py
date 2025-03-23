"""
Danh sách proxy nâng cao với khả năng auto-retry và rotation
"""
import os
import random
import logging
import time
import json
import requests
from dotenv import load_dotenv

# Thiết lập logger
logger = logging.getLogger(__name__)

# Tải biến môi trường
load_dotenv()

# Danh sách proxy mặc định
DEFAULT_PROXY_LIST = [
    "88.198.24.108:8080",
    "51.159.66.223:3128",
    "185.150.130.103:808",
    "3.220.228.103:1081",
    "159.65.146.253:8080",
    "45.77.232.126:3128",
    "121.156.109.108:8080",
    "50.174.7.155:80",
    "165.225.208.91:10605",
    "47.243.238.186:1081",
    "167.233.15.39:3128",
    "112.78.170.250:80",
    "159.89.113.155:8080"
]

# Tải danh sách proxy từ biến môi trường nếu có
PROXY_LIST = os.environ.get('PROXY_LIST', ','.join(DEFAULT_PROXY_LIST)).split(',')

def parse_proxy_url(proxy_url):
    """
    Phân tích URL proxy thành các thành phần
    
    Args:
        proxy_url (str): URL proxy theo format "host:port" hoặc "host:port:username:password"
        
    Returns:
        dict: Các thành phần của proxy
    """
    if not proxy_url:
        return None
        
    parts = proxy_url.split(':')
    
    if len(parts) >= 2:
        proxy_config = {
            "host": parts[0],
            "port": int(parts[1]),
            "auth": None
        }
        
        # Kiểm tra xem có thông tin xác thực không
        if len(parts) >= 4:
            proxy_config["auth"] = (parts[2], parts[3])
        
        return proxy_config
    
    return None

def get_proxy_url(proxy_config):
    """
    Tạo URL proxy từ cấu hình
    
    Args:
        proxy_config (dict): Cấu hình proxy
        
    Returns:
        str: URL proxy
    """
    if not proxy_config:
        return None
        
    if proxy_config.get("auth"):
        username, password = proxy_config["auth"]
        return f"http://{username}:{password}@{proxy_config['host']}:{proxy_config['port']}"
    else:
        return f"http://{proxy_config['host']}:{proxy_config['port']}"

def get_proxy_dict(proxy_config):
    """
    Tạo dictionary proxy cho requests
    
    Args:
        proxy_config (dict): Cấu hình proxy
        
    Returns:
        dict: Dictionary proxy cho requests
    """
    if not proxy_config:
        return None
        
    url = get_proxy_url(proxy_config)
    if url:
        return {
            "http": url,
            "https": url
        }
    
    return None

def get_random_proxy():
    """
    Lấy proxy ngẫu nhiên từ danh sách
    
    Returns:
        dict: Cấu hình proxy
    """
    if not PROXY_LIST:
        return None
        
    proxy_url = random.choice(PROXY_LIST)
    return parse_proxy_url(proxy_url)

def test_proxy(proxy_config, test_url="https://api.binance.com/api/v3/ping", timeout=5):
    """
    Kiểm tra proxy có hoạt động không
    
    Args:
        proxy_config (dict): Cấu hình proxy
        test_url (str): URL để kiểm tra
        timeout (int): Thời gian timeout
        
    Returns:
        bool: True nếu proxy hoạt động, False nếu không
    """
    if not proxy_config:
        return False
        
    proxies = get_proxy_dict(proxy_config)
    
    try:
        start_time = time.time()
        response = requests.get(test_url, proxies=proxies, timeout=timeout)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"Proxy {proxy_config['host']}:{proxy_config['port']} hoạt động (Phản hồi: {elapsed:.2f}s)")
            return True
    except requests.exceptions.RequestException as e:
        logger.debug(f"Proxy {proxy_config['host']}:{proxy_config['port']} không hoạt động: {str(e)}")
    
    return False

def find_working_proxy(max_attempts=5, test_url="https://api.binance.com/api/v3/ping"):
    """
    Tìm proxy hoạt động từ danh sách
    
    Args:
        max_attempts (int): Số lần thử tối đa
        test_url (str): URL để kiểm tra
        
    Returns:
        dict: Cấu hình proxy hoạt động hoặc None nếu không tìm thấy
    """
    logger.info(f"Tìm proxy hoạt động từ danh sách {len(PROXY_LIST)} proxy...")
    
    # Sắp xếp ngẫu nhiên danh sách proxy để tránh luôn dùng cùng một proxy
    shuffled_proxies = list(PROXY_LIST)
    random.shuffle(shuffled_proxies)
    
    for _ in range(max_attempts):
        for proxy_url in shuffled_proxies:
            proxy_config = parse_proxy_url(proxy_url)
            if proxy_config and test_proxy(proxy_config, test_url):
                return proxy_config
    
    logger.warning("Không tìm thấy proxy hoạt động trong danh sách")
    return None

def get_proxy_for_binance():
    """
    Lấy proxy tốt nhất cho Binance API
    
    Returns:
        dict: Cấu hình proxy
    """
    working_proxy = find_working_proxy(test_url="https://api.binance.com/api/v3/ping")
    if working_proxy:
        return working_proxy
        
    # Thử với API Futures nếu không thành công với API thông thường
    return find_working_proxy(test_url="https://fapi.binance.com/fapi/v1/ping")

def update_proxy_list(new_list):
    """
    Cập nhật danh sách proxy
    
    Args:
        new_list (list): Danh sách proxy mới
    """
    global PROXY_LIST
    PROXY_LIST = new_list
    
    # Lưu vào file cấu hình để sử dụng sau này
    try:
        with open("proxy_list.json", "w") as f:
            json.dump(PROXY_LIST, f)
    except Exception as e:
        logger.error(f"Lỗi khi lưu danh sách proxy: {str(e)}")