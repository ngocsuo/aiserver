"""
Loads environment variables from .env file into the system
"""
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_environment_variables(env_file='.env'):
    """
    Load environment variables from a .env file

    Args:
        env_file (str): Path to .env file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Kiểm tra xem file có tồn tại không
        if not os.path.isfile(env_file):
            logger.warning(f".env file not found at {env_file}. Using environment variables only.")
            return False

        # Tải biến môi trường từ file .env
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")
        return True
    except Exception as e:
        logger.error(f"Error loading environment variables: {str(e)}")
        return False

def get_api_keys():
    """
    Get API keys from environment variables

    Returns:
        tuple: (api_key, api_secret)
    """
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        logger.warning("Binance API keys not found in environment variables")
    
    return api_key, api_secret

def get_proxy_config():
    """
    Get proxy configuration from environment variables

    Returns:
        tuple: (proxy_url, proxy_list)
    """
    proxy_url = os.environ.get('PROXY_URL')
    proxy_list = os.environ.get('PROXY_LIST', '')
    
    # Nếu proxy_list là chuỗi, chuyển thành list
    if proxy_list and isinstance(proxy_list, str):
        proxy_list = proxy_list.split(',')
    
    return proxy_url, proxy_list

def get_database_config():
    """
    Get database configuration from environment variables

    Returns:
        dict: Database configuration
    """
    return {
        'host': os.environ.get('DB_HOST', 'localhost'),
        'port': int(os.environ.get('DB_PORT', 5432)),
        'user': os.environ.get('DB_USER', 'postgres'),
        'password': os.environ.get('DB_PASSWORD', 'postgres'),
        'dbname': os.environ.get('DB_NAME', 'ethusdt_dashboard')
    }

def get_system_config():
    """
    Get system configuration from environment variables

    Returns:
        dict: System configuration
    """
    return {
        'debug': os.environ.get('DEBUG', 'False').lower() == 'true',
        'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
        'force_retrain': os.environ.get('FORCE_RETRAIN', 'False').lower() == 'true',
        'training_interval': int(os.environ.get('TRAINING_INTERVAL', 30))
    }

def get_futures_config():
    """
    Get futures configuration from environment variables
    
    Returns:
        dict: Futures configuration
    """
    return {
        'enabled': os.environ.get('FUTURES_ENABLED', 'False').lower() == 'true'
    }

def initialize_environment():
    """
    Initialize environment variables and return configuration
    
    Returns:
        dict: Configuration
    """
    load_environment_variables()
    
    api_key, api_secret = get_api_keys()
    proxy_url, proxy_list = get_proxy_config()
    db_config = get_database_config()
    system_config = get_system_config()
    futures_config = get_futures_config()
    
    return {
        'api_key': api_key,
        'api_secret': api_secret,
        'proxy_url': proxy_url,
        'proxy_list': proxy_list,
        'db_config': db_config,
        'system_config': system_config,
        'futures_config': futures_config
    }