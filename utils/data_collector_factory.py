"""
Factory module cho data collector hỗ trợ proxy
"""
import os
import logging
from utils.data_collector_proxy import BinanceDataCollector
from utils.proxy_config import configure_proxy, get_proxy_url_format
import config

# Set up logging
logger = logging.getLogger("data_collector_factory")

def create_data_collector():
    """
    Tạo instance data collector phù hợp với cấu hình hiện tại.
    
    Returns:
        object: BinanceDataCollector instance
    """
    logger.info("Attempting to use Binance API data collector")
    
    # Cấu hình proxy nếu có
    proxies = configure_proxy()
    proxy_url = get_proxy_url_format()
    
    if proxies and proxy_url:
        logger.info(f"Connecting to Binance API using proxy")
    else:
        logger.info(f"Connecting directly to Binance API")
    
    try:
        # Tạo và trả về data collector
        collector = BinanceDataCollector()
        
        # Chỉ sử dụng collector nếu kết nối thành công
        if collector.connection_status["connected"]:
            logger.info("Successfully initialized Binance data collector")
            return collector
        else:
            raise Exception(f"Could not connect to Binance API: {collector.connection_status['message']}")
            
    except Exception as e:
        logger.warning(f"Could not connect to Binance API: {e}")
        
        # Trả về MockDataCollector nếu cần
        from utils.data_collector import MockDataCollector
        logger.info("Falling back to MockDataCollector")
        return MockDataCollector()