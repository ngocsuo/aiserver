"""
Factory module cho data collector - Đã loại bỏ toàn bộ proxy cho triển khai server
"""
import os
import logging
from utils.data_collector_proxy import BinanceDataCollector
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
    
    # Luôn sử dụng kết nối trực tiếp (proxy được cấu hình ở cấp server)
    logger.info("Proxy được quản lý ở cấp hệ thống/server.")
    logger.info("Connecting directly to Binance API")
    
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
        logger.error(f"Could not connect to Binance API: {e}")
        
        # Không sử dụng MockDataCollector mà báo lỗi để người dùng biết
        error_message = f"Không thể kết nối đến Binance API: {e}"
        logger.error(error_message)
        raise Exception(error_message)