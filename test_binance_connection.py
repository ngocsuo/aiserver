"""
Script kiểm tra kết nối đến Binance sử dụng proxy
"""
import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd
import requests

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_connection")

# Đảm bảo utils được thêm vào path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các module cần thiết
from utils.proxy_config import configure_proxy, get_proxy_url_format, parse_proxy_url
import config

def test_proxy_connection():
    """Kiểm tra kết nối proxy trực tiếp"""
    proxy_config = parse_proxy_url("64.176.51.107:3128:hvnteam:matkhau123")
    
    if not proxy_config:
        logger.error("Không thể phân tích URL proxy")
        return False
    
    logger.info(f"Testing proxy connection: {proxy_config['host']}:{proxy_config['port']}")
    
    # Định dạng URL proxy cho requests
    proxy_url = get_proxy_url_format()
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    
    logger.info(f"Proxy URL: {proxy_url}")
    
    # Kiểm tra kết nối
    test_urls = [
        "https://api.binance.com/api/v3/ping",
        "https://api.binance.com/api/v3/time",
        "https://fapi.binance.com/fapi/v1/ping"
    ]
    
    success = True
    
    for url in test_urls:
        try:
            logger.info(f"Testing connection to {url}")
            start_time = time.time()
            response = requests.get(url, proxies=proxies, timeout=10)
            end_time = time.time()
            
            if response.ok:
                logger.info(f"✅ Connection successful to {url}")
                logger.info(f"   Response time: {end_time - start_time:.2f}s")
                logger.info(f"   Response: {response.text}")
            else:
                logger.error(f"❌ Connection failed to {url}")
                logger.error(f"   Status code: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                success = False
        except Exception as e:
            logger.error(f"❌ Error connecting to {url}: {e}")
            success = False
    
    return success

def test_binance_api():
    """Kiểm tra kết nối Binance API với proxy"""
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    
    # Cấu hình proxy
    proxies = configure_proxy()
    
    # Nếu không có API keys, nhắc người dùng
    if not hasattr(config, 'BINANCE_API_KEY') or not config.BINANCE_API_KEY:
        logger.warning("Binance API keys not configured. Testing without authentication.")
        api_key = ""
        api_secret = ""
    else:
        api_key = config.BINANCE_API_KEY
        api_secret = config.BINANCE_API_SECRET
    
    try:
        # Khởi tạo client với proxy
        logger.info("Initializing Binance client with proxy")
        client = Client(api_key, api_secret, {'proxies': proxies, 'timeout': 30})
        
        # Test ping
        logger.info("Testing ping")
        ping_response = client.ping()
        logger.info(f"Ping successful: {ping_response}")
        
        # Test server time
        logger.info("Testing server time")
        server_time = client.get_server_time()
        logger.info(f"Server time: {server_time}")
        
        # Test lấy thông tin symbol
        logger.info("Testing get exchange info")
        exchange_info = client.get_exchange_info()
        symbol_count = len(exchange_info['symbols'])
        logger.info(f"Exchange info: {symbol_count} symbols available")
        
        # Test futures ping nếu có API keys
        if api_key and api_secret:
            try:
                logger.info("Testing futures ping")
                futures_ping = client.futures_ping()
                logger.info(f"Futures ping successful: {futures_ping}")
                
                # Test lấy klines
                logger.info("Testing get klines")
                klines = client.futures_klines(symbol="ETHUSDT", interval="5m", limit=5)
                logger.info(f"Retrieved {len(klines)} klines for ETHUSDT 5m")
                
                # In ra nến cuối cùng
                if klines:
                    last_candle = klines[-1]
                    candle_time = datetime.fromtimestamp(last_candle[0]/1000)
                    logger.info(f"Latest candle time: {candle_time}")
                    logger.info(f"Open: {last_candle[1]}, High: {last_candle[2]}, Low: {last_candle[3]}, Close: {last_candle[4]}")
                
                return True
            except BinanceAPIException as e:
                logger.error(f"Binance API error during futures tests: {e}")
                if 'APIError(code=0)' in str(e) or 'restricted location' in str(e).lower():
                    logger.error("Geographic restriction detected. Proxy might not be bypassing restrictions.")
                return False
        else:
            logger.warning("Skipping futures tests due to missing API keys")
            return True
        
    except BinanceAPIException as e:
        logger.error(f"Binance API error: {e}")
        if 'APIError(code=0)' in str(e) or 'restricted location' in str(e).lower():
            logger.error("Geographic restriction detected. Proxy might not be bypassing restrictions.")
        return False
    except Exception as e:
        logger.error(f"Error testing Binance API: {e}")
        return False

def test_binance_data_collector():
    """Test BinanceDataCollector với proxy"""
    logger.info("Testing BinanceDataCollector with proxy")
    
    try:
        # Import data collector
        from utils.data_collector_proxy import BinanceDataCollector
        
        # Khởi tạo data collector
        collector = BinanceDataCollector()
        
        # Kiểm tra trạng thái kết nối
        if collector.connection_status["connected"]:
            logger.info("✅ BinanceDataCollector connected successfully")
            
            # Kiểm tra thu thập dữ liệu
            logger.info("Testing data collection for ETHUSDT 5m")
            data = collector.collect_historical_data(symbol="ETHUSDT", timeframe="5m", limit=10)
            
            if data is not None and not data.empty:
                logger.info(f"✅ Successfully collected {len(data)} candles")
                logger.info(f"Latest candle: {data.iloc[-1]}")
                return True
            else:
                logger.error("❌ Failed to collect data")
                return False
        else:
            logger.error(f"❌ BinanceDataCollector failed to connect: {collector.connection_status['message']}")
            return False
    except Exception as e:
        logger.error(f"❌ Error testing BinanceDataCollector: {e}")
        return False

def test_data_collector_factory():
    """Test data_collector_factory với proxy"""
    logger.info("Testing data_collector_factory with proxy")
    
    try:
        # Import factory
        from utils.data_collector_factory import create_data_collector
        
        # Tạo data collector
        collector = create_data_collector()
        
        if hasattr(collector, 'connection_status'):
            if collector.connection_status["connected"]:
                logger.info("✅ Data collector factory created a connected collector")
                
                # Kiểm tra thu thập dữ liệu
                logger.info("Testing data collection for ETHUSDT 5m")
                data = collector.collect_historical_data(symbol="ETHUSDT", timeframe="5m", limit=10)
                
                if data is not None and not data.empty:
                    logger.info(f"✅ Successfully collected {len(data)} candles")
                    return True
                else:
                    logger.error("❌ Failed to collect data")
                    return False
            else:
                logger.error(f"❌ Factory created a disconnected collector: {collector.connection_status['message']}")
                return False
        else:
            logger.warning("Created a collector without connection_status (likely MockDataCollector)")
            logger.warning("This may indicate proxy config issues, check logs for details")
            return False
    except Exception as e:
        logger.error(f"❌ Error testing data_collector_factory: {e}")
        return False

if __name__ == "__main__":
    print("===== TESTING BINANCE CONNECTION WITH PROXY =====")
    
    # Kiểm tra kết nối proxy cơ bản
    print("\n----- Testing Proxy Connection -----")
    proxy_success = test_proxy_connection()
    
    # Kiểm tra API Binance
    print("\n----- Testing Binance API -----")
    api_success = test_binance_api()
    
    # Kiểm tra Data Collector
    print("\n----- Testing BinanceDataCollector -----")
    collector_success = test_binance_data_collector()
    
    # Kiểm tra Data Collector Factory
    print("\n----- Testing Data Collector Factory -----")
    factory_success = test_data_collector_factory()
    
    # Tổng hợp kết quả
    print("\n===== TEST SUMMARY =====")
    print(f"Proxy Connection: {'✅ PASSED' if proxy_success else '❌ FAILED'}")
    print(f"Binance API: {'✅ PASSED' if api_success else '❌ FAILED'}")
    print(f"BinanceDataCollector: {'✅ PASSED' if collector_success else '❌ FAILED'}")
    print(f"Data Collector Factory: {'✅ PASSED' if factory_success else '❌ FAILED'}")
    
    overall_success = proxy_success and api_success and collector_success and factory_success
    print(f"\nOverall Test Result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    
    if not overall_success:
        print("\nSuggested Actions:")
        if not proxy_success:
            print("- Check proxy configuration and connection")
            print("- Verify proxy format: host:port:username:password")
            print("- Test proxy on command line with curl command")
        if not api_success:
            print("- Check Binance API keys")
            print("- Verify proxy is correctly forwarding to Binance domains")
        if not collector_success or not factory_success:
            print("- Check logs for specific errors")
            print("- Verify data collector implementation")
            print("- Ensure utils modules are correctly configured")