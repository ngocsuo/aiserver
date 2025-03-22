"""
Module thu thập dữ liệu nâng cao với proxy linh hoạt và xoay vòng
"""
import os
import time
import logging
import random
import pandas as pd
import numpy as np
import config
import requests
import json
from datetime import datetime, timedelta

# Thiết lập logging
logger = logging.getLogger("enhanced_data_collector")

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    logger.warning("Không thể import binance package")

class EnhancedBinanceDataCollector:
    """
    Thu thập dữ liệu từ Binance API với khả năng chống block IP địa lý cao
    """
    def __init__(self):
        """
        Khởi tạo collector với hỗ trợ proxy nâng cao
        """
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')
        self.client = None
        self.connection_status = {
            "connected": False,
            "last_update": None,
            "error": None,
            "proxy_used": None,
            "proxy_location": None
        }
        
        # Tối ưu hóa proxy với retry mechanism
        self.max_retries = 5
        self.retry_delay = 5  # seconds
        self.use_rotating_proxy = True
        self.current_proxy_index = 0
        
        # Danh sách user agents để tránh bị phát hiện
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 11.5; rv:90.0) Gecko/20100101 Firefox/90.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
            "Mozilla/5.0 (iPad; CPU OS 14_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/92.0.4515.90 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"
        ]
        
        # Kết nối ban đầu
        self._connect()
        
    def _connect(self):
        """
        Kết nối đến Binance API với xử lý lỗi nâng cao
        """
        import enhanced_proxy_config as proxy_config
        
        # Thử kết nối với mỗi proxy trong danh sách, với nhiều lần thử
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Kết nối lần {attempt + 1}/{self.max_retries}")
                
                # Lấy cấu hình proxy
                proxies, proxy_details = proxy_config.configure_enhanced_proxy()
                
                if proxies and proxy_details:
                    # Chọn ngẫu nhiên một user agent
                    user_agent = random.choice(self.user_agents)
                    
                    # Tùy chỉnh headers
                    headers = {
                        "User-Agent": user_agent,
                        "Accept": "application/json",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Cache-Control": "no-cache"
                    }
                    
                    # Thêm cờ API để đánh dấu đây là ứng dụng ngoài web
                    options = {"timeout": 30, "proxies": proxies, "headers": headers}
                    
                    # Tạo client với proxy
                    client = Client(
                        self.api_key, 
                        self.api_secret,
                        {"timeout": 30, "proxies": proxies, "headers": headers}
                    )
                    
                    # Test kết nối
                    client.ping()
                    logger.info("Ping Binance API thành công")
                    
                    # Kiểm tra truy cập tới Futures API
                    test_klines = client.futures_klines(
                        symbol="ETHUSDT",
                        interval="1m",
                        limit=1
                    )
                    
                    if test_klines and len(test_klines) > 0:
                        logger.info("Đã kết nối thành công tới Binance Futures API")
                        
                        # Cập nhật thông tin kết nối
                        self.client = client
                        self.connection_status["connected"] = True
                        self.connection_status["last_update"] = datetime.now()
                        self.connection_status["error"] = None
                        self.connection_status["proxy_used"] = f"{proxy_details['host']}:{proxy_details['port']}"
                        
                        return True
                    else:
                        logger.warning("Kết nối tới Binance Futures API thất bại")
                
                # Nếu kết nối thất bại, đợi rồi thử lại
                time.sleep(self.retry_delay)
                
            except BinanceAPIException as e:
                # Lỗi API Binance
                logger.error(f"Lỗi Binance API: {e}")
                self.connection_status["error"] = str(e)
                
                if "IP has been auto-banned" in str(e) or "IP restricted" in str(e):
                    logger.error("IP bị chặn bởi Binance! Thử proxy khác...")
                    time.sleep(self.retry_delay * 2)  # Đợi lâu hơn khi bị chặn
                else:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Lỗi kết nối: {e}")
                self.connection_status["error"] = str(e)
                time.sleep(self.retry_delay)
        
        # Nếu đã thử tất cả proxy mà vẫn thất bại, thử kết nối trực tiếp
        try:
            logger.info("Thử kết nối trực tiếp đến Binance API...")
            client = Client(self.api_key, self.api_secret, {"timeout": 30})
            
            # Test kết nối
            client.ping()
            
            # Kiểm tra truy cập tới Futures API
            test_klines = client.futures_klines(
                symbol="ETHUSDT",
                interval="1m",
                limit=1
            )
            
            if test_klines and len(test_klines) > 0:
                logger.info("Đã kết nối trực tiếp thành công tới Binance Futures API")
                
                # Cập nhật thông tin kết nối
                self.client = client
                self.connection_status["connected"] = True
                self.connection_status["last_update"] = datetime.now()
                self.connection_status["error"] = None
                self.connection_status["proxy_used"] = None
                
                return True
        except Exception as e:
            logger.error(f"Kết nối trực tiếp cũng thất bại: {e}")
            self.connection_status["error"] = str(e)
        
        self.connection_status["connected"] = False
        self.connection_status["error"] = "Không thể kết nối đến Binance API sau nhiều lần thử"
        logger.error("Không thể kết nối đến Binance API sau nhiều lần thử")
        return False
        
    def _reconnect_if_needed(self):
        """
        Kiểm tra và tái kết nối nếu cần
        """
        # Kiểm tra xem kết nối có hoạt động không
        if not self.connection_status["connected"] or not self.client:
            logger.info("Không có kết nối, thử kết nối lại...")
            return self._connect()
            
        # Kiểm tra xem kết nối có quá cũ không (>30 phút)
        if self.connection_status["last_update"]:
            last_update = self.connection_status["last_update"]
            if (datetime.now() - last_update).total_seconds() > 1800:  # 30 phút
                logger.info("Kết nối đã quá cũ, thử kết nối lại...")
                return self._connect()
                
        # Kiểm tra kết nối bằng cách ping
        try:
            self.client.ping()
            self.connection_status["last_update"] = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Lỗi khi ping Binance API: {e}")
            return self._connect()
    
    def _convert_klines_to_dataframe(self, klines):
        """
        Chuyển đổi dữ liệu klines từ Binance sang DataFrame.
        
        Args:
            klines (list): Dữ liệu klines thô từ Binance
            
        Returns:
            pd.DataFrame: DataFrame với dữ liệu OHLCV
        """
        if not klines:
            return pd.DataFrame()
            
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Chuyển đổi kiểu dữ liệu
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume',
                          'taker_buy_quote_asset_volume']
                          
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
            
        # Chuyển timestamp thành datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Đặt timestamp làm index
        df.set_index('timestamp', inplace=True)
        
        return df
        
    def collect_historical_data(self, symbol=config.SYMBOL, timeframe=config.PRIMARY_TIMEFRAME, 
                              limit=config.LOOKBACK_PERIODS, start_date=None, end_date=None):
        """
        Thu thập dữ liệu lịch sử OHLCV từ Binance với retry và proxy rotation.
        
        Args:
            symbol (str): Cặp giao dịch
            timeframe (str): Khung thời gian nến
            limit (int): Số lượng nến cần thu thập
            start_date (str, optional): Ngày bắt đầu dạng "YYYY-MM-DD"
            end_date (str, optional): Ngày kết thúc dạng "YYYY-MM-DD"
            
        Returns:
            pd.DataFrame: DataFrame với dữ liệu OHLCV
        """
        if not self._reconnect_if_needed():
            logger.error("Không thể kết nối đến Binance API")
            return pd.DataFrame()  # Trả về DataFrame rỗng thay vì None
        
        for attempt in range(self.max_retries):
            try:
                if start_date and end_date:
                    # Chuyển đổi ngày thành timestamp
                    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
                    
                    logger.info(f"Thu thập dữ liệu lịch sử cho {symbol} {timeframe} từ {start_date} đến {end_date}")
                    
                    # Lấy dữ liệu với khoảng thời gian
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=timeframe,
                        startTime=start_ts,
                        endTime=end_ts,
                        limit=1000  # Giới hạn API của Binance
                    )
                else:
                    logger.info(f"Thu thập {limit} nến {timeframe} gần nhất cho {symbol}")
                    
                    # Lấy dữ liệu với số lượng giới hạn
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=timeframe,
                        limit=limit
                    )
                
                logger.info(f"Đã thu thập được {len(klines)} nến {timeframe}")
                return self._convert_klines_to_dataframe(klines)
                
            except BinanceAPIException as e:
                logger.error(f"Lỗi Binance API khi thu thập dữ liệu: {e}")
                
                if "IP has been auto-banned" in str(e) or "IP restricted" in str(e):
                    logger.warning("IP bị chặn, thử kết nối lại với proxy mới...")
                    self._connect()  # Thử kết nối lại với proxy mới
                    time.sleep(self.retry_delay * 2)
                else:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Lỗi khi thu thập dữ liệu lịch sử: {e}")
                time.sleep(self.retry_delay)
        
        logger.error(f"Không thể lấy dữ liệu sau {self.max_retries} lần thử")
        return pd.DataFrame()  # Trả về DataFrame rỗng thay vì None
            
    def update_data(self, symbol=config.SYMBOL):
        """
        Cập nhật dữ liệu cho tất cả các khung thời gian đã cấu hình.
        
        Args:
            symbol (str): Cặp giao dịch
            
        Returns:
            dict: Dictionary với DataFrame đã cập nhật cho mỗi khung thời gian
        """
        if not self._reconnect_if_needed():
            logger.error("Không thể kết nối đến Binance API")
            return {}
            
        data = {}
        
        # Cập nhật dữ liệu cho từng khung thời gian đã cấu hình
        for timeframe in config.TIMEFRAMES:
            try:
                df = self.collect_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=config.LOOKBACK_PERIODS
                )
                
                if not df.empty:
                    data[timeframe] = df
                    logger.info(f"Đã cập nhật dữ liệu cho {symbol} {timeframe}: {len(df)} nến")
                else:
                    logger.warning(f"Không thể cập nhật dữ liệu cho {symbol} {timeframe}")
                    
            except Exception as e:
                logger.error(f"Lỗi khi cập nhật dữ liệu {timeframe}: {e}")
        
        return data
    
    def get_funding_rate(self, symbol=config.SYMBOL, limit=500):
        """
        Lấy dữ liệu tỷ lệ tài trợ từ Binance.
        
        Args:
            symbol (str): Cặp giao dịch
            limit (int): Số lượng bản ghi cần lấy
            
        Returns:
            pd.DataFrame: DataFrame với dữ liệu tỷ lệ tài trợ
        """
        if not self._reconnect_if_needed():
            logger.error("Không thể kết nối đến Binance API")
            return pd.DataFrame()
            
        for attempt in range(self.max_retries):
            try:
                funding_rates = self.client.futures_funding_rate(symbol=symbol, limit=limit)
                
                if funding_rates:
                    df = pd.DataFrame(funding_rates)
                    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
                    df['fundingRate'] = pd.to_numeric(df['fundingRate'])
                    df.set_index('fundingTime', inplace=True)
                    
                    logger.info(f"Đã lấy {len(df)} bản ghi tỷ lệ tài trợ cho {symbol}")
                    return df
                    
                logger.warning(f"Không có dữ liệu tỷ lệ tài trợ cho {symbol}")
                return pd.DataFrame()
                
            except Exception as e:
                logger.error(f"Lỗi khi lấy dữ liệu tỷ lệ tài trợ: {e}")
                time.sleep(self.retry_delay)
                
        logger.error(f"Không thể lấy dữ liệu tỷ lệ tài trợ sau {self.max_retries} lần thử")
        return pd.DataFrame()
    
    def get_open_interest(self, symbol=config.SYMBOL, timeframe="5m", limit=500):
        """
        Lấy dữ liệu open interest từ Binance.
        
        Args:
            symbol (str): Cặp giao dịch
            timeframe (str): Khung thời gian
            limit (int): Số lượng bản ghi cần lấy
            
        Returns:
            pd.DataFrame: DataFrame với dữ liệu open interest
        """
        if not self._reconnect_if_needed():
            logger.error("Không thể kết nối đến Binance API")
            return pd.DataFrame()
            
        for attempt in range(self.max_retries):
            try:
                open_interest = self.client.futures_open_interest_hist(
                    symbol=symbol,
                    period=timeframe,
                    limit=limit
                )
                
                if open_interest:
                    df = pd.DataFrame(open_interest)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'])
                    df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'])
                    df.set_index('timestamp', inplace=True)
                    
                    logger.info(f"Đã lấy {len(df)} bản ghi open interest cho {symbol}")
                    return df
                    
                logger.warning(f"Không có dữ liệu open interest cho {symbol}")
                return pd.DataFrame()
                
            except Exception as e:
                logger.error(f"Lỗi khi lấy dữ liệu open interest: {e}")
                time.sleep(self.retry_delay)
                
        logger.error(f"Không thể lấy dữ liệu open interest sau {self.max_retries} lần thử")
        return pd.DataFrame()
        
    def get_connection_status(self):
        """
        Kiểm tra trạng thái kết nối hiện tại
        
        Returns:
            dict: Trạng thái kết nối
        """
        # Cập nhật trạng thái kết nối hiện tại
        if self.client:
            try:
                # Ping server
                self.client.ping()
                
                # Cập nhật trạng thái
                self.connection_status["connected"] = True
                self.connection_status["last_update"] = datetime.now()
                self.connection_status["error"] = None
            except Exception as e:
                self.connection_status["connected"] = False
                self.connection_status["error"] = str(e)
        
        return self.connection_status
        
def create_enhanced_data_collector():
    """
    Tạo instance của EnhancedBinanceDataCollector
    
    Returns:
        EnhancedBinanceDataCollector: Đối tượng thu thập dữ liệu
    """
    return EnhancedBinanceDataCollector()
    
if __name__ == "__main__":
    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Kiểm tra kết nối
    collector = create_enhanced_data_collector()
    status = collector.get_connection_status()
    print(f"Trạng thái kết nối: {status}")
    
    # Thử lấy dữ liệu
    if status["connected"]:
        # Lấy dữ liệu lịch sử
        df = collector.collect_historical_data(limit=10)
        print(f"Dữ liệu lịch sử: {len(df)} nến")
        if not df.empty:
            print(df.head())