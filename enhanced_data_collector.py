"""
Module thu thập dữ liệu nâng cao với proxy linh hoạt
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import threading
import json

# Import Binance client
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Import cấu hình proxy nâng cao
from enhanced_proxy_config import configure_binance_client

import config
from utils.thread_safe_logging import thread_safe_log

# Thiết lập logging
logger = logging.getLogger("enhanced_data_collector")

class EnhancedBinanceDataCollector:
    """
    Thu thập dữ liệu từ Binance API với khả năng chống block IP địa lý cao
    """
    def __init__(self):
        """
        Khởi tạo collector với hỗ trợ proxy nâng cao
        """
        self.last_update = None
        
        # Khởi tạo trạng thái kết nối
        self.connection_status = {
            "connected": False,
            "error": None,
            "message": "Initializing connection to Binance API",
            "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "using_proxy": False
        }
        
        # Lưu trữ dữ liệu đã thu thập
        self.data = {}
        for tf in [config.TIMEFRAMES["primary"]] + config.TIMEFRAMES["secondary"]:
            self.data[tf] = None
        
        # Xử lý khóa API
        if not (hasattr(config, 'BINANCE_API_KEY') and hasattr(config, 'BINANCE_API_SECRET') and 
                config.BINANCE_API_KEY and config.BINANCE_API_SECRET):
            # Thử đọc từ biến môi trường
            api_key = os.environ.get('BINANCE_API_KEY')
            api_secret = os.environ.get('BINANCE_API_SECRET')
            
            if api_key and api_secret:
                logger.info("Using Binance API keys from environment variables")
                config.BINANCE_API_KEY = api_key
                config.BINANCE_API_SECRET = api_secret
            else:
                logger.warning("Binance API keys not found")
                self.connection_status["message"] = "API keys not found"
                self.connection_status["error"] = "API keys missing"
                return
        
        # Kết nối đến Binance API với proxy
        self._connect()
        
    def _connect(self):
        """
        Kết nối đến Binance API
        """
        try:
            logger.info("Connecting to Binance API...")
            thread_safe_log("Connecting to Binance API...")
            
            # Sử dụng hàm helper để tạo kết nối
            self.client, self.connection_status = configure_binance_client(
                config.BINANCE_API_KEY, 
                config.BINANCE_API_SECRET
            )
            
            if self.client and self.connection_status["connected"]:
                logger.info("Binance data collector initialized successfully")
                thread_safe_log("Binance data collector initialized successfully")
            else:
                logger.error(f"Failed to connect to Binance API: {self.connection_status['error']}")
                thread_safe_log(f"Failed to connect to Binance API: {self.connection_status['error']}")
        except Exception as e:
            logger.error(f"Error initializing Binance data collector: {str(e)}")
            thread_safe_log(f"Error initializing Binance data collector: {str(e)}")
            self.connection_status["error"] = str(e)
            self.connection_status["message"] = "Error connecting to Binance API"
    
    def _reconnect_if_needed(self):
        """
        Kiểm tra và tái kết nối nếu cần
        """
        if not self.connection_status["connected"] or not self.client:
            logger.info("Reconnecting to Binance API...")
            self._connect()
            return self.connection_status["connected"]
        
        try:
            # Kiểm tra kết nối
            self.client.ping()
            return True
        except:
            logger.warning("Binance connection lost, reconnecting...")
            self._connect()
            return self.connection_status["connected"]
    
    def _convert_klines_to_dataframe(self, klines):
        """
        Chuyển đổi dữ liệu klines từ Binance sang DataFrame.
        
        Args:
            klines (list): Dữ liệu klines thô từ Binance
            
        Returns:
            pd.DataFrame: DataFrame với dữ liệu OHLCV
        """
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(klines, columns=columns)
        
        # Chuyển đổi kiểu dữ liệu
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Chuyển timestamp thành datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
        
    def collect_historical_data(self, symbol=config.SYMBOL, timeframe=config.PRIMARY_TIMEFRAME, 
                              limit=config.LOOKBACK_PERIODS, start_date=None, end_date=None):
        """
        Thu thập dữ liệu lịch sử OHLCV từ Binance.
        
        Args:
            symbol (str): Cặp giao dịch
            timeframe (str): Khung thời gian nến
            limit (int): Số lượng nến cần thu thập
            start_date (str, optional): Ngày bắt đầu dạng "YYYY-MM-DD"
            end_date (str, optional): Ngày kết thúc dạng "YYYY-MM-DD"
            
        Returns:
            pd.DataFrame: DataFrame với dữ liệu OHLCV
        """
        # Đảm bảo kết nối
        if not self._reconnect_if_needed():
            logger.error("Cannot collect historical data: No connection to Binance")
            return None
        
        # Giới hạn số lượng nến (max 1000 cho 1 request)
        if limit > 1000:
            logger.warning(f"Limiting request to 1000 candles instead of {limit}")
            limit = 1000
            
        try:
            logger.info(f"Collecting {limit} latest {timeframe} candles for {symbol}")
            
            if start_date and end_date:
                # Chuyển đổi chuỗi ngày thành timestamp
                start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
                
                logger.info(f"Collecting data from {start_date} to {end_date}")
                
                # Thu thập dữ liệu trong khoảng thời gian
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=timeframe,
                    startTime=start_timestamp,
                    endTime=end_timestamp,
                    limit=limit
                )
            else:
                # Thu thập dữ liệu mới nhất
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=limit
                )
                
            if klines and len(klines) > 0:
                df = self._convert_klines_to_dataframe(klines)
                logger.info(f"Collected {len(df)} {timeframe} candles for {symbol}")
                
                # Lưu dữ liệu vào bộ nhớ đệm
                self.data[timeframe] = df
                
                return df
            else:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return None
                
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            if "location restrictions" in str(e).lower():
                logger.error("Geographic restriction detected. Reconnecting with different proxy...")
                # Thử kết nối lại với proxy khác
                self._connect()
                # Thử lại request sau khi kết nối lại
                return self.collect_historical_data(symbol, timeframe, limit, start_date, end_date)
            return None
        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            return None
            
    def update_data(self, symbol=config.SYMBOL):
        """
        Cập nhật dữ liệu cho tất cả các khung thời gian đã cấu hình.
        
        Args:
            symbol (str): Cặp giao dịch
            
        Returns:
            dict: Dictionary với DataFrame đã cập nhật cho mỗi khung thời gian
        """
        # Đảm bảo kết nối
        if not self._reconnect_if_needed():
            logger.error("Cannot update data: No connection to Binance")
            return None
            
        updated_data = {}
        
        # Cập nhật cho khung thời gian chính
        primary_data = self.collect_historical_data(
            symbol=symbol,
            timeframe=config.TIMEFRAMES["primary"],
            limit=config.LOOKBACK_PERIODS
        )
        
        if primary_data is not None:
            updated_data[config.TIMEFRAMES["primary"]] = primary_data
            
        # Cập nhật cho các khung thời gian thứ cấp
        for tf in config.TIMEFRAMES["secondary"]:
            if tf in self.data:
                tf_data = self.collect_historical_data(
                    symbol=symbol,
                    timeframe=tf,
                    limit=config.LOOKBACK_PERIODS
                )
                
                if tf_data is not None:
                    updated_data[tf] = tf_data
                    
        # Cập nhật thời gian
        self.last_update = datetime.now()
        
        return updated_data

    def get_funding_rate(self, symbol=config.SYMBOL, limit=500):
        """
        Lấy dữ liệu tỷ lệ tài trợ từ Binance.
        
        Args:
            symbol (str): Cặp giao dịch
            limit (int): Số lượng bản ghi cần lấy
            
        Returns:
            pd.DataFrame: DataFrame với dữ liệu tỷ lệ tài trợ
        """
        # Đảm bảo kết nối
        if not self._reconnect_if_needed():
            logger.error("Cannot get funding rate: No connection to Binance")
            return None
            
        try:
            funding_rate = self.client.futures_funding_rate(symbol=symbol, limit=limit)
            
            if funding_rate:
                df = pd.DataFrame(funding_rate)
                df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
                df['fundingRate'] = pd.to_numeric(df['fundingRate'])
                df.set_index('fundingTime', inplace=True)
                df.sort_index(inplace=True)
                
                return df
            else:
                logger.warning(f"No funding rate data received for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting funding rate: {e}")
            return None
            
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
        # Đảm bảo kết nối
        if not self._reconnect_if_needed():
            logger.error("Cannot get open interest: No connection to Binance")
            return None
            
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
                df.sort_index(inplace=True)
                
                return df
            else:
                logger.warning(f"No open interest data received for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting open interest: {e}")
            return None

    def get_connection_status(self):
        """
        Kiểm tra trạng thái kết nối hiện tại
        
        Returns:
            dict: Trạng thái kết nối
        """
        # Cập nhật thời gian kiểm tra
        self.connection_status["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Kiểm tra kết nối nếu đang được kết nối
        if self.connection_status["connected"] and self.client:
            try:
                # Thử ping để kiểm tra kết nối
                self.client.ping()
                self.connection_status["message"] = "Connection active"
            except Exception as e:
                # Cập nhật trạng thái khi kết nối bị mất
                self.connection_status["connected"] = False
                self.connection_status["error"] = str(e)
                self.connection_status["message"] = "Connection lost"
        
        return self.connection_status
        
# Hàm helper để tạo collector
def create_enhanced_data_collector():
    """
    Tạo instance của EnhancedBinanceDataCollector
    
    Returns:
        EnhancedBinanceDataCollector: Đối tượng thu thập dữ liệu
    """
    try:
        collector = EnhancedBinanceDataCollector()
        if collector.connection_status["connected"]:
            return collector
        else:
            logger.error("Failed to create data collector with connection")
            thread_safe_log(f"Lỗi kết nối Binance API: {collector.connection_status['error']}")
            return None
    except Exception as e:
        logger.error(f"Error creating data collector: {e}")
        thread_safe_log(f"Lỗi tạo data collector: {str(e)}")
        return None