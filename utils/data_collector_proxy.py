"""
Module thu thập dữ liệu tích hợp proxy cho Binance API
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import random
import os
import requests
import json
import socket

# Import Binance client
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Import proxy config
from utils.proxy_config import configure_proxy, configure_socket_proxy, get_proxy_url_format

import config

# Set up logging
logger = logging.getLogger("data_collector")

class BinanceDataCollector:
    """
    Thu thập dữ liệu OHLCV từ Binance Futures API với hỗ trợ proxy
    """
    def __init__(self):
        """Khởi tạo collector với cấu hình proxy"""
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
        
        try:
            # Kiểm tra API keys
            if not config.BINANCE_API_KEY or not config.BINANCE_API_SECRET:
                logger.warning("Binance API keys not found in config")
                self.connection_status["message"] = "API keys not found in configuration"
                return
            
            # Cấu hình proxy từ utils/proxy_config.py
            proxies = configure_proxy()
            proxy_url = get_proxy_url_format()
            
            if proxies and proxy_url:
                logger.info(f"Connecting to Binance API using proxy")
                self.connection_status["using_proxy"] = True
                
                # Cấu hình proxy cho Socket nếu cần
                configure_socket_proxy()
                
                # Cấu hình client với proxy
                self.client = Client(
                    config.BINANCE_API_KEY, 
                    config.BINANCE_API_SECRET,
                    {"timeout": 30, "proxies": proxies}
                )
            else:
                logger.info("Connecting directly to Binance API without proxy")
                # Kết nối trực tiếp nếu không có proxy
                self.client = Client(
                    config.BINANCE_API_KEY, 
                    config.BINANCE_API_SECRET,
                    {"timeout": 30}
                )
            
            # Test kết nối với timeout tăng cường
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(20)  # 20 giây timeout
            
            try:
                self.client.ping()
                logger.info("Binance API connection successful")
                
                # Kiểm tra trạng thái hệ thống
                system_status = self.client.get_system_status()
                if system_status['status'] == 0:
                    logger.info("Binance system status: Normal")
                else:
                    logger.warning(f"Binance system status: Maintenance - {system_status['msg']}")
                
                # Kiểm tra xem API Futures có thể truy cập không
                test_klines = self.client.futures_klines(
                    symbol=config.SYMBOL,
                    interval=config.TIMEFRAMES["primary"],
                    limit=1
                )
                
                if test_klines:
                    logger.info("Binance Futures API accessible")
                    self.connection_status["connected"] = True
                    self.connection_status["message"] = "Connected to Binance Futures API"
                
                logger.info("Binance data collector initialized successfully")
                
            except BinanceAPIException as e:
                logger.error(f"Error initializing BinanceDataCollector: {e}")
                self.connection_status["error"] = str(e)
                self.connection_status["message"] = f"API Error: {str(e)}"
                
                if 'APIError(code=0)' in str(e) or 'restricted location' in str(e).lower():
                    # Lỗi hạn chế địa lý
                    logger.error("Geographic restriction detected. This will work when deployed on your server.")
                    logger.error(f"Error message: {e}")
                    logger.error("Lỗi khi khởi tạo Binance API collector: Hạn chế địa lý phát hiện. Hệ thống sẽ hoạt động bình thường khi triển khai trên server riêng của bạn.")
                    raise
            except Exception as e:
                logger.error(f"Error initializing BinanceDataCollector: {e}")
                self.connection_status["error"] = str(e)
                self.connection_status["message"] = f"Connection error: {str(e)}"
                raise
            finally:
                # Khôi phục timeout mặc định
                socket.setdefaulttimeout(original_timeout)
                
        except Exception as e:
            logger.error(f"Error initializing BinanceDataCollector: {e}")
            self.connection_status["error"] = str(e)
            self.connection_status["message"] = f"Initialization error: {str(e)}"
            self.connection_status["connected"] = False
            raise
    
    def _convert_klines_to_dataframe(self, klines):
        """
        Chuyển đổi dữ liệu klines từ Binance sang DataFrame.
        
        Args:
            klines (list): Dữ liệu klines thô từ Binance
            
        Returns:
            pd.DataFrame: DataFrame với dữ liệu OHLCV
        """
        try:
            columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            df = pd.DataFrame(klines, columns=columns)
            
            # Chuyển đổi kiểu dữ liệu
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume', 
                        'quote_asset_volume', 'taker_buy_base_asset_volume', 
                        'taker_buy_quote_asset_volume']:
                df[col] = df[col].astype(float)
                
            # Đặt index
            df.set_index('open_time', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting klines to DataFrame: {e}")
            return pd.DataFrame()
    
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
        if not self.connection_status["connected"]:
            logger.warning(f"Cannot collect data: {self.connection_status['message']}")
            return None
        
        try:
            # Nếu có ngày bắt đầu và kết thúc, ưu tiên dùng nó
            if start_date and end_date:
                logger.info(f"Collecting historical data for {symbol} {timeframe} from {start_date} to {end_date}")
                
                # Chuyển đổi định dạng ngày thành timestamp
                start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
                
                # Lấy dữ liệu
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=timeframe,
                    startTime=start_timestamp,
                    endTime=end_timestamp
                )
            else:
                # Nếu không có ngày cụ thể, dùng limit
                logger.info(f"Collecting {limit} latest {timeframe} candles for {symbol}")
                
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=limit
                )
            
            # Chuyển đổi thành DataFrame
            if klines:
                df = self._convert_klines_to_dataframe(klines)
                logger.info(f"Collected {len(df)} {timeframe} candles for {symbol}")
                return df
            else:
                logger.warning(f"No data returned from Binance for {symbol} {timeframe}")
                return None
                
        except BinanceAPIException as e:
            logger.error(f"Binance API error collecting historical data: {e}")
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
        if not self.connection_status["connected"]:
            logger.warning(f"Cannot update data: {self.connection_status['message']}")
            return self.data
            
        now = datetime.now()
        
        # Kiểm tra nếu cần cập nhật dựa trên chu kỳ đã cấu hình
        if self.last_update and (now - self.last_update).total_seconds() < config.UPDATE_INTERVAL:
            # Chưa đến lúc cập nhật
            return self.data
            
        try:
            # Cập nhật dữ liệu cho mỗi khung thời gian
            for tf in [config.TIMEFRAMES["primary"]] + config.TIMEFRAMES["secondary"]:
                # Nếu chưa có dữ liệu, thu thập dữ liệu lịch sử
                if self.data[tf] is None:
                    logger.info(f"Initial data collection for {symbol} {tf}")
                    
                    # Nếu có ngày bắt đầu lịch sử, lấy dữ liệu từ đó đến hiện tại
                    if hasattr(config, 'HISTORICAL_START_DATE') and config.HISTORICAL_START_DATE:
                        start_date = config.HISTORICAL_START_DATE
                        end_date = now.strftime("%Y-%m-%d")
                        self.data[tf] = self.collect_historical_data(
                            symbol=symbol, 
                            timeframe=tf,
                            start_date=start_date,
                            end_date=end_date
                        )
                    else:
                        # Nếu không, lấy số lượng nến được cấu hình
                        self.data[tf] = self.collect_historical_data(
                            symbol=symbol, 
                            timeframe=tf,
                            limit=config.LOOKBACK_PERIODS
                        )
                else:
                    # Đã có dữ liệu, chỉ cập nhật nến mới nhất
                    last_candle_time = self.data[tf].index.max()
                    
                    # Tính khoảng thời gian giữa nến cuối và hiện tại
                    time_diff = now - last_candle_time
                    
                    # Chỉ cập nhật nếu đã qua một khoảng thời gian đủ lớn
                    # Lấy 5 nến gần đây nhất để đảm bảo không bỏ sót dữ liệu
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=tf,
                        limit=5
                    )
                    
                    if klines:
                        new_df = self._convert_klines_to_dataframe(klines)
                        
                        # Ghép với dữ liệu hiện có, xóa bỏ bản sao
                        self.data[tf] = pd.concat([self.data[tf], new_df])
                        self.data[tf] = self.data[tf][~self.data[tf].index.duplicated(keep='last')]
                        
                        # Sắp xếp theo thời gian
                        self.data[tf].sort_index(inplace=True)
                        
                        logger.info(f"Updated {symbol} {tf} data, now have {len(self.data[tf])} candles")
                    else:
                        logger.warning(f"No new data available for {symbol} {tf}")
            
            # Cập nhật thời gian cập nhật cuối cùng
            self.last_update = now
            
            return self.data
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error updating data: {e}")
            return self.data
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return self.data
    
    def get_funding_rate(self, symbol=config.SYMBOL, limit=500):
        """
        Lấy dữ liệu tỷ lệ tài trợ từ Binance.
        
        Args:
            symbol (str): Cặp giao dịch
            limit (int): Số lượng bản ghi cần lấy
            
        Returns:
            pd.DataFrame: DataFrame với dữ liệu tỷ lệ tài trợ
        """
        if not self.connection_status["connected"]:
            logger.warning(f"Cannot get funding rate: {self.connection_status['message']}")
            return None
            
        try:
            # Lấy dữ liệu tỷ lệ tài trợ
            funding_rates = self.client.futures_funding_rate(symbol=symbol, limit=limit)
            
            if not funding_rates:
                logger.warning(f"No funding rate data available for {symbol}")
                return None
                
            # Chuyển đổi thành DataFrame
            df = pd.DataFrame(funding_rates)
            
            # Chuyển đổi kiểu dữ liệu
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float)
            
            # Đặt index
            df.set_index('fundingTime', inplace=True)
            
            # Sắp xếp theo thời gian
            df.sort_index(inplace=True)
            
            logger.info(f"Collected {len(df)} funding rate records for {symbol}")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting funding rate: {e}")
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
        if not self.connection_status["connected"]:
            logger.warning(f"Cannot get open interest: {self.connection_status['message']}")
            return None
            
        try:
            # Lấy dữ liệu open interest
            open_interest = self.client.futures_open_interest_hist(
                symbol=symbol, 
                period=timeframe, 
                limit=limit
            )
            
            if not open_interest:
                logger.warning(f"No open interest data available for {symbol} {timeframe}")
                return None
                
            # Chuyển đổi thành DataFrame
            df = pd.DataFrame(open_interest)
            
            # Chuyển đổi kiểu dữ liệu
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
            df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
            
            # Đặt index
            df.set_index('timestamp', inplace=True)
            
            # Sắp xếp theo thời gian
            df.sort_index(inplace=True)
            
            logger.info(f"Collected {len(df)} open interest records for {symbol} {timeframe}")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting open interest: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting open interest: {e}")
            return None