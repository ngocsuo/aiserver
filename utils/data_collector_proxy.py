"""
Module thu thập dữ liệu từ Binance API - Đã loại bỏ hoàn toàn proxy
Proxy được cấu hình ở cấp hệ thống/server thay vì trong code
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

# Import Binance client
from binance.client import Client
from binance.exceptions import BinanceAPIException

import config

# Set up logging
logger = logging.getLogger("data_collector")

class BinanceDataCollector:
    """
    Thu thập dữ liệu OHLCV từ Binance Futures API
    (Proxy được quản lý ở cấp hệ thống, không trong code)
    """
    def __init__(self):
        """Khởi tạo collector với kết nối trực tiếp"""
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
            
            # Kết nối trực tiếp đến Binance API (proxy được quản lý ở cấp hệ thống)
            logger.info("Connecting directly to Binance API without proxy")
            
            # Kết nối đến Binance API
            self.client = Client(
                config.BINANCE_API_KEY, 
                config.BINANCE_API_SECRET,
                {"timeout": 30}
            )
            
            # Test kết nối
            try:
                self.client.ping()
                logger.info("Binance API connection successful")
                
                # Kiểm tra trạng thái hệ thống
                try:
                    system_status = self.client.get_system_status()
                    if isinstance(system_status, dict) and 'status' in system_status:
                        if system_status['status'] == 0:
                            logger.info("Binance system status: Normal")
                        else:
                            logger.warning(f"Binance system status: Maintenance - {system_status.get('msg', 'Unknown message')}")
                    else:
                        logger.warning(f"Unexpected system status format: {system_status}")
                except Exception as e:
                    logger.warning(f"Could not get system status: {e}")
                
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
                
                # Báo lỗi đặc biệt cho vấn đề hạn chế địa lý
                if 'APIError(code=0)' in str(e) or 'restricted location' in str(e).lower():
                    logger.error("Geographic restriction detected. Please configure proxy at SYSTEM level")
                    self.connection_status["message"] = (
                        "Lỗi kết nối do hạn chế địa lý. Vui lòng cấu hình proxy ở cấp hệ thống/server. "
                        "Không cần chỉnh sửa code."
                    )
                
                # Các lỗi API khác
                raise
            except Exception as e:
                logger.error(f"Error initializing BinanceDataCollector: {e}")
                self.connection_status["error"] = str(e)
                self.connection_status["message"] = f"Connection error: {str(e)}"
                raise
                
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
            
            # Chuyển đổi timestamp sang datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Chuyển đổi cột số sang float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            # Chuyển đổi số giao dịch sang int
            df['number_of_trades'] = df['number_of_trades'].astype(int)
            
            # Xóa cột ignore
            df = df.drop(columns=['ignore'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting klines to DataFrame: {e}")
            raise
            
    def collect_historical_data(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAMES["primary"], 
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
        try:
            if not self.connection_status["connected"]:
                logger.warning("Attempting to collect data without established connection")
                return None
                
            logger.info(f"Collecting historical data for {symbol}, timeframe {timeframe}")
            
            # Nếu chỉ định ngày bắt đầu và kết thúc
            if start_date and end_date:
                logger.info(f"Date range: {start_date} to {end_date}")
                start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
                
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=timeframe,
                    startTime=start_ts,
                    endTime=end_ts
                )
            else:
                logger.info(f"Collecting last {limit} candles")
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=timeframe,
                    limit=limit
                )
            
            # Chuyển đổi dữ liệu klines sang DataFrame
            df = self._convert_klines_to_dataframe(klines)
            
            # Lưu trữ dữ liệu
            self.data[timeframe] = df
            
            # Cập nhật thời gian lấy dữ liệu
            self.last_update = datetime.now()
            
            logger.info(f"Successfully collected {len(df)} records")
            
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error collecting historical data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            raise
            
    def update_data(self, symbol=config.SYMBOL):
        """
        Cập nhật dữ liệu cho tất cả các khung thời gian đã cấu hình.
        
        Args:
            symbol (str): Cặp giao dịch
            
        Returns:
            dict: Dictionary với DataFrame đã cập nhật cho mỗi khung thời gian
        """
        try:
            if not self.connection_status["connected"]:
                logger.warning("Attempting to update data without established connection")
                return None
                
            updated_data = {}
            
            # Cập nhật dữ liệu cho khung thời gian chính
            primary_tf = config.TIMEFRAMES["primary"]
            updated_data[primary_tf] = self.collect_historical_data(
                symbol=symbol,
                timeframe=primary_tf,
                limit=config.LOOKBACK_PERIODS
            )
            
            # Cập nhật dữ liệu cho các khung thời gian phụ
            for secondary_tf in config.TIMEFRAMES["secondary"]:
                updated_data[secondary_tf] = self.collect_historical_data(
                    symbol=symbol,
                    timeframe=secondary_tf,
                    limit=config.LOOKBACK_PERIODS
                )
            
            # Lưu trữ dữ liệu đã cập nhật
            self.data = updated_data
            
            # Cập nhật thời gian lấy dữ liệu
            self.last_update = datetime.now()
            
            logger.info(f"Data updated for all timeframes at {self.last_update}")
            
            return updated_data
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            raise
            
    def get_funding_rate(self, symbol=config.SYMBOL, limit=500):
        """
        Lấy dữ liệu tỷ lệ tài trợ từ Binance.
        
        Args:
            symbol (str): Cặp giao dịch
            limit (int): Số lượng bản ghi cần lấy
            
        Returns:
            pd.DataFrame: DataFrame với dữ liệu tỷ lệ tài trợ
        """
        try:
            if not self.connection_status["connected"]:
                logger.warning("Attempting to get funding rate without established connection")
                return None
                
            funding_rates = self.client.futures_funding_rate(symbol=symbol, limit=limit)
            
            df = pd.DataFrame(funding_rates)
            
            # Chuyển đổi timestamp sang datetime
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            
            # Chuyển đổi fundingRate sang float
            df['fundingRate'] = df['fundingRate'].astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting funding rate: {e}")
            raise
            
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
        try:
            if not self.connection_status["connected"]:
                logger.warning("Attempting to get open interest without established connection")
                return None
                
            open_interest = self.client.futures_open_interest_hist(
                symbol=symbol,
                period=timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(open_interest)
            
            # Chuyển đổi timestamp sang datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Chuyển đổi sumOpenInterest và sumOpenInterestValue sang float
            numeric_columns = ['sumOpenInterest', 'sumOpenInterestValue']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting open interest: {e}")
            raise
            
    def get_connection_status(self):
        """
        Trả về trạng thái kết nối hiện tại.
        
        Returns:
            dict: Trạng thái kết nối
        """
        # Cập nhật thời gian kiểm tra
        self.connection_status["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Kiểm tra kết nối nếu đã được khởi tạo
        if hasattr(self, 'client'):
            try:
                self.client.ping()
                self.connection_status["connected"] = True
                self.connection_status["message"] = "Connected to Binance API"
                self.connection_status["error"] = None
            except Exception as e:
                self.connection_status["connected"] = False
                self.connection_status["message"] = f"Connection error: {str(e)}"
                self.connection_status["error"] = str(e)
        
        return self.connection_status