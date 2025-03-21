"""
Data collection module for fetching ETHUSDT data from Binance Futures.
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import random

# Import Binance client for real API access
from binance.client import Client
from binance.exceptions import BinanceAPIException

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_collector")

class MockDataCollector:
    """
    A mock data collector that generates simulated OHLCV data for testing 
    and development without requiring API keys.
    """
    def __init__(self):
        """Initialize the mock data collector."""
        # Track last update time
        self.last_update = None
        
        # Store the collected data
        self.data = {}
        for tf in [config.TIMEFRAMES["primary"]] + config.TIMEFRAMES["secondary"]:
            self.data[tf] = None
            
        # Set the base price for ETH
        self.base_price = 3500.0
        self.current_price = self.base_price
        self.price_trend = 0  # -1: downtrend, 0: sideways, 1: uptrend
        self.trend_strength = 0.3  # How strong the trend is
        self.trend_duration = 100  # How many candles the trend lasts
        self.candle_count = 0
        
        logger.info("Mock data collector initialized")
        
    def generate_candle(self, timeframe="5m", last_candle=None):
        """
        Generate a simulated OHLCV candle with realistic price action.
        
        Args:
            timeframe (str): Candle timeframe
            last_candle (dict): Previous candle data
            
        Returns:
            dict: Generated candle data
        """
        # Define minutes map for timeframes (moved out from nested block)
        minutes_map = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "1d": 1440
        }
        minutes = minutes_map.get(timeframe, 5)
        
        # Set timestamp based on timeframe
        if last_candle is None:
            # Start time for first candle
            timestamp = datetime.now() - timedelta(days=5)  # Start 5 days ago
        else:
            # Calculate next timestamp based on timeframe
            timestamp = last_candle["open_time"] + timedelta(minutes=minutes)
        
        # Check if we need to change the trend
        self.candle_count += 1
        if self.candle_count % self.trend_duration == 0:
            # Change trend randomly
            self.price_trend = random.choice([-1, 0, 1])
            self.trend_strength = random.uniform(0.2, 0.8)
            self.trend_duration = random.randint(50, 200)
            logger.info(f"Changed price trend to {self.price_trend} with strength {self.trend_strength}")
        
        # Generate price based on trend and randomness
        price_change_pct = 0
        
        # Add trend component
        if self.price_trend == 1:  # Uptrend
            price_change_pct += random.uniform(0, self.trend_strength) / 100
        elif self.price_trend == -1:  # Downtrend
            price_change_pct -= random.uniform(0, self.trend_strength) / 100
            
        # Add random component (market noise)
        price_change_pct += random.uniform(-0.5, 0.5) / 100
        
        # Update current price
        if last_candle is None:
            self.current_price = self.base_price
        else:
            self.current_price = last_candle["close"] * (1 + price_change_pct)
        
        # Generate open, high, low, close
        open_price = self.current_price
        
        # Random price action within the candle
        range_pct = random.uniform(0.1, 0.5) / 100  # 0.1% to 0.5% range
        price_range = open_price * range_pct
        
        # Determine if candle is bullish or bearish
        is_bullish = random.random() > 0.5
        
        if is_bullish:
            close_price = open_price * (1 + random.uniform(0, range_pct))
            high_price = close_price * (1 + random.uniform(0, range_pct/2))
            low_price = open_price * (1 - random.uniform(0, range_pct/2))
        else:
            close_price = open_price * (1 - random.uniform(0, range_pct))
            high_price = open_price * (1 + random.uniform(0, range_pct/2))
            low_price = close_price * (1 - random.uniform(0, range_pct/2))
        
        # Generate volume based on price volatility
        base_volume = random.uniform(1000, 5000)
        volume_multiplier = 1 + abs(price_change_pct) * 100  # More volume on volatile candles
        volume = base_volume * volume_multiplier
        
        # Create candle data
        candle = {
            "open_time": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "close_time": timestamp + timedelta(minutes=minutes - 1, seconds=59),
            "quote_asset_volume": volume * close_price,
            "number_of_trades": int(volume / 10),
            "taker_buy_base_asset_volume": volume * random.uniform(0.4, 0.6),
            "taker_buy_quote_asset_volume": volume * close_price * random.uniform(0.4, 0.6),
            "ignore": 0
        }
        
        return candle
        
    def generate_historical_data(self, timeframe="5m", num_candles=config.LOOKBACK_PERIODS):
        """
        Generate historical OHLCV data.
        
        Args:
            timeframe (str): Candle timeframe
            num_candles (int): Number of candles to generate
            
        Returns:
            pd.DataFrame: DataFrame with generated OHLCV data
        """
        candles = []
        last_candle = None
        
        for _ in range(num_candles):
            candle = self.generate_candle(timeframe, last_candle)
            candles.append(list(candle.values()))
            last_candle = candle
        
        # Convert to DataFrame
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(candles, columns=columns)
        
        # Set index to open_time
        df['open_time'] = pd.to_datetime(df['open_time'])
        df['close_time'] = pd.to_datetime(df['close_time'])
        df.set_index('open_time', inplace=True)
        
        return df
        
    def collect_historical_data(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAMES["primary"], 
                              limit=config.LOOKBACK_PERIODS):
        """
        Generate historical OHLCV data.
        
        Args:
            symbol (str): Trading pair symbol (ignored in mock)
            timeframe (str): Candle timeframe
            limit (int): Number of candles to generate
            
        Returns:
            pd.DataFrame: DataFrame with simulated OHLCV data
        """
        logger.info(f"Generating {limit} {timeframe} mock candles for {symbol}")
        return self.generate_historical_data(timeframe, limit)
        
    def update_data(self, symbol=config.SYMBOL):
        """
        Update data for all configured timeframes.
        
        Args:
            symbol (str): Trading pair symbol (ignored in mock)
            
        Returns:
            dict: Dictionary with updated dataframes for each timeframe
        """
        now = datetime.now()
        
        # Check if we need to update (based on configured interval)
        if self.last_update and (now - self.last_update).total_seconds() < config.UPDATE_INTERVAL:
            # Not time to update yet
            return self.data
        
        try:
            # Update data for each timeframe
            for tf in [config.TIMEFRAMES["primary"]] + config.TIMEFRAMES["secondary"]:
                # If we don't have data yet, collect historical data
                if self.data[tf] is None:
                    self.data[tf] = self.collect_historical_data(symbol=symbol, timeframe=tf)
                else:
                    # We have existing data, just update with recent candles
                    # Generate 1 new candle
                    last_candle_time = self.data[tf].index.max()
                    last_candle = {
                        "open_time": last_candle_time,
                        "close": self.data[tf].iloc[-1]["close"],
                        "open": self.data[tf].iloc[-1]["open"],
                        "high": self.data[tf].iloc[-1]["high"],
                        "low": self.data[tf].iloc[-1]["low"],
                    }
                    new_candle = self.generate_candle(tf, last_candle)
                    
                    # Convert to DataFrame
                    columns = [
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ]
                    new_df = pd.DataFrame([list(new_candle.values())], columns=columns)
                    new_df['open_time'] = pd.to_datetime(new_df['open_time'])
                    new_df['close_time'] = pd.to_datetime(new_df['close_time'])
                    new_df.set_index('open_time', inplace=True)
                    
                    # Append the new data
                    self.data[tf] = pd.concat([self.data[tf], new_df])
                    
                    # Remove duplicates if any
                    self.data[tf] = self.data[tf][~self.data[tf].index.duplicated(keep='last')]
                    
                    # Sort by time
                    self.data[tf].sort_index(inplace=True)
                    
            # Update last update time
            self.last_update = now
            logger.info(f"Mock data updated for all timeframes at {now}")
            
            # Return the updated data
            return self.data
            
        except Exception as e:
            logger.error(f"Error updating mock data: {e}")
            return self.data
            
    def get_funding_rate(self, symbol=config.SYMBOL, limit=500):
        """
        Generate mock funding rate data.
        
        Args:
            symbol (str): Trading pair symbol (ignored in mock)
            limit (int): Number of funding rate records to generate
            
        Returns:
            pd.DataFrame: DataFrame with mock funding rate data
        """
        funding_rates = []
        start_time = datetime.now() - timedelta(hours=limit*8)  # Funding rate every 8 hours
        
        for i in range(limit):
            # Generate random funding rate, slightly biased towards positive
            funding_rate = random.normalvariate(0.0001, 0.001)  # Mean slightly positive
            
            funding_time = start_time + timedelta(hours=i*8)
            
            funding_rates.append({
                'fundingTime': funding_time,
                'fundingRate': funding_rate,
                'symbol': symbol
            })
            
        df = pd.DataFrame(funding_rates)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'])
        df.set_index('fundingTime', inplace=True)
        
        return df
        
    def get_open_interest(self, symbol=config.SYMBOL, timeframe="5m", limit=500):
        """
        Generate mock open interest data.
        
        Args:
            symbol (str): Trading pair symbol (ignored in mock)
            timeframe (str): Data timeframe
            limit (int): Number of records to generate
            
        Returns:
            pd.DataFrame: DataFrame with mock open interest data
        """
        open_interest_data = []
        
        # Map timeframe to minutes
        minutes_map = {
            "5m": 5, "15m": 15, "30m": 30, "1h": 60,
            "2h": 120, "4h": 240, "1d": 1440
        }
        minutes = minutes_map.get(timeframe, 5)
        
        start_time = datetime.now() - timedelta(minutes=limit*minutes)
        
        # Base open interest value
        base_oi = 50000000  # 50M USDT
        
        for i in range(limit):
            # Generate open interest that correlates with price trends
            oi_change = 0
            
            # Add trend component
            if self.price_trend == 1:  # Uptrend
                oi_change += random.uniform(0, self.trend_strength * 2000000)
            elif self.price_trend == -1:  # Downtrend
                oi_change -= random.uniform(0, self.trend_strength * 2000000)
                
            # Add random component (market noise)
            oi_change += random.uniform(-1000000, 1000000)
            
            current_oi = base_oi + oi_change
            
            timestamp = start_time + timedelta(minutes=i*minutes)
            
            open_interest_data.append({
                'timestamp': timestamp,
                'sumOpenInterest': current_oi / self.current_price,  # In ETH
                'sumOpenInterestValue': current_oi,  # In USDT
                'symbol': symbol
            })
            
        df = pd.DataFrame(open_interest_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df

class BinanceDataCollector:
    """
    A data collector that fetches real OHLCV data from Binance Futures API.
    """
    def __init__(self):
        """Initialize the Binance data collector."""
        # Track last update time
        self.last_update = None
        self.client = None
        self.connection_status = {
            "connected": False,
            "error": None,
            "message": "Not initialized",
            "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store the collected data
        self.data = {}
        for tf in [config.TIMEFRAMES["primary"]] + config.TIMEFRAMES["secondary"]:
            self.data[tf] = None
            
        try:
            # Check if API keys are available
            if not config.BINANCE_API_KEY or not config.BINANCE_API_SECRET:
                logger.warning("Binance API keys not found in config")
                self.connection_status["message"] = "API keys not found in configuration"
                return
                
            # Initialize Binance client with API keys
            self.client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
            
            # Test connection with timeout
            import socket
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(5)  # 5 second timeout
            
            # Test connection
            try:
                self.client.ping()
                logger.info("Binance API connection successful")
                
                # Check system status
                system_status = self.client.get_system_status()
                if system_status['status'] == 0:
                    logger.info("Binance system status: Normal")
                else:
                    logger.warning(f"Binance system status: Maintenance - {system_status['msg']}")
                
                # Check if futures API is accessible
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
                if 'APIError(code=-1130)' in str(e):
                    # Invalid API endpoint, likely a geographic restriction
                    logger.error("Binance API error: Geographic restrictions detected")
                    self.connection_status["error"] = "Geographic restriction"
                    self.connection_status["message"] = "Binance access restricted in your region. Consider using VPN."
                elif 'APIError(code=-2015)' in str(e):
                    # Invalid API key
                    logger.error("Binance API error: Invalid API key")
                    self.connection_status["error"] = "Invalid API key"
                    self.connection_status["message"] = "Please check your API key and secret"
                else:
                    logger.error(f"Binance API error: {e}")
                    self.connection_status["error"] = str(e)
                    self.connection_status["message"] = f"API Error: {e}"
            except socket.timeout:
                logger.error("Binance API connection timeout")
                self.connection_status["error"] = "Connection timeout"
                self.connection_status["message"] = "Connection to Binance API timed out, network issues detected"
            finally:
                socket.setdefaulttimeout(original_timeout)
                
        except Exception as e:
            logger.error(f"Error initializing BinanceDataCollector: {e}")
            self.connection_status["error"] = str(e)
            self.connection_status["message"] = f"Initialization error: {e}"
            
    def _convert_klines_to_dataframe(self, klines):
        """
        Convert kline data from Binance to DataFrame.
        
        Args:
            klines (list): Raw kline data from Binance
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=columns)
        
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                        'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
                        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
            
        # Set index to open_time
        df.set_index('open_time', inplace=True)
        
        return df
            
    def collect_historical_data(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAMES["primary"], 
                              limit=config.LOOKBACK_PERIODS):
        """
        Collect historical OHLCV data from Binance.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Candle timeframe
            limit (int): Number of candles to collect
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {limit} {timeframe} candles for {symbol} from Binance")
            
            # Get klines from Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = self._convert_klines_to_dataframe(klines)
            
            logger.info(f"Fetched {len(df)} {timeframe} candles for {symbol}")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching historical data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
            
    def update_data(self, symbol=config.SYMBOL):
        """
        Update data for all configured timeframes.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            dict: Dictionary with updated dataframes for each timeframe
        """
        now = datetime.now()
        
        # Check if we need to update (based on configured interval)
        if self.last_update and (now - self.last_update).total_seconds() < config.UPDATE_INTERVAL:
            # Not time to update yet
            return self.data
        
        try:
            # Update data for each timeframe
            for tf in [config.TIMEFRAMES["primary"]] + config.TIMEFRAMES["secondary"]:
                # If we don't have data yet, collect historical data
                if self.data[tf] is None:
                    self.data[tf] = self.collect_historical_data(symbol=symbol, timeframe=tf)
                else:
                    # We have existing data, just update with recent candles
                    # Determine how many candles to fetch based on the last timestamp
                    last_timestamp = self.data[tf].index.max()
                    # Fetch a reasonable number of recent candles to ensure no gaps
                    recent_klines = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=tf,
                        start_str=int((last_timestamp - timedelta(hours=1)).timestamp() * 1000),
                        limit=100
                    )
                    
                    # Convert to DataFrame
                    if recent_klines:
                        recent_df = self._convert_klines_to_dataframe(recent_klines)
                        
                        # Filter to new data only
                        recent_df = recent_df[recent_df.index > last_timestamp]
                        
                        if not recent_df.empty:
                            # Append the new data
                            self.data[tf] = pd.concat([self.data[tf], recent_df])
                            
                            # Remove duplicates if any
                            self.data[tf] = self.data[tf][~self.data[tf].index.duplicated(keep='last')]
                            
                            # Sort by time
                            self.data[tf].sort_index(inplace=True)
                    
            # Update last update time
            self.last_update = now
            logger.info(f"Binance data updated for all timeframes at {now}")
            
            # Return the updated data
            return self.data
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error updating data: {e}")
            return self.data
        except Exception as e:
            logger.error(f"Error updating Binance data: {e}")
            return self.data
            
    def get_funding_rate(self, symbol=config.SYMBOL, limit=500):
        """
        Get funding rate data from Binance.
        
        Args:
            symbol (str): Trading pair symbol
            limit (int): Number of funding rate records to retrieve
            
        Returns:
            pd.DataFrame: DataFrame with funding rate data
        """
        try:
            logger.info(f"Fetching funding rate data for {symbol}")
            
            # Fetch funding rate history
            funding_rates = self.client.futures_funding_rate(symbol=symbol, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(funding_rates)
            
            # Convert timestamps and numeric values
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = pd.to_numeric(df['fundingRate'])
            
            df.set_index('fundingTime', inplace=True)
            
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching funding rates: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            raise
            
    def get_open_interest(self, symbol=config.SYMBOL, timeframe="5m", limit=500):
        """
        Get open interest data from Binance.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            limit (int): Number of records to retrieve
            
        Returns:
            pd.DataFrame: DataFrame with open interest data
        """
        try:
            logger.info(f"Fetching open interest data for {symbol}")
            
            # Fetch open interest history
            open_interest = self.client.futures_open_interest_hist(
                symbol=symbol,
                period=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(open_interest)
            
            # Convert timestamps and numeric values
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'])
            df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'])
            
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching open interest: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching open interest: {e}")
            raise

# This function creates the appropriate data collector based on config settings
def create_data_collector():
    """
    Factory function to create the appropriate data collector.
    Check config.USE_REAL_API and availability of API keys.
    Also checks for geographic restrictions.
    
    Returns:
        Either BinanceDataCollector or MockDataCollector instance
    """
    if config.FORCE_MOCK_DATA:
        logger.warning("FORCE_MOCK_DATA is enabled, using mock data collector")
        return MockDataCollector()
        
    if config.USE_REAL_API and config.BINANCE_API_KEY and config.BINANCE_API_SECRET:
        logger.info("Attempting to use Binance API data collector")
        try:
            # Try to create a real data collector
            collector = BinanceDataCollector()
            
            # Check if connection was successful
            if collector.connection_status["connected"]:
                logger.info("Successfully connected to Binance API")
                return collector
            else:
                # If there's a connection error, log it and fall back to mock
                error_msg = collector.connection_status["message"]
                logger.warning(f"Could not connect to Binance API: {error_msg}")
                
                if "Geographic restriction" in collector.connection_status.get("error", ""):
                    logger.error("Geographic restriction detected. Consider using VPN.")
                
                # Fall back to mock collector
                logger.warning("Falling back to mock data collector due to API connection issues")
                return MockDataCollector()
        except Exception as e:
            logger.error(f"Error initializing Binance API collector: {e}")
            logger.warning("Falling back to mock data collector")
            return MockDataCollector()
    else:
        # Either USE_REAL_API is False or API keys are not available
        if not config.USE_REAL_API:
            logger.info("Using mock data collector (USE_REAL_API is False)")
        else:
            logger.warning("API keys not available, using mock data collector")
        
        return MockDataCollector()