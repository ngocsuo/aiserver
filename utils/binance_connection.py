"""
Module cung cấp các phương thức kết nối trực tiếp tới Binance API
mà không cần sử dụng proxy (để khắc phục vấn đề kết nối khi chạy trên Replit)
"""
import os
import time
import json
import logging
import requests
from typing import Dict, Any, Optional, List, Union
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("binance_connection")

class BinanceDirectConnection:
    """
    Kết nối trực tiếp tới Binance API mà không thông qua thư viện python-binance
    để hạn chế các vấn đề khi chạy trên Replit
    """
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Khởi tạo đối tượng kết nối Binance
        
        Args:
            api_key (str): Binance API key
            api_secret (str): Binance API secret
        """
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        self.base_url = "https://api.binance.com"
        self.fapi_url = "https://fapi.binance.com"
        self.headers = {
            "X-MBX-APIKEY": self.api_key
        }
        self.client = None
        
        # Thử khởi tạo client chính thức từ python-binance
        try:
            if self.api_key and self.api_secret:
                self.client = Client(self.api_key, self.api_secret)
                logger.info("Khởi tạo thành công Binance client thông qua thư viện chính thức")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Binance client thông qua thư viện chính thức: {e}")
            logger.info("Sẽ sử dụng phương thức kết nối trực tiếp")
            self.client = None
    
    def ping(self) -> bool:
        """
        Kiểm tra kết nối tới Binance API
        
        Returns:
            bool: True nếu kết nối thành công, False nếu không
        """
        try:
            response = requests.get(f"{self.base_url}/api/v3/ping")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Lỗi khi ping Binance API: {e}")
            return False
    
    def get_server_time(self) -> int:
        """
        Lấy thời gian hiện tại của server Binance
        
        Returns:
            int: Thời gian server dạng timestamp (ms)
        """
        try:
            # Thử sử dụng client chính thức trước
            if self.client:
                return self.client.get_server_time()["serverTime"]
            
            # Nếu không được, sử dụng phương thức trực tiếp
            response = requests.get(f"{self.base_url}/api/v3/time")
            if response.status_code == 200:
                return response.json()["serverTime"]
            else:
                logger.error(f"Lỗi khi lấy thời gian server: {response.text}")
                return int(time.time() * 1000)
        except Exception as e:
            logger.error(f"Lỗi khi lấy thời gian server: {e}")
            return int(time.time() * 1000)
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500, 
                   start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[List]:
        """
        Lấy dữ liệu klines (nến) từ Binance
        
        Args:
            symbol (str): Ký hiệu cặp tiền (ví dụ: ETHUSDT)
            interval (str): Khoảng thời gian (ví dụ: 1m, 5m, 1h)
            limit (int): Số lượng nến tối đa cần lấy
            start_time (int, optional): Thời gian bắt đầu dạng timestamp (ms)
            end_time (int, optional): Thời gian kết thúc dạng timestamp (ms)
            
        Returns:
            List[List]: Danh sách các nến
        """
        try:
            # Thử sử dụng client chính thức trước
            if self.client:
                try:
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        startTime=start_time,
                        endTime=end_time
                    )
                    return klines
                except Exception as e:
                    logger.warning(f"Không thể sử dụng client chính thức để lấy klines: {e}")
            
            # Nếu không được, sử dụng phương thức trực tiếp
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
                
            response = requests.get(f"{self.fapi_url}/fapi/v1/klines", params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Lỗi khi lấy klines: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Lỗi khi lấy klines: {e}")
            return []
    
    def get_historical_klines(self, symbol: str, interval: str, 
                              start_str: str, end_str: Optional[str] = None, 
                              limit: int = 1000) -> List[List]:
        """
        Lấy dữ liệu klines lịch sử từ Binance dựa trên chuỗi thời gian
        
        Args:
            symbol (str): Ký hiệu cặp tiền (ví dụ: ETHUSDT)
            interval (str): Khoảng thời gian (ví dụ: 1m, 5m, 1h)
            start_str (str): Thời gian bắt đầu dạng chuỗi (ví dụ: "1 Jan, 2020")
            end_str (str, optional): Thời gian kết thúc dạng chuỗi
            limit (int): Số lượng nến tối đa cho mỗi request
            
        Returns:
            List[List]: Danh sách các nến
        """
        try:
            # Thử sử dụng client chính thức trước
            if self.client:
                try:
                    klines = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=start_str,
                        end_str=end_str,
                        limit=limit
                    )
                    return klines
                except Exception as e:
                    logger.warning(f"Không thể sử dụng client chính thức để lấy historical klines: {e}")
            
            # Nếu không được, cần chuyển đổi chuỗi thời gian thành timestamp
            import datetime
            if isinstance(start_str, str):
                if start_str.isdigit():
                    start_time = int(start_str)
                else:
                    start_time = int(datetime.datetime.strptime(start_str, "%d %b, %Y").timestamp() * 1000)
            else:
                start_time = int(start_str)
                
            if end_str:
                if isinstance(end_str, str):
                    if end_str.isdigit():
                        end_time = int(end_str)
                    else:
                        end_time = int(datetime.datetime.strptime(end_str, "%d %b, %Y").timestamp() * 1000)
                else:
                    end_time = int(end_str)
            else:
                end_time = int(time.time() * 1000)
            
            # Lấy dữ liệu theo từng phần nếu cần
            klines = []
            while start_time < end_time:
                temp_end_time = min(start_time + (limit * 60 * 1000), end_time)
                
                # Lấy dữ liệu cho phần hiện tại
                temp_klines = self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    start_time=start_time,
                    end_time=temp_end_time
                )
                
                if not temp_klines or len(temp_klines) == 0:
                    break
                
                klines.extend(temp_klines)
                
                # Cập nhật start_time cho lần lấy tiếp theo
                start_time = temp_klines[-1][0] + 1
                
                # Delay để tránh rate limit
                time.sleep(0.5)
            
            return klines
        except Exception as e:
            logger.error(f"Lỗi khi lấy historical klines: {e}")
            return []
    
    def get_funding_rate(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Lấy dữ liệu funding rate từ Binance
        
        Args:
            symbol (str): Ký hiệu cặp tiền (ví dụ: ETHUSDT)
            limit (int): Số lượng bản ghi tối đa
            
        Returns:
            List[Dict]: Danh sách các bản ghi funding rate
        """
        try:
            # Thử sử dụng client chính thức trước
            if self.client:
                try:
                    rates = self.client.futures_funding_rate(
                        symbol=symbol,
                        limit=limit
                    )
                    return rates
                except Exception as e:
                    logger.warning(f"Không thể sử dụng client chính thức để lấy funding rate: {e}")
            
            # Nếu không được, sử dụng phương thức trực tiếp
            params = {
                "symbol": symbol,
                "limit": limit
            }
                
            response = requests.get(f"{self.fapi_url}/fapi/v1/fundingRate", params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Lỗi khi lấy funding rate: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Lỗi khi lấy funding rate: {e}")
            return []
    
    def get_open_interest(self, symbol: str, period: str = "5m", limit: int = 500) -> List[Dict]:
        """
        Lấy dữ liệu open interest từ Binance
        
        Args:
            symbol (str): Ký hiệu cặp tiền (ví dụ: ETHUSDT)
            period (str): Khoảng thời gian (ví dụ: 5m, 1h, 1d)
            limit (int): Số lượng bản ghi tối đa
            
        Returns:
            List[Dict]: Danh sách các bản ghi open interest
        """
        try:
            # Thử sử dụng client chính thức trước
            if self.client:
                try:
                    data = self.client.futures_open_interest_hist(
                        symbol=symbol,
                        period=period,
                        limit=limit
                    )
                    return data
                except Exception as e:
                    logger.warning(f"Không thể sử dụng client chính thức để lấy open interest: {e}")
            
            # Nếu không được, sử dụng phương thức trực tiếp
            params = {
                "symbol": symbol,
                "period": period,
                "limit": limit
            }
                
            response = requests.get(f"{self.fapi_url}/futures/data/openInterestHist", params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Lỗi khi lấy open interest: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Lỗi khi lấy open interest: {e}")
            return []

def create_binance_connection(api_key: str = None, api_secret: str = None) -> BinanceDirectConnection:
    """
    Tạo đối tượng kết nối Binance
    
    Args:
        api_key (str, optional): Binance API key, mặc định lấy từ biến môi trường
        api_secret (str, optional): Binance API secret, mặc định lấy từ biến môi trường
        
    Returns:
        BinanceDirectConnection: Đối tượng kết nối Binance
    """
    return BinanceDirectConnection(api_key, api_secret)