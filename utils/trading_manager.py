"""
Module quản lý giao dịch với Binance Futures API.
"""
import os
import time
import threading
import logging
import math
from datetime import datetime, timedelta, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pytz
import config  # Thêm import config để sử dụng cấu hình proxy

# Import các hàm hỗ trợ từ module trading_manager_functions
from utils.trading_manager_functions import get_current_date_tz7, get_daily_pnl_summary

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('trading_manager')

class TradingManager:
    """
    Quản lý giao dịch tự động với Binance Futures.
    """
    def __init__(self, api_key=None, api_secret=None):
        """
        Khởi tạo trading manager.
        
        Args:
            api_key (str): Binance API key
            api_secret (str): Binance API secret
        """
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        self.client = None
        self.trading_thread = None
        self.running = False
        self.is_position_open = False
        self.position_info = None
        self.trading_logs = []
        self.status = "Khởi tạo"
        
        # Khởi tạo thống kê PNL theo ngày (múi giờ +7)
        self.daily_pnl = {
            'date': get_current_date_tz7(),
            'trades': [],
            'total_pnl': 0.0,
            'win_count': 0,
            'loss_count': 0
        }
    
    def add_log(self, message, level="info"):
        """
        Thêm thông báo vào nhật ký giao dịch.
        
        Args:
            message (str): Thông báo cần ghi lại
            level (str): Mức độ thông báo ("info", "warning", "error")
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": timestamp, "message": message, "level": level}
        self.trading_logs.append(log_entry)
        
        # Giới hạn số lượng logs giữ lại
        if len(self.trading_logs) > 100:
            self.trading_logs = self.trading_logs[-100:]
        
        # Log vào console
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
    
    def connect(self, api_key=None, api_secret=None):
        """
        Kết nối đến Binance API.
        
        Args:
            api_key (str): Binance API key
            api_secret (str): Binance API secret
            
        Returns:
            bool: Kết quả kết nối
        """
        try:
            # Cập nhật API keys nếu được cung cấp
            if api_key:
                self.api_key = api_key
            if api_secret:
                self.api_secret = api_secret
            
            # Kiểm tra xem đã có API keys chưa
            if not self.api_key or not self.api_secret:
                self.add_log("Thiếu API key hoặc API secret", "error")
                self.status = "Lỗi: Thiếu API keys"
                return False
            
            # Kết nối trực tiếp không sử dụng proxy
            proxy_settings = None
            logger.info("TradingManager kết nối trực tiếp, không sử dụng proxy")
            
            # Tạo kết nối trực tiếp
            self.client = Client(
                self.api_key, 
                self.api_secret,
                {"timeout": 120}  # Tăng timeout lên 120 giây
            )
            
            # Kiểm tra kết nối
            server_time = self.client.get_server_time()
            if server_time:
                self.add_log("Đã kết nối thành công đến Binance API")
                self.status = "Đã kết nối"
                return True
            else:
                self.add_log("Không thể kết nối đến Binance API", "error")
                self.status = "Lỗi kết nối"
                return False
        except BinanceAPIException as e:
            self.add_log(f"Lỗi API Binance: {e}", "error")
            self.status = f"Lỗi: {e}"
            return False
        except Exception as e:
            self.add_log(f"Lỗi kết nối: {e}", "error")
            self.status = f"Lỗi: {e}"
            return False
    
    def get_futures_account_balance(self):
        """
        Lấy số dư tài khoản Binance Futures.
        
        Returns:
            float: Số dư USDT, hoặc None nếu có lỗi
        """
        try:
            if not self.client:
                self.add_log("Chưa kết nối đến Binance API", "error")
                return None
            
            # Lấy số dư tài khoản future
            futures_account = self.client.futures_account_balance()
            
            # Tìm USDT balance
            for asset in futures_account:
                if asset['asset'] == 'USDT':
                    balance = float(asset['balance'])
                    self.add_log(f"Số dư tài khoản USDT: {balance}")
                    return balance
            
            self.add_log("Không tìm thấy số dư USDT", "warning")
            return None
        except BinanceAPIException as e:
            self.add_log(f"Lỗi khi lấy số dư: {e}", "error")
            return None
        except Exception as e:
            self.add_log(f"Lỗi hệ thống: {e}", "error")
            return None
    
    def set_leverage(self, symbol, leverage):
        """
        Thiết lập đòn bẩy cho một cặp giao dịch.
        
        Args:
            symbol (str): Biểu tượng giao dịch (ví dụ: ETHUSDT)
            leverage (int): Đòn bẩy (1-125)
            
        Returns:
            bool: Kết quả thiết lập
        """
        try:
            if not self.client:
                self.add_log("Chưa kết nối đến Binance API", "error")
                return False
            
            response = self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            if response and 'leverage' in response:
                self.add_log(f"Đã thiết lập đòn bẩy {response['leverage']}x cho {symbol}")
                return True
            else:
                self.add_log(f"Không thể thiết lập đòn bẩy cho {symbol}", "warning")
                return False
        except BinanceAPIException as e:
            self.add_log(f"Lỗi khi thiết lập đòn bẩy: {e}", "error")
            return False
        except Exception as e:
            self.add_log(f"Lỗi hệ thống: {e}", "error")
            return False
    
    def open_position(self, symbol, side, quantity, is_isolated=True):
        """
        Mở vị thế.
        
        Args:
            symbol (str): Biểu tượng giao dịch (ví dụ: ETHUSDT)
            side (str): Hướng giao dịch ('BUY' hoặc 'SELL')
            quantity (float): Số lượng giao dịch
            is_isolated (bool): Sử dụng chế độ isolated margin
            
        Returns:
            dict: Thông tin lệnh, hoặc None nếu có lỗi
        """
        try:
            if not self.client:
                self.add_log("Chưa kết nối đến Binance API", "error")
                return None
            
            # Mở vị thế mà không cần thay đổi margin type
            # Nếu margin type cần được đổi, API sẽ tự báo lỗi
            # Tạo lệnh thử và xử lý lỗi nếu có
            
            # Tạo lệnh
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            self.add_log(f"Đã mở vị thế {side} {quantity} {symbol} với ID: {order['orderId']}")
            self.is_position_open = True
            self.position_info = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'entry_price': 0,  # Sẽ cập nhật giá sau
                'order_id': order['orderId']
            }
            
            # Cập nhật giá vào lệnh
            time.sleep(1)  # Đợi lệnh được thực hiện
            self._update_position_price()
            
            return order
        except BinanceAPIException as e:
            self.add_log(f"Lỗi khi mở vị thế: {e}", "error")
            return None
        except Exception as e:
            self.add_log(f"Lỗi hệ thống: {e}", "error")
            return None
    
    def close_position(self, symbol=None, side=None, quantity=None):
        """
        Đóng vị thế hiện tại.
        
        Args:
            symbol (str): Biểu tượng giao dịch (ví dụ: ETHUSDT)
            side (str): Hướng giao dịch ('BUY' hoặc 'SELL')
            quantity (float): Số lượng giao dịch
            
        Returns:
            dict: Thông tin lệnh, hoặc None nếu có lỗi
        """
        try:
            if not self.client:
                self.add_log("Chưa kết nối đến Binance API", "error")
                return None
            
            if not self.is_position_open or not self.position_info:
                self.add_log("Không có vị thế nào đang mở", "warning")
                return None
            
            # Sử dụng thông tin vị thế hiện tại nếu không được cung cấp
            symbol = symbol or self.position_info['symbol']
            close_side = 'SELL' if (side or self.position_info['side']) == 'BUY' else 'BUY'
            quantity = quantity or self.position_info['quantity']
            
            # Tạo lệnh đóng vị thế
            order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type='MARKET',
                quantity=quantity
            )
            
            # Tính toán lợi nhuận và cập nhật thống kê theo ngày
            profit_pct = 0
            profit_usdt = 0
            
            if self.position_info and 'entry_price' in self.position_info:
                current_price = self._get_current_price(symbol)
                if current_price:
                    entry_price = self.position_info['entry_price']
                    quantity = self.position_info['quantity']
                    
                    if self.position_info['side'] == 'BUY':
                        profit_pct = (current_price - entry_price) / entry_price * 100
                        profit_usdt = (current_price - entry_price) * quantity
                    else:
                        profit_pct = (entry_price - current_price) / entry_price * 100
                        profit_usdt = (entry_price - current_price) * quantity
                    
                    self.add_log(f"Đã đóng vị thế {symbol} với lợi nhuận {profit_pct:.2f}% ({profit_usdt:.2f} USDT)")
                    
                    # Cập nhật PNL theo ngày
                    self._update_daily_pnl(profit_usdt, profit_pct, symbol, self.position_info['side'])
                else:
                    self.add_log(f"Đã đóng vị thế {symbol}")
            else:
                self.add_log(f"Đã đóng vị thế {symbol}")
            
            # Cập nhật trạng thái
            self.is_position_open = False
            self.position_info = None
            
            return order
        except BinanceAPIException as e:
            self.add_log(f"Lỗi khi đóng vị thế: {e}", "error")
            return None
        except Exception as e:
            self.add_log(f"Lỗi hệ thống: {e}", "error")
            return None
    
    def _get_current_price(self, symbol):
        """
        Lấy giá hiện tại của một cặp giao dịch.
        
        Args:
            symbol (str): Biểu tượng giao dịch (ví dụ: ETHUSDT)
            
        Returns:
            float: Giá hiện tại, hoặc None nếu có lỗi
        """
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except:
            return None
    
    def _update_position_price(self):
        """Cập nhật giá vào lệnh"""
        if not self.is_position_open or not self.position_info:
            return
        
        try:
            # Lấy thông tin vị thế từ Binance
            symbol = self.position_info.get('symbol')
            if not symbol:
                self.add_log("Thiếu thông tin symbol trong position_info", "warning")
                return
                
            try:
                positions = self.client.futures_position_information(symbol=symbol)
            except Exception as e:
                self.add_log(f"Không thể lấy thông tin vị thế từ Binance: {str(e)}", "warning")
                return
                
            # Tìm vị thế hiện tại và cập nhật giá
            for position in positions:
                try:
                    if (position.get('symbol') == symbol and 
                        'positionAmt' in position and 
                        float(position['positionAmt']) != 0):
                        
                        if 'entryPrice' in position:
                            self.position_info['entry_price'] = float(position['entryPrice'])
                            self.add_log(f"Cập nhật giá vào lệnh: {self.position_info['entry_price']}")
                        else:
                            self.add_log("Không tìm thấy entryPrice trong thông tin vị thế", "warning")
                        break
                except Exception as e:
                    self.add_log(f"Lỗi khi xử lý thông tin vị thế: {str(e)}", "warning")
                    continue
        except Exception as e:
            self.add_log(f"Lỗi khi cập nhật giá vị thế: {str(e)}", "warning")
    
    def calculate_position_quantity(self, symbol, account_percent, leverage=1):
        """
        Tính toán số lượng giao dịch dựa trên phần trăm tài khoản.
        
        Args:
            symbol (str): Biểu tượng giao dịch (ví dụ: ETHUSDT)
            account_percent (float): Phần trăm tài khoản sử dụng (0-100)
            leverage (int): Đòn bẩy được sử dụng
            
        Returns:
            float: Số lượng giao dịch, hoặc None nếu có lỗi
        """
        try:
            if not self.client:
                return None
            
            # Lấy số dư USDT
            balance = self.get_futures_account_balance()
            if not balance:
                return None
            
            # Tính số tiền giao dịch dựa trên phần trăm
            trade_amount = balance * (account_percent / 100.0)
            
            # Lấy giá hiện tại
            price = self._get_current_price(symbol)
            if not price:
                return None
            
            # Tính toán số lượng với đòn bẩy
            quantity = (trade_amount * leverage) / price
            
            # Lấy thông tin symbol để làm tròn số lượng
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if symbol_info:
                # Tìm bộ lọc số lượng tối thiểu và step size
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_size_filter:
                    min_qty = float(lot_size_filter['minQty'])
                    step_size = float(lot_size_filter['stepSize'])
                    
                    # Làm tròn số lượng theo step size
                    precision = 0
                    if '.' in str(step_size):
                        precision = len(str(step_size).split('.')[1])
                    
                    # Tính số lượng theo step size đúng cách
                    step_size_precision = int(round(-math.log10(step_size)))
                    quantity = max(min_qty, round(quantity, step_size_precision))
                    
                    # Làm tròn theo step size
                    quantity = float(int(quantity / step_size) * step_size)
                    # Format với đúng số chữ số thập phân
                    quantity = float(format(quantity, f'.{step_size_precision}f'))
                    
                    self.add_log(f"Số lượng giao dịch {symbol}: {quantity} (${trade_amount:.2f}, {account_percent}% tài khoản)")
                    return quantity
            
            # Nếu không tìm thấy thông tin symbol, làm tròn đến 3 chữ số thập phân
            rounded_quantity = round(quantity, 3)
            self.add_log(f"Số lượng giao dịch {symbol}: {rounded_quantity} (${trade_amount:.2f}, {account_percent}% tài khoản)")
            return rounded_quantity
        except BinanceAPIException as e:
            self.add_log(f"Lỗi khi tính toán số lượng giao dịch: {e}", "error")
            return None
        except Exception as e:
            self.add_log(f"Lỗi hệ thống: {e}", "error")
            return None
    
    def get_position_pnl(self, symbol=None):
        """
        Lấy PNL (Profit and Loss) của vị thế hiện tại.
        
        Args:
            symbol (str): Biểu tượng giao dịch (ví dụ: ETHUSDT)
            
        Returns:
            dict: Dictionary chứa thông tin PNL, hoặc None nếu có lỗi
        """
        try:
            if not self.client:
                self.add_log("Chưa kết nối đến Binance API", "error")
                return None
            
            if not self.is_position_open or not self.position_info:
                return {"has_position": False, "pnl": 0, "pnl_percent": 0}
            
            symbol = symbol or self.position_info['symbol']
            
            # Xử lý trong một khối try lớn
            try:
                try:
                    positions = self.client.futures_position_information(symbol=symbol)
                    
                    for position in positions:
                        # Xử lý cẩn thận với try-except
                        try:
                            if (position.get('symbol') == symbol and 
                                'positionAmt' in position and 
                                float(position['positionAmt']) != 0):
                                
                                # Kiểm tra các trường cần thiết
                                if not all(k in position for k in ['entryPrice', 'markPrice', 'leverage', 'positionAmt']):
                                    self.add_log(f"Thiếu trường dữ liệu vị thế PNL: {position.keys()}", "warning")
                                    continue
                                
                                entry_price = float(position['entryPrice'])
                                mark_price = float(position['markPrice'])
                                position_amt = float(position['positionAmt'])
                                leverage = float(position['leverage'])
                                
                                # Tính PNL dựa trên hướng vị thế
                                if position_amt > 0:  # Long position
                                    pnl_percent = (mark_price - entry_price) / entry_price * 100 * leverage
                                    pnl_usdt = position_amt * (mark_price - entry_price)
                                else:  # Short position
                                    pnl_percent = (entry_price - mark_price) / entry_price * 100 * leverage
                                    pnl_usdt = -position_amt * (entry_price - mark_price)
                                
                                # Lấy giá thanh lý nếu có
                                liquidation_price = None
                                if 'liquidationPrice' in position:
                                    if position['liquidationPrice'] != '0':
                                        liquidation_price = float(position['liquidationPrice'])
                                
                                return {
                                    "has_position": True,
                                    "symbol": symbol,
                                    "position_amount": position_amt,
                                    "entry_price": entry_price,
                                    "current_price": mark_price,
                                    "pnl": pnl_usdt,
                                    "pnl_percent": pnl_percent,
                                    "leverage": leverage,
                                    "liquidation_price": liquidation_price
                                }
                        except KeyError as e:
                            self.add_log(f"Thiếu trường {e} trong dữ liệu vị thế", "warning")
                            continue
                        except Exception as e:
                            self.add_log(f"Lỗi khi xử lý thông tin vị thế: {str(e)}", "warning")
                            continue
                    
                    # Không tìm thấy vị thế trong dữ liệu từ API
                    self.is_position_open = False
                    self.position_info = None
                    return {"has_position": False, "pnl": 0, "pnl_percent": 0}
                    
                except Exception as e:
                    self.add_log(f"Lỗi khi lấy thông tin vị thế: {str(e)}", "warning")
                    return {"has_position": False, "pnl": 0, "pnl_percent": 0}
            except Exception as e:
                self.add_log(f"Lỗi không xác định: {str(e)}", "error")
                return {"has_position": False, "pnl": 0, "pnl_percent": 0}
        except BinanceAPIException as e:
            self.add_log(f"Lỗi khi lấy thông tin PNL: {e}", "error")
            return None
        except Exception as e:
            self.add_log(f"Lỗi hệ thống: {e}", "error")
            return None
    
    def start_trading_bot(self, config, prediction_engine):
        """
        Bắt đầu bot giao dịch tự động.
        
        Args:
            config (dict): Cấu hình giao dịch
            prediction_engine: Engine dự đoán
            
        Returns:
            bool: Kết quả khởi động bot
        """
        if self.trading_thread and self.trading_thread.is_alive():
            self.add_log("Bot giao dịch đã đang chạy", "warning")
            return False
        
        if not self.client:
            self.add_log("Chưa kết nối đến Binance API", "error")
            return False
        
        # Kiểm tra xem khung thời gian có hợp lệ không
        import config as app_config
        if 'timeframe' not in config or config['timeframe'] not in app_config.TRADING_SETTINGS['available_timeframes']:
            self.add_log(f"Khung thời gian không hợp lệ. Sử dụng khung mặc định {app_config.TRADING_SETTINGS['default_timeframe']}", "warning")
            config['timeframe'] = app_config.TRADING_SETTINGS['default_timeframe']
        
        # Thiết lập cấu hình
        self.running = True
        self.status = "Đang chạy"
        self.trading_config = config  # Lưu trữ cấu hình để tham chiếu sau này
        
        # Bắt đầu thread giao dịch
        self.trading_thread = threading.Thread(
            target=self._trading_loop,
            args=(config, prediction_engine),
            daemon=True
        )
        self.trading_thread.start()
        
        self.add_log(f"Đã bắt đầu bot giao dịch tự động trên khung {config['timeframe']}")
        return True
    
    def stop_trading_bot(self):
        """
        Dừng bot giao dịch.
        
        Returns:
            bool: Kết quả dừng bot
        """
        if not self.running:
            self.add_log("Bot giao dịch không đang chạy", "warning")
            return False
        
        self.running = False
        self.status = "Đã dừng"
        self.add_log("Đã dừng bot giao dịch")
        
        # Đóng vị thế nếu được yêu cầu và có vị thế mở
        if self.is_position_open:
            self.add_log("Đang đóng vị thế sau khi dừng bot...")
            self.close_position()
        
        return True
    
    def _trading_loop(self, config, prediction_engine):
        """
        Vòng lặp giao dịch tự động.
        
        Args:
            config (dict): Cấu hình giao dịch
            prediction_engine: Engine dự đoán
        """
        import config as app_config
        timeframe = config.get('timeframe', app_config.TRADING_SETTINGS['default_timeframe'])
        update_interval = app_config.TRADING_SETTINGS['update_interval'].get(timeframe, 60)
        
        self.add_log("Vòng lặp giao dịch bắt đầu với cấu hình:")
        self.add_log(f"- Symbol: {config['symbol']}")
        self.add_log(f"- Khung thời gian: {timeframe}")
        self.add_log(f"- Take Profit: {config['take_profit_type']} {config['take_profit_value']}")
        self.add_log(f"- Stop Loss: {config['stop_loss_type']} {config['stop_loss_value']}")
        self.add_log(f"- Tài khoản: {config['account_percent']}%")
        self.add_log(f"- Đòn bẩy: {config['leverage']}x")
        self.add_log(f"- Confidence tối thiểu: {config['min_confidence']}%")
        self.add_log(f"- Cập nhật dự đoán mỗi: {update_interval} giây")
        
        # Thiết lập đòn bẩy
        leverage_result = self.set_leverage(config['symbol'], config['leverage'])
        if not leverage_result:
            self.add_log("Không thể thiết lập đòn bẩy, dừng bot", "error")
            self.running = False
            self.status = "Lỗi đòn bẩy"
            return
        
        # Thiết lập thời gian cho lần cập nhật dự đoán tiếp theo
        last_prediction_time = 0
        
        # Vòng lặp chính
        while self.running:
            try:
                # Kiểm tra PNL nếu có vị thế mở
                if self.is_position_open:
                    self._check_exit_conditions(config)
                    time.sleep(0.5)  # Đợi 0.5s để giảm số lượng request
                    continue
                
                # Kiểm tra xem đã đến lúc cập nhật dự đoán chưa
                current_time = time.time()
                if current_time - last_prediction_time >= update_interval:
                    # Không có vị thế, lấy dự đoán mới và kiểm tra điều kiện mở
                    # Force predicting để cập nhật dự đoán liên tục
                    try:
                        # Lấy dữ liệu mới nhất cho khung thời gian đã chọn
                        current_data = prediction_engine._get_latest_data(timeframe=timeframe)
                        
                        # Dự đoán trên khung thời gian đã chọn
                        prediction_engine.predict(current_data, use_cache=False, timeframe=timeframe)
                        self.add_log(f"Đã cập nhật dự đoán mới trên khung {timeframe}.", "info")
                        
                        # Cập nhật thời gian dự đoán cuối cùng
                        last_prediction_time = current_time
                        
                        # Kiểm tra điều kiện mở vị thế
                        self._check_entry_conditions(config, prediction_engine, timeframe)
                    except Exception as e:
                        self.add_log(f"Lỗi khi cập nhật dự đoán: {e}", "warning")
                
                # Ngủ ngắn hơn để không bỏ lỡ thời điểm cập nhật dự đoán
                time.sleep(1)
            
            except BinanceAPIException as e:
                self.add_log(f"Lỗi API trong vòng lặp giao dịch: {e}", "error")
                time.sleep(5)  # Đợi lâu hơn nếu có lỗi API
            
            except Exception as e:
                self.add_log(f"Lỗi hệ thống trong vòng lặp giao dịch: {e}", "error")
                time.sleep(5)
        
        self.status = "Đã dừng"
        self.add_log("Vòng lặp giao dịch đã kết thúc")
    
    def _check_exit_conditions(self, config):
        """
        Kiểm tra điều kiện đóng vị thế.
        
        Args:
            config (dict): Cấu hình giao dịch
        """
        if not self.is_position_open or not self.position_info:
            return
        
        symbol = self.position_info['symbol']
        
        # Lấy thông tin PNL hiện tại
        pnl_info = self.get_position_pnl(symbol)
        if not pnl_info or not pnl_info.get('has_position', False):
            self.is_position_open = False
            self.position_info = None
            self.add_log("Vị thế không còn tồn tại", "warning")
            return
        
        # Kiểm tra TP/SL
        take_profit_type = config['take_profit_type']
        take_profit_value = config['take_profit_value']
        stop_loss_type = config['stop_loss_type']
        stop_loss_value = config['stop_loss_value']
        
        if take_profit_type == 'percent':
            take_profit_triggered = pnl_info['pnl_percent'] >= take_profit_value
        else:  # 'usdt'
            take_profit_triggered = pnl_info['pnl'] >= take_profit_value
        
        if stop_loss_type == 'percent':
            stop_loss_triggered = pnl_info['pnl_percent'] <= -stop_loss_value
        else:  # 'usdt'
            stop_loss_triggered = pnl_info['pnl'] <= -stop_loss_value
        
        # Hiển thị trạng thái PNL hiện tại (logs cho lần thứ 10)
        if hasattr(self, '_pnl_check_count'):
            self._pnl_check_count += 1
            if self._pnl_check_count % 10 == 0:  # Log mỗi 10 lần kiểm tra
                self.add_log(f"PNL hiện tại: {pnl_info['pnl']:.2f} USDT ({pnl_info['pnl_percent']:.2f}%)")
        else:
            self._pnl_check_count = 0
        
        # Đóng vị thế nếu điều kiện được đáp ứng
        if take_profit_triggered:
            self.add_log(f"Take Profit kích hoạt: {pnl_info['pnl']:.2f} USDT ({pnl_info['pnl_percent']:.2f}%)")
            self.close_position()
        elif stop_loss_triggered:
            self.add_log(f"Stop Loss kích hoạt: {pnl_info['pnl']:.2f} USDT ({pnl_info['pnl_percent']:.2f}%)")
            self.close_position()
    
    def _update_daily_pnl(self, profit_usdt, profit_pct, symbol, side):
        """
        Cập nhật thống kê PNL theo ngày.
        
        Args:
            profit_usdt (float): Lợi nhuận tính bằng USDT
            profit_pct (float): Lợi nhuận tính bằng phần trăm
            symbol (str): Symbol giao dịch
            side (str): Hướng giao dịch (BUY/SELL)
        """
        # Kiểm tra ngày hiện tại, nếu khác với ngày trong daily_pnl thì reset
        current_date = get_current_date_tz7()
        if current_date != self.daily_pnl['date']:
            # Lưu thống kê ngày cũ vào lịch sử nếu cần
            self.add_log(f"Đã chuyển sang ngày mới {current_date}, reset thống kê PNL")
            
            # Khởi tạo thống kê mới
            self.daily_pnl = {
                'date': current_date,
                'trades': [],
                'total_pnl': 0.0,
                'win_count': 0,
                'loss_count': 0
            }
        
        # Thêm giao dịch mới vào danh sách
        trade_info = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'symbol': symbol,
            'side': side,
            'pnl': profit_usdt,
            'pnl_percent': profit_pct
        }
        
        self.daily_pnl['trades'].append(trade_info)
        self.daily_pnl['total_pnl'] += profit_usdt
        
        if profit_usdt > 0:
            self.daily_pnl['win_count'] += 1
        else:
            self.daily_pnl['loss_count'] += 1
            
        # Tính tỷ lệ thắng
        total_trades = self.daily_pnl['win_count'] + self.daily_pnl['loss_count']
        win_rate = (self.daily_pnl['win_count'] / total_trades * 100) if total_trades > 0 else 0
        
        self.add_log(f"PNL ngày {current_date}: {self.daily_pnl['total_pnl']:.2f} USDT, "
                    f"Win rate: {win_rate:.1f}% ({self.daily_pnl['win_count']}/{total_trades})")
        
    def get_daily_pnl_summary(self):
        """
        Lấy tóm tắt PNL theo ngày.
        
        Returns:
            dict: Thông tin tóm tắt PNL theo ngày
        """
        current_date = get_current_date_tz7()
        
        # Kiểm tra ngày trong daily_pnl có phải là ngày hiện tại không
        if current_date != self.daily_pnl['date']:
            self.daily_pnl['date'] = current_date
            self.daily_pnl['trades'] = []
            self.daily_pnl['total_pnl'] = 0.0
            self.daily_pnl['win_count'] = 0
            self.daily_pnl['loss_count'] = 0
            
        # Tính tỷ lệ thắng
        total_trades = self.daily_pnl['win_count'] + self.daily_pnl['loss_count']
        win_rate = (self.daily_pnl['win_count'] / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'date': self.daily_pnl['date'],
            'total_pnl': self.daily_pnl['total_pnl'],
            'win_count': self.daily_pnl['win_count'],
            'loss_count': self.daily_pnl['loss_count'],
            'win_rate': win_rate,
            'trades': self.daily_pnl['trades']
        }
    
    def _check_entry_conditions(self, config, prediction_engine, timeframe=None):
        """
        Kiểm tra điều kiện mở vị thế.
        
        Args:
            config (dict): Cấu hình giao dịch
            prediction_engine: Engine dự đoán
            timeframe (str, optional): Khung thời gian dự đoán. Nếu None, lấy từ config
        """
        symbol = config['symbol']
        min_confidence = config['min_confidence']  # Đã chuyển % thành phân số khi nhận config
        
        # Xác định khung thời gian
        if timeframe is None:
            import config as app_config
            timeframe = config.get('timeframe', app_config.TRADING_SETTINGS['default_timeframe'])
        
        # Lấy dự đoán mới nhất cho khung thời gian cụ thể
        prediction = prediction_engine.get_cached_prediction(timeframe)
        
        # Kiểm tra dự đoán có hợp lệ không
        if not prediction or 'trend' not in prediction or 'confidence' not in prediction:
            self.add_log(f"Không có dự đoán hợp lệ cho khung {timeframe}", "info")
            return
        
        # Kiểm tra độ tin cậy có đáp ứng yêu cầu tối thiểu
        confidence = prediction['confidence']
        if confidence < min_confidence:
            self.add_log(f"Độ tin cậy {confidence*100:.1f}% thấp hơn ngưỡng {min_confidence*100:.1f}%", "info")
            return
        
        # Kiểm tra xu hướng
        trend = prediction['trend']
        if trend == 'NEUTRAL':
            self.add_log(f"Xu hướng NEUTRAL, không mở vị thế", "info")
            return
        
        # Kiểm tra biến động giá tối thiểu (nếu được cấu hình)
        min_price_movement = config.get('min_price_movement', 0)
        if min_price_movement > 0 and 'target_price' in prediction and 'price' in prediction:
            current_price = prediction['price']
            target_price = prediction['target_price']
            price_movement = abs(target_price - current_price)
            
            # Kiểm tra xem biến động giá có đạt ngưỡng tối thiểu không
            if price_movement < min_price_movement:
                self.add_log(f"Biến động giá dự đoán {price_movement:.2f} USDT thấp hơn ngưỡng {min_price_movement:.2f} USDT", "info")
                return
            else:
                self.add_log(f"Biến động giá dự đoán đạt {price_movement:.2f} USDT (ngưỡng: {min_price_movement:.2f} USDT)")
        
        # Dự đoán hợp lệ và đáp ứng yêu cầu độ tin cậy
        self.add_log(f"Dự đoán hợp lệ trên khung {timeframe}: {trend} với độ tin cậy {confidence*100:.1f}%")
        
        # Chi tiết dự đoán kỹ thuật
        if 'technical_reason' in prediction:
            self.add_log(f"Lý do kỹ thuật: {prediction['technical_reason']}")
        
        # Tính toán số lượng giao dịch
        account_percent = config['account_percent']
        leverage = config['leverage']
        quantity = self.calculate_position_quantity(symbol, account_percent, leverage)
        
        if not quantity:
            self.add_log("Không thể tính toán số lượng giao dịch", "error")
            return
        
        # Mở vị thế
        side = 'BUY' if trend == 'LONG' else 'SELL'
        self.add_log(f"Mở vị thế {side} {quantity} {symbol} dựa trên khung {timeframe}")
        result = self.open_position(symbol, side, quantity)
        
        if result:
            self.add_log(f"Đã mở vị thế thành công: {side} {quantity} {symbol}")
            
            # Thêm thông tin khung thời gian vào position_info
            if self.position_info:
                self.position_info['timeframe'] = timeframe
                self.position_info['prediction'] = prediction
        else:
            self.add_log("Không thể mở vị thế", "error")