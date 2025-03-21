"""
Module quản lý giao dịch với Binance Futures API.
"""
import os
import time
import threading
import logging
from datetime import datetime, timedelta, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pytz

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
            'date': self._get_current_date_tz7(),
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
            
            # Thiết lập proxy để vượt qua hạn chế IP của Binance
            proxy_str = "mb105.raiproxy.com:15989:S6lnXxjtieCIA38a:XXjY9RleeBfS8AFX"
            proxy_parts = proxy_str.split(':')
            
            if len(proxy_parts) >= 4:
                host = proxy_parts[0]
                port = proxy_parts[1]
                username = proxy_parts[2]
                password = proxy_parts[3]
                
                proxy_auth = f"{username}:{password}@{host}:{port}"
                proxy_settings = {
                    'http': f'http://{proxy_auth}',
                    'https': f'http://{proxy_auth}'
                }
                
                logger.info(f"Kết nối qua proxy xác thực ({host}:{port})")
                
                # Tạo kết nối với proxy
                self.client = Client(
                    self.api_key, 
                    self.api_secret,
                    {"proxies": proxy_settings, "timeout": 60}
                )
            else:
                # Fallback to direct connection if proxy format is invalid
                logger.warning("Định dạng proxy không hợp lệ, thử kết nối trực tiếp")
                self.client = Client(self.api_key, self.api_secret)
            
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
            
            # Thiết lập chế độ margin
            if is_isolated:
                self.client.futures_change_margin_type(symbol=symbol, marginType='ISOLATED')
            
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
            
            # Tính toán lợi nhuận
            if self.position_info and 'entry_price' in self.position_info:
                current_price = self._get_current_price(symbol)
                if current_price:
                    if self.position_info['side'] == 'BUY':
                        profit_pct = (current_price - self.position_info['entry_price']) / self.position_info['entry_price'] * 100
                    else:
                        profit_pct = (self.position_info['entry_price'] - current_price) / self.position_info['entry_price'] * 100
                    
                    self.add_log(f"Đã đóng vị thế {symbol} với lợi nhuận {profit_pct:.2f}%")
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
            symbol = self.position_info['symbol']
            positions = self.client.futures_position_information(symbol=symbol)
            
            for position in positions:
                if position['symbol'] == symbol and float(position['positionAmt']) != 0:
                    self.position_info['entry_price'] = float(position['entryPrice'])
                    self.add_log(f"Cập nhật giá vào lệnh: {self.position_info['entry_price']}")
                    break
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
            exchange_info = self.client.get_exchange_info()
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
                    
                    quantity = max(min_qty, round(quantity - (quantity % step_size), precision))
                    
                    self.add_log(f"Số lượng giao dịch {symbol}: {quantity} (${trade_amount:.2f}, {account_percent}% tài khoản)")
                    return quantity
            
            # Nếu không tìm thấy thông tin symbol, làm tròn đến 4 chữ số thập phân
            rounded_quantity = round(quantity, 4)
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
            
            # Lấy thông tin vị thế
            positions = self.client.futures_position_information(symbol=symbol)
            
            for position in positions:
                if position['symbol'] == symbol and float(position['positionAmt']) != 0:
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
                    
                    result = {
                        "has_position": True,
                        "symbol": symbol,
                        "position_amount": position_amt,
                        "entry_price": entry_price,
                        "current_price": mark_price,
                        "pnl": pnl_usdt,
                        "pnl_percent": pnl_percent,
                        "leverage": leverage,
                        "liquidation_price": float(position['liquidationPrice']) if position['liquidationPrice'] != '0' else None
                    }
                    
                    return result
            
            # Không tìm thấy vị thế
            self.is_position_open = False
            self.position_info = None
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
        
        # Thiết lập cấu hình
        self.running = True
        self.status = "Đang chạy"
        
        # Bắt đầu thread giao dịch
        self.trading_thread = threading.Thread(
            target=self._trading_loop,
            args=(config, prediction_engine),
            daemon=True
        )
        self.trading_thread.start()
        
        self.add_log("Đã bắt đầu bot giao dịch tự động")
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
        self.add_log("Vòng lặp giao dịch bắt đầu với cấu hình:")
        self.add_log(f"- Symbol: {config['symbol']}")
        self.add_log(f"- Take Profit: {config['take_profit_type']} {config['take_profit_value']}")
        self.add_log(f"- Stop Loss: {config['stop_loss_type']} {config['stop_loss_value']}")
        self.add_log(f"- Tài khoản: {config['account_percent']}%")
        self.add_log(f"- Đòn bẩy: {config['leverage']}x")
        self.add_log(f"- Confidence tối thiểu: {config['min_confidence']}%")
        
        # Thiết lập đòn bẩy
        leverage_result = self.set_leverage(config['symbol'], config['leverage'])
        if not leverage_result:
            self.add_log("Không thể thiết lập đòn bẩy, dừng bot", "error")
            self.running = False
            self.status = "Lỗi đòn bẩy"
            return
        
        # Vòng lặp chính
        while self.running:
            try:
                # Kiểm tra PNL nếu có vị thế mở
                if self.is_position_open:
                    self._check_exit_conditions(config)
                    time.sleep(0.5)  # Đợi 0.5s để giảm số lượng request
                    continue
                
                # Không có vị thế, kiểm tra dự đoán để mở vị thế mới
                self._check_entry_conditions(config, prediction_engine)
                time.sleep(1)  # Đợi 1s trước khi kiểm tra lại
            
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
    
    def _check_entry_conditions(self, config, prediction_engine):
        """
        Kiểm tra điều kiện mở vị thế.
        
        Args:
            config (dict): Cấu hình giao dịch
            prediction_engine: Engine dự đoán
        """
        symbol = config['symbol']
        min_confidence = config['min_confidence'] / 100.0  # Chuyển % thành phân số
        
        # Lấy dự đoán mới nhất
        prediction = prediction_engine.get_cached_prediction()
        
        # Kiểm tra dự đoán có hợp lệ không
        if not prediction or 'trend' not in prediction or 'confidence' not in prediction:
            return
        
        # Kiểm tra độ tin cậy có đáp ứng yêu cầu tối thiểu
        confidence = prediction['confidence']
        if confidence < min_confidence:
            return
        
        # Kiểm tra xu hướng
        trend = prediction['trend']
        if trend == 'NEUTRAL':
            return
        
        # Dự đoán hợp lệ và đáp ứng yêu cầu độ tin cậy
        self.add_log(f"Dự đoán hợp lệ: {trend} với độ tin cậy {confidence*100:.1f}%")
        
        # Tính toán số lượng giao dịch
        account_percent = config['account_percent']
        leverage = config['leverage']
        quantity = self.calculate_position_quantity(symbol, account_percent, leverage)
        
        if not quantity:
            self.add_log("Không thể tính toán số lượng giao dịch", "error")
            return
        
        # Mở vị thế
        side = 'BUY' if trend == 'LONG' else 'SELL'
        self.add_log(f"Mở vị thế {side} {quantity} {symbol}")
        result = self.open_position(symbol, side, quantity)
        
        if result:
            self.add_log(f"Đã mở vị thế thành công: {side} {quantity} {symbol}")
        else:
            self.add_log("Không thể mở vị thế", "error")