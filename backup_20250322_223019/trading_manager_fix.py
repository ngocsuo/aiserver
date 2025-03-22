"""
Sửa lỗi cho TradingManager để giải quyết vấn đề không thể dừng bot
"""

# Lỗi này thường xảy ra khi bot đang chạy trong một thread riêng biệt và 
# không có cơ chế dừng an toàn. Dưới đây là cách sửa:

# Tìm file utils/trading_manager.py và tìm hàm stop_trading_bot và _trading_loop

# Sửa hàm stop_trading_bot như sau:
def stop_trading_bot(self):
    """
    Dừng bot giao dịch.
    
    Returns:
        bool: Kết quả dừng bot
    """
    if self._trading_thread and self._trading_thread.is_alive():
        self._stop_trading = True  # Set flag stop
        
        # Thêm timeout để không bị treo vô hạn
        self._trading_thread.join(timeout=10)
        
        # Force kill thread nếu vẫn còn sống sau timeout
        if self._trading_thread.is_alive():
            import ctypes
            thread_id = self._trading_thread.ident
            if thread_id:
                try:
                    # Thử terminate thread nếu không dừng được
                    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread_id),
                        ctypes.py_object(SystemExit)
                    )
                    if res > 1:
                        # Nếu có lỗi, clean up
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_long(thread_id), 
                            ctypes.c_long(0)
                        )
                except Exception as e:
                    self.add_log(f"Không thể terminate thread: {str(e)}", "error")
            
        self._trading_thread = None
        self.add_log("Bot giao dịch đã dừng", "info")
        return True
    else:
        self.add_log("Bot giao dịch chưa chạy", "warning")
        return False

# Và đảm bảo hàm _trading_loop có kiểm tra flag stop:
def _trading_loop(self, config, prediction_engine):
    """
    Vòng lặp giao dịch tự động.
    
    Args:
        config (dict): Cấu hình giao dịch
        prediction_engine: Engine dự đoán
    """
    self._stop_trading = False
    
    while not self._stop_trading:
        try:
            # Kiểm tra nếu nên dừng
            if self._stop_trading:
                break
                
            # Code hiện tại của bạn...
            
            # Thêm sleep để giảm tải CPU và cho phép kiểm tra flag _stop_trading thường xuyên hơn
            time.sleep(1)
        except Exception as e:
            self.add_log(f"Lỗi trong vòng lặp giao dịch: {str(e)}", "error")
            # Thêm sleep ngắn để tránh loop liên tục khi có lỗi
            time.sleep(5)
            
    self.add_log("Vòng lặp giao dịch đã kết thúc", "info")