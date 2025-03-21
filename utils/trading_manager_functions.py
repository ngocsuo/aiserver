"""
Các hàm bổ sung cho TradingManager
"""
import os
from datetime import datetime, timedelta, timezone
import pytz
import logging

# Các hàm liên quan đến múi giờ và thời gian
def get_current_date_tz7():
    """Lấy ngày hiện tại theo múi giờ +7 (Việt Nam)"""
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    now = datetime.now(vietnam_tz)
    return now.strftime("%Y-%m-%d")

def get_daily_pnl_summary(trades):
    """Tổng hợp PNL theo ngày"""
    if not trades:
        return {
            'date': get_current_date_tz7(),
            'total_pnl': 0.0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0,
            'trades': []
        }
    
    total_pnl = sum(trade.get('pnl', 0) for trade in trades)
    win_count = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    loss_count = sum(1 for trade in trades if trade.get('pnl', 0) < 0)
    
    total_trades = win_count + loss_count
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'date': get_current_date_tz7(),
        'total_pnl': total_pnl,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'trades': trades
    }