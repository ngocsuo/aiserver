"""
Các hàm bổ sung cho TradingManager
"""

import datetime
import pytz

def get_current_date_tz7():
    """Lấy ngày hiện tại theo múi giờ +7 (Việt Nam)"""
    tz = pytz.timezone('Asia/Bangkok')  # Bangkok cùng múi giờ +7 như Việt Nam
    now = datetime.datetime.now(tz)
    return now.strftime("%Y-%m-%d")

def get_daily_pnl_summary(trades):
    """Tổng hợp PNL theo ngày"""
    if not trades:
        return {
            'total_pnl': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0
        }
        
    total_pnl = sum(trade['pnl'] for trade in trades)
    win_count = sum(1 for trade in trades if trade['pnl'] > 0)
    loss_count = len(trades) - win_count
    win_rate = (win_count / len(trades) * 100) if trades else 0
    
    return {
        'total_pnl': total_pnl,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate
    }