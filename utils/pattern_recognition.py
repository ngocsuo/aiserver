"""
Module nhận diện mẫu hình nến và phân tích xu hướng.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def detect_candlestick_patterns(df):
    """
    Phát hiện các mẫu hình nến phổ biến trong DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame với dữ liệu OHLCV
        
    Returns:
        list: Danh sách các mẫu hình nến được phát hiện
    """
    patterns = []
    
    # Chuyển DataFrame thành array và làm việc với nó cho hiệu suất cao hơn
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    # Lấy thân nến và bóng
    candle_body = abs(closes - opens)
    body_avg = np.mean(candle_body)
    
    # Tính độ dài thân và bóng nến
    upper_shadows = highs - np.maximum(opens, closes)
    lower_shadows = np.minimum(opens, closes) - lows
    
    # Kiểm tra xem giá đang trong xu hướng tăng hay giảm
    trend = "uptrend" if np.mean(closes[-5:]) > np.mean(closes[-10:-5]) else "downtrend"
    
    # Mẫu hình Doji
    for i in range(len(df)-1, max(len(df)-3, 0), -1):
        if candle_body[i] <= 0.1 * body_avg:
            pattern = {
                "name": "Doji",
                "description": "Nến có thân rất nhỏ, chỉ ra sự lưỡng lự giữa người mua và người bán.",
                "direction": "neutral",
                "reliability": 60,
                "index": i
            }
            patterns.append(pattern)
            break
    
    # Mẫu hình Hammer (Búa)
    for i in range(len(df)-1, max(len(df)-3, 0), -1):
        if (lower_shadows[i] >= 2 * candle_body[i] and 
            upper_shadows[i] <= 0.1 * lower_shadows[i] and
            trend == "downtrend"):
            pattern = {
                "name": "Hammer",
                "description": "Nến có bóng dưới dài và thân nhỏ, xuất hiện ở đáy xu hướng giảm.",
                "direction": "bullish",
                "reliability": 70,
                "index": i
            }
            patterns.append(pattern)
            break
    
    # Mẫu hình Shooting Star (Sao Băng)
    for i in range(len(df)-1, max(len(df)-3, 0), -1):
        if (upper_shadows[i] >= 2 * candle_body[i] and 
            lower_shadows[i] <= 0.1 * upper_shadows[i] and
            trend == "uptrend"):
            pattern = {
                "name": "Shooting Star",
                "description": "Nến có bóng trên dài và thân nhỏ, xuất hiện ở đỉnh xu hướng tăng.",
                "direction": "bearish",
                "reliability": 70,
                "index": i
            }
            patterns.append(pattern)
            break
    
    # Mẫu hình Engulfing (Bao phủ)
    if len(df) >= 2:
        for i in range(len(df)-1, max(len(df)-3, 0), -1):
            # Bullish Engulfing
            if (opens[i] < closes[i] and  # Nến tăng
                opens[i-1] > closes[i-1] and  # Nến giảm
                opens[i] <= closes[i-1] and 
                closes[i] >= opens[i-1]):
                pattern = {
                    "name": "Bullish Engulfing",
                    "description": "Nến tăng bao phủ hoàn toàn nến giảm trước đó, chỉ ra sự đảo chiều tăng.",
                    "direction": "bullish",
                    "reliability": 80,
                    "index": i
                }
                patterns.append(pattern)
                break
                
            # Bearish Engulfing
            elif (opens[i] > closes[i] and  # Nến giảm
                  opens[i-1] < closes[i-1] and  # Nến tăng
                  opens[i] >= closes[i-1] and 
                  closes[i] <= opens[i-1]):
                pattern = {
                    "name": "Bearish Engulfing",
                    "description": "Nến giảm bao phủ hoàn toàn nến tăng trước đó, chỉ ra sự đảo chiều giảm.",
                    "direction": "bearish",
                    "reliability": 80,
                    "index": i
                }
                patterns.append(pattern)
                break
    
    # Mẫu hình Morning Star (Sao Mai)
    if len(df) >= 3:
        for i in range(len(df)-1, max(len(df)-3, 0), -1):
            if (i >= 2 and
                opens[i-2] > closes[i-2] and  # Nến giảm
                abs(opens[i-1] - closes[i-1]) < body_avg * 0.5 and  # Nến nhỏ
                opens[i] < closes[i] and  # Nến tăng
                closes[i] > (opens[i-2] + closes[i-2]) / 2):  # Đóng cửa trên giữa nến đầu tiên
                pattern = {
                    "name": "Morning Star",
                    "description": "Mẫu hình 3 nến với nến giảm, nến nhỏ và nến tăng, chỉ ra sự đảo chiều tăng mạnh.",
                    "direction": "bullish",
                    "reliability": 85,
                    "index": i
                }
                patterns.append(pattern)
                break
    
    # Mẫu hình Evening Star (Sao Hôm)
    if len(df) >= 3:
        for i in range(len(df)-1, max(len(df)-3, 0), -1):
            if (i >= 2 and
                opens[i-2] < closes[i-2] and  # Nến tăng
                abs(opens[i-1] - closes[i-1]) < body_avg * 0.5 and  # Nến nhỏ
                opens[i] > closes[i] and  # Nến giảm
                closes[i] < (opens[i-2] + closes[i-2]) / 2):  # Đóng cửa dưới giữa nến đầu tiên
                pattern = {
                    "name": "Evening Star",
                    "description": "Mẫu hình 3 nến với nến tăng, nến nhỏ và nến giảm, chỉ ra sự đảo chiều giảm mạnh.",
                    "direction": "bearish",
                    "reliability": 85,
                    "index": i
                }
                patterns.append(pattern)
                break
    
    # Mẫu hình Harami (Thai Nghén)
    if len(df) >= 2:
        for i in range(len(df)-1, max(len(df)-3, 0), -1):
            # Bullish Harami
            if (opens[i-1] > closes[i-1] and  # Nến giảm
                opens[i] < closes[i] and  # Nến tăng
                opens[i] > closes[i-1] and 
                closes[i] < opens[i-1]):
                pattern = {
                    "name": "Bullish Harami",
                    "description": "Nến tăng nhỏ nằm hoàn toàn trong thân nến giảm lớn trước đó, chỉ ra khả năng đảo chiều tăng.",
                    "direction": "bullish",
                    "reliability": 65,
                    "index": i
                }
                patterns.append(pattern)
                break
                
            # Bearish Harami
            elif (opens[i-1] < closes[i-1] and  # Nến tăng
                  opens[i] > closes[i] and  # Nến giảm
                  opens[i] < closes[i-1] and 
                  closes[i] > opens[i-1]):
                pattern = {
                    "name": "Bearish Harami",
                    "description": "Nến giảm nhỏ nằm hoàn toàn trong thân nến tăng lớn trước đó, chỉ ra khả năng đảo chiều giảm.",
                    "direction": "bearish",
                    "reliability": 65,
                    "index": i
                }
                patterns.append(pattern)
                break
    
    return patterns


def calculate_support_resistance(df, window=30, sensitivity=0.01):
    """
    Tính toán các mức hỗ trợ và kháng cự dựa trên các đỉnh và đáy gần đây.
    
    Args:
        df (pd.DataFrame): DataFrame với dữ liệu OHLCV
        window (int): Số nến để tìm đỉnh và đáy
        sensitivity (float): Ngưỡng nhạy cảm để xác định mức
        
    Returns:
        dict: Dictionary chứa các mức hỗ trợ và kháng cự
    """
    if len(df) < window:
        return None
    
    price_range = df['high'].max() - df['low'].min()
    min_distance = price_range * sensitivity
    
    # Tìm các đỉnh cục bộ
    highs = df['high'].values
    lows = df['low'].values
    
    peaks = []
    troughs = []
    
    for i in range(window, len(df)-window):
        # Đỉnh cao
        if highs[i] == max(highs[i-window:i+window+1]):
            peaks.append(highs[i])
        
        # Đáy thấp
        if lows[i] == min(lows[i-window:i+window+1]):
            troughs.append(lows[i])
    
    # Gom nhóm các mức gần nhau
    grouped_peaks = _group_levels(peaks, min_distance)
    grouped_troughs = _group_levels(troughs, min_distance)
    
    # Lấy giá hiện tại
    current_price = df['close'].iloc[-1]
    
    # Phân loại thành hỗ trợ và kháng cự dựa trên giá hiện tại
    support_levels = [level for level in grouped_troughs if level < current_price]
    resistance_levels = [level for level in grouped_peaks if level > current_price]
    
    # Sắp xếp theo độ mạnh (số lần chạm)
    support_levels.sort(reverse=True)
    resistance_levels.sort()
    
    # Thêm các mức Fibonacci Retracement nếu có đủ dữ liệu
    if len(df) > 100:
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        
        fib_levels = {
            0.236: low + 0.236 * diff,
            0.382: low + 0.382 * diff,
            0.5: low + 0.5 * diff,
            0.618: low + 0.618 * diff,
            0.786: low + 0.786 * diff
        }
        
        # Thêm các mức Fibonacci vào mức hỗ trợ và kháng cự
        for level_value in fib_levels.values():
            if level_value < current_price:
                support_levels.append(level_value)
            else:
                resistance_levels.append(level_value)
        
        # Sắp xếp lại
        support_levels = sorted(list(set(support_levels)), reverse=True)
        resistance_levels = sorted(list(set(resistance_levels)))
    
    return {
        'support': support_levels,
        'resistance': resistance_levels
    }


def _group_levels(levels, min_distance):
    """
    Gom nhóm các mức gần nhau.
    
    Args:
        levels (list): Danh sách các mức
        min_distance (float): Khoảng cách tối thiểu để coi là riêng biệt
        
    Returns:
        list: Danh sách các mức đã gom nhóm
    """
    if not levels:
        return []
    
    sorted_levels = sorted(levels)
    grouped = []
    
    current_group = [sorted_levels[0]]
    
    for level in sorted_levels[1:]:
        if level - current_group[-1] <= min_distance:
            current_group.append(level)
        else:
            # Tính trung bình nhóm
            grouped.append(sum(current_group) / len(current_group))
            current_group = [level]
    
    # Thêm nhóm cuối cùng
    if current_group:
        grouped.append(sum(current_group) / len(current_group))
    
    return grouped


def analyze_price_trend(df, window=14):
    """
    Phân tích xu hướng giá.
    
    Args:
        df (pd.DataFrame): DataFrame với dữ liệu OHLCV
        window (int): Cửa sổ trượt để phân tích xu hướng
        
    Returns:
        dict: Thông tin về xu hướng
    """
    if len(df) < 2 * window:
        return None
    
    # Lấy giá đóng cửa
    closes = df['close'].values
    
    # Tính xu hướng dựa trên trung bình động
    ma_short = np.mean(closes[-window:])
    ma_long = np.mean(closes[-2*window:-window])
    
    # Xác định xu hướng
    if ma_short > ma_long * 1.01:  # Tăng ít nhất 1%
        trend = "uptrend"
    elif ma_short < ma_long * 0.99:  # Giảm ít nhất 1%
        trend = "downtrend"
    else:
        trend = "sideways"
    
    # Tính độ dốc của xu hướng
    x = np.arange(len(closes[-window:]))
    slope, _ = np.polyfit(x, closes[-window:], 1)
    
    # Tính độ dài xu hướng hiện tại
    trend_duration = 0
    current_direction = 1 if closes[-1] > closes[-2] else -1
    
    for i in range(len(closes)-2, -1, -1):
        direction = 1 if closes[i] > closes[i-1] else -1
        if direction == current_direction:
            trend_duration += 1
        else:
            break
    
    # Tính độ mạnh xu hướng (1-10)
    trend_strength = 0
    
    # Dựa trên độ dốc
    normalized_slope = abs(slope) / np.mean(closes[-window:]) * 100
    slope_strength = min(5, normalized_slope * 20)
    
    # Dựa trên sự nhất quán (có bao nhiêu nến đi theo xu hướng)
    consistent_candles = sum(1 for i in range(1, window) if (closes[-i] - closes[-i-1]) * current_direction > 0)
    consistency_strength = consistent_candles / window * 5
    
    trend_strength = round(slope_strength + consistency_strength)
    
    # Tạo đường xu hướng
    trendline = x * slope + np.polyfit(x, closes[-window:], 1)[1]
    
    # Phát hiện các mức hỗ trợ/kháng cự trong xu hướng
    sr_levels = calculate_support_resistance(df)
    
    return {
        'trend': trend,
        'strength': trend_strength,
        'duration': trend_duration,
        'slope': slope,
        'trendline': trendline,
        'support_levels': sr_levels['support'] if sr_levels else [],
        'resistance_levels': sr_levels['resistance'] if sr_levels else []
    }