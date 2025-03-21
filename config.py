"""
Configuration settings for the ETHUSDT prediction system.
"""
import os
from datetime import datetime

# Binance API Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Proxy Configuration
USE_PROXY = os.environ.get("USE_PROXY", "False").lower() in ("true", "1", "yes")  # Tắt proxy mặc định (người dùng sẽ bật khi cần)
PROXY_HOST = os.environ.get("PROXY_HOST", "")
PROXY_PORT = os.environ.get("PROXY_PORT", "")
PROXY_USERNAME = os.environ.get("PROXY_USERNAME", "")
PROXY_PASSWORD = os.environ.get("PROXY_PASSWORD", "")

# Feature flags
USE_REAL_API = True  # Luôn sử dụng real Binance API, không dùng mock data
DEBUG_MODE = True     # Enable additional logging and debug information
FORCE_MOCK_DATA = False  # KHÔNG sử dụng mock data trong mọi trường hợp

# Trading Symbol and Timeframes
SYMBOL = "ETHUSDT"

# Cấu trúc khung thời gian đơn giản, rõ ràng hơn
PRIMARY_TIMEFRAME = "1m"  # Timeframe chính mặc định
SECONDARY_TIMEFRAME = "5m"  # Timeframe thứ hai cho dự đoán dài hạn
ALL_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]  # Tất cả khung thời gian hỗ trợ

# Cấu trúc cũ được giữ lại để tương thích
TIMEFRAMES = {
    "primary": PRIMARY_TIMEFRAME,
    "secondary": ["5m", "15m", "1h", "4h"]
}

# Cấu hình dự đoán cho mỗi khung thời gian
PREDICTION_SETTINGS = {
    "1m": {
        "horizons": {
            "10m": 10,  # 10 phút = 10 candles của khung 1m
            "15m": 15,  # 15 phút = 15 candles của khung 1m
        }
    },
    "5m": {
        "horizons": {
            "30m": 6,   # 30 phút = 6 candles của khung 5m
            "1h": 12,   # 1 giờ = 12 candles của khung 5m
        }
    }
}

# Cấu hình giao dịch tự động
TRADING_SETTINGS = {
    "available_timeframes": ["1m", "5m"],  # Các khung thời gian có thể sử dụng cho bot
    "default_timeframe": "5m",             # Khung thời gian giao dịch mặc định
    "min_price_movement": 0.0,             # Biến động giá tối thiểu (USDT) để vào lệnh
    "default_min_price_movement": 6.0,     # Giá trị mặc định cho biến động giá tối thiểu
    "update_interval": {
        "1m": 30,                          # Cập nhật dự đoán mỗi 30 giây cho khung 1m
        "5m": 60                           # Cập nhật dự đoán mỗi 60 giây cho khung 5m
    }
}

# Timeframe mặc định và horizon dự đoán mặc định
DEFAULT_TIMEFRAME = "1m"
DEFAULT_PREDICTION_HORIZON = "10m"

# Data Collection Configuration
LOOKBACK_PERIODS = 5000  # Number of candles to collect initially

# Cấu hình thời gian dữ liệu
DATA_RANGE_OPTIONS = {
    "realtime": 3,          # 3 ngày dữ liệu cho biểu đồ thời gian thực (giảm để load nhanh hơn)
    "training": 90,         # 3 tháng (90 ngày) dữ liệu cho huấn luyện mặc định (giảm để training nhanh hơn)
    "max_history": 1200     # Tối đa khoảng 3-4 năm dữ liệu
}

# Tính ngày bắt đầu dữ liệu huấn luyện (3 tháng gần nhất)
from datetime import datetime, timedelta
today = datetime.now()
DEFAULT_TRAINING_START_DATE = (today - timedelta(days=DATA_RANGE_OPTIONS["training"])).strftime("%Y-%m-%d")

# Giá trị mặc định cho ngày bắt đầu dữ liệu huấn luyện
# Giá trị này có thể được cập nhật trong thời gian chạy từ UI
HISTORICAL_START_DATE = "2023-12-01"  # Sử dụng ngày 1/12/2023 làm giá trị mặc định theo yêu cầu

ENABLE_BACKTESTING = True  # Enable backtesting functionality
BACKTEST_PERIOD_START = "2022-01-01"  # Start date for backtesting
BACKTEST_PERIOD_END = "2022-12-31"   # End date for backtesting
UPDATE_INTERVAL = 10  # Seconds between data updates (giảm xuống để cập nhật thường xuyên hơn)

# Feature Engineering Parameters
TECHNICAL_INDICATORS = {
    "rsi": {"window": 14},
    "ema": {"windows": [9, 21, 55, 200]},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bbands": {"window": 20, "std_dev": 2},
    "atr": {"window": 14},
    "vwap": {"window": 14},
    # Advanced indicators added
    "supertrend": {"period": 10, "multiplier": 3.0},
    "ichimoku": {
        "tenkan_period": 9, 
        "kijun_period": 26, 
        "senkou_b_period": 52, 
        "displacement": 26
    },
    "adx": {"window": 14},
    "pivot_points": {"method": "fibonacci"},
    "volume_profile": {"divisions": 10}
}

# Data Labeling Configuration
PRICE_MOVEMENT_THRESHOLD = 0.003  # 0.3% move for labeling
TARGET_PNL_THRESHOLD = 5  # $5 for alternative labeling
PREDICTION_WINDOW = 12   # Số nến 5m (1 giờ) để dự đoán trong tương lai

# Model Configuration
SEQUENCE_LENGTH = 60    # Số nến 5m cho mô hình dãy
TRAINING_SPLIT = 0.7  # 70% training data
VALIDATION_SPLIT = 0.15  # 15% validation data
TEST_SPLIT = 0.15  # 15% test data
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Continuous Training Configuration
CONTINUOUS_TRAINING = True  # Enable continuous model retraining
TRAINING_SCHEDULE = {
    "frequency": "custom",  # Options: 'hourly', 'daily', 'weekly', 'custom'
    "interval_minutes": 30,  # Train every 30 minutes
    "hour": 2,            # For daily/weekly: hour of day (0-23) to train
    "minute": 30,         # Minute of hour to train
    "day_of_week": 1      # For weekly: day of week (0=Monday, 6=Sunday)
}
MINIMUM_NEW_DATA_POINTS = 30   # Số nến 1-phút tối thiểu cần có để huấn luyện lại (30 phút)

# Chunked Training Configuration
CHUNK_BY_MONTHS = True  # Enable chunked training by months
MAX_CHUNK_SIZE = 5000   # Giảm kích thước chunk từ 10000 xuống 5000 để tăng tốc độ training

# Model Paths
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")

# Prediction Engine Configuration
CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence for definitive prediction
VALIDITY_MINUTES = 15  # How long predictions are valid for

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Classes for model output
CLASSES = ["SHORT", "NEUTRAL", "LONG"]
