"""
Configuration settings for the ETHUSDT prediction system.
"""
import os
from datetime import datetime

# Binance API Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Feature flags
USE_REAL_API = True  # Set to True to use real Binance API instead of mock data
DEBUG_MODE = True     # Enable additional logging and debug information
FORCE_MOCK_DATA = False  # Set to False to use real data with configured proxy

# Trading Symbol and Timeframes
SYMBOL = "ETHUSDT"
TIMEFRAMES = {
    "primary": "1m",   # Cập nhật từ 5m sang 1m theo yêu cầu
    "secondary": ["15m", "1h", "4h"]
}

# Data Collection Configuration
LOOKBACK_PERIODS = 5000  # Number of candles to collect initially

# Cấu hình thời gian dữ liệu
DATA_RANGE_OPTIONS = {
    "realtime": 30,         # 30 ngày dữ liệu cho biểu đồ thời gian thực
    "training": 365,        # 12 tháng (365 ngày) dữ liệu cho huấn luyện mặc định
    "max_history": 1200     # Tối đa khoảng 3-4 năm dữ liệu
}

# Tính ngày bắt đầu dữ liệu huấn luyện (12 tháng gần nhất)
from datetime import datetime, timedelta
today = datetime.now()
DEFAULT_TRAINING_START_DATE = (today - timedelta(days=DATA_RANGE_OPTIONS["training"])).strftime("%Y-%m-%d")

# Sử dụng dữ liệu 12 tháng gần nhất thay vì từ 2022
HISTORICAL_START_DATE = DEFAULT_TRAINING_START_DATE  # Start date for historical data training

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
    "vwap": {"window": 14}
}

# Data Labeling Configuration
PRICE_MOVEMENT_THRESHOLD = 0.003  # 0.3% move for labeling
TARGET_PNL_THRESHOLD = 5  # $5 for alternative labeling
PREDICTION_WINDOW = 60  # Số nến 1m (1 giờ) để dự đoán trong tương lai

# Model Configuration
SEQUENCE_LENGTH = 120  # Số nến 1m cho mô hình dãy (tăng từ 60 nến 5m lên 120 nến 1m)
TRAINING_SPLIT = 0.7  # 70% training data
VALIDATION_SPLIT = 0.15  # 15% validation data
TEST_SPLIT = 0.15  # 15% test data
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

# Continuous Training Configuration
CONTINUOUS_TRAINING = True  # Enable continuous model retraining
TRAINING_SCHEDULE = {
    "frequency": "daily",  # Options: 'hourly', 'daily', 'weekly'
    "hour": 2,            # For daily/weekly: hour of day (0-23) to train
    "minute": 30,         # Minute of hour to train
    "day_of_week": 1      # For weekly: day of week (0=Monday, 6=Sunday)
}
MINIMUM_NEW_DATA_POINTS = 1440  # Số nến 1-phút tối thiểu cần có để huấn luyện lại (= 24h)

# Chunked Training Configuration
CHUNK_BY_MONTHS = True  # Enable chunked training by months
MAX_CHUNK_SIZE = 10000  # Maximum number of candles per chunk

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
