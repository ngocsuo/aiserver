"""
Configuration settings for the ETHUSDT prediction system.
"""
import os
from datetime import datetime, timedelta

# Binance API settings
SYMBOL = "ETHUSDT"
PRIMARY_TIMEFRAME = "5m"
SECONDARY_TIMEFRAME = "1m"
TIMEFRAMES = {
    "primary": PRIMARY_TIMEFRAME,
    "secondary": [SECONDARY_TIMEFRAME]
}
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
LOOKBACK_PERIODS = 1000
HISTORICAL_DAYS = 30  # For initial training data

# Model settings
PREDICTION_HORIZON = {
    "short": "10m",
    "medium": "30m",
    "long": "1h"
}
PREDICTION_WINDOW = 10  # Số period dự đoán
VALIDITY_MINUTES = 30   # Thời gian hiệu lực dự đoán (phút)

# Model training settings
TRAINING_INTERVAL = 1800  # In seconds (30 minutes)
SEQUENCE_LENGTH = 60  # Number of periods to use for sequence models
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
VALIDATION_SPLIT = VALIDATION_SIZE  # Alias for compatibility
TEST_SPLIT = TEST_SIZE  # Alias for compatibility
BATCH_SIZE = 32
EPOCHS = 20
CONFIDENCE_THRESHOLD = 0.6
PRICE_MOVEMENT_THRESHOLD = 0.1  # Phần trăm thay đổi giá tối thiểu (1%)
MODEL_VERSION = "v1.0"
CLASSES = ["UP", "DOWN", "NEUTRAL"]  # Các lớp dự đoán
EARLY_STOPPING_PATIENCE = 5
DEFAULT_TRAINING_START_DATE = "2022-01-01"
SECONDARY_TIMEFRAME = "1m"  # Thời gian thứ cấp để lấy dữ liệu

# Data update settings
DATA_UPDATE_INTERVAL = 60  # In seconds

# System paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = MODELS_DIR  # Alias for compatibility

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Proxy settings
USE_PROXY = True
PROXY_ROTATION_INTERVAL = 600  # In seconds (10 minutes)
PROXY_URL = "64.176.51.107:3128:hvnteam:matkhau123"  # Cấu hình proxy

# Additional configuration
HISTORICAL_START_DATE = "2022-01-01"
USE_REAL_API = True
DEFAULT_TIMEFRAME = PRIMARY_TIMEFRAME
DEFAULT_PREDICTION_HORIZON = "medium"
PREDICTION_SETTINGS = {
    "confidence_threshold": 0.65,
    "minimum_samples": 100,
    "use_ensemble": True
}

# Advanced settings
DEBUG_MODE = True
ENABLE_LOGGING = True
LOG_LEVEL = "INFO"

# Training settings
TRAINING_SCHEDULE = {
    "frequency": "hourly",
    "interval": 1,  # Train every 1 hour
    "start_hour": 0,
    "end_hour": 23,
    "days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
}
CONTINUOUS_TRAINING = True
MINIMUM_NEW_DATA_POINTS = 100
CHUNK_BY_MONTHS = True
UPDATE_INTERVAL = 60  # In seconds

# Feature engineering settings
TECHNICAL_FEATURES = [
    "sma_fast", "sma_medium", "sma_slow",
    "ema_fast", "ema_medium", "ema_slow",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bbands_upper", "bbands_middle", "bbands_lower",
    "obv", "atr", "adx"
]

CHART_PATTERN_FEATURES = [
    "doji", "hammer", "engulfing_bullish", "engulfing_bearish",
    "three_white_soldiers", "three_black_crows"
]

# Trading settings
DEFAULT_TRADING_CONFIG = {
    "strategy": "AI Prediction",
    "risk_level": "Medium",
    "leverage": 5,
    "position_size": 0.1,  # 10% of available balance
    "take_profit": 1.5,  # 1.5%
    "stop_loss": 0.5,  # 0.5%
    "trailing_stop": 0.3,  # 0.3%
    "max_positions": 3
}

TARGET_PNL_THRESHOLD = 0.005  # 0.5% threshold for profit target
TRADING_SETTINGS = DEFAULT_TRADING_CONFIG  # Alias for compatibility

# If you're using PostgreSQL database
DB_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": os.environ.get("PGPORT", "5432"),
    "database": os.environ.get("PGDATABASE", "trading_db"),
    "user": os.environ.get("PGUSER", "postgres"),
    "password": os.environ.get("PGPASSWORD", "password")
}

# Get the Database URL from environment variable or construct it
DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)