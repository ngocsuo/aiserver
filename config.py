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
    "primary": "5m",
    "secondary": ["30m", "4h"]
}

# Data Collection Configuration
LOOKBACK_PERIODS = 5000  # Number of candles to collect initially
HISTORICAL_START_DATE = "2022-01-01"  # Start date for historical data training
ENABLE_BACKTESTING = True  # Enable backtesting functionality
BACKTEST_PERIOD_START = "2022-01-01"  # Start date for backtesting
BACKTEST_PERIOD_END = "2022-12-31"   # End date for backtesting
UPDATE_INTERVAL = 60  # Seconds between data updates

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
PREDICTION_WINDOW = 12  # Number of 5m candles (1 hour) to look ahead

# Model Configuration
SEQUENCE_LENGTH = 60  # Number of candles for sequence models
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
MINIMUM_NEW_DATA_POINTS = 288  # Minimum new 5-min candles required for retraining (equals 24h)

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
