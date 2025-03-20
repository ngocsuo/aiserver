"""
Configuration settings for the ETHUSDT prediction system.
"""
import os
from datetime import datetime

# Binance API Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Trading Symbol and Timeframes
SYMBOL = "ETHUSDT"
TIMEFRAMES = {
    "primary": "5m",
    "secondary": ["30m", "4h"]
}

# Data Collection Configuration
LOOKBACK_PERIODS = 5000  # Number of candles to collect initially
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
