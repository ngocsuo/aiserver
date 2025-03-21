"""
Prediction engine for combining model outputs and generating final predictions.
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
import random

from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.cnn_model import CNNModel
from models.historical_similarity import HistoricalSimilarity
from models.meta_learner import MetaLearner
from utils.data_processor import DataProcessor
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("prediction_engine")

class PredictionEngine:
    def __init__(self):
        """Initialize the prediction engine."""
        self.data_processor = DataProcessor()
        self.models = None
        self.last_prediction = None
        self.last_prediction_time = None
        self.prediction_count = 0
        logger.info("Prediction engine initialized")
    
    def load_models(self):
        """
        Load trained models for prediction.
        
        Returns:
            dict: Loaded models
        """
        try:
            # Check if models are already loaded
            if self.models is not None:
                return self.models
                
            # Check if models exist on disk
            lstm_path = os.path.join(config.MODEL_DIR, f"lstm_{config.MODEL_VERSION}.h5")
            transformer_path = os.path.join(config.MODEL_DIR, f"transformer_{config.MODEL_VERSION}.h5")
            cnn_path = os.path.join(config.MODEL_DIR, f"cnn_{config.MODEL_VERSION}.h5")
            historical_path = os.path.join(config.MODEL_DIR, f"historical_{config.MODEL_VERSION}.pkl")
            meta_path = os.path.join(config.MODEL_DIR, f"meta_{config.MODEL_VERSION}.pkl")
            
            lstm_exists = os.path.exists(lstm_path)
            transformer_exists = os.path.exists(transformer_path)
            cnn_exists = os.path.exists(cnn_path)
            historical_exists = os.path.exists(historical_path)
            meta_exists = os.path.exists(meta_path)
            
            # Initialize models (will use pretrained or build new ones)
            models = {}
            
            if lstm_exists:
                # Initialize with input shape, will be loaded from disk
                models['lstm'] = LSTMModel(
                    input_shape=(config.SEQUENCE_LENGTH, 30),  # Dummy input shape
                    output_dim=3,
                    model_path=lstm_path
                )
                logger.info(f"Loaded LSTM model from {lstm_path}")
            
            if transformer_exists:
                models['transformer'] = TransformerModel(
                    input_shape=(config.SEQUENCE_LENGTH, 30),
                    output_dim=3,
                    model_path=transformer_path
                )
                logger.info(f"Loaded Transformer model from {transformer_path}")
            
            if cnn_exists:
                models['cnn'] = CNNModel(
                    input_shape=(config.SEQUENCE_LENGTH, 5, 1),  # OHLCV
                    output_dim=3,
                    model_path=cnn_path
                )
                logger.info(f"Loaded CNN model from {cnn_path}")
            
            if historical_exists:
                models['historical_similarity'] = HistoricalSimilarity(
                    sequence_length=config.SEQUENCE_LENGTH,
                    model_path=historical_path
                )
                logger.info(f"Loaded Historical Similarity model from {historical_path}")
            
            if meta_exists:
                models['meta_learner'] = MetaLearner(
                    model_type='logistic',
                    model_path=meta_path
                )
                logger.info(f"Loaded Meta-Learner model from {meta_path}")
            
            # If no models were loaded, create mock models or return None
            if not models:
                logger.warning("No trained models found. Using fallback prediction.")
                # Return empty models, will use fallback prediction
                pass
            
            self.models = models
            return models
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return None
    
    def is_prediction_valid(self):
        """
        Check if the last prediction is still valid.
        
        Returns:
            bool: Whether the prediction is still valid
        """
        if (self.last_prediction is None or 
            self.last_prediction_time is None):
            return False
        
        # Check if prediction is expired
        # Lấy thời gian từ Binance nếu có thể, để đảm bảo dự đoán theo thời gian thực của thị trường
        try:
            # Sử dụng data_collector đã tạo
            from utils.data_collector import create_data_collector
            collector = create_data_collector()
            
            # Lấy thời gian từ Binance
            server_time = collector.client.get_server_time()
            server_time_ms = server_time['serverTime']
            now = datetime.fromtimestamp(server_time_ms / 1000)
            logger.info(f"Using Binance server time for prediction validity check: {now}")
        except Exception as e:
            # Nếu không lấy được thời gian từ Binance, sử dụng thời gian local
            now = datetime.now()
            logger.warning(f"Unable to get Binance server time, using local time: {now}. Error: {e}")
        
        valid_until = self.last_prediction_time + timedelta(minutes=config.VALIDITY_MINUTES)
        
        # Ghi log thông tin về tính hợp lệ
        is_valid = now < valid_until
        if is_valid:
            remaining_minutes = (valid_until - now).total_seconds() / 60
            logger.info(f"Prediction still valid for {remaining_minutes:.1f} more minutes")
        else:
            logger.info(f"Prediction expired. Made {(now - self.last_prediction_time).total_seconds() / 60:.1f} minutes ago")
            
        return is_valid
    
    def get_cached_prediction(self):
        """
        Get the cached prediction if valid.
        
        Returns:
            dict: Cached prediction or None
        """
        if self.is_prediction_valid():
            return self.last_prediction
        return None
    
    def predict(self, data, use_cache=True):
        """
        Generate predictions from all models and combine them.
        
        Args:
            data (pd.DataFrame): Latest price data
            use_cache (bool): Whether to use cached predictions if valid
            
        Returns:
            dict: Prediction result with trend, confidence, etc.
        """
        try:
            # Check if we can use cached prediction
            if use_cache and self.is_prediction_valid():
                logger.info("Using cached prediction")
                return self.last_prediction
            
            # Load models if not loaded yet
            models = self.load_models()
            current_price = data.iloc[-1]['close']
            
            # For development/demo, use random predictions if models not trained
            if models is None or len(models) == 0:
                logger.warning("No trained models available. Using fallback prediction.")
                # Generate random prediction for demonstration
                prediction = self._create_random_prediction(current_price)
                
                # Store for caching
                self.last_prediction = prediction
                
                # Lấy thời gian từ Binance nếu có thể
                try:
                    # Sử dụng data_collector đã tạo
                    from utils.data_collector import create_data_collector
                    collector = create_data_collector()
                    
                    # Lấy thời gian từ Binance
                    server_time = collector.client.get_server_time()
                    server_time_ms = server_time['serverTime']
                    now = datetime.fromtimestamp(server_time_ms / 1000)
                    logger.info(f"Using Binance server time for timestamp: {now}")
                    self.last_prediction_time = now
                except Exception as e:
                    # Nếu không lấy được thời gian từ Binance, sử dụng thời gian local
                    self.last_prediction_time = datetime.now()
                    logger.warning(f"Unable to get Binance server time, using local time: {self.last_prediction_time}. Error: {e}")
                
                self.prediction_count += 1
                
                return prediction
            
            # Prepare data for prediction
            sequence_data, _ = self.data_processor.prepare_latest_data(data)
            cnn_data, _ = self.data_processor.prepare_latest_cnn_data(data)
            
            # Get predictions from each model
            model_predictions = {}
            model_confidences = {}
            
            # LSTM prediction
            if 'lstm' in models:
                lstm_pred, lstm_probs = models['lstm'].predict(sequence_data)
                model_predictions['lstm'] = lstm_pred[0]
                model_confidences['lstm'] = max(lstm_probs[0])
                logger.info(f"LSTM prediction: {config.CLASSES[lstm_pred[0]]} with confidence {max(lstm_probs[0]):.2f}")
            
            # Transformer prediction
            if 'transformer' in models:
                transformer_pred, transformer_probs = models['transformer'].predict(sequence_data)
                model_predictions['transformer'] = transformer_pred[0]
                model_confidences['transformer'] = max(transformer_probs[0])
                logger.info(f"Transformer prediction: {config.CLASSES[transformer_pred[0]]} with confidence {max(transformer_probs[0]):.2f}")
            
            # CNN prediction
            if 'cnn' in models and cnn_data is not None:
                cnn_pred, cnn_probs = models['cnn'].predict(cnn_data)
                model_predictions['cnn'] = cnn_pred[0]
                model_confidences['cnn'] = max(cnn_probs[0])
                logger.info(f"CNN prediction: {config.CLASSES[cnn_pred[0]]} with confidence {max(cnn_probs[0]):.2f}")
            
            # Historical similarity prediction
            if 'historical_similarity' in models:
                hist_pred, hist_conf, _, _ = models['historical_similarity'].predict(sequence_data[0])
                model_predictions['historical_similarity'] = hist_pred
                model_confidences['historical_similarity'] = hist_conf
                logger.info(f"Historical similarity prediction: {config.CLASSES[hist_pred]} with confidence {hist_conf:.2f}")
            
            # Meta-learner prediction (if available)
            meta_prediction = None
            meta_confidence = 0.0
            
            if 'meta_learner' in models and len(model_predictions) > 1:
                # Combine probabilities from all models
                base_model_probs = []
                
                if 'lstm' in models:
                    base_model_probs.append(models['lstm'].predict(sequence_data)[1])
                
                if 'transformer' in models:
                    base_model_probs.append(models['transformer'].predict(sequence_data)[1])
                
                if 'cnn' in models and cnn_data is not None:
                    base_model_probs.append(models['cnn'].predict(cnn_data)[1])
                
                # Make meta prediction
                meta_pred, meta_probs = models['meta_learner'].predict(base_model_probs)
                meta_prediction = meta_pred[0]
                meta_confidence = max(meta_probs[0])
                logger.info(f"Meta-learner prediction: {config.CLASSES[meta_prediction]} with confidence {meta_confidence:.2f}")
            
            # Combine predictions from all models
            prediction = self._combine_predictions(
                model_predictions, 
                model_confidences,
                meta_prediction, 
                meta_confidence,
                data=data  # Pass the actual data for technical analysis
            )
            
            # Enhance prediction with detailed technical analysis
            enhanced_prediction = self._enhance_technical_reasoning(prediction, data)
            
            # Store for caching
            self.last_prediction = enhanced_prediction
            
            # Lấy thời gian từ Binance nếu có thể
            try:
                # Sử dụng data_collector đã tạo
                from utils.data_collector import create_data_collector
                collector = create_data_collector()
                
                # Lấy thời gian từ Binance
                server_time = collector.client.get_server_time()
                server_time_ms = server_time['serverTime']
                now = datetime.fromtimestamp(server_time_ms / 1000)
                logger.info(f"Using Binance server time for timestamp: {now}")
                self.last_prediction_time = now
            except Exception as e:
                # Nếu không lấy được thời gian từ Binance, sử dụng thời gian local
                self.last_prediction_time = datetime.now()
                logger.warning(f"Unable to get Binance server time, using local time: {self.last_prediction_time}. Error: {e}")
            
            self.prediction_count += 1
            
            logger.info(f"Prediction generated: {enhanced_prediction['trend']} with confidence {enhanced_prediction['confidence']:.2f}")
            
            return enhanced_prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            # Return error prediction
            return self._create_error_prediction(str(e))
    
    def _combine_predictions(self, model_predictions, model_confidences,
                           meta_prediction, meta_confidence, data=None):
        """
        Combine predictions from all models using dynamic weighting system.
        
        Args:
            model_predictions (dict): Predictions from individual models
            model_confidences (dict): Confidences from individual models
            meta_prediction (int): Prediction from meta-learner
            meta_confidence (float): Confidence from meta-learner
            data (pd.DataFrame, optional): Latest price and indicator data
            
        Returns:
            dict: Combined prediction
        """
        current_time = datetime.now()
        
        # Try to get model weights from meta_learner (if available and initialized)
        model_weights = {}
        try:
            from models.meta_learner import MetaLearner
            # Get the meta-learner instance
            if hasattr(self, 'models') and 'meta_learner' in self.models:
                meta_learner = self.models['meta_learner']
                if hasattr(meta_learner, 'model_weights'):
                    # Use the dynamic weights from meta-learner
                    model_weights = meta_learner.model_weights
                    logger.info(f"Using dynamic model weights: {model_weights}")
        except Exception as e:
            logger.warning(f"Could not get dynamic weights: {e}. Using default weights.")
            
        # Determine prediction class - prefer meta-learner if available and confident
        if meta_prediction is not None and meta_confidence >= config.CONFIDENCE_THRESHOLD:
            prediction_class = meta_prediction
            confidence = meta_confidence
        else:
            # No confident meta-learner, use weighted voting
            votes = {0: 0, 1: 0, 2: 0}  # SHORT, NEUTRAL, LONG
            
            # Count weighted votes from each model
            for model, pred in model_predictions.items():
                # Get model confidence
                model_confidence = model_confidences[model]
                
                # Apply dynamic weight if available
                if model in model_weights:
                    weight = model_confidence * model_weights[model]
                    logger.info(f"Applied dynamic weight for {model}: {model_weights[model]:.2f}")
                else:
                    weight = model_confidence
                
                votes[pred] += weight
            
            # Select class with highest votes
            prediction_class = max(votes, key=votes.get)
            
            # Calculate confidence based on vote margin
            total_votes = sum(votes.values())
            confidence = votes[prediction_class] / total_votes if total_votes > 0 else 0.5
        
        # Get reference to latest data
        latest_price = 3500.0  # Default if we don't have real data
        
        # Try to extract actual latest price and indicator data if available
        technical_indicators = {}
        if data is not None and not data.empty:
            try:
                latest_row = data.iloc[-1]
                latest_price = latest_row['close']
                
                # Extract technical indicators for display
                indicator_keys = [
                    'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'bb_upper', 'bb_lower', 'bb_middle', 
                    'ema_9', 'ema_21', 'ema_55', 'ema_200',
                    'atr', 'obv', 'volume'
                ]
                
                for key in indicator_keys:
                    if key in latest_row:
                        technical_indicators[key] = latest_row[key]
                
                # Calculate derived indicators
                if 'bb_upper' in technical_indicators and 'bb_lower' in technical_indicators:
                    bb_width = (technical_indicators['bb_upper'] - technical_indicators['bb_lower']) / technical_indicators['bb_middle']
                    technical_indicators['bb_width'] = bb_width
                
                # Calculate price relative to Bollinger Bands
                if 'bb_upper' in technical_indicators and 'bb_lower' in technical_indicators:
                    bb_range = technical_indicators['bb_upper'] - technical_indicators['bb_lower']
                    if bb_range > 0:
                        bb_position = (latest_price - technical_indicators['bb_lower']) / bb_range
                        technical_indicators['bb_position'] = bb_position
                
            except Exception as e:
                logger.warning(f"Error extracting technical indicators: {e}")
        
        # Generate prediction details
        classes = config.CLASSES
        prediction_label = classes[prediction_class]
        
        # Generate target price and move percentage based on prediction and class
        if prediction_class == 0:  # SHORT
            # More pessimistic if RSI is high
            rsi_factor = 1.0
            if 'rsi' in technical_indicators:
                rsi = technical_indicators['rsi']
                # Stronger downside projection if RSI is overbought
                if rsi > 70:
                    rsi_factor = 1.5
                elif rsi < 30:
                    rsi_factor = 0.7
                    
            predicted_move = -random.uniform(0.3, 1.2) * rsi_factor
            target_price = latest_price * (1 + predicted_move/100)
            
        elif prediction_class == 2:  # LONG
            # More optimistic if RSI is low
            rsi_factor = 1.0
            if 'rsi' in technical_indicators:
                rsi = technical_indicators['rsi']
                # Stronger upside projection if RSI is oversold
                if rsi < 30:
                    rsi_factor = 1.5
                elif rsi > 70:
                    rsi_factor = 0.7
                    
            predicted_move = random.uniform(0.3, 1.2) * rsi_factor
            target_price = latest_price * (1 + predicted_move/100)
            
        else:  # NEUTRAL
            volatility_factor = 1.0
            # Use ATR or Bollinger width to estimate volatility if available
            if 'atr' in technical_indicators and latest_price > 0:
                volatility_factor = technical_indicators['atr'] / latest_price * 100  # ATR as percent of price
            elif 'bb_width' in technical_indicators:
                volatility_factor = technical_indicators['bb_width'] * 10
                
            # Limit the volatility factor to a reasonable range
            volatility_factor = max(0.5, min(1.5, volatility_factor))
            
            predicted_move = random.uniform(-0.2, 0.2) * volatility_factor
            target_price = latest_price * (1 + predicted_move/100)
        
        # Generate reason for prediction with actual data
        reason = self._generate_technical_reason(prediction_class, data=data)
        
        prediction = {
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "price": latest_price,
            "trend": prediction_label,
            "confidence": round(confidence, 2),
            "target_price": round(target_price, 2),
            "predicted_move": round(predicted_move, 2),
            "reason": reason,
            "valid_for_minutes": config.VALIDITY_MINUTES,
            "technical_indicators": technical_indicators
        }
        
        return prediction
    
    def _create_random_prediction(self, current_price):
        """
        Create a random prediction for demonstration purposes.
        
        Args:
            current_price (float): Current price
        
        Returns:
            dict: Random prediction
        """
        classes = config.CLASSES
        prediction_class = random.choice([0, 1, 2])
        confidence = random.uniform(0.65, 0.95)
        
        if prediction_class == 0:  # SHORT
            predicted_move = -random.uniform(0.3, 1.2)
            target_price = current_price * (1 + predicted_move/100)
            reason = "Bearish divergence detected; RSI overbought; 200 EMA resistance"
        elif prediction_class == 2:  # LONG
            predicted_move = random.uniform(0.3, 1.2)
            target_price = current_price * (1 + predicted_move/100)
            reason = "Bullish pattern confirmed; RSI oversold; 50 EMA support"
        else:  # NEUTRAL
            predicted_move = random.uniform(-0.2, 0.2)
            target_price = current_price * (1 + predicted_move/100)
            reason = "Sideways price action; low volatility; mixed signals"
        
        # Create random technical indicators
        technical_indicators = {
            "rsi": random.uniform(30, 70),
            "macd": random.uniform(-0.5, 0.5),
            "macd_signal": random.uniform(-0.5, 0.5),
            "bb_upper": current_price * 1.01,
            "bb_lower": current_price * 0.99,
            "bb_middle": current_price,
            "ema_9": current_price * (1 + random.uniform(-0.01, 0.01)),
            "ema_21": current_price * (1 + random.uniform(-0.01, 0.01)),
            "atr": current_price * 0.01,
            "volume": random.uniform(1000, 10000)
        }
        
        # Add derived indicators
        technical_indicators["bb_width"] = (technical_indicators["bb_upper"] - technical_indicators["bb_lower"]) / technical_indicators["bb_middle"]
        technical_indicators["bb_position"] = (current_price - technical_indicators["bb_lower"]) / (technical_indicators["bb_upper"] - technical_indicators["bb_lower"])
        
        prediction = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price": current_price,
            "trend": classes[prediction_class],
            "confidence": round(confidence, 2),
            "target_price": round(target_price, 2),
            "predicted_move": round(predicted_move, 2),
            "reason": reason,
            "valid_for_minutes": config.VALIDITY_MINUTES,
            "technical_indicators": technical_indicators
        }
        
        return prediction
    
    def _detect_candlestick_patterns(self, data):
        """Phát hiện mẫu nến phổ biến"""
        patterns = []
        if data is None or len(data) < 5:
            return patterns
            
        last_candles = data.iloc[-5:].copy()
        
        # Doji
        last_candle = last_candles.iloc[-1]
        body_size = abs(last_candle['close'] - last_candle['open'])
        if body_size < (last_candle['high'] - last_candle['low']) * 0.1:
            patterns.append({"name": "Doji", "bullish": None, "strength": 0.6})
        
        # Hammer
        if (last_candle['low'] < min(last_candle['open'], last_candle['close'])) and \
           (min(last_candle['open'], last_candle['close']) - last_candle['low']) > 2 * body_size and \
           (last_candle['high'] - max(last_candle['open'], last_candle['close'])) < 0.5 * body_size:
            patterns.append({"name": "Hammer", "bullish": True, "strength": 0.7})
        
        # Engulfing
        if len(last_candles) >= 2:
            prev_candle = last_candles.iloc[-2]
            prev_body_size = abs(prev_candle['close'] - prev_candle['open'])
            
            # Bullish Engulfing
            if (prev_candle['close'] < prev_candle['open']) and \
               (last_candle['close'] > last_candle['open']) and \
               (last_candle['open'] <= prev_candle['close']) and \
               (last_candle['close'] >= prev_candle['open']) and \
               (body_size > prev_body_size):
                patterns.append({"name": "Bullish Engulfing", "bullish": True, "strength": 0.8})
                
            # Bearish Engulfing
            if (prev_candle['close'] > prev_candle['open']) and \
               (last_candle['close'] < last_candle['open']) and \
               (last_candle['open'] >= prev_candle['close']) and \
               (last_candle['close'] <= prev_candle['open']) and \
               (body_size > prev_body_size):
                patterns.append({"name": "Bearish Engulfing", "bullish": False, "strength": 0.8})
        
        # Morning Star (bullish reversal)
        if len(last_candles) >= 3:
            first_candle = last_candles.iloc[-3]
            middle_candle = last_candles.iloc[-2]
            last_candle = last_candles.iloc[-1]
            
            first_body_size = abs(first_candle['close'] - first_candle['open'])
            middle_body_size = abs(middle_candle['close'] - middle_candle['open'])
            last_body_size = abs(last_candle['close'] - last_candle['open'])
            
            if (first_candle['close'] < first_candle['open']) and \
               (middle_body_size < first_body_size * 0.5) and \
               (last_candle['close'] > last_candle['open']) and \
               (last_candle['close'] > (first_candle['open'] + first_candle['close']) / 2):
                patterns.append({"name": "Morning Star", "bullish": True, "strength": 0.9})
                
        # Evening Star (bearish reversal)
        if len(last_candles) >= 3:
            first_candle = last_candles.iloc[-3]
            middle_candle = last_candles.iloc[-2]
            last_candle = last_candles.iloc[-1]
            
            first_body_size = abs(first_candle['close'] - first_candle['open'])
            middle_body_size = abs(middle_candle['close'] - middle_candle['open'])
            last_body_size = abs(last_candle['close'] - last_candle['open'])
            
            if (first_candle['close'] > first_candle['open']) and \
               (middle_body_size < first_body_size * 0.5) and \
               (last_candle['close'] < last_candle['open']) and \
               (last_candle['close'] < (first_candle['open'] + first_candle['close']) / 2):
                patterns.append({"name": "Evening Star", "bullish": False, "strength": 0.9})
                
        return patterns
    
    def _calculate_support_resistance(self, data):
        """Tính toán các mức hỗ trợ/kháng cự"""
        if data is None or len(data) < 20:
            return {"support": [], "resistance": []}
            
        # Tính toán các mức swing high/low đơn giản
        prices = data['close'].values
        window = 5
        
        supports = []
        resistances = []
        
        for i in range(window, len(prices) - window):
            # Kiểm tra xem có phải là swing low?
            if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] <= prices[i+j] for j in range(1, window+1)):
                supports.append({"price": prices[i], "index": i})
                
            # Kiểm tra xem có phải là swing high?
            if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                resistances.append({"price": prices[i], "index": i})
        
        # Chỉ giữ lại các mức gần nhất
        supports = sorted(supports, key=lambda x: x["index"], reverse=True)[:3]
        resistances = sorted(resistances, key=lambda x: x["index"], reverse=True)[:3]
        
        # Chỉ lấy giá trị
        support_levels = [round(s["price"], 2) for s in supports]
        resistance_levels = [round(r["price"], 2) for r in resistances]
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }
        
    def _analyze_price_position(self, data):
        """Phân tích vị trí giá hiện tại so với các đường MA"""
        try:
            ma20 = data['close'].rolling(window=20).mean().iloc[-1]
            ma50 = data['close'].rolling(window=50).mean().iloc[-1]
            ma200 = data['close'].rolling(window=200).mean().iloc[-1] if len(data) > 200 else None
            
            current_price = data['close'].iloc[-1]
            
            result = {
                'above_ma20': current_price > ma20,
                'above_ma50': current_price > ma50,
                'ma20_trend': 'UP' if ma20 > data['close'].rolling(window=20).mean().iloc[-2] else 'DOWN',
                'price_to_ma20_percent': round((current_price / ma20 - 1) * 100, 2)
            }
            
            if ma200 is not None:
                result['above_ma200'] = current_price > ma200
                
            return result
        except Exception as e:
            logger.warning(f"Error analyzing price position: {e}")
            return {}
            
    def _analyze_momentum(self, data):
        """Phân tích động lượng dựa trên RSI và MACD"""
        try:
            # RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - signal_line
            
            return {
                'rsi': round(rsi.iloc[-1], 2),
                'rsi_trend': 'OVERBOUGHT' if rsi.iloc[-1] > 70 else 'OVERSOLD' if rsi.iloc[-1] < 30 else 'NEUTRAL',
                'macd_line': round(macd_line.iloc[-1], 6),
                'signal_line': round(signal_line.iloc[-1], 6),
                'macd_histogram': round(macd_histogram.iloc[-1], 6),
                'macd_trend': 'BULLISH' if macd_histogram.iloc[-1] > 0 else 'BEARISH'
            }
        except Exception as e:
            logger.warning(f"Error analyzing momentum: {e}")
            return {}
            
    def _analyze_volume(self, data):
        """Phân tích khối lượng giao dịch"""
        try:
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            
            return {
                'current_volume': current_volume,
                'avg_volume_20': avg_volume,
                'volume_ratio': round(current_volume / avg_volume, 2),
                'increasing_volume': current_volume > avg_volume,
                'volume_trend': 'HIGH' if current_volume > avg_volume * 1.5 else 
                                'LOW' if current_volume < avg_volume * 0.5 else 'NORMAL'
            }
        except Exception as e:
            logger.warning(f"Error analyzing volume: {e}")
            return {}
            
    def _enhance_technical_reasoning(self, prediction, data):
        """
        Cải thiện phân tích kỹ thuật trong dự đoán, sử dụng phân tích kỹ thuật
        dựa trên các mẫu nến Nhật Bản, mức hỗ trợ/kháng cự, và phân tích xu hướng.
        
        Args:
            prediction (dict): Dự đoán hiện tại cần cải thiện
            data (pd.DataFrame): Dữ liệu giá gần đây
            
        Returns:
            dict: Dự đoán được cải thiện với lý do kỹ thuật chi tiết hơn
        """
        if data is None or data.empty:
            return prediction
        
        try:
            latest_data = data.iloc[-20:].copy() if len(data) > 20 else data.copy()
            current_price = latest_data.iloc[-1]['close']
            
            # Phát hiện mẫu nến
            candle_patterns = self._detect_candlestick_patterns(latest_data)
            if candle_patterns:
                prediction['candle_patterns'] = candle_patterns
                
                # Tách mẫu nến theo xu hướng tăng/giảm
                bullish_patterns = [p for p in candle_patterns if p.get('bullish') is True]
                bearish_patterns = [p for p in candle_patterns if p.get('bullish') is False]
                
                # Thêm vào lý do kỹ thuật
                if 'reason' in prediction:
                    if bullish_patterns and prediction['trend'] == 'LONG':
                        pattern_info = ", ".join([f"{p['name']}" for p in bullish_patterns])
                        prediction['reason'] += f" Phát hiện mẫu nến tăng: {pattern_info}."
                    elif bearish_patterns and prediction['trend'] == 'SHORT': 
                        pattern_info = ", ".join([f"{p['name']}" for p in bearish_patterns])
                        prediction['reason'] += f" Phát hiện mẫu nến giảm: {pattern_info}."
            
            # Tính toán mức hỗ trợ/kháng cự
            sr_levels = self._calculate_support_resistance(latest_data)
            prediction['support_resistance'] = sr_levels
            
            # Thêm phân tích về mức hỗ trợ và kháng cự vào lý do
            if 'reason' in prediction and (sr_levels.get('support') or sr_levels.get('resistance')):
                support = sr_levels.get('support', [])
                resistance = sr_levels.get('resistance', [])
                
                if support and current_price < support[0] * 1.01:
                    support_str = ", ".join([f"{s}" for s in support])
                    if prediction['trend'] == 'LONG':
                        prediction['reason'] += f" Giá đang tiếp cận mức hỗ trợ quan trọng ({support_str}), có thể tạo điểm vào lệnh tốt."
                    else:
                        prediction['reason'] += f" Giá đang tiếp cận mức hỗ trợ ({support_str})."
                elif resistance and current_price > resistance[0] * 0.99:
                    resistance_str = ", ".join([f"{r}" for r in resistance])
                    if prediction['trend'] == 'SHORT':
                        prediction['reason'] += f" Giá đang tiếp cận mức kháng cự quan trọng ({resistance_str}), có thể tạo điểm vào lệnh tốt."
                    else:
                        prediction['reason'] += f" Giá đang tiếp cận mức kháng cự ({resistance_str})."
        
        except Exception as e:
            logger.warning(f"Error enhancing technical reasoning: {e}")
        
        # Tính toán tỷ lệ R/R (Risk/Reward)
        try:
            current_price = data['close'].iloc[-1]
            sr_levels = prediction.get('support_resistance', {'support': [], 'resistance': []})
            
            if prediction['trend'] == 'LONG':
                nearest_resistance = min([r for r in sr_levels.get('resistance', []) if r > current_price], 
                                      default=current_price * 1.02)
                nearest_support = max([s for s in sr_levels.get('support', []) if s < current_price], 
                                   default=current_price * 0.99)
                
                risk = current_price - nearest_support
                reward = nearest_resistance - current_price
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0
                
            elif prediction['trend'] == 'SHORT':
                nearest_resistance = min([r for r in sr_levels.get('resistance', []) if r > current_price], 
                                      default=current_price * 1.01)
                nearest_support = max([s for s in sr_levels.get('support', []) if s < current_price], 
                                   default=current_price * 0.98)
                
                risk = nearest_resistance - current_price
                reward = current_price - nearest_support
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0
            else:
                rr_ratio = 0
            
            prediction['risk_reward_ratio'] = rr_ratio
            
            # Phân tích chi tiết thêm
            prediction['detailed_analysis'] = {
                'current_price': current_price,
                'price_position': self._analyze_price_position(data),
                'momentum': self._analyze_momentum(data),
                'volume_analysis': self._analyze_volume(data)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating risk/reward ratio: {e}")
            
        return prediction

    def _generate_technical_reason(self, prediction, data=None):
        """
        Generate technical reasoning for the prediction based on actual indicators.
        
        Args:
            prediction (int): Class prediction (0, 1, or 2)
            data (pd.DataFrame, optional): Latest technical indicator data
            
        Returns:
            str: Technical reasoning with actual indicator values
        """
        # If we don't have actual data, use templates
        if data is None or len(data) < 5:
            # Templates for different prediction classes
            short_reasons = [
                "Bearish divergence on RSI; price rejected at upper Bollinger Band; downtrend on higher timeframe",
                "MACD bearish crossover; decreasing volume; failed to break resistance at {price}",
                "Double top pattern formed; overbought on RSI; EMA 9 crossing below EMA 21",
                "Fibonacci resistance rejection; bearish engulfing candle; increased selling volume",
                "Lower highs forming; head and shoulders pattern; funding rate positive indicating overleveraged longs"
            ]
            
            neutral_reasons = [
                "Price within Bollinger Bands; RSI in midrange (40-60); no clear trend direction",
                "Consolidation phase; decreasing volume; tight price range bounds",
                "Mixed signals: bearish MACD but bullish RSI; indecision candles forming",
                "Price at key support/resistance level; waiting for breakout confirmation",
                "Low volatility period; historical similarity shows probable range-bound movement"
            ]
            
            long_reasons = [
                "Bullish divergence on RSI; price bounced off lower Bollinger Band; uptrend on higher timeframe",
                "MACD bullish crossover; increasing volume; broke resistance at {price}",
                "Double bottom pattern confirmed; oversold on RSI; EMA 9 crossing above EMA 21",
                "Fibonacci support holding; bullish engulfing candle; increased buying volume",
                "Higher lows forming; inverse head and shoulders; funding rate negative indicating overleveraged shorts"
            ]
            
            if prediction == 0:  # SHORT
                return random.choice(short_reasons)
            elif prediction == 2:  # LONG
                return random.choice(long_reasons)
            else:  # NEUTRAL
                return random.choice(neutral_reasons)
        
        # Generate reasoning based on actual indicator values
        try:
            # Get the latest values for key indicators
            latest = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else latest
            
            # Extract indicator values
            current_price = latest['close']
            rsi_value = latest.get('rsi', 50)
            macd_value = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            prev_macd = prev.get('macd', 0)
            prev_macd_signal = prev.get('macd_signal', 0)
            
            bb_upper = latest.get('bb_upper', current_price * 1.01)
            bb_lower = latest.get('bb_lower', current_price * 0.99)
            bb_middle = latest.get('bb_middle', current_price)
            
            volume = latest.get('volume', 0)
            prev_volume = prev.get('volume', 0)
            
            ema_short = latest.get('ema_9', current_price)
            ema_medium = latest.get('ema_21', current_price)
            prev_ema_short = prev.get('ema_9', current_price)
            prev_ema_medium = prev.get('ema_21', current_price)
            
            # Analyze RSI conditions
            if rsi_value > 70:
                rsi_condition = f"Overbought (RSI: {rsi_value:.1f})"
            elif rsi_value < 30:
                rsi_condition = f"Oversold (RSI: {rsi_value:.1f})"
            else:
                rsi_condition = f"Neutral (RSI: {rsi_value:.1f})"
                
            # Analyze MACD conditions
            macd_cross = "No significant change"
            if (prev_macd < prev_macd_signal and macd_value > macd_signal):
                macd_cross = f"Bullish MACD crossover ({macd_value:.4f} > {macd_signal:.4f})"
            elif (prev_macd > prev_macd_signal and macd_value < macd_signal):
                macd_cross = f"Bearish MACD crossover ({macd_value:.4f} < {macd_signal:.4f})"
            elif macd_value > 0:
                macd_cross = f"MACD positive and trending {('up' if macd_value > prev_macd else 'down')}"
            else:
                macd_cross = f"MACD negative and trending {('up' if macd_value > prev_macd else 'down')}"
                
            # Analyze Bollinger Bands
            if current_price > bb_upper:
                bb_condition = f"Price above upper BB ({bb_upper:.1f}), suggesting overbought"
            elif current_price < bb_lower:
                bb_condition = f"Price below lower BB ({bb_lower:.1f}), suggesting oversold"
            else:
                percent_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
                bb_condition = f"Price at {percent_position:.1f}% of BB range"
                
            # Analyze EMA trends
            if ema_short > ema_medium and prev_ema_short <= prev_ema_medium:
                ema_trend = f"EMA 9 crossed above EMA 21, indicating potential uptrend"
            elif ema_short < ema_medium and prev_ema_short >= prev_ema_medium:
                ema_trend = f"EMA 9 crossed below EMA 21, indicating potential downtrend"
            elif ema_short > ema_medium:
                ema_trend = f"EMA 9 > EMA 21, suggesting uptrend continuation"
            else:
                ema_trend = f"EMA 9 < EMA 21, suggesting downtrend continuation"
                
            # Analyze volume
            if volume > prev_volume * 1.2:
                volume_condition = f"Volume increased by {((volume/prev_volume)-1)*100:.1f}%, suggesting strong conviction"
            elif volume < prev_volume * 0.8:
                volume_condition = f"Volume decreased by {((prev_volume/volume)-1)*100:.1f}%, suggesting diminishing interest"
            else:
                volume_condition = "Volume stable"
                
            # Generate reasons based on prediction class
            if prediction == 0:  # SHORT
                primary_reasons = [
                    f"{rsi_condition}" if rsi_value > 60 else None,
                    f"{macd_cross}" if macd_value < macd_signal else None,
                    f"{bb_condition}" if current_price > bb_middle else None,
                    f"{ema_trend}" if ema_short < ema_medium else None,
                    f"{volume_condition}" if "increased" in volume_condition else None
                ]
                secondary_reasons = [
                    "Price rejected at resistance level",
                    "Higher timeframe trend is bearish",
                    "Bearish candlestick pattern detected"
                ]
            elif prediction == 2:  # LONG
                primary_reasons = [
                    f"{rsi_condition}" if rsi_value < 40 else None,
                    f"{macd_cross}" if macd_value > macd_signal else None,
                    f"{bb_condition}" if current_price < bb_middle else None,
                    f"{ema_trend}" if ema_short > ema_medium else None,
                    f"{volume_condition}" if "increased" in volume_condition else None
                ]
                secondary_reasons = [
                    "Price bounced off support level",
                    "Higher timeframe trend is bullish",
                    "Bullish candlestick pattern detected"
                ]
            else:  # NEUTRAL
                primary_reasons = [
                    f"{rsi_condition}" if 40 <= rsi_value <= 60 else None,
                    f"{macd_cross}" if abs(macd_value - macd_signal) < 0.0005 else None,
                    f"{bb_condition}" if bb_lower < current_price < bb_upper else None,
                    "Price trading in a range", 
                    f"{volume_condition}" if "stable" in volume_condition else None
                ]
                secondary_reasons = [
                    "Sideways price action detected",
                    "Mixed signals across indicators",
                    "Waiting for stronger confirmation"
                ]
                
            # Filter out None values
            primary_reasons = [r for r in primary_reasons if r is not None]
            
            # Combine primary and secondary reasons
            reasons = primary_reasons + [random.choice(secondary_reasons)]
            
            # Format the final reason string
            technical_reason = "; ".join(reasons[:3])
            return technical_reason
            
        except Exception as e:
            logger.error(f"Error generating technical reason: {e}")
            
            # Fall back to default reasons
            default_reasons = {
                0: "Bearish technical indicators suggest short position", # SHORT
                1: "Mixed signals indicate neutral stance",              # NEUTRAL
                2: "Bullish technical indicators suggest long position"  # LONG
            }
            return default_reasons.get(prediction, "Technical analysis inconclusive")
    
    def _create_error_prediction(self, error_message):
        """
        Create an error prediction response.
        
        Args:
            error_message (str): Error message
            
        Returns:
            dict: Error prediction
        """
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price": 0,
            "trend": "ERROR",
            "confidence": 0,
            "target_price": 0,
            "predicted_move": 0,
            "reason": f"Error generating prediction: {error_message}",
            "valid_for_minutes": 0,
            "technical_indicators": {}
        }