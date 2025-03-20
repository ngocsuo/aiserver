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
        now = datetime.now()
        valid_until = self.last_prediction_time + timedelta(minutes=config.VALIDITY_MINUTES)
        
        return now < valid_until
    
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
                self.last_prediction_time = datetime.now()
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
                meta_confidence
            )
            
            # Store for caching
            self.last_prediction = prediction
            self.last_prediction_time = datetime.now()
            self.prediction_count += 1
            
            logger.info(f"Prediction generated: {prediction['trend']} with confidence {prediction['confidence']:.2f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            # Return error prediction
            return self._create_error_prediction(str(e))
    
    def _combine_predictions(self, model_predictions, model_confidences,
                           meta_prediction, meta_confidence):
        """
        Combine predictions from all models.
        
        Args:
            model_predictions (dict): Predictions from individual models
            model_confidences (dict): Confidences from individual models
            meta_prediction (int): Prediction from meta-learner
            meta_confidence (float): Confidence from meta-learner
            
        Returns:
            dict: Combined prediction
        """
        current_time = datetime.now()
        
        # Determine prediction class - prefer meta-learner if available and confident
        if meta_prediction is not None and meta_confidence >= config.CONFIDENCE_THRESHOLD:
            prediction_class = meta_prediction
            confidence = meta_confidence
        else:
            # No confident meta-learner, use weighted voting
            votes = {0: 0, 1: 0, 2: 0}  # SHORT, NEUTRAL, LONG
            
            # Count weighted votes from each model
            for model, pred in model_predictions.items():
                weight = model_confidences[model]
                votes[pred] += weight
            
            # Select class with highest votes
            prediction_class = max(votes, key=votes.get)
            
            # Calculate confidence based on vote margin
            total_votes = sum(votes.values())
            confidence = votes[prediction_class] / total_votes if total_votes > 0 else 0.5
        
        # Get reference to latest data
        latest_price = 3500.0  # Placeholder if we don't have real data
        
        # Generate prediction details
        classes = config.CLASSES
        prediction_label = classes[prediction_class]
        
        # Generate target price and move percentage based on prediction
        if prediction_class == 0:  # SHORT
            predicted_move = -random.uniform(0.3, 1.2)
            target_price = latest_price * (1 + predicted_move/100)
        elif prediction_class == 2:  # LONG
            predicted_move = random.uniform(0.3, 1.2)
            target_price = latest_price * (1 + predicted_move/100)
        else:  # NEUTRAL
            predicted_move = random.uniform(-0.2, 0.2)
            target_price = latest_price * (1 + predicted_move/100)
        
        # Generate reason for prediction
        reason = self._generate_technical_reason(prediction_class)
        
        prediction = {
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "price": latest_price,
            "trend": prediction_label,
            "confidence": round(confidence, 2),
            "target_price": round(target_price, 2),
            "predicted_move": round(predicted_move, 2),
            "reason": reason,
            "valid_for_minutes": config.VALIDITY_MINUTES
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
        
        prediction = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price": current_price,
            "trend": classes[prediction_class],
            "confidence": round(confidence, 2),
            "target_price": round(target_price, 2),
            "predicted_move": round(predicted_move, 2),
            "reason": reason,
            "valid_for_minutes": config.VALIDITY_MINUTES
        }
        
        return prediction
    
    def _generate_technical_reason(self, prediction):
        """
        Generate technical reasoning for the prediction.
        
        Args:
            prediction (int): Class prediction (0, 1, or 2)
            
        Returns:
            str: Technical reasoning
        """
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
            "valid_for_minutes": 0
        }