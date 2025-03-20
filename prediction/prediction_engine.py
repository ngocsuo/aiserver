"""
Prediction engine for combining model outputs and generating final predictions.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

import config
from utils.data_processor import DataProcessor
from models.model_trainer import ModelTrainer

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
        self.model_trainer = ModelTrainer()
        self.models = {}
        self.last_prediction = None
        self.last_prediction_time = None
        self.prediction_validity = config.VALIDITY_MINUTES  # minutes
        
    def load_models(self):
        """
        Load trained models for prediction.
        
        Returns:
            dict: Loaded models
        """
        try:
            # Load models from disk
            self.models = self.model_trainer.load_models()
            
            if not self.models:
                logger.warning("No models loaded. Training may be needed.")
                
            return self.models
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return {}
            
    def is_prediction_valid(self):
        """
        Check if the last prediction is still valid.
        
        Returns:
            bool: Whether the prediction is still valid
        """
        if self.last_prediction is None or self.last_prediction_time is None:
            return False
            
        # Check if prediction is expired
        current_time = datetime.now()
        elapsed = (current_time - self.last_prediction_time).total_seconds() / 60
        
        return elapsed < self.prediction_validity
        
    def get_cached_prediction(self):
        """
        Get the cached prediction if valid.
        
        Returns:
            dict: Cached prediction or None
        """
        if self.is_prediction_valid():
            # Add remaining validity time
            remaining = self.prediction_validity - (datetime.now() - self.last_prediction_time).total_seconds() / 60
            self.last_prediction['valid_for_minutes'] = round(remaining, 1)
            self.last_prediction['cached'] = True
            
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
            # Check if we have a valid cached prediction
            if use_cache:
                cached = self.get_cached_prediction()
                if cached is not None:
                    logger.info("Using cached prediction")
                    return cached
                    
            # Ensure models are loaded
            if not self.models:
                self.load_models()
                
            if not self.models:
                logger.error("No models available for prediction")
                return self._create_error_prediction("No models available")
                
            # Prepare data for each model type
            sequence_data, original_data = self.data_processor.prepare_latest_data(
                data, lookback=config.SEQUENCE_LENGTH
            )
            
            image_data, _ = self.data_processor.prepare_latest_cnn_data(
                data, lookback=config.SEQUENCE_LENGTH
            )
            
            if sequence_data is None or image_data is None:
                logger.error("Failed to prepare data for prediction")
                return self._create_error_prediction("Data preparation failed")
                
            # Get predictions from each model
            model_predictions = {}
            model_confidences = {}
            model_probs = []
            
            # LSTM prediction
            if 'lstm' in self.models:
                pred_lstm, probs_lstm = self.models['lstm'].predict(sequence_data)
                if pred_lstm is not None:
                    model_predictions['lstm'] = pred_lstm[0]
                    model_confidences['lstm'] = np.max(probs_lstm[0])
                    model_probs.append(probs_lstm)
                    
            # Transformer prediction
            if 'transformer' in self.models:
                pred_transformer, probs_transformer = self.models['transformer'].predict(sequence_data)
                if pred_transformer is not None:
                    model_predictions['transformer'] = pred_transformer[0]
                    model_confidences['transformer'] = np.max(probs_transformer[0])
                    model_probs.append(probs_transformer)
                    
            # CNN prediction
            if 'cnn' in self.models:
                pred_cnn, probs_cnn = self.models['cnn'].predict(image_data)
                if pred_cnn is not None:
                    model_predictions['cnn'] = pred_cnn[0]
                    model_confidences['cnn'] = np.max(probs_cnn[0])
                    model_probs.append(probs_cnn)
                    
            # Historical similarity prediction
            if 'historical' in self.models:
                pred_hist, probs_hist, _, _ = self.models['historical'].predict(sequence_data)
                if pred_hist is not None:
                    model_predictions['historical'] = pred_hist
                    model_confidences['historical'] = np.max(probs_hist)
                    model_probs.append(probs_hist.reshape(1, -1))
                    
            # Meta-learner prediction (if available and all other models provided predictions)
            meta_prediction = None
            meta_confidence = None
            meta_probs = None
            
            if 'meta' in self.models and len(model_probs) >= 2:
                pred_meta, probs_meta = self.models['meta'].predict(model_probs)
                if pred_meta is not None:
                    meta_prediction = pred_meta[0]
                    meta_confidence = np.max(probs_meta[0])
                    meta_probs = probs_meta[0]
                    
            # Generate combined prediction
            final_prediction = self._combine_predictions(
                model_predictions, model_confidences,
                meta_prediction, meta_confidence
            )
            
            # Calculate predicted price
            if original_data is not None and not original_data.empty:
                current_price = original_data['close'].iloc[-1]
                
                # Simple price prediction based on trend
                if final_prediction['trend'].upper() == 'LONG':
                    # Predict price increase based on average ATR
                    atr = original_data['high'].iloc[-14:] - original_data['low'].iloc[-14:]
                    avg_atr = atr.mean()
                    predicted_price = current_price + (avg_atr * 0.5)  # Move 0.5 ATR up
                elif final_prediction['trend'].upper() == 'SHORT':
                    # Predict price decrease based on average ATR
                    atr = original_data['high'].iloc[-14:] - original_data['low'].iloc[-14:]
                    avg_atr = atr.mean()
                    predicted_price = current_price - (avg_atr * 0.5)  # Move 0.5 ATR down
                else:
                    predicted_price = current_price  # No change for NEUTRAL
                    
                final_prediction['price'] = round(predicted_price, 2)
                
            # Cache the prediction
            self.last_prediction = final_prediction
            self.last_prediction_time = datetime.now()
            
            return final_prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
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
        try:
            # If we have meta-learner prediction, use it as primary
            if meta_prediction is not None and meta_confidence >= config.CONFIDENCE_THRESHOLD:
                prediction = meta_prediction
                confidence = meta_confidence
                reason = "Meta-learner prediction with high confidence"
                
                # Add which models agreed with meta-learner
                agreeing_models = [model for model, pred in model_predictions.items() 
                               if pred == meta_prediction]
                if agreeing_models:
                    reason += f"; agrees with {', '.join(agreeing_models)}"
                    
            # Otherwise, use majority voting with confidence weighting
            else:
                # Count weighted votes for each class
                class_votes = {0: 0.0, 1: 0.0, 2: 0.0}  # SHORT, NEUTRAL, LONG
                
                for model, pred in model_predictions.items():
                    weight = model_confidences.get(model, 0.5)  # Default weight if no confidence
                    class_votes[pred] += weight
                    
                # Get the class with the most votes
                prediction = max(class_votes.items(), key=lambda x: x[1])[0]
                
                # Calculate confidence as normalized vote strength
                total_votes = sum(class_votes.values())
                confidence = class_votes[prediction] / total_votes if total_votes > 0 else 0.0
                
                # Generate reason
                voting_models = [model for model, pred in model_predictions.items() 
                             if pred == prediction]
                reason = f"Majority voting (weighted): {', '.join(voting_models)} agree"
                
            # Map prediction to trend
            trend_map = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
            trend = trend_map[prediction]
            
            # Adjust confidence calculation
            adjusted_confidence = confidence
            
            # Check for strong agreement among models
            if all(pred == prediction for pred in model_predictions.values()):
                adjusted_confidence = min(1.0, confidence * 1.2)  # Boost confidence for unanimous
                reason += "; all models agree"
                
            # Check for disagreement
            elif len(set(model_predictions.values())) == len(model_predictions):
                adjusted_confidence = confidence * 0.8  # Reduce confidence for disagreement
                reason += "; models disagree"
                
            # Append technical reasons if confidence is high
            if adjusted_confidence >= 0.7:
                reason += self._generate_technical_reason(prediction)
                
            # Create the prediction dict
            prediction_dict = {
                "trend": trend.lower(),
                "confidence": round(adjusted_confidence, 2),
                "valid_for_minutes": config.VALIDITY_MINUTES,
                "reason": reason,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return prediction_dict
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return self._create_error_prediction("Error combining model predictions")
            
    def _generate_technical_reason(self, prediction):
        """
        Generate technical reasoning for the prediction.
        
        Args:
            prediction (int): Class prediction (0, 1, or 2)
            
        Returns:
            str: Technical reasoning
        """
        # This is a placeholder for real technical analysis based on the actual data
        # In a real system, you would analyze the indicators from the processed data
        # For demo, we'll use hardcoded examples for each prediction type
        
        if prediction == 0:  # SHORT
            reasons = [
                "; RSI overbought; price near resistance",
                "; bearish engulfing pattern; MACD bearish cross",
                "; price above upper Bollinger Band; bearish divergence"
            ]
        elif prediction == 2:  # LONG
            reasons = [
                "; RSI oversold; price near support",
                "; bullish engulfing pattern; MACD bullish cross",
                "; price below lower Bollinger Band; bullish divergence"
            ]
        else:  # NEUTRAL
            reasons = [
                "; price within Bollinger Bands; RSI neutral",
                "; no clear pattern; low volatility",
                "; mixed signals; waiting for confirmation"
            ]
            
        # Return a random reason from the list
        import random
        return random.choice(reasons)
            
    def _create_error_prediction(self, error_message):
        """
        Create an error prediction response.
        
        Args:
            error_message (str): Error message
            
        Returns:
            dict: Error prediction
        """
        return {
            "trend": "neutral",
            "confidence": 0.0,
            "valid_for_minutes": 5,
            "reason": f"Error: {error_message}",
            "error": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
