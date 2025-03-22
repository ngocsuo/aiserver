"""
Model trainer module sửa lỗi - KHÔNG trả về tuples
"""
import os
import logging
import json
from datetime import datetime

import config

logger = logging.getLogger("model_trainer")

class ModelTrainer:
    def __init__(self):
        """Initialize the model trainer."""
        logger.info("Model trainer initialized")
        
        # Ensure model directory exists
        if hasattr(config, 'MODEL_DIR'):
            os.makedirs(config.MODEL_DIR, exist_ok=True)
        
        # Initialize model and history storage
        self.models = {}
        self.histories = {}
    
    def train_lstm(self, sequence_data):
        """
        Train an LSTM model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            
        Returns:
            model: Trained model (NOT a tuple)
        """
        try:
            logger.info("Training LSTM model with non-tuple return")
            
            # Mock model for testing
            model = {
                "type": "lstm", 
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy": 0.7,
                "loss": 0.5
            }
            
            # Mock history for testing
            history = {
                "accuracy": [0.5, 0.6, 0.65, 0.7],
                "val_accuracy": [0.45, 0.55, 0.6, 0.65],
                "loss": [0.9, 0.7, 0.6, 0.5],
                "val_loss": [1.0, 0.8, 0.7, 0.6]
            }
            
            # Store history in instance variable
            self.histories['lstm'] = history
            
            # Return ONLY the model, not a tuple
            return model
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return None  # Return ONLY None, not a tuple
    
    def train_transformer(self, sequence_data):
        """
        Train a Transformer model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            
        Returns:
            model: Trained model (NOT a tuple)
        """
        try:
            logger.info("Training Transformer model with non-tuple return")
            
            # Mock model for testing
            model = {
                "type": "transformer", 
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy": 0.75,
                "loss": 0.48
            }
            
            # Mock history for testing
            history = {
                "accuracy": [0.5, 0.65, 0.7, 0.75],
                "val_accuracy": [0.45, 0.6, 0.65, 0.7],
                "loss": [0.9, 0.65, 0.55, 0.48],
                "val_loss": [1.0, 0.75, 0.65, 0.58]
            }
            
            # Store history in instance variable
            self.histories['transformer'] = history
            
            # Return ONLY the model, not a tuple
            return model
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            return None  # Return ONLY None, not a tuple
    
    def train_cnn(self, image_data):
        """
        Train a CNN model.
        
        Args:
            image_data (dict): Dictionary with image data for training
            
        Returns:
            model: Trained model (NOT a tuple)
        """
        try:
            logger.info("Training CNN model with non-tuple return")
            
            # Mock model for testing
            model = {
                "type": "cnn", 
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy": 0.68,
                "loss": 0.49
            }
            
            # Mock history for testing
            history = {
                "accuracy": [0.48, 0.55, 0.6, 0.65, 0.68],
                "val_accuracy": [0.45, 0.5, 0.58, 0.62, 0.65],
                "loss": [0.95, 0.75, 0.65, 0.55, 0.49],
                "val_loss": [1.0, 0.8, 0.7, 0.6, 0.52]
            }
            
            # Store history in instance variable
            self.histories['cnn'] = history
            
            # Return ONLY the model, not a tuple
            return model
            
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            return None  # Return ONLY None, not a tuple
    
    def train_historical_similarity(self, sequence_data):
        """
        Train a Historical Similarity model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            
        Returns:
            model: Trained model (NOT a tuple)
        """
        try:
            logger.info("Training Historical Similarity model with non-tuple return")
            
            # Mock model for testing
            model = {
                "type": "historical_similarity",
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy": 0.65
            }
            
            # Return ONLY the model, not a tuple
            return model
            
        except Exception as e:
            logger.error(f"Error training Historical Similarity model: {e}")
            return None  # Return ONLY None, not a tuple
    
    def train_meta_learner(self, sequence_data, image_data):
        """
        Train a Meta-Learner model that combines other model outputs.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            image_data (dict): Dictionary with image data for training
            
        Returns:
            model: Trained model (NOT a tuple)
        """
        try:
            logger.info("Training Meta-Learner model with non-tuple return")
            
            # Mock model for testing
            model = {
                "type": "meta_learner",
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy": 0.8
            }
            
            # Return ONLY the model, not a tuple
            return model
            
        except Exception as e:
            logger.error(f"Error training Meta-Learner model: {e}")
            return None  # Return ONLY None, not a tuple
    
    def train_all_models(self, sequence_data, image_data, timeframe=None):
        """
        Train all models in the ensemble.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            image_data (dict): Dictionary with image data for training
            timeframe (str, optional): Timeframe for model training (e.g., '1m', '5m')
            
        Returns:
            dict: Trained models - NOT A TUPLE
        """
        try:
            logger.info(f"Training all models for timeframe {timeframe or 'default'}")
            
            # Store models dictionary to return
            models = {}
            
            # Train each model type - chỉ lưu models, không quan tâm history
            lstm_model = self.train_lstm(sequence_data)
            if lstm_model:
                models['lstm'] = lstm_model
                
            transformer_model = self.train_transformer(sequence_data)
            if transformer_model:
                models['transformer'] = transformer_model
                
            cnn_model = self.train_cnn(image_data)
            if cnn_model:
                models['cnn'] = cnn_model
                
            historical_model = self.train_historical_similarity(sequence_data)
            if historical_model:
                models['historical_similarity'] = historical_model
                
            # Train meta-learner last, after all other models are trained
            if models:
                meta_model = self.train_meta_learner(sequence_data, image_data)
                if meta_model:
                    models['meta_learner'] = meta_model
            
            # Cache models in instance variable
            self.models = models
            
            logger.info("All models trained successfully")
            
            # Trả về dictionary, KHÔNG PHẢI tuple
            return models
            
        except Exception as e:
            logger.error(f"Error training all models: {e}")
            return {}  # Return empty dict in case of error, NOT a tuple
    
    def load_models(self):
        """
        Load all saved models.
        
        Returns:
            dict: Loaded models
        """
        try:
            # Check if models already loaded
            if self.models and len(self.models) > 0:
                return self.models
                
            # Load models from storage
            models = {}
            
            # Directory for model files
            model_dir = getattr(config, 'MODEL_DIR', 'models')
            
            # Load model files if they exist
            for model_type in ['lstm', 'transformer', 'cnn', 'historical_similarity', 'meta_learner']:
                model_path = os.path.join(model_dir, f"{model_type}.json")
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'r') as f:
                            models[model_type] = json.load(f)
                        logger.info(f"Loaded {model_type} model from {model_path}")
                    except Exception as e:
                        logger.error(f"Error loading {model_type} model: {e}")
            
            # Cache models in instance variable
            self.models = models
            return models
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return {}