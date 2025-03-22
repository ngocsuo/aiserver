"""
Model trainer module to coordinate the training of all models.
"""
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_trainer")

class ModelTrainer:
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.histories = {}
        logger.info("Model trainer initialized")
    
    def train_lstm(self, sequence_data):
        """
        Train an LSTM model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            
        Returns:
            tuple: (model, history)
        """
        try:
            logger.info("Training LSTM model (placeholder for actual training)")
            
            # For development/demonstration, we'll return a mock model and history
            # In a production environment, this would use the actual LSTM model
            
            model = {
                'name': 'LSTM',
                'accuracy': 0.72,
                'loss': 0.45
            }
            
            history = {
                'accuracy': [0.5, 0.6, 0.65, 0.7, 0.72],
                'val_accuracy': [0.48, 0.55, 0.62, 0.67, 0.69],
                'loss': [0.9, 0.7, 0.6, 0.5, 0.45],
                'val_loss': [0.95, 0.8, 0.65, 0.55, 0.5]
            }
            
            # Save the mock model for demonstration purposes
            model_path = os.path.join(config.MODEL_DIR, f"lstm_{config.MODEL_VERSION}.pkl")
            pd.to_pickle(model, model_path)
            
            return model, history
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return None, None
    
    def train_transformer(self, sequence_data):
        """
        Train a Transformer model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            
        Returns:
            tuple: (model, history)
        """
        try:
            logger.info("Training Transformer model (placeholder for actual training)")
            
            # For development/demonstration, we'll return a mock model and history
            
            model = {
                'name': 'Transformer',
                'accuracy': 0.76,
                'loss': 0.41
            }
            
            history = {
                'accuracy': [0.55, 0.65, 0.7, 0.73, 0.76],
                'val_accuracy': [0.52, 0.6, 0.65, 0.7, 0.73],
                'loss': [0.85, 0.65, 0.55, 0.45, 0.41],
                'val_loss': [0.9, 0.7, 0.6, 0.5, 0.45]
            }
            
            # Save the mock model for demonstration purposes
            model_path = os.path.join(config.MODEL_DIR, f"transformer_{config.MODEL_VERSION}.pkl")
            pd.to_pickle(model, model_path)
            
            return model, history
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            return None, None
    
    def train_cnn(self, image_data):
        """
        Train a CNN model.
        
        Args:
            image_data (dict): Dictionary with image data for training
            
        Returns:
            tuple: (model, history)
        """
        try:
            logger.info("Training CNN model (placeholder for actual training)")
            
            # For development/demonstration, we'll return a mock model and history
            
            model = {
                'name': 'CNN',
                'accuracy': 0.68,
                'loss': 0.49
            }
            
            history = {
                'accuracy': [0.48, 0.55, 0.6, 0.65, 0.68],
                'val_accuracy': [0.45, 0.5, 0.58, 0.62, 0.65],
                'loss': [0.95, 0.75, 0.65, 0.55, 0.49],
                'val_loss': [1.0, 0.8, 0.7, 0.6, 0.52]
            }
            
            # Save the mock model for demonstration purposes
            model_path = os.path.join(config.MODEL_DIR, f"cnn_{config.MODEL_VERSION}.pkl")
            pd.to_pickle(model, model_path)
            
            return model, history
            
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            return None, None
    
    def train_historical_similarity(self, sequence_data):
        """
        Train a Historical Similarity model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            
        Returns:
            tuple: (model, None)
        """
        try:
            logger.info("Training Historical Similarity model (placeholder for actual training)")
            
            # For development/demonstration, we'll return a mock model
            
            model = {
                'name': 'Historical Similarity',
                'accuracy': 0.65,
                'loss': None
            }
            
            # Save the mock model for demonstration purposes
            model_path = os.path.join(config.MODEL_DIR, f"historical_{config.MODEL_VERSION}.pkl")
            pd.to_pickle(model, model_path)
            
            return model, None
            
        except Exception as e:
            logger.error(f"Error training Historical Similarity model: {e}")
            return None, None
    
    def train_meta_learner(self, sequence_data, image_data):
        """
        Train a Meta-Learner model that combines other model outputs.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            image_data (dict): Dictionary with image data for training
            
        Returns:
            tuple: (model, None)
        """
        try:
            logger.info("Training Meta-Learner model (placeholder for actual training)")
            
            # For development/demonstration, we'll return a mock model
            
            model = {
                'name': 'Meta-Learner',
                'accuracy': 0.81,
                'loss': 0.35
            }
            
            # Save the mock model for demonstration purposes
            model_path = os.path.join(config.MODEL_DIR, f"meta_{config.MODEL_VERSION}.pkl")
            pd.to_pickle(model, model_path)
            
            return model, None
            
        except Exception as e:
            logger.error(f"Error training Meta-Learner model: {e}")
            return None, None
    
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
            
            # Train each model type
            try:
                # Fix: Use result_list to handle tuple unpacking error
                result = self.train_lstm(sequence_data)
                if isinstance(result, tuple) and len(result) >= 1:
                    lstm_model = result[0]
                    lstm_history = result[1] if len(result) > 1 else None
                else:
                    lstm_model = result
                    lstm_history = None
                    
                if lstm_model:
                    models['lstm'] = lstm_model
            except Exception as e:
                logger.error(f"Error training LSTM model: {e}")
                
            try:
                # Fix: Use result_list to handle tuple unpacking error
                result = self.train_transformer(sequence_data)
                if isinstance(result, tuple) and len(result) >= 1:
                    transformer_model = result[0]
                    transformer_history = result[1] if len(result) > 1 else None
                else:
                    transformer_model = result
                    transformer_history = None
                    
                if transformer_model:
                    models['transformer'] = transformer_model
            except Exception as e:
                logger.error(f"Error training Transformer model: {e}")
                
            try:
                # Fix: Use result_list to handle tuple unpacking error
                result = self.train_cnn(image_data)
                if isinstance(result, tuple) and len(result) >= 1:
                    cnn_model = result[0]
                    cnn_history = result[1] if len(result) > 1 else None
                else:
                    cnn_model = result
                    cnn_history = None
                    
                if cnn_model:
                    models['cnn'] = cnn_model
            except Exception as e:
                logger.error(f"Error training CNN model: {e}")
                
            try:
                # Fix: Use result_list to handle tuple unpacking error
                result = self.train_historical_similarity(sequence_data)
                if isinstance(result, tuple) and len(result) >= 1:
                    historical_model = result[0]
                else:
                    historical_model = result
                    
                if historical_model:
                    models['historical_similarity'] = historical_model
            except Exception as e:
                logger.error(f"Error training Historical Similarity model: {e}")
                
            try:
                # Fix: Use result_list to handle tuple unpacking error
                result = self.train_meta_learner(sequence_data, image_data)
                if isinstance(result, tuple) and len(result) >= 1:
                    meta_model = result[0]
                else:
                    meta_model = result
                    
                if meta_model:
                    models['meta_learner'] = meta_model
            except Exception as e:
                logger.error(f"Error training Meta-Learner model: {e}")
                
            # Store models and training histories
            self.models = models
            
            # Lưu histories nếu có
            self.histories = {}
            if 'lstm' in models and lstm_history is not None:
                self.histories['lstm'] = lstm_history
            if 'transformer' in models and transformer_history is not None:
                self.histories['transformer'] = transformer_history
            if 'cnn' in models and cnn_history is not None:
                self.histories['cnn'] = cnn_history
            
            self.histories = {
                'lstm': lstm_history,
                'transformer': transformer_history,
                'cnn': cnn_history
            }
            
            logger.info("All models trained successfully")
            
            # Sửa lỗi: Đảm bảo trả về dict models, KHÔNG phải tuple
            return self.models  # Chỉ trả về dictionary models, không phải tuple
            
        except Exception as e:
            logger.error(f"Error training all models: {e}")
            return None
    
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
                
            # Check if models exist on disk
            lstm_path = os.path.join(config.MODEL_DIR, f"lstm_{config.MODEL_VERSION}.pkl")
            transformer_path = os.path.join(config.MODEL_DIR, f"transformer_{config.MODEL_VERSION}.pkl")
            cnn_path = os.path.join(config.MODEL_DIR, f"cnn_{config.MODEL_VERSION}.pkl")
            historical_path = os.path.join(config.MODEL_DIR, f"historical_{config.MODEL_VERSION}.pkl")
            meta_path = os.path.join(config.MODEL_DIR, f"meta_{config.MODEL_VERSION}.pkl")
            
            # Load models if they exist
            models = {}
            
            if os.path.exists(lstm_path):
                models['lstm'] = pd.read_pickle(lstm_path)
                logger.info(f"Loaded LSTM model from {lstm_path}")
            
            if os.path.exists(transformer_path):
                models['transformer'] = pd.read_pickle(transformer_path)
                logger.info(f"Loaded Transformer model from {transformer_path}")
            
            if os.path.exists(cnn_path):
                models['cnn'] = pd.read_pickle(cnn_path)
                logger.info(f"Loaded CNN model from {cnn_path}")
            
            if os.path.exists(historical_path):
                models['historical_similarity'] = pd.read_pickle(historical_path)
                logger.info(f"Loaded Historical Similarity model from {historical_path}")
            
            if os.path.exists(meta_path):
                models['meta_learner'] = pd.read_pickle(meta_path)
                logger.info(f"Loaded Meta-Learner model from {meta_path}")
            
            self.models = models
            return models
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return {}