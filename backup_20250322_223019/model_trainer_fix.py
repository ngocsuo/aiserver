"""
Model trainer module to coordinate the training of all models.
PHIÊN BẢN ĐÃ SỬA - bổ sung tham số timeframe
"""
import os
import logging
import numpy as np
import config
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.cnn_model import CNNModel
from models.historical_similarity import HistoricalSimilarity
from models.meta_learner import MetaLearner

# Thiết lập logger
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        """Initialize the model trainer."""
        self.models = None
        self.histories = None
        
    def train_lstm(self, sequence_data):
        """
        Train an LSTM model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            
        Returns:
            tuple: (model, history)
        """
        try:
            # Extract data
            X_train = sequence_data.get('X_train')
            y_train = sequence_data.get('y_train')
            X_val = sequence_data.get('X_val')
            y_val = sequence_data.get('y_val')
            
            if X_train is None or y_train is None:
                logger.error("Training data is None")
                return None, None
                
            # Get input shape from data
            input_shape = X_train.shape[1:]
            
            # Create and build model
            lstm_model = LSTMModel(input_shape=input_shape)
            lstm_model.build()
            
            # Train model
            history = lstm_model.train(X_train, y_train, X_val, y_val, 
                                     epochs=config.EPOCHS, 
                                     batch_size=config.BATCH_SIZE)
            
            # Save model
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            lstm_model.save(os.path.join(config.MODELS_DIR, 'lstm_model.keras'))
            
            return lstm_model, history
            
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
            # Extract data
            X_train = sequence_data.get('X_train')
            y_train = sequence_data.get('y_train')
            X_val = sequence_data.get('X_val')
            y_val = sequence_data.get('y_val')
            
            if X_train is None or y_train is None:
                logger.error("Training data is None")
                return None, None
                
            # Get input shape from data
            input_shape = X_train.shape[1:]
            
            # Create and build model
            transformer_model = TransformerModel(input_shape=input_shape)
            transformer_model.build()
            
            # Train model
            history = transformer_model.train(X_train, y_train, X_val, y_val, 
                                           epochs=config.EPOCHS, 
                                           batch_size=config.BATCH_SIZE)
            
            # Save model
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            transformer_model.save(os.path.join(config.MODELS_DIR, 'transformer_model.keras'))
            
            return transformer_model, history
            
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
            # Extract data
            X_train = image_data.get('X_train')
            y_train = image_data.get('y_train')
            X_val = image_data.get('X_val')
            y_val = image_data.get('y_val')
            
            if X_train is None or y_train is None:
                logger.error("Training data is None")
                return None, None
                
            # Get input shape from data
            input_shape = X_train.shape[1:]
            
            # Create and build model
            cnn_model = CNNModel(input_shape=input_shape)
            cnn_model.build()
            
            # Train model
            history = cnn_model.train(X_train, y_train, X_val, y_val, 
                                    epochs=config.EPOCHS, 
                                    batch_size=config.BATCH_SIZE)
            
            # Save model
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            cnn_model.save(os.path.join(config.MODELS_DIR, 'cnn_model.keras'))
            
            return cnn_model, history
            
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
            # Extract data
            X_train = sequence_data.get('X_train')
            y_train = sequence_data.get('y_train')
            
            if X_train is None or y_train is None:
                logger.error("Training data is None")
                return None, None
                
            # Create model
            historical_model = HistoricalSimilarity(
                sequence_length=config.SEQUENCE_LENGTH
            )
            
            # Train model
            historical_model.train(X_train, y_train)
            
            # Save model
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            historical_model.save(os.path.join(config.MODELS_DIR, 'historical_model.pkl'))
            
            return historical_model, None
            
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
            # Extract data
            X_train = sequence_data.get('X_train')
            y_train = sequence_data.get('y_train')
            X_val = sequence_data.get('X_val')
            y_val = sequence_data.get('y_val')
            
            if X_train is None or y_train is None:
                logger.error("Training data is None")
                return None, None
                
            # Load or create individual models
            input_shape_seq = X_train.shape[1:]
            input_shape_img = image_data.get('X_train').shape[1:]
            
            lstm_model = LSTMModel(input_shape=input_shape_seq)
            lstm_model.build()
            
            transformer_model = TransformerModel(input_shape=input_shape_seq)
            transformer_model.build()
            
            cnn_model = CNNModel(input_shape=input_shape_img)
            cnn_model.build()
            
            historical_model = HistoricalSimilarity(
                sequence_length=config.SEQUENCE_LENGTH
            )
            
            # Get predictions from each model
            lstm_preds, lstm_probs = lstm_model.predict(X_val)
            transformer_preds, transformer_probs = transformer_model.predict(X_val)
            cnn_preds, cnn_probs = cnn_model.predict(image_data.get('X_val'))
            
            # Create meta-learner
            meta_model = MetaLearner(model_type='logistic')
            meta_model.build()
            
            # Prepare meta-features
            base_model_probs = [lstm_probs, transformer_probs, cnn_probs]
            
            # Train meta-learner
            meta_model.train(base_model_probs, y_val)
            
            # Save model
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            meta_model.save(os.path.join(config.MODELS_DIR, 'meta_model.pkl'))
            
            return meta_model, None
            
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
            dict: Trained models
        """
        try:
            if timeframe:
                logger.info(f"Training all models for timeframe: {timeframe}")
            else:
                logger.info("Training all models")
            
            # Train each model type
            lstm_model, lstm_history = self.train_lstm(sequence_data)
            transformer_model, transformer_history = self.train_transformer(sequence_data)
            cnn_model, cnn_history = self.train_cnn(image_data)
            historical_model, _ = self.train_historical_similarity(sequence_data)
            meta_model, _ = self.train_meta_learner(sequence_data, image_data)
            
            # Store models and training histories
            self.models = {
                'lstm': lstm_model,
                'transformer': transformer_model,
                'cnn': cnn_model,
                'historical_similarity': historical_model,
                'meta_learner': meta_model
            }
            
            self.histories = {
                'lstm': lstm_history,
                'transformer': transformer_history,
                'cnn': cnn_history
            }
            
            # Lưu thông tin về timeframe
            if timeframe:
                self.timeframe = timeframe
            
            logger.info("All models trained successfully")
            
            return self.models
            
        except Exception as e:
            logger.error(f"Error training all models: {e}")
            return None