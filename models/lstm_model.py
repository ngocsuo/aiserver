"""
LSTM model for sequence-based price prediction.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import logging

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lstm_model")

class LSTMModel:
    def __init__(self, input_shape, output_dim=3, model_path=None):
        """
        Initialize the LSTM model.
        
        Args:
            input_shape (tuple): Shape of input data (seq_length, n_features)
            output_dim (int): Number of output classes (default: 3 for SHORT/NEUTRAL/LONG)
            model_path (str): Path to load a pre-trained model
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self.build()
            
    def build(self):
        """Build the LSTM model architecture."""
        try:
            # Create a sequential model
            model = Sequential([
                # Bidirectional LSTM layers with increasing complexity
                Bidirectional(LSTM(64, return_sequences=True), 
                             input_shape=self.input_shape),
                BatchNormalization(),
                Dropout(0.2),
                
                Bidirectional(LSTM(128, return_sequences=True)),
                BatchNormalization(),
                Dropout(0.3),
                
                Bidirectional(LSTM(64, return_sequences=False)),
                BatchNormalization(),
                Dropout(0.2),
                
                # Output layer
                Dense(self.output_dim, activation='softmax')
            ])
            
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info(f"LSTM model built with input shape {self.input_shape}")
            logger.info(f"Model summary: {model.summary()}")
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise
            
    def train(self, X_train, y_train, X_val, y_val, epochs=config.EPOCHS, 
             batch_size=config.BATCH_SIZE):
        """
        Train the LSTM model.
        
        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation data
            y_val (np.ndarray): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            logger.error("Model not initialized. Call build() first.")
            return None
            
        try:
            # Create model checkpoint callback
            model_dir = os.path.join(config.MODEL_DIR, f"lstm_{config.MODEL_VERSION}")
            os.makedirs(model_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(model_dir, "lstm_model_best.h5")
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )
            
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            )
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[checkpoint, early_stopping],
                verbose=1
            )
            
            # Save the final model
            final_path = os.path.join(model_dir, "lstm_model_final.h5")
            self.model.save(final_path)
            
            logger.info(f"LSTM model trained for {len(history.history['loss'])} epochs and saved to {model_dir}")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise
            
    def predict(self, X):
        """
        Make predictions with the LSTM model.
        
        Args:
            X (np.ndarray): Input data for prediction
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            logger.error("Model not initialized. Call build() or load() first.")
            return None, None
            
        try:
            # Get class probabilities
            probabilities = self.model.predict(X)
            
            # Get class predictions
            predictions = np.argmax(probabilities, axis=1)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions with LSTM model: {e}")
            return None, None
            
    def evaluate(self, X_test, y_test):
        """
        Evaluate the LSTM model.
        
        Args:
            X_test (np.ndarray): Test data
            y_test (np.ndarray): Test labels
            
        Returns:
            tuple: (loss, accuracy)
        """
        if self.model is None:
            logger.error("Model not initialized. Call build() or load() first.")
            return None, None
            
        try:
            # Evaluate the model
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=1)
            
            logger.info(f"LSTM model evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            return loss, accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating LSTM model: {e}")
            return None, None
            
    def save(self, path):
        """
        Save the LSTM model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            logger.error("No model to save. Call build() first.")
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            self.model.save(path)
            
            logger.info(f"LSTM model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
            
    def load(self, path):
        """
        Load an LSTM model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        try:
            # Load the model
            self.model = load_model(path)
            
            # Update input shape from loaded model
            self.input_shape = self.model.input_shape[1:]
            
            # Update output dimension from loaded model
            self.output_dim = self.model.output_shape[1]
            
            logger.info(f"LSTM model loaded from {path}")
            logger.info(f"Loaded model input shape: {self.input_shape}")
            logger.info(f"Loaded model output shape: {self.output_dim}")
            
        except Exception as e:
            logger.error(f"Error loading LSTM model from {path}: {e}")
            raise
