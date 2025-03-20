"""
CNN model for image-based price prediction from candlestick patterns.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import logging

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cnn_model")

class CNNModel:
    def __init__(self, input_shape, output_dim=3, model_path=None):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input data (seq_length, channels, 1)
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
        """Build the CNN model architecture."""
        try:
            # Input layer
            inputs = Input(shape=self.input_shape)
            
            # First convolutional block
            x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 1))(x)
            x = Dropout(0.2)(x)
            
            # Second convolutional block
            x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 1))(x)
            x = Dropout(0.2)(x)
            
            # Third convolutional block
            x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 1))(x)
            x = Dropout(0.3)(x)
            
            # Flatten and fully connected layers
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
            # Output layer
            outputs = Dense(self.output_dim, activation='softmax')(x)
            
            # Create the model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info(f"CNN model built with input shape {self.input_shape}")
            logger.info(f"Model summary: {model.summary()}")
            
        except Exception as e:
            logger.error(f"Error building CNN model: {e}")
            raise
            
    def train(self, X_train, y_train, X_val, y_val, epochs=config.EPOCHS, 
             batch_size=config.BATCH_SIZE):
        """
        Train the CNN model.
        
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
            model_dir = os.path.join(config.MODEL_DIR, f"cnn_{config.MODEL_VERSION}")
            os.makedirs(model_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(model_dir, "cnn_model_best.h5")
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
            final_path = os.path.join(model_dir, "cnn_model_final.h5")
            self.model.save(final_path)
            
            logger.info(f"CNN model trained for {len(history.history['loss'])} epochs and saved to {model_dir}")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            raise
            
    def predict(self, X):
        """
        Make predictions with the CNN model.
        
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
            logger.error(f"Error making predictions with CNN model: {e}")
            return None, None
            
    def evaluate(self, X_test, y_test):
        """
        Evaluate the CNN model.
        
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
            
            logger.info(f"CNN model evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            return loss, accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating CNN model: {e}")
            return None, None
            
    def save(self, path):
        """
        Save the CNN model to disk.
        
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
            
            logger.info(f"CNN model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving CNN model: {e}")
            
    def load(self, path):
        """
        Load a CNN model from disk.
        
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
            
            logger.info(f"CNN model loaded from {path}")
            logger.info(f"Loaded model input shape: {self.input_shape}")
            logger.info(f"Loaded model output shape: {self.output_dim}")
            
        except Exception as e:
            logger.error(f"Error loading CNN model from {path}: {e}")
            raise
