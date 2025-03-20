"""
Transformer model for sequence-based price prediction.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import logging

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transformer_model")

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head attention and feed-forward network."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerModel:
    def __init__(self, input_shape, output_dim=3, model_path=None):
        """
        Initialize the Transformer model.
        
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
        """Build the Transformer model architecture."""
        try:
            # Get dimensions from input shape
            seq_length, n_features = self.input_shape
            
            # Input layer
            inputs = Input(shape=self.input_shape)
            
            # Transformer blocks
            x = inputs
            
            # First transformer block
            x = TransformerBlock(n_features, num_heads=4, ff_dim=64, rate=0.1)(x)
            
            # Second transformer block
            x = TransformerBlock(n_features, num_heads=4, ff_dim=128, rate=0.1)(x)
            
            # Global average pooling
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Fully connected layers
            x = Dense(128, activation="relu")(x)
            x = Dropout(0.2)(x)
            x = Dense(64, activation="relu")(x)
            x = Dropout(0.2)(x)
            
            # Output layer
            outputs = Dense(self.output_dim, activation="softmax")(x)
            
            # Create the model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info(f"Transformer model built with input shape {self.input_shape}")
            logger.info(f"Model summary: {model.summary()}")
            
        except Exception as e:
            logger.error(f"Error building Transformer model: {e}")
            raise
            
    def train(self, X_train, y_train, X_val, y_val, epochs=config.EPOCHS, 
             batch_size=config.BATCH_SIZE):
        """
        Train the Transformer model.
        
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
            model_dir = os.path.join(config.MODEL_DIR, f"transformer_{config.MODEL_VERSION}")
            os.makedirs(model_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(model_dir, "transformer_model_best.h5")
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
            final_path = os.path.join(model_dir, "transformer_model_final.h5")
            self.model.save(final_path)
            
            logger.info(f"Transformer model trained for {len(history.history['loss'])} epochs and saved to {model_dir}")
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            raise
            
    def predict(self, X):
        """
        Make predictions with the Transformer model.
        
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
            logger.error(f"Error making predictions with Transformer model: {e}")
            return None, None
            
    def evaluate(self, X_test, y_test):
        """
        Evaluate the Transformer model.
        
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
            
            logger.info(f"Transformer model evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            return loss, accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating Transformer model: {e}")
            return None, None
            
    def save(self, path):
        """
        Save the Transformer model to disk.
        
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
            
            logger.info(f"Transformer model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving Transformer model: {e}")
            
    def load(self, path):
        """
        Load a Transformer model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        try:
            # Load the model
            self.model = load_model(
                path, 
                custom_objects={"TransformerBlock": TransformerBlock}
            )
            
            # Update input shape from loaded model
            self.input_shape = self.model.input_shape[1:]
            
            # Update output dimension from loaded model
            self.output_dim = self.model.output_shape[1]
            
            logger.info(f"Transformer model loaded from {path}")
            logger.info(f"Loaded model input shape: {self.input_shape}")
            logger.info(f"Loaded model output shape: {self.output_dim}")
            
        except Exception as e:
            logger.error(f"Error loading Transformer model from {path}: {e}")
            raise
