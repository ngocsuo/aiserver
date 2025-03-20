"""
Historical similarity model to find and match price patterns.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("historical_similarity")

class HistoricalSimilarity:
    def __init__(self, sequence_length=config.SEQUENCE_LENGTH, model_path=None):
        """
        Initialize the historical similarity model.
        
        Args:
            sequence_length (int): Length of price sequences to compare
            model_path (str): Path to load historical patterns
        """
        self.sequence_length = sequence_length
        self.historical_patterns = None
        self.historical_labels = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            
    def normalize_pattern(self, pattern, fit=False):
        """
        Normalize a price pattern to make it comparable.
        
        Args:
            pattern (np.ndarray): Price pattern to normalize
            fit (bool): Whether to fit the scaler
            
        Returns:
            np.ndarray: Normalized pattern
        """
        try:
            # Reshape for scaler
            reshaped = pattern.reshape(-1, pattern.shape[-1])
            
            if fit:
                normalized = self.scaler.fit_transform(reshaped)
            else:
                normalized = self.scaler.transform(reshaped)
                
            # Reshape back
            normalized = normalized.reshape(pattern.shape)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing pattern: {e}")
            return None
            
    def extract_patterns(self, data, labels):
        """
        Extract historical patterns from price data.
        
        Args:
            data (np.ndarray): Sequential price data
            labels (np.ndarray): Corresponding labels
            
        Returns:
            tuple: (patterns, pattern_labels)
        """
        try:
            n_samples = data.shape[0]
            
            # Extract key features for pattern matching
            # Using a small subset of features for better pattern recognition
            price_indices = [0, 1, 2, 3]  # Assuming OHLC are first 4 features
            patterns = data[:, :, price_indices]
            
            # Normalize patterns
            normalized_patterns = self.normalize_pattern(patterns, fit=True)
            
            logger.info(f"Extracted {n_samples} historical patterns")
            
            return normalized_patterns, labels
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            return None, None
            
    def train(self, data, labels):
        """
        Train the historical similarity model by storing patterns.
        
        Args:
            data (np.ndarray): Sequential price data
            labels (np.ndarray): Corresponding labels
            
        Returns:
            tuple: (historical_patterns, historical_labels)
        """
        try:
            # Extract and store patterns
            self.historical_patterns, self.historical_labels = self.extract_patterns(data, labels)
            
            # Save the model
            model_dir = os.path.join(config.MODEL_DIR, f"historical_{config.MODEL_VERSION}")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, "historical_patterns.pkl")
            self.save(model_path)
            
            logger.info(f"Historical similarity model trained with {len(self.historical_labels)} patterns")
            
            return self.historical_patterns, self.historical_labels
            
        except Exception as e:
            logger.error(f"Error training historical similarity model: {e}")
            raise
            
    def find_similar_patterns(self, query_pattern, top_k=5):
        """
        Find the most similar historical patterns.
        
        Args:
            query_pattern (np.ndarray): Query price pattern
            top_k (int): Number of similar patterns to return
            
        Returns:
            tuple: (indices, similarities, majority_label, confidence)
        """
        if self.historical_patterns is None or self.historical_labels is None:
            logger.error("No historical patterns. Call train() first.")
            return None, None, None, 0
            
        try:
            # Extract key features from query pattern (same as in training)
            price_indices = [0, 1, 2, 3]  # Assuming OHLC are first 4 features
            query = query_pattern[:, price_indices]
            
            # Normalize query pattern
            query_norm = self.normalize_pattern(query)
            
            if query_norm is None:
                return None, None, None, 0
                
            # Flatten patterns for comparison
            n_patterns = self.historical_patterns.shape[0]
            
            # Reshape for comparison
            query_flat = query_norm.reshape(1, -1)
            historical_flat = self.historical_patterns.reshape(n_patterns, -1)
            
            # Calculate similarities
            similarities = cosine_similarity(query_flat, historical_flat)[0]
            
            # Get top-k most similar patterns
            top_indices = np.argsort(similarities)[-top_k:]
            top_similarities = similarities[top_indices]
            
            # Get labels of similar patterns
            similar_labels = self.historical_labels[top_indices]
            
            # Count label frequencies
            label_counts = np.bincount(similar_labels.astype(int))
            majority_label = np.argmax(label_counts)
            
            # Calculate confidence as normalized count of majority label
            confidence = label_counts[majority_label] / top_k
            
            return top_indices, top_similarities, majority_label, confidence
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return None, None, None, 0
            
    def predict(self, query_pattern, top_k=5):
        """
        Make predictions based on historical pattern similarity.
        
        Args:
            query_pattern (np.ndarray): Query price pattern
            top_k (int): Number of similar patterns to consider
            
        Returns:
            tuple: (prediction, confidence, similar_indices, similarities)
        """
        try:
            # Find similar patterns
            indices, similarities, label, confidence = self.find_similar_patterns(
                query_pattern, top_k
            )
            
            if indices is None:
                return None, 0, None, None
                
            logger.info(f"Prediction based on {top_k} similar patterns: {label} with confidence {confidence:.2f}")
            
            # For compatibility with other models
            probabilities = np.zeros(3)  # 3 classes: SHORT, NEUTRAL, LONG
            probabilities[label] = confidence
            
            return label, probabilities, indices, similarities
            
        except Exception as e:
            logger.error(f"Error making predictions with historical similarity: {e}")
            return None, None, None, None
            
    def save(self, path):
        """
        Save the historical patterns to disk.
        
        Args:
            path (str): Path to save the model
        """
        if self.historical_patterns is None or self.historical_labels is None:
            logger.error("No patterns to save. Call train() first.")
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model components
            model_data = {
                'patterns': self.historical_patterns,
                'labels': self.historical_labels,
                'scaler': self.scaler,
                'sequence_length': self.sequence_length
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Historical similarity model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving historical similarity model: {e}")
            
    def load(self, path):
        """
        Load historical patterns from disk.
        
        Args:
            path (str): Path to the saved patterns
        """
        try:
            # Load the model components
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.historical_patterns = model_data['patterns']
            self.historical_labels = model_data['labels']
            self.scaler = model_data['scaler']
            self.sequence_length = model_data['sequence_length']
            
            logger.info(f"Historical similarity model loaded from {path} with {len(self.historical_labels)} patterns")
            
        except Exception as e:
            logger.error(f"Error loading historical similarity model from {path}: {e}")
            raise
