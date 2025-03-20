"""
Historical similarity model to find and match price patterns.
"""
import os
import logging
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
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
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.historical_patterns = []
        self.historical_labels = []
        self.mock_model = {
            'name': 'Historical Similarity',
            'accuracy': 0.65,
            'patterns': np.random.random((10, sequence_length)),
            'labels': np.random.randint(0, 3, 10)
        }
        
        # Try to load the model if path is provided
        if model_path is not None and os.path.exists(model_path):
            self.load(model_path)
            logger.info(f"Historical patterns loaded from {model_path}")
    
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
            pattern = pattern.reshape(-1, 1)
            
            if fit:
                return self.scaler.fit_transform(pattern).flatten()
            else:
                return self.scaler.transform(pattern).flatten()
                
        except Exception as e:
            logger.error(f"Error normalizing pattern: {e}")
            return pattern
    
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
            
            patterns = []
            pattern_labels = []
            
            for i in range(n_samples):
                # Extract closing prices from sequence
                # In a real implementation, this would use the actual close price column
                pattern = data[i, :, 0]  # Assume close price is the first feature
                
                # Normalize pattern
                normalized_pattern = self.normalize_pattern(pattern, fit=(i == 0))
                
                patterns.append(normalized_pattern)
                pattern_labels.append(labels[i])
            
            return np.array(patterns), np.array(pattern_labels)
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            return np.array([]), np.array([])
    
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
            logger.info("Training historical similarity model (placeholder)")
            
            # Extract patterns from data
            patterns, pattern_labels = self.extract_patterns(data, labels)
            
            # Store patterns for later use
            self.historical_patterns = patterns
            self.historical_labels = pattern_labels
            
            logger.info(f"Stored {len(patterns)} historical patterns")
            
            return patterns, pattern_labels
            
        except Exception as e:
            logger.error(f"Error training historical similarity model: {e}")
            # Use mock patterns for demonstration
            self.historical_patterns = self.mock_model['patterns']
            self.historical_labels = self.mock_model['labels']
            return self.historical_patterns, self.historical_labels
    
    def find_similar_patterns(self, query_pattern, top_k=5):
        """
        Find the most similar historical patterns.
        
        Args:
            query_pattern (np.ndarray): Query price pattern
            top_k (int): Number of similar patterns to return
            
        Returns:
            tuple: (indices, similarities, majority_label, confidence)
        """
        try:
            if len(self.historical_patterns) == 0:
                logger.warning("No historical patterns available")
                # Return dummy values
                return (
                    np.array([]),
                    np.array([]),
                    1,  # NEUTRAL
                    0.5
                )
            
            # Normalize query pattern
            query_pattern = self.normalize_pattern(query_pattern)
            
            # Calculate Euclidean distance between query and all historical patterns
            distances = np.array([
                np.sum((query_pattern - pattern) ** 2) 
                for pattern in self.historical_patterns
            ])
            
            # Convert distances to similarities
            similarities = 1.0 / (1.0 + distances)
            
            # Find top-k similar patterns
            if top_k > len(similarities):
                top_k = len(similarities)
                
            top_indices = np.argsort(similarities)[-top_k:]
            top_similarities = similarities[top_indices]
            top_labels = self.historical_labels[top_indices]
            
            # Determine majority label
            if len(top_labels) > 0:
                label_counts = np.bincount(top_labels)
                majority_label = np.argmax(label_counts)
                confidence = label_counts[majority_label] / len(top_labels)
            else:
                majority_label = 1  # NEUTRAL
                confidence = 0.5
            
            return top_indices, top_similarities, majority_label, confidence
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return np.array([]), np.array([]), 1, 0.5
    
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
            logger.info("Making prediction with historical similarity model (placeholder)")
            
            # For demonstration, generate random predictions
            # In a real implementation, this would use the actual pattern matching logic
            
            if len(self.historical_patterns) == 0:
                # Use the mock model's patterns
                self.historical_patterns = self.mock_model['patterns']
                self.historical_labels = self.mock_model['labels']
            
            # Find similar patterns
            indices, similarities, prediction, confidence = self.find_similar_patterns(
                query_pattern, top_k
            )
            
            # If no similar patterns found, use random prediction
            if len(indices) == 0:
                prediction = np.random.randint(0, 3)
                confidence = np.random.uniform(0.6, 0.7)
                
            return prediction, confidence, indices, similarities
            
        except Exception as e:
            logger.error(f"Error making predictions with historical similarity model: {e}")
            # Return fallback prediction (NEUTRAL)
            return 1, 0.5, np.array([]), np.array([])
    
    def save(self, path):
        """
        Save the historical patterns to disk.
        
        Args:
            path (str): Path to save the model
        """
        try:
            # Save patterns and labels
            model_data = {
                'historical_patterns': self.historical_patterns,
                'historical_labels': self.historical_labels,
                'scaler': self.scaler
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
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.historical_patterns = model_data.get('historical_patterns', [])
            self.historical_labels = model_data.get('historical_labels', [])
            self.scaler = model_data.get('scaler', MinMaxScaler(feature_range=(-1, 1)))
            
            logger.info(f"Loaded {len(self.historical_patterns)} historical patterns from {path}")
            
        except Exception as e:
            logger.error(f"Error loading historical similarity model: {e}")
            # Use the mock model for demonstration
            self.historical_patterns = self.mock_model['patterns']
            self.historical_labels = self.mock_model['labels']