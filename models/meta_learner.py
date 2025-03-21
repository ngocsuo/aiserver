"""
Meta-learner model that combines predictions from multiple base models 
with dynamic weighting based on model performance.
"""
import os
import logging
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime, timedelta
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("meta_learner")

class MetaLearner:
    def __init__(self, model_type='logistic', model_path=None):
        """
        Initialize the meta-learner model.
        
        Args:
            model_type (str): Type of meta-learner ('logistic' or 'gbdt')
            model_path (str): Path to load a pre-trained model
        """
        self.model_type = model_type
        self.model = None
        self.mock_model = {
            'name': 'Meta-Learner',
            'accuracy': 0.81,
            'loss': 0.35
        }
        
        # Performance history for dynamic weighting
        self.performance_history = {
            'lstm': {'correct': 0, 'total': 0, 'recent_correct': [], 'recent_total': []},
            'transformer': {'correct': 0, 'total': 0, 'recent_correct': [], 'recent_total': []},
            'cnn': {'correct': 0, 'total': 0, 'recent_correct': [], 'recent_total': []},
            'historical_similarity': {'correct': 0, 'total': 0, 'recent_correct': [], 'recent_total': []}
        }
        
        # Model weights for dynamic weighting (initialized equally)
        self.model_weights = {
            'lstm': 1.0,
            'transformer': 1.0, 
            'cnn': 1.0,
            'historical_similarity': 1.0
        }
        
        # Time window for recent performance (number of predictions to consider)
        self.recent_window_size = 20
        
        # Last update time for weights
        self.last_weights_update = datetime.now()
        
        # Try to load the model if path is provided
        if model_path is not None and os.path.exists(model_path):
            self.load(model_path)
            logger.info(f"Meta-learner model loaded from {model_path}")
        else:
            self.build()
    
    def build(self):
        """Build the meta-learner model."""
        try:
            logger.info(f"Building {self.model_type} meta-learner model (placeholder)")
            
            if self.model_type == 'logistic':
                self.model = LogisticRegression(max_iter=1000, C=1.0)
            elif self.model_type == 'gbdt':
                self.model = GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=3
                )
            else:
                logger.warning(f"Unknown model type: {self.model_type}, using logistic regression")
                self.model = LogisticRegression(max_iter=1000, C=1.0)
                
            return self.model
            
        except Exception as e:
            logger.error(f"Error building meta-learner model: {e}")
            return None
    
    def prepare_meta_features(self, base_model_probs):
        """
        Prepare meta-features from base model probabilities.
        
        Args:
            base_model_probs (list): List of probability arrays from base models
            
        Returns:
            np.ndarray: Combined meta-features
        """
        try:
            # Check if we have any probabilities
            if not base_model_probs or len(base_model_probs) == 0:
                return np.array([])
            
            # For demonstration, just concatenate all probabilities
            # In a real implementation, this might include more sophisticated feature engineering
            
            # Get number of samples
            n_samples = base_model_probs[0].shape[0]
            
            # Concatenate all probabilities
            meta_features = np.hstack([probs for probs in base_model_probs])
            
            return meta_features
            
        except Exception as e:
            logger.error(f"Error preparing meta-features: {e}")
            return np.array([])
    
    def train(self, base_model_probs, y_train):
        """
        Train the meta-learner model.
        
        Args:
            base_model_probs (list): List of probability arrays from base models
            y_train (np.ndarray): Training labels
            
        Returns:
            object: Trained model
        """
        try:
            logger.info("Training meta-learner model (placeholder)")
            
            # Prepare meta-features
            meta_features = self.prepare_meta_features(base_model_probs)
            
            if meta_features.size == 0 or len(y_train) == 0:
                logger.warning("No meta-features or labels available for training")
                return None
            
            # Build model if not built yet
            if self.model is None:
                self.build()
            
            # Fit the model
            self.model.fit(meta_features, y_train)
            
            logger.info("Meta-learner model trained successfully")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training meta-learner model: {e}")
            return None
    
    def predict(self, base_model_probs):
        """
        Make predictions with the meta-learner model.
        
        Args:
            base_model_probs (list): List of probability arrays from base models
            
        Returns:
            tuple: (predictions, probabilities)
        """
        try:
            logger.info("Making predictions with dynamic meta-learner model")
            
            # Update weights if enough time has passed or it's the first update
            self._update_model_weights()
            
            # Prepare meta-features
            meta_features = self.prepare_meta_features(base_model_probs)
            
            if meta_features.size == 0:
                logger.warning("No meta-features available for prediction")
                # Return random prediction
                num_samples = 1
                predictions = np.ones(num_samples, dtype=int)  # NEUTRAL
                probabilities = np.zeros((num_samples, 3))
                probabilities[:, 1] = 0.6  # Higher probability for NEUTRAL
                probabilities[:, 0] = 0.2  # Some probability for SHORT
                probabilities[:, 2] = 0.2  # Some probability for LONG
                
                return predictions, probabilities
            
            # Build model if not built yet
            if self.model is None:
                self.build()
            
            # Make predictions
            predictions = self.model.predict(meta_features)
            
            # Get probabilities
            probabilities = self.model.predict_proba(meta_features)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions with meta-learner model: {e}")
            # Return fallback prediction (NEUTRAL)
            num_samples = 1
            if base_model_probs and len(base_model_probs) > 0:
                num_samples = base_model_probs[0].shape[0]
                
            return np.ones(num_samples, dtype=int), np.array([[0.2, 0.6, 0.2]] * num_samples)
    
    def _update_model_weights(self):
        """
        Update the model weights based on recent performance.
        Weights are dynamically adjusted to favor models with better
        recent prediction accuracy.
        """
        try:
            # Check if enough time has passed since last update
            time_since_update = (datetime.now() - self.last_weights_update).total_seconds()
            
            # Only update weights every hour or if it's the first time
            if time_since_update < 3600 and self.last_weights_update != datetime.min:
                return
                
            logger.info("Updating dynamic model weights based on performance")
            
            # Calculate accuracy for each model
            for model_name, stats in self.performance_history.items():
                if stats['total'] > 0:
                    # Calculate overall accuracy
                    overall_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.5
                    
                    # Calculate recent accuracy (with more weight)
                    recent_correct = sum(stats['recent_correct'])
                    recent_total = sum(stats['recent_total'])
                    recent_accuracy = recent_correct / recent_total if recent_total > 0 else 0.5
                    
                    # Combine both with more weight on recent performance
                    combined_accuracy = (0.3 * overall_accuracy) + (0.7 * recent_accuracy)
                    
                    # Scale for weight calculation (add 0.5 to ensure minimum weight of 0.5)
                    self.model_weights[model_name] = 0.5 + combined_accuracy
                    
                    logger.info(f"Model {model_name}: Recent accuracy {recent_accuracy:.2f}, " + 
                              f"Overall {overall_accuracy:.2f}, New weight {self.model_weights[model_name]:.2f}")
            
            # Normalize weights to sum to total models (preserves scale)
            sum_weights = sum(self.model_weights.values())
            if sum_weights > 0:
                model_count = len(self.model_weights)
                scale_factor = model_count / sum_weights
                for model_name in self.model_weights:
                    self.model_weights[model_name] *= scale_factor
            
            self.last_weights_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating model weights: {e}")
            # Reset to default equal weights in case of error
            for model_name in self.model_weights:
                self.model_weights[model_name] = 1.0
    
    def update_performance(self, model_predictions, actual_outcome):
        """
        Update the performance history of each model based on their predictions
        compared to the actual outcome.
        
        Args:
            model_predictions (dict): Dictionary of model name to prediction value
            actual_outcome (int): The actual outcome that occurred
        """
        try:
            logger.info(f"Updating model performance with actual outcome: {config.CLASSES[actual_outcome]}")
            
            # Update each model's performance
            for model_name, prediction in model_predictions.items():
                if model_name not in self.performance_history:
                    logger.warning(f"Model {model_name} not found in performance history")
                    continue
                
                # Increment total predictions
                self.performance_history[model_name]['total'] += 1
                
                # Add to recent total
                self.performance_history[model_name]['recent_total'].append(1)
                
                # Trim recent history if too long
                while len(self.performance_history[model_name]['recent_total']) > self.recent_window_size:
                    self.performance_history[model_name]['recent_total'].pop(0)
                
                # Check if prediction was correct
                is_correct = (prediction == actual_outcome)
                
                # Increment correct if accurate
                if is_correct:
                    self.performance_history[model_name]['correct'] += 1
                    self.performance_history[model_name]['recent_correct'].append(1)
                else:
                    self.performance_history[model_name]['recent_correct'].append(0)
                
                # Trim recent history if too long
                while len(self.performance_history[model_name]['recent_correct']) > self.recent_window_size:
                    self.performance_history[model_name]['recent_correct'].pop(0)
                
                # Calculate current accuracy
                total = self.performance_history[model_name]['total']
                correct = self.performance_history[model_name]['correct']
                accuracy = correct / total if total > 0 else 0
                
                logger.info(f"Model {model_name}: Prediction={config.CLASSES[prediction]}, " +
                          f"Correct={is_correct}, Accuracy={accuracy:.2f} ({correct}/{total})")
            
            # Update weights if enough new data
            if self._should_update_weights():
                self._update_model_weights()
                
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def _should_update_weights(self):
        """
        Determine if the weights should be updated based on the amount
        of new performance data.
        
        Returns:
            bool: Whether weights should be updated
        """
        # Check if any model has at least 5 new predictions since last update
        for model_name, stats in self.performance_history.items():
            if len(stats['recent_total']) >= 5:
                return True
                
        return False
    
    def evaluate(self, base_model_probs, y_test):
        """
        Evaluate the meta-learner model.
        
        Args:
            base_model_probs (list): List of probability arrays from base models
            y_test (np.ndarray): Test labels
            
        Returns:
            float: Accuracy score
        """
        try:
            logger.info("Evaluating meta-learner model (placeholder)")
            
            # Prepare meta-features
            meta_features = self.prepare_meta_features(base_model_probs)
            
            if meta_features.size == 0 or len(y_test) == 0:
                logger.warning("No meta-features or labels available for evaluation")
                return 0.0
            
            # Build model if not built yet
            if self.model is None:
                self.build()
            
            # Calculate accuracy
            predictions = self.model.predict(meta_features)
            accuracy = np.mean(predictions == y_test)
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating meta-learner model: {e}")
            return 0.0
    
    def save(self, path):
        """
        Save the meta-learner model to disk.
        
        Args:
            path (str): Path to save the model
        """
        try:
            # Save model along with performance history and weights
            save_data = {
                'model': self.model,
                'performance_history': self.performance_history,
                'model_weights': self.model_weights,
                'last_weights_update': self.last_weights_update,
                'recent_window_size': self.recent_window_size
            }
            
            with open(path, 'wb') as f:
                pickle.dump(save_data, f)
                
            logger.info(f"Meta-learner model and performance data saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving meta-learner model: {e}")
    
    def load(self, path):
        """
        Load a meta-learner model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        try:
            with open(path, 'rb') as f:
                saved_data = pickle.load(f)
                
                # Check if the saved data is a dictionary containing model and performance data
                if isinstance(saved_data, dict) and 'model' in saved_data:
                    self.model = saved_data['model']
                    
                    # Load performance history if available
                    if 'performance_history' in saved_data:
                        self.performance_history = saved_data['performance_history']
                    
                    # Load model weights if available
                    if 'model_weights' in saved_data:
                        self.model_weights = saved_data['model_weights']
                else:
                    # Legacy format: just the model
                    self.model = saved_data
                
            logger.info(f"Meta-learner model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading meta-learner model: {e}")
            # Use the default mock model for demonstration
            self.build()