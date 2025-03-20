"""
Meta-learner model that combines predictions from multiple base models.
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import logging

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
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self.build()
            
    def build(self):
        """Build the meta-learner model."""
        try:
            if self.model_type == 'logistic':
                # Logistic Regression meta-learner
                self.model = LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight='balanced',
                    multi_class='multinomial',
                    solver='lbfgs',
                    random_state=42
                )
            elif self.model_type == 'gbdt':
                # Gradient Boosting Decision Tree meta-learner
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            logger.info(f"Meta-learner model ({self.model_type}) initialized")
            
        except Exception as e:
            logger.error(f"Error building meta-learner model: {e}")
            raise
            
    def prepare_meta_features(self, base_model_probs):
        """
        Prepare meta-features from base model probabilities.
        
        Args:
            base_model_probs (list): List of probability arrays from base models
            
        Returns:
            np.ndarray: Combined meta-features
        """
        try:
            # Stack all probability outputs as features
            meta_features = np.hstack(base_model_probs)
            
            return meta_features
            
        except Exception as e:
            logger.error(f"Error preparing meta-features: {e}")
            return None
            
    def train(self, base_model_probs, y_train):
        """
        Train the meta-learner model.
        
        Args:
            base_model_probs (list): List of probability arrays from base models
            y_train (np.ndarray): Training labels
            
        Returns:
            object: Trained model
        """
        if self.model is None:
            logger.error("Model not initialized. Call build() first.")
            return None
            
        try:
            # Prepare meta-features
            meta_features = self.prepare_meta_features(base_model_probs)
            
            if meta_features is None:
                return None
                
            # Train the meta-learner
            self.model.fit(meta_features, y_train)
            
            logger.info(f"Meta-learner model trained on {meta_features.shape[0]} samples with {meta_features.shape[1]} features")
            
            # Save the model
            model_dir = os.path.join(config.MODEL_DIR, f"meta_{config.MODEL_VERSION}")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f"meta_learner_{self.model_type}.pkl")
            self.save(model_path)
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training meta-learner model: {e}")
            raise
            
    def predict(self, base_model_probs):
        """
        Make predictions with the meta-learner model.
        
        Args:
            base_model_probs (list): List of probability arrays from base models
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            logger.error("Model not initialized. Call build() or load() first.")
            return None, None
            
        try:
            # Prepare meta-features
            meta_features = self.prepare_meta_features(base_model_probs)
            
            if meta_features is None:
                return None, None
                
            # Get class probabilities (if the model supports predict_proba)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(meta_features)
            else:
                # For models without predict_proba, create dummy probabilities
                predictions = self.model.predict(meta_features)
                n_classes = len(np.unique(predictions))
                probabilities = np.zeros((len(predictions), n_classes))
                probabilities[np.arange(len(predictions)), predictions] = 1
                
            # Get class predictions
            predictions = self.model.predict(meta_features)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions with meta-learner model: {e}")
            return None, None
            
    def evaluate(self, base_model_probs, y_test):
        """
        Evaluate the meta-learner model.
        
        Args:
            base_model_probs (list): List of probability arrays from base models
            y_test (np.ndarray): Test labels
            
        Returns:
            float: Accuracy score
        """
        if self.model is None:
            logger.error("Model not initialized. Call build() or load() first.")
            return None
            
        try:
            # Prepare meta-features
            meta_features = self.prepare_meta_features(base_model_probs)
            
            if meta_features is None:
                return None
                
            # Evaluate the model
            accuracy = self.model.score(meta_features, y_test)
            
            logger.info(f"Meta-learner model evaluation - Accuracy: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating meta-learner model: {e}")
            return None
            
    def save(self, path):
        """
        Save the meta-learner model to disk.
        
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
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
                
            logger.info(f"Meta-learner model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving meta-learner model: {e}")
            
    def load(self, path):
        """
        Load a meta-learner model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        try:
            # Load the model
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
                
            # Determine model type from loaded model
            if isinstance(self.model, LogisticRegression):
                self.model_type = 'logistic'
            elif isinstance(self.model, GradientBoostingClassifier):
                self.model_type = 'gbdt'
            else:
                self.model_type = 'unknown'
                
            logger.info(f"Meta-learner model ({self.model_type}) loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading meta-learner model from {path}: {e}")
            raise
