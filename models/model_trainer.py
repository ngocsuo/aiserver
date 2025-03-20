"""
Model trainer module to coordinate the training of all models.
"""
import os
import numpy as np
import logging
from datetime import datetime

import config
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.cnn_model import CNNModel
from models.meta_learner import MetaLearner
from models.historical_similarity import HistoricalSimilarity

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
        self.training_histories = {}
        self.evaluation_results = {}
        self.model_paths = {}
        
    def train_lstm(self, sequence_data):
        """
        Train an LSTM model.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            
        Returns:
            tuple: (model, history)
        """
        try:
            # Unpack data
            X_train, y_train = sequence_data['train']
            X_val, y_val = sequence_data['val']
            X_test, y_test = sequence_data['test']
            
            if X_train is None or X_train.shape[0] == 0:
                logger.error("No training data for LSTM")
                return None, None
                
            # Get input shape (sequence_length, n_features)
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            # Create and train model
            lstm_model = LSTMModel(input_shape=input_shape, output_dim=3)
            history = lstm_model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE
            )
            
            # Evaluate model
            loss, accuracy = lstm_model.evaluate(X_test, y_test)
            
            # Store results
            self.models['lstm'] = lstm_model
            self.training_histories['lstm'] = history
            self.evaluation_results['lstm'] = {'loss': loss, 'accuracy': accuracy}
            
            # Save model path
            model_dir = os.path.join(config.MODEL_DIR, f"lstm_{config.MODEL_VERSION}")
            model_path = os.path.join(model_dir, "lstm_model_best.h5")
            self.model_paths['lstm'] = model_path
            
            logger.info(f"LSTM model trained with accuracy: {accuracy:.4f}")
            
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
            # Unpack data
            X_train, y_train = sequence_data['train']
            X_val, y_val = sequence_data['val']
            X_test, y_test = sequence_data['test']
            
            if X_train is None or X_train.shape[0] == 0:
                logger.error("No training data for Transformer")
                return None, None
                
            # Get input shape (sequence_length, n_features)
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            # Create and train model
            transformer_model = TransformerModel(input_shape=input_shape, output_dim=3)
            history = transformer_model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE
            )
            
            # Evaluate model
            loss, accuracy = transformer_model.evaluate(X_test, y_test)
            
            # Store results
            self.models['transformer'] = transformer_model
            self.training_histories['transformer'] = history
            self.evaluation_results['transformer'] = {'loss': loss, 'accuracy': accuracy}
            
            # Save model path
            model_dir = os.path.join(config.MODEL_DIR, f"transformer_{config.MODEL_VERSION}")
            model_path = os.path.join(model_dir, "transformer_model_best.h5")
            self.model_paths['transformer'] = model_path
            
            logger.info(f"Transformer model trained with accuracy: {accuracy:.4f}")
            
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
            # Unpack data
            X_train, y_train = image_data['train']
            X_val, y_val = image_data['val']
            X_test, y_test = image_data['test']
            
            if X_train is None or X_train.shape[0] == 0:
                logger.error("No training data for CNN")
                return None, None
                
            # Get input shape (seq_length, channels, 1)
            input_shape = X_train.shape[1:]
            
            # Create and train model
            cnn_model = CNNModel(input_shape=input_shape, output_dim=3)
            history = cnn_model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE
            )
            
            # Evaluate model
            loss, accuracy = cnn_model.evaluate(X_test, y_test)
            
            # Store results
            self.models['cnn'] = cnn_model
            self.training_histories['cnn'] = history
            self.evaluation_results['cnn'] = {'loss': loss, 'accuracy': accuracy}
            
            # Save model path
            model_dir = os.path.join(config.MODEL_DIR, f"cnn_{config.MODEL_VERSION}")
            model_path = os.path.join(model_dir, "cnn_model_best.h5")
            self.model_paths['cnn'] = model_path
            
            logger.info(f"CNN model trained with accuracy: {accuracy:.4f}")
            
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
            # Unpack data
            X_train, y_train = sequence_data['train']
            X_test, y_test = sequence_data['test']
            
            if X_train is None or X_train.shape[0] == 0:
                logger.error("No training data for Historical Similarity")
                return None, None
                
            # Create and train model
            historical_model = HistoricalSimilarity(
                sequence_length=config.SEQUENCE_LENGTH
            )
            
            # Train with full data
            historical_model.train(X_train, y_train)
            
            # Test with random samples
            test_samples = min(100, X_test.shape[0])
            random_indices = np.random.choice(X_test.shape[0], test_samples, replace=False)
            
            X_test_sample = X_test[random_indices]
            y_test_sample = y_test[random_indices]
            
            # Evaluate
            correct = 0
            for i in range(test_samples):
                pred, _, _, _ = historical_model.predict(X_test_sample[i:i+1])
                if pred == y_test_sample[i]:
                    correct += 1
                    
            accuracy = correct / test_samples
            
            # Store results
            self.models['historical'] = historical_model
            self.training_histories['historical'] = None
            self.evaluation_results['historical'] = {'accuracy': accuracy}
            
            # Save model path
            model_dir = os.path.join(config.MODEL_DIR, f"historical_{config.MODEL_VERSION}")
            model_path = os.path.join(model_dir, "historical_patterns.pkl")
            self.model_paths['historical'] = model_path
            
            logger.info(f"Historical Similarity model trained with accuracy: {accuracy:.4f}")
            
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
            # Unpack data
            X_train_seq, y_train = sequence_data['train']
            X_val_seq, y_val = sequence_data['val']
            X_test_seq, y_test = sequence_data['test']
            
            X_train_img = image_data['train'][0]
            X_val_img = image_data['val'][0]
            X_test_img = image_data['test'][0]
            
            if X_train_seq is None or X_train_seq.shape[0] == 0:
                logger.error("No training data for Meta-Learner")
                return None, None
                
            # Make predictions with base models
            base_models = ['lstm', 'transformer', 'cnn', 'historical']
            train_probs = []
            val_probs = []
            test_probs = []
            
            for model_name in base_models:
                if model_name in self.models:
                    model = self.models[model_name]
                    
                    if model_name == 'lstm' or model_name == 'transformer':
                        _, train_prob = model.predict(X_train_seq)
                        _, val_prob = model.predict(X_val_seq)
                        _, test_prob = model.predict(X_test_seq)
                    elif model_name == 'cnn':
                        _, train_prob = model.predict(X_train_img)
                        _, val_prob = model.predict(X_val_img)
                        _, test_prob = model.predict(X_test_img)
                    elif model_name == 'historical':
                        # For historical model, generate probabilities for each sample
                        train_prob = np.zeros((X_train_seq.shape[0], 3))
                        val_prob = np.zeros((X_val_seq.shape[0], 3))
                        test_prob = np.zeros((X_test_seq.shape[0], 3))
                        
                        for i in range(X_train_seq.shape[0]):
                            _, prob, _, _ = model.predict(X_train_seq[i:i+1])
                            if prob is not None:
                                train_prob[i] = prob
                                
                        for i in range(X_val_seq.shape[0]):
                            _, prob, _, _ = model.predict(X_val_seq[i:i+1])
                            if prob is not None:
                                val_prob[i] = prob
                                
                        for i in range(X_test_seq.shape[0]):
                            _, prob, _, _ = model.predict(X_test_seq[i:i+1])
                            if prob is not None:
                                test_prob[i] = prob
                    
                    train_probs.append(train_prob)
                    val_probs.append(val_prob)
                    test_probs.append(test_prob)
            
            if not train_probs:
                logger.error("No base model predictions available for Meta-Learner")
                return None, None
                
            # Create and train meta-learner
            meta_model = MetaLearner(model_type='gbdt')
            meta_model.train(train_probs, y_train)
            
            # Evaluate
            accuracy = meta_model.evaluate(test_probs, y_test)
            
            # Store results
            self.models['meta'] = meta_model
            self.training_histories['meta'] = None
            self.evaluation_results['meta'] = {'accuracy': accuracy}
            
            # Save model path
            model_dir = os.path.join(config.MODEL_DIR, f"meta_{config.MODEL_VERSION}")
            model_path = os.path.join(model_dir, "meta_learner_gbdt.pkl")
            self.model_paths['meta'] = model_path
            
            logger.info(f"Meta-Learner model trained with accuracy: {accuracy:.4f}")
            
            return meta_model, None
            
        except Exception as e:
            logger.error(f"Error training Meta-Learner model: {e}")
            return None, None
            
    def train_all_models(self, sequence_data, image_data):
        """
        Train all models in the ensemble.
        
        Args:
            sequence_data (dict): Dictionary with sequence data for training
            image_data (dict): Dictionary with image data for training
            
        Returns:
            dict: Trained models
        """
        try:
            logger.info("Starting training of all models")
            
            # Train base models
            self.train_lstm(sequence_data)
            self.train_transformer(sequence_data)
            self.train_cnn(image_data)
            self.train_historical_similarity(sequence_data)
            
            # Train meta-learner
            self.train_meta_learner(sequence_data, image_data)
            
            # Log overall results
            logger.info("All models trained successfully")
            logger.info(f"Model evaluation results: {self.evaluation_results}")
            
            return self.models
            
        except Exception as e:
            logger.error(f"Error training all models: {e}")
            return {}
            
    def load_models(self):
        """
        Load all saved models.
        
        Returns:
            dict: Loaded models
        """
        try:
            logger.info("Loading saved models")
            
            # Check if model dir exists
            if not os.path.exists(config.MODEL_DIR):
                logger.warning(f"Model directory {config.MODEL_DIR} does not exist")
                return {}
                
            # List version directories
            version_dirs = [d for d in os.listdir(config.MODEL_DIR) if os.path.isdir(os.path.join(config.MODEL_DIR, d))]
            
            if not version_dirs:
                logger.warning("No model versions found")
                return {}
                
            # Get the latest version by parsing timestamp from directory name
            latest_version = max(version_dirs, key=lambda x: x.split('_')[-1] if '_' in x else '0')
            
            # Load models by type
            models = {}
            
            # LSTM
            lstm_dir = os.path.join(config.MODEL_DIR, f"lstm_{latest_version}")
            lstm_path = os.path.join(lstm_dir, "lstm_model_best.h5")
            if os.path.exists(lstm_path):
                models['lstm'] = LSTMModel(input_shape=None, model_path=lstm_path)
                logger.info(f"Loaded LSTM model from {lstm_path}")
                
            # Transformer
            transformer_dir = os.path.join(config.MODEL_DIR, f"transformer_{latest_version}")
            transformer_path = os.path.join(transformer_dir, "transformer_model_best.h5")
            if os.path.exists(transformer_path):
                models['transformer'] = TransformerModel(input_shape=None, model_path=transformer_path)
                logger.info(f"Loaded Transformer model from {transformer_path}")
                
            # CNN
            cnn_dir = os.path.join(config.MODEL_DIR, f"cnn_{latest_version}")
            cnn_path = os.path.join(cnn_dir, "cnn_model_best.h5")
            if os.path.exists(cnn_path):
                models['cnn'] = CNNModel(input_shape=None, model_path=cnn_path)
                logger.info(f"Loaded CNN model from {cnn_path}")
                
            # Historical Similarity
            historical_dir = os.path.join(config.MODEL_DIR, f"historical_{latest_version}")
            historical_path = os.path.join(historical_dir, "historical_patterns.pkl")
            if os.path.exists(historical_path):
                models['historical'] = HistoricalSimilarity(model_path=historical_path)
                logger.info(f"Loaded Historical Similarity model from {historical_path}")
                
            # Meta-Learner
            meta_dir = os.path.join(config.MODEL_DIR, f"meta_{latest_version}")
            meta_path = os.path.join(meta_dir, "meta_learner_gbdt.pkl")
            if os.path.exists(meta_path):
                models['meta'] = MetaLearner(model_path=meta_path)
                logger.info(f"Loaded Meta-Learner model from {meta_path}")
                
            self.models = models
            logger.info(f"Loaded {len(models)} models from version {latest_version}")
            
            return models
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return {}
