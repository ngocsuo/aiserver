"""
Data processing module for preparing datasets for training and prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

import config
from utils.feature_engineering import FeatureEngineer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_processor")

class DataProcessor:
    def __init__(self):
        """Initialize the data processor with a feature engineer."""
        self.feature_engineer = FeatureEngineer()
        self.processed_data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.sequence_length = config.SEQUENCE_LENGTH
        
    def process_data(self, raw_data):
        """
        Process raw OHLCV data for model training.
        
        Args:
            raw_data (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: Processed data with features and targets
        """
        if raw_data is None or raw_data.empty:
            logger.error("No data provided for processing")
            return None
            
        try:
            # Apply full preprocessing pipeline
            self.processed_data = self.feature_engineer.preprocess_data(raw_data)
            logger.info(f"Data processed successfully: {len(self.processed_data)} samples")
            return self.processed_data
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
            
    def split_train_val_test(self, data=None, target_col='target_class'):
        """
        Split data into training, validation, and test sets.
        
        Args:
            data (pd.DataFrame): Processed data (uses self.processed_data if None)
            target_col (str): Target column name
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if data is None:
            data = self.processed_data
            
        if data is None or data.empty:
            logger.error("No data available for splitting")
            return None, None, None, None, None, None
            
        if target_col not in data.columns:
            logger.error(f"Target column '{target_col}' not found in data")
            return None, None, None, None, None, None
            
        try:
            # Get feature and target data
            feature_cols = [col for col in data.columns if not col.startswith('target_')]
            X = data[feature_cols]
            y = data[target_col]
            
            # First split: training vs (validation + test)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, 
                test_size=(config.VALIDATION_SPLIT + config.TEST_SPLIT),
                shuffle=False  # Preserve time order for time series data
            )
            
            # Second split: validation vs test
            test_ratio = config.TEST_SPLIT / (config.VALIDATION_SPLIT + config.TEST_SPLIT)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=test_ratio,
                shuffle=False  # Preserve time order
            )
            
            # Store the splits
            self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
            self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
            
            logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return None, None, None, None, None, None
            
    def prepare_sequence_data(self, data=None, target_col='target_class'):
        """
        Prepare sequence data for LSTM and Transformer models.
        
        Args:
            data (pd.DataFrame): Processed data (uses processed_data if None)
            target_col (str): Target column name
            
        Returns:
            dict: Dictionary with sequence data for training and evaluation
        """
        if data is None:
            data = self.processed_data
            
        if data is None or data.empty:
            logger.error("No data available for sequence preparation")
            return None
            
        try:
            # Split the data first
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_train_val_test(
                data, target_col
            )
            
            if X_train is None:
                return None
                
            # Create sequences for each split
            train_data = pd.concat([X_train, pd.Series(y_train, index=X_train.index, name=target_col)], axis=1)
            val_data = pd.concat([X_val, pd.Series(y_val, index=X_val.index, name=target_col)], axis=1)
            test_data = pd.concat([X_test, pd.Series(y_test, index=X_test.index, name=target_col)], axis=1)
            
            # Create sequences
            X_train_seq, y_train_seq = self.feature_engineer.create_sequences(
                train_data, target_col, self.sequence_length
            )
            X_val_seq, y_val_seq = self.feature_engineer.create_sequences(
                val_data, target_col, self.sequence_length
            )
            X_test_seq, y_test_seq = self.feature_engineer.create_sequences(
                test_data, target_col, self.sequence_length
            )
            
            # Package the data
            sequence_data = {
                'train': (X_train_seq, y_train_seq),
                'val': (X_val_seq, y_val_seq),
                'test': (X_test_seq, y_test_seq)
            }
            
            logger.info("Sequence data prepared successfully")
            
            return sequence_data
            
        except Exception as e:
            logger.error(f"Error preparing sequence data: {e}")
            return None
            
    def prepare_cnn_data(self, data=None, target_col='target_class'):
        """
        Prepare image-like data for CNN models.
        
        Args:
            data (pd.DataFrame): Processed data (uses processed_data if None)
            target_col (str): Target column name
            
        Returns:
            dict: Dictionary with image data for training and evaluation
        """
        if data is None:
            data = self.processed_data
            
        if data is None or data.empty:
            logger.error("No data available for CNN data preparation")
            return None
            
        try:
            # Split the data first
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_train_val_test(
                data, target_col
            )
            
            if X_train is None:
                return None
                
            # Create image data for each split
            train_data = pd.concat([X_train, pd.Series(y_train, index=X_train.index, name=target_col)], axis=1)
            val_data = pd.concat([X_val, pd.Series(y_val, index=X_val.index, name=target_col)], axis=1)
            test_data = pd.concat([X_test, pd.Series(y_test, index=X_test.index, name=target_col)], axis=1)
            
            # Create image-like data
            X_train_img, y_train_img = self.feature_engineer.create_image_data(
                train_data, self.sequence_length
            )
            X_val_img, y_val_img = self.feature_engineer.create_image_data(
                val_data, self.sequence_length
            )
            X_test_img, y_test_img = self.feature_engineer.create_image_data(
                test_data, self.sequence_length
            )
            
            # Package the data
            image_data = {
                'train': (X_train_img, y_train_img),
                'val': (X_val_img, y_val_img),
                'test': (X_test_img, y_test_img)
            }
            
            logger.info("CNN data prepared successfully")
            
            return image_data
            
        except Exception as e:
            logger.error(f"Error preparing CNN data: {e}")
            return None
            
    def prepare_latest_data(self, data, lookback=config.SEQUENCE_LENGTH):
        """
        Prepare the latest data for prediction.
        
        Args:
            data (pd.DataFrame): Latest OHLCV data
            lookback (int): Number of candles to include in sequence
            
        Returns:
            tuple: (X_sequence, original_data) for prediction
        """
        if data is None or data.empty:
            logger.error("No data provided for prediction")
            return None, None
            
        try:
            # Get the latest 'lookback' candles
            latest_data = data.iloc[-lookback:].copy()
            
            # Process the data (without adding target labels)
            processed_data = self.feature_engineer.add_basic_price_features(latest_data)
            processed_data = self.feature_engineer.add_technical_indicators(processed_data)
            processed_data = self.feature_engineer.normalize_features(processed_data, fit=False)
            
            # Create a single sequence
            feature_cols = [col for col in processed_data.columns if not col.startswith('target_')]
            X_sequence = processed_data[feature_cols].values.reshape(1, lookback, -1)
            
            logger.info(f"Latest data prepared for prediction, shape: {X_sequence.shape}")
            
            return X_sequence, latest_data
            
        except Exception as e:
            logger.error(f"Error preparing latest data for prediction: {e}")
            return None, None
            
    def prepare_latest_cnn_data(self, data, lookback=config.SEQUENCE_LENGTH):
        """
        Prepare the latest data for CNN prediction.
        
        Args:
            data (pd.DataFrame): Latest OHLCV data
            lookback (int): Number of candles to include in image
            
        Returns:
            tuple: (X_image, original_data) for prediction
        """
        if data is None or data.empty:
            logger.error("No data provided for CNN prediction")
            return None, None
            
        try:
            # Get the latest 'lookback' candles
            latest_data = data.iloc[-lookback:].copy()
            
            # Process the data (without adding target labels)
            processed_data = self.feature_engineer.add_basic_price_features(latest_data)
            processed_data = self.feature_engineer.add_technical_indicators(processed_data)
            processed_data = self.feature_engineer.normalize_features(processed_data, fit=False)
            
            # Extract OHLCV data for the image
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            X_image = np.zeros((1, lookback, 5, 1))  # 1 sample, lookback timepoints, 5 channels, 1 depth
            
            for i, col in enumerate(ohlcv_cols):
                if col in processed_data.columns:
                    X_image[0, :, i, 0] = processed_data[col].values
            
            logger.info(f"Latest CNN data prepared for prediction, shape: {X_image.shape}")
            
            return X_image, latest_data
            
        except Exception as e:
            logger.error(f"Error preparing latest CNN data for prediction: {e}")
            return None, None
