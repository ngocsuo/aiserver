"""
Data processing module for preparing datasets for training and prediction.
"""
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
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
        logger.info("Data processor initialized")
    
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
            logger.info(f"Processing {len(raw_data)} candles of data")
            
            # Apply full preprocessing pipeline
            processed_data = self.feature_engineer.preprocess_data(
                raw_data, 
                add_basic=True, 
                add_technical=True,
                add_labels=True,
                normalize=True
            )
            
            self.processed_data = processed_data
            
            logger.info(f"Data processing complete: {len(processed_data)} samples with {len(processed_data.columns)} features")
            return processed_data
            
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
            return None
        
        try:
            # Get features and target
            y = data[target_col]
            
            # Exclude target columns and any non-feature columns
            exclude_cols = [col for col in data.columns if col.startswith('target_') or 
                          col in ['open_time', 'close_time']]
            X = data.drop(exclude_cols, axis=1)
            
            # First split into train and temp (val+test)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, 
                test_size=(config.VALIDATION_SPLIT + config.TEST_SPLIT),
                shuffle=False  # Time series data, maintain chronological order
            )
            
            # Split temp into val and test
            test_size = config.TEST_SPLIT / (config.VALIDATION_SPLIT + config.TEST_SPLIT)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, 
                test_size=test_size,
                shuffle=False  # Time series data, maintain chronological order
            )
            
            logger.info(f"Data split complete: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return None
    
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
            # Split data into train, val, test
            split_data = self.split_train_val_test(data, target_col)
            
            if split_data is None:
                return None
                
            X_train, X_val, X_test, y_train, y_val, y_test = split_data
            
            # Create sequences
            X_train_seq = self.feature_engineer.create_sequences(
                pd.concat([X_train, y_train.to_frame()], axis=1),
                target_col=target_col,
                seq_length=config.SEQUENCE_LENGTH
            )
            
            X_val_seq = self.feature_engineer.create_sequences(
                pd.concat([X_val, y_val.to_frame()], axis=1),
                target_col=target_col,
                seq_length=config.SEQUENCE_LENGTH
            )
            
            X_test_seq = self.feature_engineer.create_sequences(
                pd.concat([X_test, y_test.to_frame()], axis=1),
                target_col=target_col,
                seq_length=config.SEQUENCE_LENGTH
            )
            
            logger.info(f"Sequence data preparation complete")
            
            return {
                'X_train': X_train_seq[0],
                'y_train': X_train_seq[1],
                'X_val': X_val_seq[0],
                'y_val': X_val_seq[1],
                'X_test': X_test_seq[0],
                'y_test': X_test_seq[1],
                'feature_names': X_train.columns.tolist()
            }
            
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
            # Split data into train, val, test
            split_data = self.split_train_val_test(data, target_col)
            
            if split_data is None:
                return None
                
            X_train, X_val, X_test, y_train, y_val, y_test = split_data
            
            # Create image-like data
            X_train_img, y_train_img = self.feature_engineer.create_image_data(
                pd.concat([X_train, y_train.to_frame()], axis=1),
                seq_length=config.SEQUENCE_LENGTH
            )
            
            X_val_img, y_val_img = self.feature_engineer.create_image_data(
                pd.concat([X_val, y_val.to_frame()], axis=1),
                seq_length=config.SEQUENCE_LENGTH
            )
            
            X_test_img, y_test_img = self.feature_engineer.create_image_data(
                pd.concat([X_test, y_test.to_frame()], axis=1),
                seq_length=config.SEQUENCE_LENGTH
            )
            
            logger.info(f"CNN data preparation complete")
            
            return {
                'X_train': X_train_img,
                'y_train': y_train_img,
                'X_val': X_val_img,
                'y_val': y_val_img,
                'X_test': X_test_img,
                'y_test': y_test_img
            }
            
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
            return None
        
        try:
            # Process the data using the feature engineer
            processed_data = self.feature_engineer.preprocess_data(
                data,
                add_basic=True,
                add_technical=True,
                add_labels=False,  # No labels for prediction
                normalize=True
            )
            
            # Get most recent lookback candles
            recent_data = processed_data.iloc[-lookback:]
            
            # Check if we have enough data
            if len(recent_data) < lookback:
                logger.warning(f"Not enough data for prediction. Need {lookback} candles, got {len(recent_data)}")
                # Pad with zeros if needed
                pad_size = lookback - len(recent_data)
                pad_data = pd.DataFrame(0, index=range(pad_size), columns=recent_data.columns)
                recent_data = pd.concat([pad_data, recent_data])
            
            # Convert to numpy array for model input
            X_sequence = recent_data.to_numpy()
            
            # Reshape for sequence models [samples, timesteps, features]
            X_sequence = X_sequence.reshape(1, lookback, X_sequence.shape[1])
            
            logger.info(f"Latest data prepared for prediction")
            
            return X_sequence, data.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error preparing latest data: {e}")
            return None
    
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
            return None
        
        try:
            # Process the data using the feature engineer
            processed_data = self.feature_engineer.preprocess_data(
                data,
                add_basic=True,
                add_technical=True,
                add_labels=False,  # No labels for prediction
                normalize=True
            )
            
            # Get most recent lookback candles
            recent_data = processed_data.iloc[-lookback:]
            
            # Check if we have enough data
            if len(recent_data) < lookback:
                logger.warning(f"Not enough data for CNN prediction. Need {lookback} candles, got {len(recent_data)}")
                # Not enough data for CNN prediction
                return None
            
            # Create image data
            X_image, _ = self.feature_engineer.create_image_data(recent_data, seq_length=lookback)
            
            # We need a single sample
            X_image = X_image[-1:] 
            
            logger.info(f"Latest data prepared for CNN prediction")
            
            return X_image, data.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error preparing latest CNN data: {e}")
            return None