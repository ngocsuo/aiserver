"""
Data processing module for preparing datasets for training and prediction.
Optimized with parallel processing and memory-efficient operations.
"""
import pandas as pd
import numpy as np
import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
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
    
    def _optimize_dataframe(self, df):
        """
        Tối ưu hóa bộ nhớ của DataFrame bằng cách giảm kích thước kiểu dữ liệu.
        
        Args:
            df (pd.DataFrame): DataFrame cần tối ưu
            
        Returns:
            pd.DataFrame: DataFrame đã tối ưu
        """
        # Tối ưu số nguyên
        for col in df.select_dtypes(include=['int']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
                
        # Tối ưu số thực
        for col in df.select_dtypes(include=['float']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
                
        # Tối ưu boolean
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype('int8')
            
        return df
    
    def _process_chunk(self, chunk, add_basic=True, add_technical=True, add_labels=True, normalize=True):
        """
        Xử lý một phần dữ liệu. Dùng cho xử lý song song.
        
        Args:
            chunk (pd.DataFrame): Phần dữ liệu cần xử lý
            add_basic (bool): Thêm tính năng cơ bản
            add_technical (bool): Thêm chỉ báo kỹ thuật
            add_labels (bool): Thêm nhãn mục tiêu
            normalize (bool): Chuẩn hóa dữ liệu
            
        Returns:
            pd.DataFrame: Dữ liệu đã xử lý
        """
        return self.feature_engineer.preprocess_data(
            chunk, 
            add_basic=add_basic, 
            add_technical=add_technical,
            add_labels=add_labels,
            normalize=normalize
        )
        
    def process_data(self, raw_data):
        """
        Xử lý dữ liệu OHLCV thô cho việc huấn luyện mô hình sử dụng xử lý song song.
        
        Args:
            raw_data (pd.DataFrame): Dữ liệu OHLCV thô
            
        Returns:
            pd.DataFrame: Dữ liệu đã xử lý với các tính năng và mục tiêu
        """
        if raw_data is None or raw_data.empty:
            logger.error("No data provided for processing")
            return None
        
        try:
            data_length = len(raw_data)
            logger.info(f"Processing {data_length} candles of data")
            
            # Đối với dữ liệu nhỏ, sử dụng phương pháp đơn luồng
            if data_length < 5000:
                logger.info("Using single-process data processing for small dataset")
                processed_data = self.feature_engineer.preprocess_data(
                    raw_data, 
                    add_basic=True, 
                    add_technical=True,
                    add_labels=True,
                    normalize=True
                )
                self.processed_data = processed_data
                return processed_data
                
            # Đối với dữ liệu lớn, chia nhỏ và xử lý song song
            # Xác định số lượng CPU sẽ sử dụng
            cpu_count = multiprocessing.cpu_count()
            workers = max(1, min(cpu_count - 1, 4))  # Sử dụng tối đa 4 CPU hoặc (n_cores - 1)
            
            # Xác định kích thước phần (đảm bảo chia hết để tránh các vấn đề với dãy thời gian)
            chunk_size = max(1000, data_length // workers)
            
            # Chia dữ liệu thành các phần
            chunks = []
            for i in range(0, data_length, chunk_size):
                end = min(i + chunk_size, data_length)
                # Đảm bảo chồng lấn các phần để tránh mất thông tin tại ranh giới
                if i > 0:
                    # Chồng lấn 100 điểm dữ liệu để tính toán chỉ báo kỹ thuật chính xác
                    overlap = 100
                    start = max(0, i - overlap)
                else:
                    start = i
                chunks.append(raw_data.iloc[start:end])
                
            logger.info(f"Data split into {len(chunks)} chunks for parallel processing using {workers} workers")
            
            # Xử lý song song
            partial_preprocess = partial(
                self._process_chunk, 
                add_basic=True, 
                add_technical=True,
                add_labels=True,
                normalize=True
            )
            
            processed_chunks = []
            
            # Sử dụng ProcessPoolExecutor cho xử lý song song
            with ProcessPoolExecutor(max_workers=workers) as executor:
                processed_chunks = list(executor.map(partial_preprocess, chunks))
                
            # Kết hợp các phần đã xử lý
            if processed_chunks:
                # Loại bỏ các phần chồng lấn
                for i in range(1, len(processed_chunks)):
                    # Xác định điểm bắt đầu bỏ qua phần chồng lấn
                    if len(chunks[i-1]) >= chunk_size:
                        # Giữ lại các điểm dữ liệu không chồng lấn
                        processed_chunks[i] = processed_chunks[i].iloc[100:]
                
                # Kết hợp các phần
                processed_data = pd.concat(processed_chunks)
                
                # Loại bỏ bản sao nếu có
                processed_data = processed_data[~processed_data.index.duplicated(keep='first')]
                
                # Tối ưu hóa bộ nhớ
                processed_data = self._optimize_dataframe(processed_data)
                
                logger.info(f"Parallel processing complete: {len(processed_data)} samples with {len(processed_data.columns)} features")
                self.processed_data = processed_data
                return processed_data
            else:
                logger.error("No data processed")
                return None
            
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
            
            # Check if we have enough data for the full dataset
            # We need at least lookback+1 candles for create_image_data to work
            # because it needs to create at least one sequence
            if len(processed_data) < lookback + 1:
                logger.warning(f"Not enough processed data for CNN prediction. Need at least {lookback+1} candles, got {len(processed_data)}")
                
                # For prediction when we don't have enough historical data, we'll create a single sample
                # by padding the data if necessary
                if len(processed_data) < lookback:
                    logger.warning(f"Padding data for CNN prediction. Need {lookback} candles, got {len(processed_data)}")
                    # Get all available processed data
                    recent_data = processed_data.copy()
                    
                    # Pad with duplicated first row if needed
                    pad_size = lookback - len(recent_data)
                    first_row = recent_data.iloc[0:1]
                    pad_data = pd.concat([first_row] * pad_size)
                    recent_data = pd.concat([pad_data, recent_data]).reset_index(drop=True)
                else:
                    # Get most recent lookback candles
                    recent_data = processed_data.iloc[-lookback:].reset_index(drop=True)
                
                # Create a manual image representation
                # Shape: (1, lookback, 5, 1) for a single sample
                X_image = np.zeros((1, lookback, 5, 1))
                
                # Fill with OHLCV data
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                for j, col in enumerate(ohlcv_cols):
                    if col in recent_data.columns:
                        X_image[0, :, j, 0] = recent_data[col].values
                    else:
                        logger.warning(f"Column {col} not found in data. Using zeros.")
                
                logger.info(f"Created manual image data of shape {X_image.shape} for prediction")
            else:
                # Create image data using regular method
                # We use lookback+1 to ensure we have at least one sequence
                X_image, _ = self.feature_engineer.create_image_data(processed_data, seq_length=lookback)
                
                if X_image is None or len(X_image) == 0:
                    logger.error("Failed to create image data for prediction")
                    return None
                
                # We need just the last sample for prediction
                X_image = X_image[-1:] 
            
            logger.info(f"Latest data prepared for CNN prediction with shape {X_image.shape}")
            
            return X_image, data.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error preparing latest CNN data: {e}")
            return None