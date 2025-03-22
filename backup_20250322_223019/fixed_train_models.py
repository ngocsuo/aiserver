"""
Phiên bản đã sửa lỗi cho continuous_trainer.py
Lỗi: too many values to unpack (expected 2)
"""
import os
import sys
import time
import logging
import threading
import queue
from datetime import datetime, timedelta
import calendar
import numpy as np
import pandas as pd

import config
from utils.thread_safe_logging import thread_safe_log
from utils.data_collector import create_data_collector
from utils.data_processor import DataProcessor
from models.model_trainer import ModelTrainer

# Thiết lập logger
logger = logging.getLogger(__name__)

def _train_for_timeframe(self, timeframe):
    """
    Train models for a specific timeframe.
    
    Args:
        timeframe (str): Timeframe to train for (e.g., "1m", "5m")
    """
    try:
        # Import dependencies here to avoid circular imports
        from utils.data_collector import create_data_collector
        from utils.data_processor import DataProcessor
        from models.model_trainer import ModelTrainer
        from utils.thread_safe_logging import thread_safe_log
        
        # Create necessary objects
        data_collector = create_data_collector()
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        
        # Collect data
        self._add_log(f"Collecting data for {timeframe}")
        thread_safe_log(f"Collecting data for {timeframe}")
        
        if hasattr(config, 'HISTORICAL_START_DATE') and config.HISTORICAL_START_DATE:
            # For historical training
            data = data_collector.collect_historical_data(
                timeframe=timeframe,
                start_date=config.HISTORICAL_START_DATE
            )
        else:
            # For recent data only
            data = data_collector.collect_historical_data(
                timeframe=timeframe, 
                limit=config.LOOKBACK_PERIODS
            )
        
        if data is None or len(data) == 0:
            self._add_log(f"No data available for {timeframe}")
            thread_safe_log(f"No data available for {timeframe}")
            return
            
        self._add_log(f"Collected {len(data)} candles for {timeframe}")
        thread_safe_log(f"Collected {len(data)} candles for {timeframe}")
        
        # Process data
        self._add_log(f"Processing data for {timeframe}")
        thread_safe_log(f"Processing data for {timeframe}")
        processed_data = data_processor.process_data(data)
        
        # Prepare sequence data
        self._add_log(f"Preparing sequence data for {timeframe}")
        thread_safe_log(f"Preparing sequence data for {timeframe}")
        sequence_data = data_processor.prepare_sequence_data(processed_data)
        
        # Prepare image data
        self._add_log(f"Preparing image data for {timeframe}")
        thread_safe_log(f"Preparing image data for {timeframe}")
        image_data = data_processor.prepare_cnn_data(processed_data)
        
        # Train models
        self._add_log(f"Training models for {timeframe}")
        thread_safe_log(f"Training models for {timeframe}")
        
        # QUAN TRỌNG: SỬA LỖI Ở ĐÂY
        try:
            # Xử lý lỗi nếu hàm train_all_models trả về tuple hoặc nhiều giá trị
            result = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
            # Kiểm tra xem kết quả có phải là tuple với nhiều giá trị không
            if isinstance(result, tuple):
                # Nếu là tuple, lấy phần tử đầu tiên (models)
                models = result[0]
            else:
                # Nếu không, sử dụng kết quả trực tiếp
                models = result
        except ValueError as e:
            if "too many values to unpack" in str(e):
                # Ghi log lỗi và thử lại với cách khác
                thread_safe_log(f"Lỗi khi unpack giá trị: {e}. Thử lại với cách lấy phần tử đầu tiên.")
                result = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
                # Ép kiểu kết quả về list và lấy phần tử đầu tiên
                models = list(result)[0] if isinstance(result, (list, tuple)) else result
            else:
                # Nếu là lỗi ValueError khác, ném lại ngoại lệ
                raise
        
        if models:
            self._add_log(f"Models trained successfully for {timeframe}")
            thread_safe_log(f"Models trained successfully for {timeframe}")
            return models
        else:
            self._add_log(f"No models trained for {timeframe}")
            thread_safe_log(f"No models trained for {timeframe}")
            return None
            
    except Exception as e:
        self._add_log(f"Error training for timeframe {timeframe}: {e}")
        thread_safe_log(f"Error training for timeframe {timeframe}: {e}")
        logger.error(f"Error training for timeframe {timeframe}: {e}")
        raise

def _execute_training(self, force=False):
    """
    Execute the actual training process.
    
    Args:
        force (bool): If True, force training regardless of whether enough new data is available
    """
    if self.training_in_progress:
        logger.warning("Training already in progress, skipping")
        return
        
    try:
        self.training_in_progress = True
        start_time = datetime.now()
        
        logger.info(f"Starting training process at {start_time}")
        thread_safe_log(f"Bắt đầu quá trình huấn luyện lúc {start_time}")
        
        # Check if we have enough new data to justify retraining
        if not force and self.new_data_count < config.MINIMUM_NEW_DATA_POINTS:
            logger.info(f"Not enough new data ({self.new_data_count} points) for retraining, minimum required: {config.MINIMUM_NEW_DATA_POINTS}")
            self.training_in_progress = False
            return
            
        # Train by monthly chunks if enabled
        if config.CHUNK_BY_MONTHS and self.chunk_start_dates:
            result = self._train_by_monthly_chunks()
            # Thêm xử lý kết quả nếu cần thiết
        else:
            result = self._train_with_all_data()
            # Thêm xử lý kết quả nếu cần thiết
            
        # Record training event
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.last_training_time = end_time
        self.training_history.append({
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "data_points": self.new_data_count,
            "forced": force
        })
        
        # Reset new data counter
        self.new_data_count = 0
        
        # Log completion with duration
        mins = int(duration // 60)
        secs = int(duration % 60)
        time_str = f"{mins} phút {secs} giây" if mins > 0 else f"{secs} giây"
        self._add_log(f"✅ Quá trình huấn luyện hoàn tất trong {time_str}")
        thread_safe_log(f"Quá trình huấn luyện hoàn tất trong {time_str}")
        
        # Calculate next training time
        next_time = end_time + timedelta(seconds=30*60)  # 30 minutes
        self._add_log(f"⏱️ Đợt huấn luyện tiếp theo dự kiến: {next_time.strftime('%H:%M:%S')}")
        thread_safe_log(f"Đợt huấn luyện tiếp theo dự kiến: {next_time.strftime('%H:%M:%S')}")
        
        logger.info(f"Training process completed in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Error during training execution: {e}")
        thread_safe_log(f"Lỗi trong quá trình huấn luyện: {e}")
        self._add_log(f"❌ Lỗi huấn luyện: {str(e)}")
    finally:
        self.training_in_progress = False
        
def _train_with_all_data(self):
    """Train models using all data at once."""
    logger.info("Training with all data at once")
    
    try:
        # Collect all historical data
        raw_data = None
        
        self._add_log("🔄 Đang thu thập dữ liệu lịch sử...")
        thread_safe_log("Đang thu thập dữ liệu lịch sử...")
        
        if hasattr(config, 'HISTORICAL_START_DATE') and config.HISTORICAL_START_DATE:
            raw_data = self.data_collector.collect_historical_data(
                timeframe=config.TIMEFRAMES["primary"],
                start_date=config.HISTORICAL_START_DATE
            )
        else:
            raw_data = self.data_collector.collect_historical_data(
                timeframe=config.TIMEFRAMES["primary"],
                limit=config.LOOKBACK_PERIODS
            )
            
        if raw_data is not None and not raw_data.empty:
            # Process the data
            self._add_log(f"🔧 Đang xử lý {len(raw_data)} điểm dữ liệu lịch sử...")
            thread_safe_log(f"Đang xử lý {len(raw_data)} điểm dữ liệu lịch sử...")
            processed_data = self.data_processor.process_data(raw_data)
            
            # Prepare data for different model types
            self._add_log("📊 Đang chuẩn bị dữ liệu đầu vào cho các mô hình...")
            thread_safe_log("Đang chuẩn bị dữ liệu đầu vào cho các mô hình...")
            sequence_data = self.data_processor.prepare_sequence_data(processed_data)
            image_data = self.data_processor.prepare_cnn_data(processed_data)
            
            # Train all models
            self._add_log(f"🧠 Bắt đầu huấn luyện các mô hình với {len(processed_data)} điểm dữ liệu")
            thread_safe_log(f"Bắt đầu huấn luyện các mô hình với {len(processed_data)} điểm dữ liệu")
            
            # QUAN TRỌNG: SỬA LỖI Ở ĐÂY
            try:
                # Xử lý lỗi nếu hàm train_all_models trả về tuple hoặc nhiều giá trị
                result = self.model_trainer.train_all_models(sequence_data, image_data)
                # Kiểm tra xem kết quả có phải là tuple với nhiều giá trị không
                if isinstance(result, tuple):
                    # Nếu là tuple, lấy phần tử đầu tiên (models)
                    models = result[0]
                else:
                    # Nếu không, sử dụng kết quả trực tiếp
                    models = result
            except ValueError as e:
                if "too many values to unpack" in str(e):
                    # Ghi log lỗi và thử lại với cách khác
                    thread_safe_log(f"Lỗi khi unpack giá trị: {e}. Thử lại với cách lấy phần tử đầu tiên.")
                    result = self.model_trainer.train_all_models(sequence_data, image_data)
                    # Ép kiểu kết quả về list và lấy phần tử đầu tiên
                    models = list(result)[0] if isinstance(result, (list, tuple)) else result
                else:
                    # Nếu là lỗi ValueError khác, ném lại ngoại lệ
                    raise
            
            self._add_log(f"✅ Đã huấn luyện thành công {len(models) if models else 0} mô hình")
            thread_safe_log(f"Đã huấn luyện thành công {len(models) if models else 0} mô hình")
            logger.info(f"Trained {len(models) if models else 0} models with {len(processed_data)} data points")
            
            return models
        else:
            self._add_log("❌ Không thể thu thập dữ liệu lịch sử cho việc huấn luyện")
            thread_safe_log("Không thể thu thập dữ liệu lịch sử cho việc huấn luyện")
            logger.error("No data collected for training")
            return None
            
    except Exception as e:
        self._add_log(f"❌ Lỗi huấn luyện: {str(e)}")
        thread_safe_log(f"Lỗi huấn luyện: {str(e)}")
        logger.error(f"Error training with all data: {e}")
        return None