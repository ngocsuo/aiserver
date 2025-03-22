"""
Continuous training module for scheduling and executing model retraining.
ĐÃ SỬA LỖI 'too many values to unpack (expected 2)'
"""
import os
import time
import logging
import threading
import datetime
import numpy as np
import pandas as pd
import config
from utils.thread_safe_logging import thread_safe_log

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("continuous_trainer")

class ContinuousTrainer:
    """
    Manager for continuous model training.
    Schedules and executes training jobs based on configured schedule.
    """
    def __init__(self):
        """Initialize the continuous training manager."""
        self.is_running = False
        self.last_training_time = None
        self.training_thread = None
        self.logs = []
        self.current_status = "Stopped"
        self._lock = threading.Lock()
        
        # Create data directories if they don't exist
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        logger.info("Continuous trainer initialized")
        
    def _generate_monthly_chunks(self):
        """
        Generate a list of date ranges for monthly chunks from the
        configured historical start date to the present.
        
        Returns:
            list: List of (start_date, end_date) tuples for monthly chunks
        """
        if not hasattr(config, 'HISTORICAL_START_DATE') or not config.HISTORICAL_START_DATE:
            return []
            
        try:
            start_date = datetime.datetime.strptime(config.HISTORICAL_START_DATE, "%Y-%m-%d")
            end_date = datetime.datetime.now()
            
            # Generate monthly chunks
            chunks = []
            current_date = start_date
            
            while current_date < end_date:
                # Calculate next month
                if current_date.month == 12:
                    next_date = datetime.datetime(current_date.year + 1, 1, 1)
                else:
                    next_date = datetime.datetime(current_date.year, current_date.month + 1, 1)
                
                # If next_date would be after end_date, use end_date instead
                if next_date > end_date:
                    next_date = end_date
                    
                # Format dates to strings
                start_str = current_date.strftime("%Y-%m-%d")
                end_str = next_date.strftime("%Y-%m-%d")
                
                chunks.append((start_str, end_str))
                current_date = next_date
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error generating monthly chunks: {e}")
            return []
    
    def start(self):
        """Start the continuous training manager."""
        if self.is_running:
            logger.warning("Continuous trainer is already running")
            return False
            
        self.is_running = True
        self.current_status = "Running"
        
        # Start background thread
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        logger.info("Continuous trainer started")
        return True
    
    def stop(self):
        """Stop the continuous training manager."""
        if not self.is_running:
            logger.warning("Continuous trainer is not running")
            return False
            
        self.is_running = False
        self.current_status = "Stopped"
        
        # Wait for thread to complete
        if self.training_thread:
            self.training_thread.join(timeout=2.0)
            
        logger.info("Continuous trainer stopped")
        return True
    
    def schedule_training(self, force=False):
        """
        Schedule a training job.
        
        Args:
            force (bool): If True, force training regardless of schedule
        """
        if not self.is_running and not force:
            logger.warning("Continuous trainer is not running. Call start() or use force=True")
            return False
            
        self._add_log("Training job scheduled")
        
        if force:
            # Execute training immediately
            self.current_status = "Training (forced)"
            thread = threading.Thread(target=self._execute_training, args=(True,))
            thread.daemon = True
            thread.start()
            return True
            
        return True
    
    def _add_log(self, message):
        """Add a log message to the internal log storage for UI display"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": timestamp, "message": message}
        
        with self._lock:
            self.logs.append(log_entry)
            # Keep only the last 100 log entries
            if len(self.logs) > 100:
                self.logs = self.logs[-100:]
                
        # Also log to file via thread-safe logging
        thread_safe_log(f"CONTINUOUS TRAINER: {message}")
        
        logger.info(message)
    
    def get_training_status(self):
        """
        Get the current training status.
        
        Returns:
            dict: Training status information
        """
        with self._lock:
            status = {
                "status": self.current_status,
                "last_training": self.last_training_time,
                "is_running": self.is_running,
                "logs": self.logs.copy()  # Return a copy to avoid race conditions
            }
            
        return status
    
    def _is_training_scheduled(self):
        """
        Check if training is due according to the configured schedule.
        
        Returns:
            bool: Whether training should occur now
        """
        # If no prior training, always return True
        if not self.last_training_time:
            return True
            
        # Get current time
        current_time = datetime.datetime.now()
        
        # Calculate time since last training
        time_since_last = current_time - self.last_training_time
        
        # Check if time since last training exceeds schedule interval
        # Default to 30 minutes if not set
        schedule_interval = getattr(config, 'TRAINING_SCHEDULE_INTERVAL', 30)
        
        return time_since_last.total_seconds() >= (schedule_interval * 60)
    
    def _training_loop(self):
        """Main training loop that runs in a background thread."""
        while self.is_running:
            try:
                # Check if training is scheduled
                if self._is_training_scheduled():
                    self.current_status = "Training"
                    self._execute_training()
                    self.current_status = "Running"
                
                # Sleep for a while before checking again
                # Check every minute if training should run
                for _ in range(60):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                self.current_status = "Error"
                self._add_log(f"Error in training loop: {e}")
                
                # Sleep before retrying
                time.sleep(60)
    
    def _execute_training(self, force=False):
        """
        Execute the actual training process.
        
        Args:
            force (bool): If True, force training regardless of whether enough new data is available
        """
        try:
            self._add_log("Starting training execution")
            
            # Check if we need all timeframes
            timeframes_to_train = list(config.TIMEFRAMES.values())
            
            # Train for each timeframe
            for timeframe in timeframes_to_train:
                try:
                    if not self.is_running and not force:
                        self._add_log("Training interrupted - system stopping")
                        break
                        
                    self._add_log(f"Processing data for timeframe: {timeframe}")
                    self._train_for_timeframe(timeframe)
                    
                except Exception as e:
                    self._add_log(f"Error training for timeframe {timeframe}: {e}")
                    logger.error(f"Error training for timeframe {timeframe}: {e}")
            
            # Update last training time
            self.last_training_time = datetime.datetime.now()
            self._add_log(f"Training complete at {self.last_training_time}")
            
        except Exception as e:
            self._add_log(f"Error during training execution: {e}")
            logger.error(f"Error during training execution: {e}")
    
    def _train_by_monthly_chunks(self):
        """Train models using monthly data chunks to manage memory usage for both timeframes."""
        pass # Implement if needed
    
    def _get_existing_data_ranges(self):
        """
        Kiểm tra dữ liệu đã tải từ trước đó
        
        Returns:
            list: Danh sách các khoảng thời gian đã tải, dạng [(start_date, end_date), ...]
        """
        pass # Implement if needed
    
    def _is_data_range_covered(self, start_date, end_date, existing_ranges):
        """
        Kiểm tra xem khoảng thời gian đã được tải trước đó chưa
        
        Args:
            start_date (str): Ngày bắt đầu khoảng thời gian mới
            end_date (str): Ngày kết thúc khoảng thời gian mới
            existing_ranges (list): Danh sách các khoảng thời gian đã tải
            
        Returns:
            bool: True nếu khoảng thời gian đã được tải, False nếu chưa
        """
        pass # Implement if needed
    
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
            
            # Create necessary objects
            data_collector = create_data_collector()
            data_processor = DataProcessor()
            model_trainer = ModelTrainer()
            
            # Collect data
            self._add_log(f"Collecting data for {timeframe}")
            
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
                return
                
            self._add_log(f"Collected {len(data)} candles for {timeframe}")
            
            # Process data
            self._add_log(f"Processing data for {timeframe}")
            processed_data = data_processor.process_data(data)
            
            # Prepare sequence data
            self._add_log(f"Preparing sequence data for {timeframe}")
            sequence_data = data_processor.prepare_sequence_data(processed_data)
            
            # Prepare image data
            self._add_log(f"Preparing image data for {timeframe}")
            image_data = data_processor.prepare_cnn_data(processed_data)
            
            # Train models
            self._add_log(f"Training models for {timeframe}")
            
            # QUAN TRỌNG: Đây là dòng code có lỗi, sửa lại để gọi đúng phương thức train_all_models
            # Trước đây, cố gắng unpacking giá trị: models, histories = model_trainer.train_all_models(...)
            # Nhưng train_all_models chỉ trả về models, không trả về histories
            
            # Thay vì:
            # models, histories = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
            
            # Sửa thành:
            models = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
            
            if models:
                self._add_log(f"Models trained successfully for {timeframe}")
            else:
                self._add_log(f"No models trained for {timeframe}")
                
        except Exception as e:
            self._add_log(f"Error training for timeframe {timeframe}: {e}")
            logger.error(f"Error training for timeframe {timeframe}: {e}")
            raise