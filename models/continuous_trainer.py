"""
Continuous training module for scheduling and executing model retraining.
"""
import logging
import threading
import time
import queue
import calendar
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np

import config
from utils.data_collector import create_data_collector
from utils.data_processor import DataProcessor
from models.model_trainer import ModelTrainer

# Set up logging
logger = logging.getLogger("continuous_trainer")

class ContinuousTrainer:
    """
    Manager for continuous model training.
    Schedules and executes training jobs based on configured schedule.
    """
    def __init__(self):
        """Initialize the continuous training manager."""
        self.data_collector = create_data_collector()
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        
        self.training_thread = None
        self.stop_training = threading.Event()
        self.training_queue = queue.Queue()
        
        self.last_training_time = None
        self.training_in_progress = False
        self.training_history = []
        self.new_data_count = 0
        
        # Log storage for UI display
        self.log_messages = []
        
        # Training progress tracking
        self.current_progress = 0
        self.total_chunks = 0
        self.current_chunk = 0
        
        self.chunk_start_dates = self._generate_monthly_chunks()
        self._add_log("Continuous trainer initialized with schedule: " + config.TRAINING_SCHEDULE['frequency'])
        
    def _generate_monthly_chunks(self):
        """
        Generate a list of date ranges for monthly chunks from the
        configured historical start date to the present.
        
        Returns:
            list: List of (start_date, end_date) tuples for monthly chunks
        """
        if not hasattr(config, 'HISTORICAL_START_DATE') or not config.HISTORICAL_START_DATE:
            return []
            
        start = datetime.strptime(config.HISTORICAL_START_DATE, "%Y-%m-%d")
        end = datetime.now()
        
        chunks = []
        current = start
        
        while current < end:
            # Get the last day of the current month
            _, last_day = calendar.monthrange(current.year, current.month)
            month_end = datetime(current.year, current.month, last_day)
            
            # If this would go beyond the end, use the end date
            if month_end > end:
                month_end = end
                
            # Format for Binance API
            start_date = current.strftime("%Y-%m-%d")
            end_date = month_end.strftime("%Y-%m-%d")
            
            chunks.append((start_date, end_date))
            
            # Move to first day of next month
            current = datetime(current.year, current.month, last_day) + timedelta(days=1)
            
        logger.info(f"Generated {len(chunks)} monthly chunks from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        return chunks
        
    def start(self):
        """Start the continuous training manager."""
        if self.training_thread is not None and self.training_thread.is_alive():
            logger.warning("Training thread is already running")
            return
            
        # Reset the stop flag
        self.stop_training.clear()
        
        # Create and start the training thread
        self.training_thread = threading.Thread(
            target=self._training_loop, 
            name="ContinuousTrainingThread",
            daemon=True
        )
        self.training_thread.start()
        
        logger.info("Continuous training manager started")
        
    def stop(self):
        """Stop the continuous training manager."""
        if self.training_thread is None or not self.training_thread.is_alive():
            logger.warning("Training thread is not running")
            return
            
        # Set the stop flag
        self.stop_training.set()
        
        # Wait for the thread to finish
        self.training_thread.join(timeout=5.0)
        
        if self.training_thread.is_alive():
            logger.warning("Training thread did not stop cleanly")
        else:
            logger.info("Continuous training manager stopped")
            
        self.training_thread = None
        
    def schedule_training(self, force=False):
        """
        Schedule a training job.
        
        Args:
            force (bool): If True, force training regardless of schedule
        """
        if force:
            logger.info("Forced training scheduled")
            self.training_queue.put("FORCE_TRAIN")
        else:
            logger.info("Training scheduled according to configured frequency")
            self.training_queue.put("TRAIN")
            
    def _add_log(self, message):
        """Add a log message to the internal log storage for UI display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"{timestamp} - {message}"
        self.log_messages.append(log_entry)
        # Keep only the most recent 500 messages
        if len(self.log_messages) > 500:
            self.log_messages = self.log_messages[-500:]
            
    def get_training_status(self):
        """
        Get the current training status.
        
        Returns:
            dict: Training status information
        """
        # Calculate progress percentage if training is in progress
        progress = 0
        if self.training_in_progress and self.total_chunks > 0:
            progress = min(100, int((self.current_chunk / self.total_chunks) * 100))
            
        return {
            "enabled": config.CONTINUOUS_TRAINING,
            "in_progress": self.training_in_progress,
            "last_training_time": self.last_training_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_training_time else None,
            "training_history": self.training_history[-10:],  # Last 10 training events
            "new_data_points": self.new_data_count,
            "schedule": config.TRAINING_SCHEDULE,
            "is_training": self.training_in_progress,
            "progress": progress,
            "status": "Training in progress" if self.training_in_progress else "Idle"
        }
        
    def _is_training_scheduled(self):
        """
        Check if training is due according to the configured schedule.
        
        Returns:
            bool: Whether training should occur now
        """
        if not config.CONTINUOUS_TRAINING:
            return False
            
        now = datetime.now()
        schedule = config.TRAINING_SCHEDULE
        
        if schedule["frequency"] == "hourly":
            # Check if the current minute matches the scheduled minute
            return now.minute == schedule["minute"]
            
        elif schedule["frequency"] == "daily":
            # Check if the current hour and minute match the scheduled time
            return (now.hour == schedule["hour"] and 
                    now.minute == schedule["minute"])
                    
        elif schedule["frequency"] == "weekly":
            # Check if the current day, hour, and minute match the scheduled time
            # Note: schedule uses Monday=0, Sunday=6, but datetime uses Monday=0, Sunday=6
            return (now.weekday() == schedule["day_of_week"] and
                    now.hour == schedule["hour"] and 
                    now.minute == schedule["minute"])
        
        return False
        
    def _training_loop(self):
        """Main training loop that runs in a background thread."""
        logger.info("Training loop started")
        
        check_interval = 60  # Check schedule every minute
        
        # Immediately schedule the first training to process historical data
        self.schedule_training(force=True)
        
        # Keep training continuously
        continuous_training_interval = 30 * 60  # 30 minutes between continuous training cycles
        last_continuous_training = time.time()
        
        while not self.stop_training.is_set():
            try:
                # For continuous training, don't wait for scheduled time
                current_time = time.time()
                if current_time - last_continuous_training >= continuous_training_interval:
                    # Start a new training cycle after the interval
                    logger.info(f"Starting continuous training cycle after {continuous_training_interval//60} minutes")
                    self.schedule_training(force=True)
                    last_continuous_training = current_time
                
                # Check if there's a training job in the queue
                try:
                    # Non-blocking queue check with timeout
                    job = self.training_queue.get(block=True, timeout=1.0)
                    
                    # Process the training job
                    self._execute_training(force=(job == "FORCE_TRAIN"))
                    
                    # Mark job as done
                    self.training_queue.task_done()
                    
                except queue.Empty:
                    # No training job in the queue, continue
                    pass
                    
                # Sleep for a while before checking again
                for _ in range(check_interval):
                    if self.stop_training.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                # Sleep a bit before retrying to avoid tight error loops
                time.sleep(10)
                
        logger.info("Training loop stopped")
        
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
            
            # Check if we have enough new data to justify retraining
            if not force and self.new_data_count < config.MINIMUM_NEW_DATA_POINTS:
                logger.info(f"Not enough new data ({self.new_data_count} points) for retraining, minimum required: {config.MINIMUM_NEW_DATA_POINTS}")
                self.training_in_progress = False
                return
                
            # Train by monthly chunks if enabled
            if config.CHUNK_BY_MONTHS and self.chunk_start_dates:
                self._train_by_monthly_chunks()
            else:
                self._train_with_all_data()
                
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
            
            logger.info(f"Training process completed in {duration:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Error during training execution: {e}")
        finally:
            self.training_in_progress = False
            
    def _train_by_monthly_chunks(self):
        """Train models using monthly data chunks to manage memory usage."""
        logger.info(f"Training with {len(self.chunk_start_dates)} monthly chunks")
        self._add_log(f"Bắt đầu huấn luyện với {len(self.chunk_start_dates)} đoạn dữ liệu tháng")
        
        all_processed_data = []
        # Set total chunks for progress tracking
        self.total_chunks = len(self.chunk_start_dates)
        self.current_chunk = 0
        
        # Process each monthly chunk
        for i, (start_date, end_date) in enumerate(self.chunk_start_dates):
            logger.info(f"Processing chunk {i+1}/{len(self.chunk_start_dates)}: {start_date} to {end_date}")
            
            try:
                # Collect data for this month
                raw_data = self.data_collector.collect_historical_data(
                    timeframe=config.TIMEFRAMES["primary"],
                    start_date=start_date,
                    end_date=end_date
                )
                
                if raw_data is not None and not raw_data.empty:
                    # Process the data
                    processed_chunk = self.data_processor.process_data(raw_data)
                    all_processed_data.append(processed_chunk)
                    logger.info(f"Chunk {i+1}: Processed {len(processed_chunk)} data points")
                else:
                    logger.warning(f"Chunk {i+1}: No data collected for period {start_date} to {end_date}")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                
        # Combine all processed chunks
        if all_processed_data:
            combined_data = pd.concat(all_processed_data)
            
            # Remove duplicates
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            
            # Sort by time
            combined_data.sort_index(inplace=True)
            
            logger.info(f"Combined data: {len(combined_data)} data points")
            
            # Prepare data for different model types
            sequence_data = self.data_processor.prepare_sequence_data(combined_data)
            image_data = self.data_processor.prepare_cnn_data(combined_data)
            
            # Train all models
            models = self.model_trainer.train_all_models(sequence_data, image_data)
            
            logger.info(f"Trained {len(models)} models with chunked data")
        else:
            logger.error("No processed data available after processing all chunks")
            
    def _train_with_all_data(self):
        """Train models using all data at once."""
        logger.info("Training with all data at once")
        
        try:
            # Collect all historical data
            raw_data = None
            
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
                processed_data = self.data_processor.process_data(raw_data)
                
                # Prepare data for different model types
                sequence_data = self.data_processor.prepare_sequence_data(processed_data)
                image_data = self.data_processor.prepare_cnn_data(processed_data)
                
                # Train all models
                models = self.model_trainer.train_all_models(sequence_data, image_data)
                
                logger.info(f"Trained {len(models)} models with {len(processed_data)} data points")
            else:
                logger.error("No data collected for training")
                
        except Exception as e:
            logger.error(f"Error training with all data: {e}")
            
    def increment_new_data_count(self, count=1):
        """
        Increment the counter of new data points.
        
        Args:
            count (int): Number of new data points to add
        """
        self.new_data_count += count

# Singleton instance
_instance = None

def get_continuous_trainer():
    """
    Get the singleton continuous trainer instance.
    
    Returns:
        ContinuousTrainer: The continuous trainer instance
    """
    global _instance
    if _instance is None:
        _instance = ContinuousTrainer()
    return _instance