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
import json
import pandas as pd
import numpy as np

import config
from utils.data_collector import create_data_collector
from utils.data_processor import DataProcessor
from models.model_trainer import ModelTrainer

# Set up logging
logger = logging.getLogger("continuous_trainer")

import pandas as pd
import numpy as np

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
        
        # Danh sách các khung thời gian cần huấn luyện
        self.timeframes_to_train = [config.PRIMARY_TIMEFRAME, config.SECONDARY_TIMEFRAME]
        
        # Historical start date for training (can be updated at runtime)
        self.historical_start_date = config.HISTORICAL_START_DATE
        # Monthly chunks for training
        self.monthly_chunks = self._generate_monthly_chunks()
        
        # Lưu danh sách các ngày bắt đầu chunks để dùng sau này
        self.chunk_start_dates = self.monthly_chunks.copy()
        self._add_log("Continuous trainer initialized with schedule: " + config.TRAINING_SCHEDULE['frequency'] + 
                     f" for timeframes: {', '.join(self.timeframes_to_train)}")
        
    def _generate_monthly_chunks(self):
        """
        Generate a list of date ranges for monthly chunks from the
        configured historical start date to the present.
        
        Returns:
            list: List of tuples containing (start_date, end_date) for monthly chunks
        """
        # Sử dụng giá trị historical_start_date của đối tượng continuous_trainer
        # hoặc backup từ config nếu không có
        if hasattr(self, 'historical_start_date') and self.historical_start_date:
            start = datetime.strptime(self.historical_start_date, "%Y-%m-%d")
            logger.info(f"Using custom historical start date: {self.historical_start_date}")
            self._add_log(f"🔍 Sử dụng ngày bắt đầu tùy chỉnh: {self.historical_start_date}")
        elif hasattr(config, 'DEFAULT_TRAINING_START_DATE') and config.DEFAULT_TRAINING_START_DATE:
            # Sử dụng dữ liệu 12 tháng gần nhất cho huấn luyện
            start = datetime.strptime(config.DEFAULT_TRAINING_START_DATE, "%Y-%m-%d")
            logger.info(f"Using 12-month data for training: starting from {config.DEFAULT_TRAINING_START_DATE}")
            self._add_log(f"🔍 Sử dụng dữ liệu 12 tháng gần nhất cho huấn luyện (từ {config.DEFAULT_TRAINING_START_DATE})")
        elif hasattr(config, 'HISTORICAL_START_DATE') and config.HISTORICAL_START_DATE:
            # Sử dụng ngày bắt đầu lịch sử cũ nếu không có cài đặt mới
            start = datetime.strptime(config.HISTORICAL_START_DATE, "%Y-%m-%d")
            logger.info(f"Using historical data from {config.HISTORICAL_START_DATE}")
        else:
            return []
            
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
            
            # Đảm bảo thêm tuple có đúng 2 phần tử
            chunks.append((start_date, end_date))
            
            # Move to first day of next month
            current = datetime(current.year, current.month, last_day) + timedelta(days=1)
            
        logger.info(f"Generated {len(chunks)} monthly chunks from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        
        # In ra kiểm tra định dạng các chunk
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i}: {chunk}, type: {type(chunk)}, length: {len(chunk) if isinstance(chunk, tuple) else 'not tuple'}")
            
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
        
        self._add_log("🚀 Quá trình huấn luyện liên tục đã được khởi động")
        logger.info("Continuous training manager started")
        
    def stop(self):
        """Stop the continuous training manager."""
        if self.training_thread is None or not self.training_thread.is_alive():
            logger.warning("Training thread is not running")
            return
            
        # Set the stop flag
        self.stop_training.set()
        
        # Wait for the thread to finish
        self._add_log("⏱️ Đang dừng quá trình huấn luyện liên tục...")
        self.training_thread.join(timeout=5.0)
        
        if self.training_thread.is_alive():
            self._add_log("⚠️ Không thể dừng tiến trình huấn luyện sạch sẽ")
            logger.warning("Training thread did not stop cleanly")
        else:
            self._add_log("✅ Đã dừng quá trình huấn luyện liên tục")
            logger.info("Continuous training manager stopped")
            
        self.training_thread = None
        
    def schedule_training(self, force=False):
        """
        Schedule a training job.
        
        Args:
            force (bool): If True, force training regardless of schedule
        """
        if force:
            self._add_log("🔄 Đã lên lịch huấn luyện thủ công")
            logger.info("Forced training scheduled")
            self.training_queue.put("FORCE_TRAIN")
        else:
            self._add_log("🔄 Đã lên lịch huấn luyện theo thời gian đã cấu hình")
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
            
        # Thêm thông tin về current_chunk và total_chunks cho update_status
        status_data = {
            "enabled": config.CONTINUOUS_TRAINING,
            "in_progress": self.training_in_progress,
            "last_training_time": self.last_training_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_training_time else None,
            "training_history": self.training_history[-10:],  # Last 10 training events
            "new_data_points": self.new_data_count,
            "schedule": config.TRAINING_SCHEDULE,
            "is_training": self.training_in_progress,
            "progress": progress,
            "models_trained": self.last_training_time is not None,
            "current_chunk": self.current_chunk,
            "total_chunks": self.total_chunks,
            "status": "Training in progress" if self.training_in_progress else "Idle"
        }
        
        # Lưu trạng thái vào file để thread chính có thể đọc được
        try:
            import json
            with open("training_status.json", "w") as f:
                json.dump(status_data, f)
        except Exception as e:
            logger.error(f"Error saving training status to file: {e}")
            
        return status_data
        
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
                    mins = continuous_training_interval//60
                    self._add_log(f"🕒 Bắt đầu huấn luyện định kỳ sau {mins} phút")
                    logger.info(f"Starting continuous training cycle after {mins} minutes")
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
            
            # Lưu trữ kết quả mô hình    
            model_results = {}
                
            # Train by monthly chunks if enabled
            try:
                if config.CHUNK_BY_MONTHS and self.chunk_start_dates:
                    # SỬA LỖI: Bọc trong try-except để xử lý lỗi "too many values to unpack"
                    model_results = self._train_by_monthly_chunks()
                else:
                    # Có thể _train_with_all_data cũng trả về model_results
                    model_results = self._train_with_all_data()
            except ValueError as e:
                if "too many values to unpack" in str(e):
                    logger.error(f"Error in _execute_training: {e}")
                    self._add_log(f"⚠️ Lỗi khi thực thi huấn luyện: {e}")
                    # Tạo model_results rỗng để tránh lỗi khi truy cập
                    model_results = {tf: {} for tf in self.timeframes_to_train}
                else:
                    # Ném lại ngoại lệ nếu không phải lỗi unpacking
                    raise
                
            # Ghi nhật ký kết quả
            if model_results:
                timeframes = list(model_results.keys())
                models_count = sum(len(models) for models in model_results.values() if isinstance(models, dict))
                self._add_log(f"✅ Đã huấn luyện thành công {models_count} mô hình cho {len(timeframes)} khung thời gian")
                logger.info(f"Trained {models_count} models for {timeframes}")
                
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
            
            # Calculate next training time
            next_time = end_time + timedelta(seconds=30*60)  # 30 minutes
            self._add_log(f"⏱️ Đợt huấn luyện tiếp theo dự kiến: {next_time.strftime('%H:%M:%S')}")
            
            logger.info(f"Training process completed in {duration:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Error during training execution: {e}")
        finally:
            self.training_in_progress = False
            
    def _train_by_monthly_chunks(self):
        """
        Train models using monthly data chunks to manage memory usage for both timeframes.
        
        Returns:
            dict: Dictionary của các mô hình đã huấn luyện cho mỗi khung thời gian
        """
        logger.info(f"Training with {len(self.monthly_chunks)} monthly chunks from {self.historical_start_date} for timeframes: {', '.join(self.timeframes_to_train)}")
        self._add_log(f"Bắt đầu huấn luyện với {len(self.monthly_chunks)} đoạn dữ liệu tháng từ {self.historical_start_date}")
        
        # Dictionary để lưu trữ dữ liệu đã xử lý cho mỗi khung thời gian
        all_processed_data = {timeframe: [] for timeframe in self.timeframes_to_train}
        
        # Dictionary để lưu trữ kết quả mô hình cho mỗi khung thời gian
        model_results = {}
        
        # Set total chunks for progress tracking (tổng số chunks nhân với số khung thời gian)
        self.total_chunks = len(self.monthly_chunks) * len(self.timeframes_to_train)
        self.current_chunk = 0
        
        # Kiểm tra xem đã có dữ liệu đã tải trước đó chưa
        existing_data_ranges = self._get_existing_data_ranges()
        
        # Xử lý từng khung thời gian
        for timeframe in self.timeframes_to_train:
            self._add_log(f"🕒 Đang xử lý dữ liệu cho khung thời gian: {timeframe}")
            logger.info(f"Processing data for timeframe: {timeframe}")
            
            # Process each monthly chunk for this timeframe
            for i, chunk in enumerate(self.monthly_chunks):
                # Đảm bảo chunk là tuple với 2 phần tử
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    start_date, end_date = chunk
                else:
                    # Xử lý trường hợp chunk không phải là tuple 2 phần tử
                    self._add_log(f"⚠️ Định dạng chunk không hợp lệ: {chunk}")
                    logger.warning(f"Invalid chunk format: {chunk}")
                    continue

                self.current_chunk += 1
                chunk_progress = int((self.current_chunk / self.total_chunks) * 100)
                
                # Khóa tài nguyên cho khung thời gian và khoảng thời gian cụ thể
                cache_key = f"{timeframe}_{start_date}_{end_date}"
                
                # Kiểm tra xem dữ liệu cho khoảng thời gian này đã được tải trước đó chưa
                if self._is_data_range_covered(start_date, end_date, existing_data_ranges):
                    # Dữ liệu đã tồn tại, sử dụng lại
                    log_msg = f"⏩ Bỏ qua đoạn {i+1}/{len(self.monthly_chunks)} ({timeframe}): từ {start_date} đến {end_date} - đã có dữ liệu"
                    self._add_log(log_msg)
                    logger.info(f"Skipping chunk {i+1}/{len(self.monthly_chunks)} ({timeframe}): {start_date} to {end_date} - data already exists")
                    
                    # Tải dữ liệu đã lưu từ tệp cache
                    try:
                        cached_data = self._load_cached_data(start_date, end_date, timeframe)
                        if cached_data is not None and not cached_data.empty:
                            if timeframe not in all_processed_data:
                                all_processed_data[timeframe] = []
                            all_processed_data[timeframe].append(cached_data)
                            self._add_log(f"✅ Đoạn {i+1} ({timeframe}): Đã tải {len(cached_data)} điểm dữ liệu từ bộ nhớ đệm")
                    except Exception as e:
                        # Nếu không thể tải dữ liệu từ cache, tải lại từ API
                        log_msg = f"⚠️ Không thể tải dữ liệu đệm cho đoạn {i+1} ({timeframe}): {str(e)} - Đang tải lại từ Binance"
                        self._add_log(log_msg)
                        logger.warning(f"Could not load cached data for chunk {i+1} ({timeframe}): {e} - Redownloading")
                        # Tiếp tục với quy trình tải mới dưới đây
                
                # Nếu không có dữ liệu đệm hoặc không thể tải, tải mới từ API
                if len(all_processed_data[timeframe]) <= i:
                    log_msg = f"📥 Đang tải đoạn dữ liệu {i+1}/{len(self.monthly_chunks)} ({timeframe}): từ {start_date} đến {end_date} - {chunk_progress}% hoàn thành"
                    self._add_log(log_msg)
                    logger.info(f"Downloading chunk {i+1}/{len(self.monthly_chunks)} ({timeframe}): {start_date} to {end_date}")
                
                    try:
                        # Collect data for this month with the specific timeframe
                        raw_data = self.data_collector.collect_historical_data(
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if raw_data is not None and not raw_data.empty:
                            # Process the data
                            processed_chunk = self.data_processor.process_data(raw_data)
                            all_processed_data[timeframe].append(processed_chunk)
                            
                            # Lưu dữ liệu đã xử lý vào bộ nhớ đệm với khung thời gian
                            self._save_cached_data(processed_chunk, start_date, end_date, timeframe=timeframe)
                            
                            # Cập nhật danh sách các khoảng thời gian đã tải
                            existing_data_ranges.append((start_date, end_date, timeframe))
                            
                            self._add_log(f"✅ Đoạn {i+1} ({timeframe}): Đã xử lý {len(processed_chunk)} điểm dữ liệu thành công")
                            logger.info(f"Chunk {i+1} ({timeframe}): Processed {len(processed_chunk)} data points")
                        else:
                            error_msg = f"⚠️ Đoạn {i+1} ({timeframe}): Không có dữ liệu cho giai đoạn {start_date} đến {end_date}"
                            self._add_log(error_msg)
                            logger.warning(f"Chunk {i+1} ({timeframe}): No data collected for period {start_date} to {end_date}")
                            
                    except Exception as e:
                        error_msg = f"❌ Lỗi xử lý đoạn {i+1} ({timeframe}): {str(e)}"
                        self._add_log(error_msg)
                        logger.error(f"Error processing chunk {i+1} ({timeframe}): {e}")
        
        # Sau khi xử lý toàn bộ dữ liệu cho tất cả các khung thời gian
        for timeframe, data_chunks in all_processed_data.items():
            if data_chunks:
                # Kết hợp tất cả các đoạn dữ liệu đã xử lý cho khung thời gian này
                combined_data = pd.concat(data_chunks)
                
                # Loại bỏ các dòng trùng lặp
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                
                # Sắp xếp theo thời gian
                combined_data.sort_index(inplace=True)
                
                self._add_log(f"📊 Tổng hợp dữ liệu ({timeframe}): {len(combined_data)} điểm dữ liệu đã được xử lý")
                logger.info(f"Combined data for {timeframe}: {len(combined_data)} data points")
                
                # Chuẩn bị dữ liệu cho các loại mô hình khác nhau
                sequence_data = self.data_processor.prepare_sequence_data(combined_data)
                image_data = self.data_processor.prepare_cnn_data(combined_data)
                
                # Huấn luyện tất cả các mô hình với khung thời gian cụ thể
                self._add_log(f"🧠 Bắt đầu huấn luyện các mô hình cho {timeframe} với {len(combined_data)} điểm dữ liệu")
                
                # Lưu thông tin về khung thời gian vào dữ liệu huấn luyện
                for data_dict in [sequence_data, image_data]:
                    for key in data_dict:
                        if isinstance(data_dict[key], dict):
                            data_dict[key]['timeframe'] = timeframe
                
                # SỬA LỖI: Xử lý lỗi "too many values to unpack (expected 2)"
                try:
                    # Fix #1: Lưu kết quả vào biến tạm trước để kiểm tra loại dữ liệu
                    result = self.model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
                    
                    # Fix #2: Kiểm tra xem kết quả có phải là tuple không, nếu có thì lấy phần tử đầu tiên
                    if isinstance(result, tuple) and len(result) > 0:
                        models = result[0]  # Lấy phần tử đầu tiên (models)
                        self._add_log(f"⚠️ Đã tự động xử lý kết quả tuple từ train_all_models")
                    else:
                        # Nếu không phải tuple, sử dụng kết quả trực tiếp
                        models = result
                        
                    # Fix #3: Đảm bảo models không None trước khi lưu vào kết quả
                    if models is not None:
                        model_results[timeframe] = models
                        self._add_log(f"✅ Đã huấn luyện thành công mô hình cho {timeframe}")
                    else:
                        self._add_log(f"⚠️ Huấn luyện cho {timeframe} trả về None")
                        model_results[timeframe] = {}
                        
                except ValueError as e:
                    # Fix #4: Xử lý lỗi 'too many values to unpack' nếu vẫn xảy ra
                    if "too many values to unpack" in str(e):
                        self._add_log(f"⚠️ Lỗi định dạng kết quả: {str(e)}")
                        logger.warning(f"Value unpacking error in train_all_models: {str(e)}")
                        
                        try:
                            # Lấy kết quả trực tiếp, không chuyển sang list
                            result = self.model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
                            
                            # Kiểm tra trường hợp kết quả là một tuple
                            if isinstance(result, tuple) and len(result) > 0:
                                models = result[0]  # Lấy phần tử đầu tiên nếu là tuple
                            else:
                                models = result  # Sử dụng kết quả trực tiếp nếu không phải tuple
                                
                            if models is not None:
                                model_results[timeframe] = models
                                self._add_log(f"✅ Đã khắc phục lỗi và huấn luyện thành công mô hình cho {timeframe}")
                            else:
                                self._add_log(f"⚠️ Kết quả huấn luyện rỗng cho {timeframe}")
                                model_results[timeframe] = {}
                        except Exception as inner_e:
                            self._add_log(f"❌ Lỗi khi xử lý kết quả huấn luyện: {str(inner_e)}")
                            logger.error(f"Error processing training result: {inner_e}")
                            model_results[timeframe] = {}
                    else:
                        # Lỗi ValueError khác
                        self._add_log(f"❌ Lỗi không xác định: {str(e)}")
                        logger.error(f"Unknown error in train_all_models: {e}")
                        model_results[timeframe] = {}
                except Exception as e:
                    # Fix #5: Xử lý các ngoại lệ khác
                    self._add_log(f"❌ Lỗi trong quá trình huấn luyện: {str(e)}")
                    logger.error(f"Error in training process: {e}")
                    model_results[timeframe] = {}
                
                # Thông báo kết quả sau khi xử lý
                if timeframe in model_results and model_results[timeframe]:
                    try:
                        model_count = len(model_results[timeframe])
                        self._add_log(f"✅ Đã huấn luyện thành công {model_count} mô hình cho {timeframe}")
                    except:
                        self._add_log(f"✅ Đã huấn luyện thành công mô hình cho {timeframe}")
                else:
                    self._add_log(f"⚠️ Không có mô hình nào được huấn luyện cho {timeframe}")
                logger.info(f"Trained models for {timeframe} with {len(combined_data)} data points")
            else:
                self._add_log(f"❌ Không có dữ liệu khả dụng cho {timeframe} sau khi xử lý tất cả các đoạn")
                logger.error(f"No processed data available for {timeframe} after processing all chunks")
        
        # Trả về kết quả huấn luyện cho tất cả các khung thời gian
        return model_results
    
    def _get_existing_data_ranges(self):
        """
        Kiểm tra dữ liệu đã tải từ trước đó
        
        Returns:
            list: Danh sách các khoảng thời gian đã tải, dạng [(start_date, end_date), ...]
        """
        # Kiểm tra thư mục lưu trữ dữ liệu
        cache_dir = os.path.join(config.MODEL_DIR, "data_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            return []
        
        # Đọc danh sách khoảng thời gian đã tải
        ranges_file = os.path.join(cache_dir, "data_ranges.json")
        if os.path.exists(ranges_file):
            try:
                with open(ranges_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing data ranges: {e}")
                return []
        return []
    
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
        if not existing_ranges:
            return False
            
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            for data_range in existing_ranges:
                try:
                    # Xử lý trường hợp data_range là dict (định dạng mới)
                    if isinstance(data_range, dict) and 'start_date' in data_range and 'end_date' in data_range:
                        exist_start = data_range['start_date']
                        exist_end = data_range['end_date']
                        
                        # Kiểm tra định dạng khớp với timeframe
                        if hasattr(self, 'current_timeframe') and hasattr(data_range, 'timeframe'):
                            if self.current_timeframe != data_range.get('timeframe'):
                                continue
                                
                        exist_start_date = datetime.strptime(exist_start, "%Y-%m-%d")
                        exist_end_date = datetime.strptime(exist_end, "%Y-%m-%d")
                        
                        # Khoảng thời gian đã được bao phủ bởi một khoảng thời gian hiện có
                        if start >= exist_start_date and end <= exist_end_date:
                            logger.info(f"Khoảng thời gian {start_date} đến {end_date} đã tồn tại trong cache")
                            return True
                    # Xử lý trường hợp data_range là list/tuple (định dạng cũ)
                    elif isinstance(data_range, (list, tuple)) and len(data_range) >= 2:
                        exist_start = data_range[0]
                        exist_end = data_range[1]
                        
                        # Kiểm tra định dạng khớp với timeframe
                        if len(data_range) >= 3:
                            range_timeframe = data_range[2]
                            # Bỏ qua nếu không cùng timeframe
                            if hasattr(self, 'current_timeframe') and self.current_timeframe != range_timeframe:
                                continue
                                
                        exist_start_date = datetime.strptime(exist_start, "%Y-%m-%d")
                        exist_end_date = datetime.strptime(exist_end, "%Y-%m-%d")
                        
                        # Khoảng thời gian đã được bao phủ bởi một khoảng thời gian hiện có
                        if start >= exist_start_date and end <= exist_end_date:
                            logger.info(f"Khoảng thời gian {start_date} đến {end_date} đã tồn tại trong cache")
                            return True
                except (ValueError, IndexError, TypeError, KeyError) as e:
                    logger.warning(f"Lỗi khi xử lý khoảng dữ liệu {data_range}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra khoảng dữ liệu {start_date} - {end_date}: {e}")
            # Trước đây trả về giá trị 0 gây lỗi "too many values to unpack (expected 2)"
            # Trả về False để xử lý lỗi một cách an toàn
            return False
            
        # Trả về False nếu không tìm thấy khoảng thời gian phù hợp
        return False
    
    def _save_cached_data(self, data, start_date, end_date, timeframe=None):
        """
        Lưu dữ liệu đã xử lý vào bộ nhớ đệm với tính năng nén để tiết kiệm không gian.
        
        Args:
            data (pd.DataFrame): Dữ liệu đã xử lý
            start_date (str): Ngày bắt đầu khoảng thời gian
            end_date (str): Ngày kết thúc khoảng thời gian
            timeframe (str, optional): Khung thời gian của dữ liệu
        """
        try:
            # Đảm bảo thư mục cache tồn tại
            cache_dir = os.path.join(config.MODEL_DIR, "data_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Tạo tên tệp dựa trên khoảng thời gian và khung thời gian
            file_name = f"{start_date}_to_{end_date}"
            if timeframe:
                file_name += f"_{timeframe}"
            cache_file = os.path.join(cache_dir, f"{file_name}.pkl.gz")
            
            # Tối ưu hóa kiểu dữ liệu trước khi lưu để giảm kích thước
            optimized_data = self._optimize_dataframe_types(data.copy())
            
            # Lưu DataFrame vào tệp với tính năng nén
            optimized_data.to_pickle(cache_file, compression='gzip')
            
            # Tính kích thước đã tiết kiệm
            normal_size = data.memory_usage(deep=True).sum()
            optimized_size = optimized_data.memory_usage(deep=True).sum()
            savings_percent = ((normal_size - optimized_size) / normal_size * 100) if normal_size > 0 else 0
            
            # Cập nhật danh sách khoảng thời gian đã tải
            ranges_file = os.path.join(cache_dir, "data_ranges.json")
            existing_ranges = self._get_existing_data_ranges()
            
            # Thêm khoảng thời gian mới kèm thông tin về kích thước và ngày tạo
            if not self._is_data_range_covered(start_date, end_date, existing_ranges):
                existing_ranges.append({
                    "start_date": start_date,
                    "end_date": end_date,
                    "created_at": datetime.now().isoformat(),
                    "size_bytes": os.path.getsize(cache_file) if os.path.exists(cache_file) else 0,
                    "rows": len(optimized_data)
                })
                
                # Lưu danh sách cập nhật
                with open(ranges_file, 'w') as f:
                    json.dump(existing_ranges, f, indent=2)
            
            logger.info(f"Cached data saved for period {start_date} to {end_date} (Memory optimized: {savings_percent:.1f}% saved)")
        except Exception as e:
            logger.error(f"Error saving cached data: {e}")
    
    def _optimize_dataframe_types(self, df):
        """
        Tối ưu kiểu dữ liệu của DataFrame để giảm bộ nhớ sử dụng.
        
        Args:
            df (pd.DataFrame): DataFrame cần tối ưu
            
        Returns:
            pd.DataFrame: DataFrame đã tối ưu
        """
        # Danh sách các cột để bỏ qua quá trình tối ưu (các cột mục tiêu)
        exclude_cols = ['target', 'target_class', 'target_binary']
        
        try:
            # Loại bỏ các giá trị NaN trước khi tối ưu kiểu dữ liệu
            df = df.fillna(0)
            
            # Loại bỏ các giá trị vô cùng (inf)
            df = df.replace([np.inf, -np.inf], 0)
            
            # Tối ưu cột numeric
            for col in df.select_dtypes(include=['float']).columns:
                if col not in exclude_cols:
                    try:
                        col_min = df[col].min()
                        col_max = df[col].max()
                        
                        # Kiểm tra xem dữ liệu có thể chuyển đổi sang int không
                        if df[col].equals(df[col].astype(int)):
                            if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                                df[col] = df[col].astype(np.int32)
                            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                                df[col] = df[col].astype(np.int16)
                            elif col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                                df[col] = df[col].astype(np.int8)
                        else:
                            # Tối ưu cột float
                            if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                                df[col] = df[col].astype(np.float32)
                    except Exception as e:
                        logger.warning(f"Không thể tối ưu hóa cột {col}: {e}")
                        continue
            
            # Tối ưu cột integer
            for col in df.select_dtypes(include=['int']).columns:
                if col not in exclude_cols:
                    try:
                        col_min = df[col].min()
                        col_max = df[col].max()
                        
                        if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                    except Exception as e:
                        logger.warning(f"Không thể tối ưu hóa cột {col}: {e}")
                        continue
            
            # Tối ưu cột boolean
            for col in df.select_dtypes(include=['bool']).columns:
                if col not in exclude_cols:
                    try:
                        df[col] = df[col].astype('int8')  # int8 tiết kiệm hơn bool
                    except Exception as e:
                        logger.warning(f"Không thể tối ưu hóa cột {col}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Lỗi khi tối ưu hóa DataFrame: {e}")
            # Trả về DataFrame gốc nếu xảy ra lỗi
            return df
            
        return df
    
    def _load_cached_data(self, start_date, end_date, timeframe=None):
        """
        Tải dữ liệu đã lưu từ bộ nhớ đệm với hỗ trợ cho cả định dạng nén và không nén.
        
        Args:
            start_date (str): Ngày bắt đầu khoảng thời gian
            end_date (str): Ngày kết thúc khoảng thời gian
            timeframe (str, optional): Khung thời gian của dữ liệu
            
        Returns:
            pd.DataFrame: Dữ liệu đã xử lý hoặc None nếu không tìm thấy
        """
        try:
            # Tạo các đường dẫn tệp cache có thể (nén và không nén)
            cache_dir = os.path.join(config.MODEL_DIR, "data_cache")
            
            # Tạo tên tệp dựa trên khoảng thời gian và khung thời gian
            file_name = f"{start_date}_to_{end_date}"
            if timeframe:
                file_name += f"_{timeframe}"
            
            cache_file_gz = os.path.join(cache_dir, f"{file_name}.pkl.gz")
            cache_file = os.path.join(cache_dir, f"{file_name}.pkl")
            
            # Kiểm tra tệp nén trước
            if os.path.exists(cache_file_gz):
                # Tải dữ liệu từ tệp nén
                data = pd.read_pickle(cache_file_gz, compression='gzip')
                file_size = os.path.getsize(cache_file_gz) / (1024 * 1024)  # Convert to MB
                logger.info(f"Loaded compressed cached data for period {start_date} to {end_date} ({file_size:.2f} MB)")
                return data
            # Kiểm tra tệp không nén
            elif os.path.exists(cache_file):
                # Tải dữ liệu từ tệp không nén
                data = pd.read_pickle(cache_file)
                
                # Tối ưu và lưu tệp nén cho lần sau
                optimized_data = self._optimize_dataframe_types(data.copy())
                optimized_data.to_pickle(cache_file_gz, compression='gzip')
                
                logger.info(f"Loaded and migrated cached data for period {start_date} to {end_date}")
                return data
            
            # Nếu không tìm thấy tệp cache nào
            logger.info(f"No cached data found for period {start_date} to {end_date}")
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
        
        return None
    
    # Phần này đã được di chuyển vào phương thức _train_by_monthly_chunks
        # Sau khi xử lý toàn bộ dữ liệu cho tất cả các khung thời gian
        model_results = {}
        for timeframe, data_chunks in all_processed_data.items():
            if data_chunks:
                # Kết hợp tất cả các đoạn dữ liệu đã xử lý cho khung thời gian này
                combined_data = pd.concat(data_chunks)
                
                # Loại bỏ các dòng trùng lặp
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                
                # Sắp xếp theo thời gian
                combined_data.sort_index(inplace=True)
                
                self._add_log(f"📊 Tổng hợp dữ liệu ({timeframe}): {len(combined_data)} điểm dữ liệu đã được xử lý")
                logger.info(f"Combined data for {timeframe}: {len(combined_data)} data points")
                
                # Chuẩn bị dữ liệu cho các loại mô hình khác nhau
                sequence_data = self.data_processor.prepare_sequence_data(combined_data)
                image_data = self.data_processor.prepare_cnn_data(combined_data)
                
                # Huấn luyện tất cả các mô hình với khung thời gian cụ thể
                self._add_log(f"🧠 Bắt đầu huấn luyện các mô hình cho {timeframe} với {len(combined_data)} điểm dữ liệu")
                
                # Lưu thông tin về khung thời gian vào dữ liệu huấn luyện
                for data_dict in [sequence_data, image_data]:
                    for key in data_dict:
                        if isinstance(data_dict[key], dict):
                            data_dict[key]['timeframe'] = timeframe
                
                # SỬA LỖI: Xử lý lỗi "too many values to unpack (expected 2)"
                try:
                    # Fix #1: Lưu kết quả vào biến tạm trước để kiểm tra loại dữ liệu
                    result = self.model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
                    
                    # Fix #2: Kiểm tra xem kết quả có phải là tuple không, nếu có thì lấy phần tử đầu tiên
                    if isinstance(result, tuple) and len(result) > 0:
                        models = result[0]  # Lấy phần tử đầu tiên (models)
                        self._add_log(f"⚠️ Đã tự động xử lý kết quả tuple từ train_all_models")
                    else:
                        # Nếu không phải tuple, sử dụng kết quả trực tiếp
                        models = result
                        
                    # Fix #3: Đảm bảo models không None trước khi lưu vào kết quả
                    if models is not None:
                        model_results[timeframe] = models
                        self._add_log(f"✅ Đã huấn luyện thành công mô hình cho {timeframe}")
                    else:
                        self._add_log(f"⚠️ Huấn luyện cho {timeframe} trả về None")
                        model_results[timeframe] = {}
                        
                except ValueError as e:
                    # Fix #4: Xử lý lỗi 'too many values to unpack' nếu vẫn xảy ra
                    if "too many values to unpack" in str(e):
                        self._add_log(f"⚠️ Lỗi định dạng kết quả: {str(e)}")
                        logger.warning(f"Value unpacking error in train_all_models: {str(e)}")
                        
                        try:
                            # Lấy kết quả trực tiếp, không chuyển sang list
                            result = self.model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
                            
                            # Kiểm tra trường hợp kết quả là một tuple
                            if isinstance(result, tuple) and len(result) > 0:
                                models = result[0]  # Lấy phần tử đầu tiên nếu là tuple
                            else:
                                models = result  # Sử dụng kết quả trực tiếp nếu không phải tuple
                                
                            if models is not None:
                                model_results[timeframe] = models
                                self._add_log(f"✅ Đã khắc phục lỗi và huấn luyện thành công mô hình cho {timeframe}")
                            else:
                                self._add_log(f"⚠️ Kết quả huấn luyện rỗng cho {timeframe}")
                                model_results[timeframe] = {}
                        except Exception as inner_e:
                            self._add_log(f"❌ Lỗi khi xử lý kết quả huấn luyện: {str(inner_e)}")
                            logger.error(f"Error processing training result: {inner_e}")
                            model_results[timeframe] = {}
                    else:
                        # Lỗi ValueError khác
                        self._add_log(f"❌ Lỗi không xác định: {str(e)}")
                        logger.error(f"Unknown error in train_all_models: {e}")
                        model_results[timeframe] = {}
                except Exception as e:
                    # Fix #5: Xử lý các ngoại lệ khác
                    self._add_log(f"❌ Lỗi trong quá trình huấn luyện: {str(e)}")
                    logger.error(f"Error in training process: {e}")
                    model_results[timeframe] = {}
                
                # Thông báo kết quả sau khi xử lý
                if timeframe in model_results and model_results[timeframe]:
                    try:
                        model_count = len(model_results[timeframe])
                        self._add_log(f"✅ Đã huấn luyện thành công {model_count} mô hình cho {timeframe}")
                    except:
                        self._add_log(f"✅ Đã huấn luyện thành công mô hình cho {timeframe}")
                else:
                    self._add_log(f"⚠️ Không có mô hình nào được huấn luyện cho {timeframe}")
                logger.info(f"Trained models for {timeframe} with {len(combined_data)} data points")
            else:
                self._add_log(f"❌ Không có dữ liệu khả dụng cho {timeframe} sau khi xử lý tất cả các đoạn")
                logger.error(f"No processed data available for {timeframe} after processing all chunks")
        
        return model_results
            
    def _train_with_all_data(self):
        """Train models using all data at once."""
        logger.info("Training with all data at once")
        
        # Dictionary để lưu trữ kết quả mô hình cho mỗi khung thời gian
        model_results = {}
        
        try:
            # Lặp qua từng khung thời gian để huấn luyện
            for timeframe in self.timeframes_to_train:
                # Lưu thông tin về timeframe hiện tại để sử dụng trong các phương thức khác
                self.current_timeframe = timeframe
                
                # Collect all historical data
                raw_data = None
                
                self._add_log(f"🔄 Đang thu thập dữ liệu lịch sử cho {timeframe}...")
                
                try:
                    if hasattr(config, 'HISTORICAL_START_DATE') and config.HISTORICAL_START_DATE:
                        raw_data = self.data_collector.collect_historical_data(
                            timeframe=timeframe,
                            start_date=config.HISTORICAL_START_DATE
                        )
                    else:
                        raw_data = self.data_collector.collect_historical_data(
                            timeframe=timeframe,
                            limit=config.LOOKBACK_PERIODS
                        )
                except Exception as e:
                    self._add_log(f"❌ Lỗi khi thu thập dữ liệu cho {timeframe}: {str(e)}")
                    logger.error(f"Error collecting data for {timeframe}: {e}")
                    model_results[timeframe] = {}
                    continue
                    
                if raw_data is None or raw_data.empty:
                    self._add_log(f"⚠️ Không thu thập được dữ liệu cho {timeframe}")
                    model_results[timeframe] = {}
                    continue
                
                try:
                    # Process the data
                    self._add_log(f"🔧 Đang xử lý {len(raw_data)} điểm dữ liệu lịch sử cho {timeframe}...")
                    processed_data = self.data_processor.process_data(raw_data)
                    
                    if processed_data is None or processed_data.empty:
                        self._add_log(f"⚠️ Dữ liệu sau khi xử lý trống cho {timeframe}")
                        model_results[timeframe] = {}
                        continue
                        
                    # Prepare data for different model types
                    self._add_log(f"📊 Đang chuẩn bị dữ liệu đầu vào cho các mô hình ({timeframe})...")
                    sequence_data = self.data_processor.prepare_sequence_data(processed_data)
                    image_data = self.data_processor.prepare_cnn_data(processed_data)
                    
                    # Lưu thông tin về khung thời gian vào dữ liệu huấn luyện
                    for data_dict in [sequence_data, image_data]:
                        for key in data_dict:
                            if isinstance(data_dict[key], dict):
                                data_dict[key]['timeframe'] = timeframe
                    
                    # Train all models
                    self._add_log(f"🧠 Bắt đầu huấn luyện các mô hình cho {timeframe} với {len(processed_data)} điểm dữ liệu")
                    
                    # SỬA LỖI: Xử lý lỗi "too many values to unpack (expected 2)"
                    try:
                        # Ghi log để debug
                        self._add_log(f"🔍 Bắt đầu gọi train_all_models với timeframe={timeframe}")
                        
                        # Lưu kết quả vào biến tạm trước để kiểm tra loại dữ liệu
                        result = self.model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
                        
                        self._add_log(f"📋 Kết quả trả về từ train_all_models: {type(result)}")
                        
                        # Kiểm tra kỹ hơn xem kết quả có phải là tuple không và có chiều dài đủ không
                        if result is not None:
                            if isinstance(result, tuple) and len(result) > 0:
                                models = result[0]  # Lấy phần tử đầu tiên (models)
                                self._add_log(f"⚠️ Đã tự động xử lý kết quả tuple từ train_all_models")
                            else:
                                # Nếu không phải tuple, sử dụng kết quả trực tiếp
                                models = result
                                
                            # Đảm bảo models không None trước khi lưu vào kết quả
                            if models is not None:
                                model_results[timeframe] = models
                                self._add_log(f"✅ Đã huấn luyện thành công mô hình cho {timeframe}")
                            else:
                                self._add_log(f"❌ Kết quả models là None sau khi xử lý")
                                model_results[timeframe] = {}
                        else:
                            self._add_log(f"❌ train_all_models trả về None cho {timeframe}")
                            model_results[timeframe] = {}
                            
                    except ValueError as e:
                        # Xử lý lỗi 'too many values to unpack' nếu vẫn xảy ra
                        if "too many values to unpack" in str(e):
                            self._add_log(f"⚠️ Lỗi định dạng kết quả: {str(e)}")
                            logger.warning(f"Value unpacking error in train_all_models: {str(e)}")
                            
                            try:
                                # Lấy kết quả trực tiếp, không chuyển sang list
                                result = self.model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
                                
                                # Kiểm tra trường hợp kết quả là một tuple
                                if isinstance(result, tuple) and len(result) > 0:
                                    models = result[0]  # Lấy phần tử đầu tiên nếu là tuple
                                else:
                                    models = result  # Sử dụng kết quả trực tiếp nếu không phải tuple
                                    
                                if models is not None:
                                    model_results[timeframe] = models
                                    self._add_log(f"✅ Đã khắc phục lỗi và huấn luyện thành công mô hình cho {timeframe}")
                                else:
                                    self._add_log(f"⚠️ Kết quả huấn luyện rỗng cho {timeframe}")
                                    model_results[timeframe] = {}
                            except Exception as inner_e:
                                self._add_log(f"❌ Lỗi khi xử lý kết quả huấn luyện: {str(inner_e)}")
                                logger.error(f"Error processing training result: {inner_e}")
                                model_results[timeframe] = {}
                        else:
                            # Lỗi ValueError khác
                            self._add_log(f"❌ Lỗi không xác định: {str(e)}")
                            logger.error(f"Unknown error in train_all_models: {e}")
                            model_results[timeframe] = {}
                    except Exception as e:
                        # Xử lý các ngoại lệ khác
                        self._add_log(f"❌ Lỗi trong quá trình huấn luyện: {str(e)}")
                        logger.error(f"Error in training process: {e}")
                        model_results[timeframe] = {}
                    
                    # Thông báo kết quả sau khi xử lý
                    if timeframe in model_results and model_results[timeframe]:
                        try:
                            model_count = len(model_results[timeframe])
                            self._add_log(f"✅ Đã huấn luyện thành công {model_count} mô hình cho {timeframe}")
                        except:
                            self._add_log(f"✅ Đã huấn luyện thành công mô hình cho {timeframe}")
                    else:
                        self._add_log(f"⚠️ Không có mô hình nào được huấn luyện cho {timeframe}")
                except Exception as data_process_error:
                    self._add_log(f"❌ Lỗi khi xử lý dữ liệu: {str(data_process_error)}")
                    logger.error(f"Error processing data for {timeframe}: {data_process_error}")
                    model_results[timeframe] = {}
                    
                # Kiểm tra nếu không có dữ liệu
                if raw_data is None or raw_data.empty:
                    self._add_log(f"❌ Không thể thu thập dữ liệu lịch sử cho khung thời gian {timeframe}")
                    logger.error(f"No data collected for training timeframe {timeframe}")
                    model_results[timeframe] = {}
                
        except Exception as e:
            self._add_log(f"❌ Lỗi huấn luyện: {str(e)}")
            logger.error(f"Error training with all data: {e}")
            
        return model_results
            
    def increment_new_data_count(self, count=1):
        """
        Increment the counter of new data points.
        
        Args:
            count (int): Number of new data points to add
        """
        self.new_data_count += count
    
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
            
            # QUAN TRỌNG: SỬA LỖI Ở ĐÂY - Vấn đề "too many values to unpack"
            try:
                # Cách 1: Gán trực tiếp kết quả cho models
                models = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
            except ValueError as e:
                # Nếu gặp lỗi "too many values", đây là phương pháp sửa thay thế
                if "too many values to unpack" in str(e):
                    thread_safe_log(f"Lỗi giá trị khi huấn luyện: {e}, đang thử phương pháp khác")
                    # Lưu kết quả vào biến tạm thời
                    result = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
                    # Lấy phần tử đầu tiên nếu là một tuple
                    if isinstance(result, tuple) and len(result) > 0:
                        models = result[0]
                    else:
                        models = result
                else:
                    # Nếu là lỗi khác, ném lại ngoại lệ
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