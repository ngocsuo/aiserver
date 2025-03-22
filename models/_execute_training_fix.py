"""
Phiên bản đã sửa lỗi của phương thức _execute_training
"""

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
            
        # Train for each timeframe separately instead of using chunks
        for timeframe in self.timeframes_to_train:
            try:
                self._add_log(f"🔄 Huấn luyện cho khung thời gian {timeframe}")
                logger.info(f"Training for timeframe {timeframe}")
                
                # Sử dụng phương thức được sửa lỗi _train_for_timeframe
                models = self._train_for_timeframe(timeframe)
                
                if models:
                    self._add_log(f"✅ Huấn luyện thành công cho {timeframe}")
                    logger.info(f"Training successful for {timeframe}")
                else:
                    self._add_log(f"⚠️ Không huấn luyện được mô hình cho {timeframe}")
                    logger.warning(f"No models trained for {timeframe}")
            except Exception as e:
                self._add_log(f"❌ Lỗi huấn luyện cho {timeframe}: {str(e)}")
                logger.error(f"Error training for {timeframe}: {e}")
                # Tiếp tục với timeframe tiếp theo
                continue
                
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