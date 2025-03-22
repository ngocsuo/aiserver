"""
Phiên bản sửa lỗi cho phương thức _train_for_timeframe trong ContinuousTrainer
"""

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
        # Hàm train_all_models chỉ trả về một giá trị (models), không trả về hai giá trị (models, histories)
        # Thay vì: models, histories = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
        # Sửa thành:
        models = model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
        
        if models:
            self._add_log(f"Models trained successfully for {timeframe}")
            thread_safe_log(f"Models trained successfully for {timeframe}")
        else:
            self._add_log(f"No models trained for {timeframe}")
            thread_safe_log(f"No models trained for {timeframe}")
            
    except Exception as e:
        self._add_log(f"Error training for timeframe {timeframe}: {e}")
        thread_safe_log(f"Error training for timeframe {timeframe}: {e}")
        logger.error(f"Error training for timeframe {timeframe}: {e}")
        raise