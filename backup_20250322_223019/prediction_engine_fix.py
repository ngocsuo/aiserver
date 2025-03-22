"""
Sửa lỗi cho PredictionEngine để giải quyết vấn đề "No trained models available"
"""

# Cần sửa trong file prediction/prediction_engine.py

# Tìm và sửa hàm load_models() để nó tạo dự đoán ngẫu nhiên an toàn khi không có mô hình
def load_models(self):
    """
    Load trained models for prediction.
    
    Returns:
        dict: Loaded models
    """
    try:
        if not os.path.exists(config.MODELS_DIR):
            logger.warning(f"Models directory {config.MODELS_DIR} does not exist")
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            return None
            
        models = {}
        
        # Đường dẫn đến các mô hình đã lưu
        lstm_path = os.path.join(config.MODELS_DIR, 'lstm_model.keras')
        transformer_path = os.path.join(config.MODELS_DIR, 'transformer_model.keras')
        cnn_path = os.path.join(config.MODELS_DIR, 'cnn_model.keras')
        historical_path = os.path.join(config.MODELS_DIR, 'historical_model.pkl')
        meta_path = os.path.join(config.MODELS_DIR, 'meta_model.pkl')
        
        # Danh sách các mô hình đã tìm thấy
        found_models = []
        
        # Tải mô hình LSTM nếu tồn tại
        if os.path.exists(lstm_path):
            try:
                from models.lstm_model import LSTMModel
                input_shape = (config.SEQUENCE_LENGTH, config.NUM_FEATURES)
                lstm_model = LSTMModel(input_shape=input_shape)
                lstm_model.load(lstm_path)
                models['lstm'] = lstm_model
                found_models.append('LSTM')
            except Exception as e:
                logger.error(f"Error loading LSTM model: {e}")
        
        # Tải mô hình Transformer nếu tồn tại
        if os.path.exists(transformer_path):
            try:
                from models.transformer_model import TransformerModel
                input_shape = (config.SEQUENCE_LENGTH, config.NUM_FEATURES)
                transformer_model = TransformerModel(input_shape=input_shape)
                transformer_model.load(transformer_path)
                models['transformer'] = transformer_model
                found_models.append('Transformer')
            except Exception as e:
                logger.error(f"Error loading Transformer model: {e}")
        
        # Tải mô hình CNN nếu tồn tại
        if os.path.exists(cnn_path):
            try:
                from models.cnn_model import CNNModel
                input_shape = (config.SEQUENCE_LENGTH, config.NUM_FEATURES, 1)
                cnn_model = CNNModel(input_shape=input_shape)
                cnn_model.load(cnn_path)
                models['cnn'] = cnn_model
                found_models.append('CNN')
            except Exception as e:
                logger.error(f"Error loading CNN model: {e}")
        
        # Tải mô hình Historical Similarity nếu tồn tại
        if os.path.exists(historical_path):
            try:
                from models.historical_similarity import HistoricalSimilarity
                import pickle
                with open(historical_path, 'rb') as f:
                    historical_model = pickle.load(f)
                models['historical_similarity'] = historical_model
                found_models.append('Historical')
            except Exception as e:
                logger.error(f"Error loading Historical Similarity model: {e}")
        
        # Tải mô hình Meta-Learner nếu tồn tại
        if os.path.exists(meta_path):
            try:
                import pickle
                with open(meta_path, 'rb') as f:
                    meta_model = pickle.load(f)
                models['meta_learner'] = meta_model
                found_models.append('Meta-Learner')
            except Exception as e:
                logger.error(f"Error loading Meta-Learner model: {e}")
        
        if found_models:
            logger.info(f"Loaded models: {', '.join(found_models)}")
            return models
        else:
            logger.warning("No trained models found. Creating placeholder predictions.")
            return None
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None

# Thêm giải pháp tạm thời khi không có mô hình 
# Tìm và sửa hàm predict để nó xử lý tốt hơn khi không có mô hình
def predict(self, data, use_cache=True, timeframe=None):
    """
    Generate predictions from all models and combine them.
    
    Args:
        data (pd.DataFrame): Latest price data
        use_cache (bool): Whether to use cached predictions if valid
        timeframe (str, optional): Khung thời gian dự đoán. Nếu None, sử dụng mặc định
        
    Returns:
        dict: Prediction result with trend, confidence, etc.
    """
    timeframe = timeframe or config.TIMEFRAMES["primary"]
    
    # Check if we can use cached prediction
    if use_cache and self.is_prediction_valid_for_timeframe(timeframe):
        cached_prediction = self.get_cached_prediction(timeframe)
        if cached_prediction:
            logger.info(f"Using cached prediction for {timeframe}")
            return cached_prediction
    
    # Process data and get latest candle
    if data is None or len(data) < config.SEQUENCE_LENGTH:
        logger.warning(f"Insufficient data for prediction (needed {config.SEQUENCE_LENGTH}, got {len(data) if data is not None else 0})")
        # Create a random prediction in case of insufficient data
        return self._create_fallback_prediction(self._get_current_price(timeframe))
    
    # Tải mô hình nếu chưa có
    if not self.models:
        self.models = self.load_models()
    
    # Nếu vẫn không có mô hình sau khi tải, sử dụng dự đoán dự phòng
    # Nhưng thêm thông tin rõ ràng hơn
    if not self.models:
        logger.warning("No trained models available. Using fallback prediction.")
        current_price = self._get_current_price(timeframe)
        fallback_pred = self._create_fallback_prediction(current_price)
        
        # Thêm thông tin trạng thái rõ ràng
        fallback_pred["model_status"] = "not_available"
        fallback_pred["recommendation"] = "Vui lòng huấn luyện mô hình trước khi giao dịch"
        
        # Lưu vào bộ nhớ đệm
        if timeframe not in self.cached_predictions:
            self.cached_predictions[timeframe] = {}
        self.cached_predictions[timeframe]["data"] = fallback_pred
        self.cached_predictions[timeframe]["timestamp"] = self._get_current_timestamp()
        
        return fallback_pred
    
    # (phần code còn lại giữ nguyên)

# Sửa hàm _create_fallback_prediction để có thông tin hơn
def _create_fallback_prediction(self, current_price):
    """
    Create a random prediction for demonstration purposes or when models aren't available.
    
    Args:
        current_price (float): Current price
    
    Returns:
        dict: Random prediction with status information
    """
    # Luôn tạo dự đoán neutral khi không có mô hình
    trend = "NEUTRAL"
    confidence = 0.51  # Chỉ cao hơn 50% một chút
    
    prediction = {
        "timestamp": self._get_current_timestamp(),
        "trend": trend,
        "confidence": confidence,
        "entry_price": current_price,
        "stop_loss": current_price * 0.99 if trend == "LONG" else current_price * 1.01,
        "take_profit": current_price * 1.01 if trend == "LONG" else current_price * 0.99,
        "reason": "Dự đoán tạm thời. Vui lòng huấn luyện mô hình AI.",
        "indicators": {
            "rsi": 50,
            "macd": 0,
            "bollinger": "MIDDLE",
            "trend_strength": "WEAK"
        },
        "models": {
            "lstm": {"prediction": "NEUTRAL", "confidence": 0.5},
            "transformer": {"prediction": "NEUTRAL", "confidence": 0.5},
            "cnn": {"prediction": "NEUTRAL", "confidence": 0.5},
            "historical": {"prediction": "NEUTRAL", "confidence": 0.5},
            "meta": {"prediction": "NEUTRAL", "confidence": 0.5}
        },
        "fallback": True,
        "model_status": "not_trained",
        "recommendation": "Vui lòng huấn luyện mô hình AI trước khi giao dịch"
    }
    
    return prediction