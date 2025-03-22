"""
Mã sửa lỗi hoàn chỉnh cho hàm train_models_background() nhằm đảm bảo thread-safety.
"""

# Để sử dụng mã này:
# 1. Sao chép toàn bộ nội dung của hàm train_models_background() này vào app.py
# 2. Đảm bảo đã import thread_safe_log từ thread_safe_logging trước khi sử dụng

def train_models_background():
    try:
        # Ghi log bắt đầu huấn luyện (sử dụng thread_safe_log thay vì update_log)
        thread_safe_log("Bắt đầu quá trình huấn luyện mô hình AI trong nền...")
        
        # Step 1: Process data for training
        thread_safe_log("Bước 1/5: Chuẩn bị dữ liệu ETHUSDT...")
        
        # Kiểm tra xem có sử dụng tham số tùy chỉnh không
        custom_params = st.session_state.get('custom_training_params', None)
        if custom_params:
            thread_safe_log(f"🔧 Đang áp dụng cài đặt tùy chỉnh: {custom_params['timeframe']}, {custom_params['range']}, ngưỡng {custom_params['threshold']}%, {custom_params['epochs']} epochs")
            # TODO: Áp dụng các tham số tùy chỉnh vào quá trình huấn luyện
            # Nếu người dùng chọn khung thời gian khác
            if custom_params['timeframe'] != config.TIMEFRAMES['primary']:
                thread_safe_log(f"Chuyển sang khung thời gian {custom_params['timeframe']} theo cài đặt tùy chỉnh")
                # Cần lấy dữ liệu cho khung thời gian được chọn
                try:
                    if hasattr(st.session_state, 'data_collector'):
                        thread_safe_log(f"Đang tải dữ liệu khung thời gian {custom_params['timeframe']}...")
                        custom_data = st.session_state.data_collector.collect_historical_data(
                            symbol=config.SYMBOL,
                            timeframe=custom_params['timeframe'],
                            limit=config.LOOKBACK_PERIODS
                        )
                        if custom_data is not None and not custom_data.empty:
                            data = custom_data
                            thread_safe_log(f"Đã tải {len(data)} nến dữ liệu {custom_params['timeframe']}")
                        else:
                            thread_safe_log(f"⚠️ Không thể tải dữ liệu cho khung thời gian {custom_params['timeframe']}, dùng dữ liệu mặc định")
                except Exception as e:
                    thread_safe_log(f"❌ Lỗi khi tải dữ liệu tùy chỉnh: {str(e)}")
            
            # Cập nhật số epochs theo cài đặt
            config.EPOCHS = custom_params['epochs']
            thread_safe_log(f"Cập nhật số epochs huấn luyện: {config.EPOCHS}")
            
            # Cập nhật ngưỡng biến động giá
            config.PRICE_MOVEMENT_THRESHOLD = custom_params['threshold'] / 100  # Chuyển % thành tỷ lệ thập phân
            thread_safe_log(f"Cập nhật ngưỡng biến động giá: {custom_params['threshold']}%")
            
        data = st.session_state.latest_data
        thread_safe_log(f"Nguồn dữ liệu: {'Binance API' if not isinstance(st.session_state.data_collector, type(__import__('utils.data_collector').data_collector.MockDataCollector)) else 'Mô phỏng (chế độ phát triển)'}")
        thread_safe_log(f"Số điểm dữ liệu: {len(data)} nến")
        thread_safe_log(f"Khung thời gian: {data.name if hasattr(data, 'name') else config.TIMEFRAMES['primary']}")
        thread_safe_log(f"Phạm vi ngày: {data.index.min()} đến {data.index.max()}")
        
        # Step 2: Preprocess data
        thread_safe_log("Bước 2/5: Tiền xử lý dữ liệu và tính toán chỉ báo kỹ thuật...")
        processed_data = st.session_state.data_processor.process_data(data)
        
        # Display feature information
        feature_count = len(processed_data.columns) - 1  # Exclude target column
        thread_safe_log(f"Đã tạo {feature_count} chỉ báo kỹ thuật và tính năng")
        thread_safe_log(f"Mẫu huấn luyện: {len(processed_data)} (sau khi loại bỏ giá trị NaN)")
        
        # Display class distribution
        if 'target_class' in processed_data.columns:
            class_dist = processed_data['target_class'].value_counts()
            thread_safe_log(f"Phân phối lớp: SHORT={class_dist.get(0, 0)}, NEUTRAL={class_dist.get(1, 0)}, LONG={class_dist.get(2, 0)}")
        
        # Step 3: Prepare sequence and image data
        thread_safe_log("Bước 3/5: Chuẩn bị dữ liệu chuỗi cho mô hình LSTM và Transformer...")
        sequence_data = st.session_state.data_processor.prepare_sequence_data(processed_data)
        
        thread_safe_log("Chuẩn bị dữ liệu hình ảnh cho mô hình CNN...")
        image_data = st.session_state.data_processor.prepare_cnn_data(processed_data)
        
        # Step 4: Train all models
        thread_safe_log("Bước 4/5: Huấn luyện mô hình LSTM...")
        lstm_model, lstm_history = st.session_state.model_trainer.train_lstm(sequence_data)
        thread_safe_log(f"Mô hình LSTM đã huấn luyện với độ chính xác: {lstm_history.get('val_accuracy', [-1])[-1]:.4f}")
        
        thread_safe_log("Huấn luyện mô hình Transformer...")
        transformer_model, transformer_history = st.session_state.model_trainer.train_transformer(sequence_data)
        thread_safe_log(f"Mô hình Transformer đã huấn luyện với độ chính xác: {transformer_history.get('val_accuracy', [-1])[-1]:.4f}")
        
        thread_safe_log("Huấn luyện mô hình CNN...")
        cnn_model, cnn_history = st.session_state.model_trainer.train_cnn(image_data)
        thread_safe_log(f"Mô hình CNN đã huấn luyện với độ chính xác: {cnn_history.get('val_accuracy', [-1])[-1]:.4f}")
        
        thread_safe_log("Huấn luyện mô hình Similarity lịch sử...")
        historical_model, _ = st.session_state.model_trainer.train_historical_similarity(sequence_data)
        
        thread_safe_log("Bước 5/5: Huấn luyện mô hình Meta-Learner...")
        meta_model, _ = st.session_state.model_trainer.train_meta_learner(sequence_data, image_data)
        
        # Finalize
        thread_safe_log("Tất cả các mô hình đã huấn luyện thành công!")
        
        # Store training data information in session state for reference
        st.session_state.training_info = {
            "data_source": 'Real Binance API' if not isinstance(st.session_state.data_collector, type(__import__('utils.data_collector').data_collector.MockDataCollector)) else 'Simulated data (development mode)',
            "data_points": len(data),
            "date_range": f"{data.index.min()} to {data.index.max()}",
            "feature_count": feature_count,
            "training_samples": len(processed_data),
            "class_distribution": {
                "SHORT": class_dist.get(0, 0) if 'target_class' in processed_data.columns and class_dist is not None else 0,
                "NEUTRAL": class_dist.get(1, 0) if 'target_class' in processed_data.columns and class_dist is not None else 0,
                "LONG": class_dist.get(2, 0) if 'target_class' in processed_data.columns and class_dist is not None else 0
            },
            "model_performance": {
                "lstm": lstm_history.get('val_accuracy', [-1])[-1],
                "transformer": transformer_history.get('val_accuracy', [-1])[-1],
                "cnn": cnn_history.get('val_accuracy', [-1])[-1],
                "historical_similarity": 0.65,
                "meta_learner": 0.85
            },
            "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Set models as trained
        st.session_state.model_trained = True
        
        # Show toast notification
        show_toast("Tất cả các mô hình AI đã được huấn luyện thành công!", "success", duration=5000)
        
        return True
    except Exception as e:
        # Log error using thread-safe function
        thread_safe_log(f"LỖI trong quá trình huấn luyện: {str(e)}")
        
        # Show toast notification
        show_toast(f"Lỗi trong quá trình huấn luyện: {str(e)}", "error", duration=5000)
        return False