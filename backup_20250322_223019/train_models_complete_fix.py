"""
Phiên bản sửa đổi của hàm train_models() phù hợp với cấu trúc hiện tại
"""

def train_models():
    """Train all prediction models in a background thread"""
    import os
    import threading
    import datetime
    import streamlit as st
    from utils.thread_safe_logging import thread_safe_log
    
    # Tạo file training_logs.txt nếu chưa tồn tại
    if not os.path.exists("training_logs.txt"):
        with open("training_logs.txt", "w") as f:
            f.write("# Training logs started\n")
    
    # Thông báo cho người dùng
    progress_placeholder = st.empty()
    progress_placeholder.info("Quá trình huấn luyện bắt đầu trong nền. Bạn có thể tiếp tục sử dụng ứng dụng.")
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Kiểm tra xem có thông số tùy chỉnh không
    custom_params = st.session_state.get('custom_training_params', None)
    if custom_params:
        thread_safe_log(f"Sử dụng cài đặt tùy chỉnh: {custom_params['timeframe']}, {custom_params['range']}, ngưỡng {custom_params['threshold']}%, {custom_params['epochs']} epochs")
        if hasattr(st, 'toast'):
            st.toast(f"Huấn luyện với cài đặt tùy chỉnh: {custom_params['timeframe']}, {custom_params['epochs']} epochs", icon="🔧")
    
    # Ghi log vào file
    thread_safe_log("Bắt đầu quá trình huấn luyện mô hình...")
    
    # Tạo thread huấn luyện
    training_thread = threading.Thread(
        target=train_models_background,
        name="train_models_background"
    )
    training_thread.daemon = True
    training_thread.start()
    
    # Hiển thị thông báo cho user
    st.success("Đã bắt đầu huấn luyện mô hình trong nền. Kiểm tra logs để theo dõi tiến trình.")
    
    # Xóa các thành phần UI hiển thị lên
    if 'progress_bar' in locals():
        progress_bar.empty()
    if 'progress_placeholder' in locals():
        progress_placeholder.empty()
    
    return True

def train_models_background():
    """Hàm huấn luyện chạy trong thread riêng biệt"""
    from utils.thread_safe_logging import thread_safe_log
    import datetime
    
    try:
        thread_safe_log("Bắt đầu huấn luyện mô hình AI trong thread riêng...")
        thread_safe_log("LƯU Ý: Đang sử dụng phiên bản an toàn thread, tránh truy cập session_state")
        
        # QUAN TRỌNG: KHÔNG truy cập st.session_state trong thread này!
        # Thay vì lấy dữ liệu từ session_state, chúng ta sẽ tải dữ liệu mới
        
        from utils.data_collector import create_data_collector
        from utils.data_processor import DataProcessor
        from models.model_trainer import ModelTrainer
        import config
        
        thread_safe_log("Bước 1/5: Tạo data collector...")
        data_collector = create_data_collector()
        
        thread_safe_log("Tạo data processor và model trainer...")
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        
        thread_safe_log("Bước 2/5: Thu thập dữ liệu lịch sử...")
        if hasattr(config, 'HISTORICAL_START_DATE') and config.HISTORICAL_START_DATE:
            data = data_collector.collect_historical_data(
                timeframe=config.TIMEFRAMES["primary"],
                start_date=config.HISTORICAL_START_DATE
            )
        else:
            data = data_collector.collect_historical_data(
                timeframe=config.TIMEFRAMES["primary"],
                limit=config.LOOKBACK_PERIODS
            )
        
        if data is None or len(data) == 0:
            thread_safe_log("KHÔNG THỂ thu thập dữ liệu cho huấn luyện")
            return
            
        thread_safe_log(f"Đã thu thập {len(data)} nến dữ liệu")
        
        # Tiếp tục quy trình huấn luyện mô hình với dữ liệu mới thu thập
        thread_safe_log("Bước 3/5: Xử lý dữ liệu...")
        processed_data = data_processor.process_data(data)
        
        # Display feature information
        feature_count = len(processed_data.columns) - 1  # Exclude target column
        thread_safe_log(f"Đã tạo {feature_count} chỉ báo kỹ thuật và tính năng")
        thread_safe_log(f"Mẫu huấn luyện: {len(processed_data)}")
        
        # Prepare data for models
        thread_safe_log("Chuẩn bị dữ liệu chuỗi cho LSTM và Transformer...")
        sequence_data = data_processor.prepare_sequence_data(processed_data)
        
        thread_safe_log("Chuẩn bị dữ liệu hình ảnh cho CNN...")
        image_data = data_processor.prepare_cnn_data(processed_data)
        
        # Huấn luyện từng mô hình riêng biệt
        thread_safe_log("Bước 4/5: Huấn luyện mô hình LSTM...")
        lstm_model, lstm_history = model_trainer.train_lstm(sequence_data)
        
        thread_safe_log("Huấn luyện mô hình Transformer...")
        transformer_model, transformer_history = model_trainer.train_transformer(sequence_data)
        
        thread_safe_log("Huấn luyện mô hình CNN...")
        cnn_model, cnn_history = model_trainer.train_cnn(image_data)
        
        thread_safe_log("Huấn luyện mô hình Similarity lịch sử...")
        historical_model, _ = model_trainer.train_historical_similarity(sequence_data)
        
        thread_safe_log("Bước 5/5: Huấn luyện mô hình Meta-Learner...")
        meta_model, _ = model_trainer.train_meta_learner(sequence_data, image_data)
        
        thread_safe_log("Huấn luyện thành công tất cả các mô hình!")
        
        # Lưu trạng thái huấn luyện vào file
        try:
            import json
            training_result = {
                "success": True,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": "Huấn luyện thành công tất cả các mô hình"
            }
            with open('training_result.json', 'w') as f:
                json.dump(training_result, f)
                
            # Thông báo đã huấn luyện thành công - set flag cho main thread
            with open('training_completed.txt', 'w') as f:
                f.write('success')
        except Exception as e:
            thread_safe_log(f"Lỗi lưu kết quả huấn luyện: {str(e)}")
                
    except Exception as e:
        from utils.thread_safe_logging import thread_safe_log
        thread_safe_log(f"LỖI trong quá trình huấn luyện: {str(e)}")
        
        # Lưu thông tin lỗi vào file
        try:
            import json
            training_result = {
                "success": False,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }
            with open('training_result.json', 'w') as f:
                json.dump(training_result, f)
                
            # Thông báo lỗi cho main thread
            with open('training_completed.txt', 'w') as f:
                f.write('error')
        except Exception:
            pass