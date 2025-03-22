"""
Phiên bản đã sửa lỗi của quá trình huấn luyện mô hình.
- Đã sửa lỗi "too many values to unpack"
- Đã thêm thread-safe logging
"""

def _train_models():
    # Thực thi các lệnh sau để tạo các tệp tin cần thiết
    """
    # Tạo thư mục utils nếu chưa tồn tại
    mkdir -p utils
    
    # Tạo tệp thread_safe_logging.py trong thư mục utils nếu chưa tồn tại
    cat > utils/thread_safe_logging.py << 'EOL'
    '''
    Thread-safe logging functions for AI Trading System
    '''
    import os
    import threading
    import logging
    from datetime import datetime

    # Thiết lập cơ bản cho logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Khóa thread để đảm bảo an toàn khi ghi file
    file_lock = threading.Lock()
    console_lock = threading.Lock()

    def log_to_file(message, log_file="training_logs.txt"):
        """Thread-safe function to log messages to a file"""
        with file_lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a") as f:
                f.write(f"{timestamp} - {message}\n")

    def log_to_console(message):
        """Thread-safe function to log messages to console"""
        with console_lock:
            logger = logging.getLogger("training")
            logger.info(message)
            
    def thread_safe_log(message, log_file="training_logs.txt"):
        """Combined logging function that logs to both file and console"""
        log_to_file(message, log_file)
        log_to_console(message)
        
    def read_logs_from_file(log_file="training_logs.txt", max_lines=100):
        """Read log entries from file with a maximum number of lines"""
        try:
            with file_lock:
                # Đảm bảo tệp tin tồn tại
                if not os.path.exists(log_file):
                    with open(log_file, 'w') as f:
                        pass
                    return []
                
                # Đọc tất cả các dòng
                with open(log_file, "r") as f:
                    lines = f.readlines()
                
                # Lấy các dòng cuối cùng
                return lines[-max_lines:] if lines else []
        except Exception as e:
            logging.error(f"Error reading logs: {e}")
            return []
    EOL
    
    # Tạo file training_logs.txt nếu chưa tồn tại
    touch training_logs.txt
    chmod 666 training_logs.txt
    """
    
    # Để sửa lỗi huấn luyện, thay thế hàm train_models và train_models_background
    # Sao chép đoạn mã sau vào app.py, thay thế các hàm hiện có:
    
    def train_models_background():
        """Hàm huấn luyện chạy trong thread riêng biệt"""
        from utils.thread_safe_logging import thread_safe_log
        import time
        import os
        from datetime import datetime
        import sys
        import traceback
        import config
        
        try:
            thread_safe_log("Bắt đầu quá trình huấn luyện mô hình...")
            
            # Ghi lại thời gian bắt đầu
            start_time = time.time()
            thread_safe_log(f"Thời gian bắt đầu: {datetime.now().strftime('%H:%M:%S')}")
            
            # Đảm bảo thư mục models tồn tại
            if not os.path.exists('models'):
                os.makedirs('models')
                thread_safe_log("Đã tạo thư mục models")
                
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
                thread_safe_log("Đã tạo thư mục saved_models")
            
            # Thu thập dữ liệu huấn luyện từ Binance
            thread_safe_log("Đang thu thập dữ liệu từ Binance...")
            
            from utils.data_collector import create_data_collector
            data_collector = create_data_collector()
            
            # Thu thập dữ liệu cho timeframe chính (5m)
            thread_safe_log(f"Thu thập dữ liệu cho khung thời gian {config.PRIMARY_TIMEFRAME}...")
            
            try:
                df_5m = data_collector.collect_historical_data(
                    timeframe=config.PRIMARY_TIMEFRAME,
                    limit=config.LOOKBACK_PERIODS
                )
                thread_safe_log(f"Đã thu thập {len(df_5m)} điểm dữ liệu 5m")
            except Exception as e:
                thread_safe_log(f"Lỗi khi thu thập dữ liệu 5m: {str(e)}")
                raise
                
            # Thu thập dữ liệu cho timeframe thứ cấp (1m)
            thread_safe_log(f"Thu thập dữ liệu cho khung thời gian {config.SECONDARY_TIMEFRAME}...")
            try:
                df_1m = data_collector.collect_historical_data(
                    timeframe=config.SECONDARY_TIMEFRAME,
                    limit=config.LOOKBACK_PERIODS
                )
                thread_safe_log(f"Đã thu thập {len(df_1m)} điểm dữ liệu 1m")
            except Exception as e:
                thread_safe_log(f"Lỗi khi thu thập dữ liệu 1m: {str(e)}")
                raise
            
            # Xử lý dữ liệu
            thread_safe_log("Đang xử lý dữ liệu...")
            from utils.data_processor import DataProcessor
            data_processor = DataProcessor()
            
            # Xử lý dữ liệu cho 5m
            try:
                processed_data_5m = data_processor.process_data(df_5m)
                thread_safe_log(f"Đã xử lý {len(processed_data_5m)} điểm dữ liệu 5m")
            except Exception as e:
                thread_safe_log(f"Lỗi khi xử lý dữ liệu 5m: {str(e)}")
                raise
                
            # Xử lý dữ liệu cho 1m
            try:
                processed_data_1m = data_processor.process_data(df_1m)
                thread_safe_log(f"Đã xử lý {len(processed_data_1m)} điểm dữ liệu 1m")
            except Exception as e:
                thread_safe_log(f"Lỗi khi xử lý dữ liệu 1m: {str(e)}")
                raise
                
            # Chuẩn bị dữ liệu cho LSTM và Transformer (5m)
            thread_safe_log("Chuẩn bị dữ liệu cho mô hình sequence (5m)...")
            sequence_data_5m = data_processor.prepare_sequence_data(processed_data_5m)
            
            # Chuẩn bị dữ liệu cho LSTM và Transformer (1m)
            thread_safe_log("Chuẩn bị dữ liệu cho mô hình sequence (1m)...")
            sequence_data_1m = data_processor.prepare_sequence_data(processed_data_1m)
            
            # Chuẩn bị dữ liệu cho CNN (5m)
            thread_safe_log("Chuẩn bị dữ liệu cho mô hình CNN (5m)...")
            image_data_5m = data_processor.prepare_cnn_data(processed_data_5m)
            
            # Chuẩn bị dữ liệu cho CNN (1m)
            thread_safe_log("Chuẩn bị dữ liệu cho mô hình CNN (1m)...")
            image_data_1m = data_processor.prepare_cnn_data(processed_data_1m)
            
            # Huấn luyện mô hình
            thread_safe_log("Bắt đầu huấn luyện mô hình...")
            from models.model_trainer import ModelTrainer
            
            # Hiển thị kích thước dữ liệu huấn luyện
            if 'X_train' in sequence_data_5m:
                thread_safe_log(f"Kích thước dữ liệu huấn luyện 5m: {sequence_data_5m['X_train'].shape}")
            if 'X_train' in sequence_data_1m:
                thread_safe_log(f"Kích thước dữ liệu huấn luyện 1m: {sequence_data_1m['X_train'].shape}")
                
            # Hiển thị phân phối lớp
            if 'y_train' in sequence_data_5m:
                classes_5m, counts_5m = np.unique(sequence_data_5m['y_train'], return_counts=True)
                class_dist = {int(classes_5m[i]): int(counts_5m[i]) for i in range(len(classes_5m))}
                thread_safe_log(f"Phân phối lớp 5m: SHORT: {class_dist.get(0, 0)}, NEUTRAL: {class_dist.get(1, 0)}, LONG: {class_dist.get(2, 0)}")
            
            if 'y_train' in sequence_data_1m:
                classes_1m, counts_1m = np.unique(sequence_data_1m['y_train'], return_counts=True)
                class_dist = {int(classes_1m[i]): int(counts_1m[i]) for i in range(len(classes_1m))}
                thread_safe_log(f"Phân phối lớp 1m: SHORT: {class_dist.get(0, 0)}, NEUTRAL: {class_dist.get(1, 0)}, LONG: {class_dist.get(2, 0)}")
            
            # Huấn luyện mô hình 5m
            thread_safe_log("Huấn luyện mô hình cho timeframe 5m...")
            model_trainer_5m = ModelTrainer()
            
            try:
                # SỬA LỖI: Sửa cách gọi hàm train_all_models để tránh lỗi "too many values to unpack"
                models_5m = model_trainer_5m.train_all_models(sequence_data_5m, image_data_5m, timeframe="5m")
                thread_safe_log(f"Đã huấn luyện thành công {len(models_5m) if models_5m else 0} mô hình cho timeframe 5m")
            except ValueError as e:
                if "too many values to unpack" in str(e):
                    thread_safe_log(f"Lỗi giá trị khi huấn luyện 5m: {e}, đang thử phương pháp khác")
                    # Lưu kết quả vào biến tạm thời
                    result = model_trainer_5m.train_all_models(sequence_data_5m, image_data_5m, timeframe="5m")
                    # Lấy phần tử đầu tiên nếu là một tuple
                    if isinstance(result, tuple) and len(result) > 0:
                        models_5m = result[0]
                    else:
                        models_5m = result
                    thread_safe_log(f"Đã khắc phục lỗi và huấn luyện thành công {len(models_5m) if models_5m else 0} mô hình cho timeframe 5m")
                else:
                    thread_safe_log(f"Lỗi khi huấn luyện mô hình 5m: {str(e)}")
                    raise
            except Exception as e:
                thread_safe_log(f"Lỗi khi huấn luyện mô hình 5m: {str(e)}")
                raise
            
            # Huấn luyện mô hình 1m
            thread_safe_log("Huấn luyện mô hình cho timeframe 1m...")
            model_trainer_1m = ModelTrainer()
            
            try:
                # SỬA LỖI: Sửa cách gọi hàm train_all_models để tránh lỗi "too many values to unpack"
                models_1m = model_trainer_1m.train_all_models(sequence_data_1m, image_data_1m, timeframe="1m")
                thread_safe_log(f"Đã huấn luyện thành công {len(models_1m) if models_1m else 0} mô hình cho timeframe 1m")
            except ValueError as e:
                if "too many values to unpack" in str(e):
                    thread_safe_log(f"Lỗi giá trị khi huấn luyện 1m: {e}, đang thử phương pháp khác")
                    # Lưu kết quả vào biến tạm thời
                    result = model_trainer_1m.train_all_models(sequence_data_1m, image_data_1m, timeframe="1m")
                    # Lấy phần tử đầu tiên nếu là một tuple
                    if isinstance(result, tuple) and len(result) > 0:
                        models_1m = result[0]
                    else:
                        models_1m = result
                    thread_safe_log(f"Đã khắc phục lỗi và huấn luyện thành công {len(models_1m) if models_1m else 0} mô hình cho timeframe 1m")
                else:
                    thread_safe_log(f"Lỗi khi huấn luyện mô hình 1m: {str(e)}")
                    raise
            except Exception as e:
                thread_safe_log(f"Lỗi khi huấn luyện mô hình 1m: {str(e)}")
                raise
            
            # Ghi lại thời gian kết thúc
            end_time = time.time()
            execution_time = end_time - start_time
            minutes = int(execution_time // 60)
            seconds = int(execution_time % 60)
            
            thread_safe_log(f"Hoàn tất quá trình huấn luyện mô hình trong {minutes} phút {seconds} giây")
            thread_safe_log(f"Thời gian kết thúc: {datetime.now().strftime('%H:%M:%S')}")
            
            # Cập nhật thông tin huấn luyện
            thread_safe_log("Cập nhật thông tin huấn luyện")
            training_info = {
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "models_5m": models_5m is not None,
                "models_1m": models_1m is not None,
                "execution_time_seconds": execution_time,
                "data_points_5m": len(df_5m) if df_5m is not None else 0,
                "data_points_1m": len(df_1m) if df_1m is not None else 0
            }
            
            # Trả về thông báo thành công
            show_toast("Huấn luyện mô hình thành công", type="success")
            thread_safe_log("Đã hoàn tất quá trình huấn luyện")
            return True
            
        except Exception as e:
            thread_safe_log(f"Lỗi trong quá trình huấn luyện: {str(e)}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            trace = traceback.format_exc()
            thread_safe_log(f"Chi tiết lỗi: {trace}")
            show_toast(f"Lỗi huấn luyện: {str(e)}", type="error")
            return False
    
    def train_models():
        """Train all prediction models in a background thread"""
        from utils.thread_safe_logging import thread_safe_log, read_logs_from_file
        
        # Kiểm tra xem quá trình huấn luyện đã đang chạy hay chưa
        if st.session_state.get('is_training', False):
            st.warning("Quá trình huấn luyện đang diễn ra, vui lòng đợi...")
            return False
            
        # Bắt đầu quá trình huấn luyện trong một luồng riêng biệt
        st.session_state['is_training'] = True
        st.session_state['training_start_time'] = time.time()
        st.session_state['training_logs'] = []
        
        # Tạo và khởi động luồng huấn luyện
        thread_safe_log("Khởi động luồng huấn luyện mô hình...")
        train_thread = threading.Thread(
            target=train_models_background,
            name="Training-Thread",
            daemon=True
        )
        train_thread.start()
        
        # Hiển thị thông báo
        st.success("Quá trình huấn luyện đã bắt đầu trong nền. Bạn có thể tiếp tục sử dụng ứng dụng.")
        
        # Hiển thị logs huấn luyện hiện tại
        logs = read_logs_from_file("training_logs.txt", max_lines=100)
        if logs:
            st.session_state['training_logs'] = logs
            with st.expander("Xem logs huấn luyện", expanded=True):
                for log in logs:
                    st.text(log.strip())
                    
        return True