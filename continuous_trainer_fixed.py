"""
Phiên bản đã sửa lỗi của ContinuousTrainer để tương thích với model_trainer trả về tuple
"""
import logging
import threading
import time
import os
from datetime import datetime, timedelta

from utils.thread_safe_logging import thread_safe_log

logger = logging.getLogger("continuous_trainer_fixed")

def _train_for_timeframe(self, timeframe):
    """
    Train models for a specific timeframe.
    
    Args:
        timeframe (str): Timeframe to train for (e.g., "1m", "5m")
    """
    thread_safe_log(f"Bắt đầu xử lý dữ liệu cho khung thời gian {timeframe}")
    
    # Thu thập dữ liệu từ tất cả các đoạn và xử lý
    all_processed_data = []
    
    # Xử lý từng đoạn thời gian để tránh quá tải bộ nhớ
    for i, (start_date, end_date) in enumerate(self.monthly_chunks):
        thread_safe_log(f"Đang xử lý đoạn {i+1}/{len(self.monthly_chunks)} cho {timeframe}: {start_date} đến {end_date}")
        
        # Thu thập dữ liệu cho đoạn hiện tại
        raw_data = self.data_collector.collect_historical_data(
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Kiểm tra dữ liệu hợp lệ
        if raw_data is not None and len(raw_data) > 0:
            # Xử lý dữ liệu
            processed_data = self.data_processor.process_data(raw_data)
            
            # Nếu có dữ liệu đã xử lý, thêm vào danh sách
            if processed_data is not None and len(processed_data) > 0:
                all_processed_data.append(processed_data)
                thread_safe_log(f"✅ Đã xử lý {len(processed_data)} điểm dữ liệu cho {timeframe} từ {start_date} đến {end_date}")
            else:
                thread_safe_log(f"⚠️ Không có dữ liệu nào được xử lý cho {timeframe} từ {start_date} đến {end_date}")
        else:
            thread_safe_log(f"❌ Không thể thu thập dữ liệu cho {timeframe} từ {start_date} đến {end_date}")
            
        # Cập nhật tiến độ xử lý
        self.current_chunk += 1
        self.current_progress = (self.current_chunk / self.total_chunks) * 100
    
    # Tiếp tục nếu có dữ liệu đã xử lý
    if all_processed_data:
        # Kết hợp tất cả dữ liệu đã xử lý
        combined_data = pd.concat(all_processed_data)
        thread_safe_log(f"✅ Đã kết hợp {len(combined_data)} điểm dữ liệu cho {timeframe}")
        
        # Chuẩn bị dữ liệu cho các loại mô hình khác nhau
        sequence_data = self.data_processor.prepare_sequence_data(combined_data)
        image_data = self.data_processor.prepare_cnn_data(combined_data)
        
        # SỬA LỖI: Xử lý lỗi "too many values to unpack (expected 2)"
        try:
            # SỬA LỖI: Chỉ cần models, không cần history
            models = self.model_trainer.train_all_models(sequence_data, image_data, timeframe=timeframe)
            
            # SỬA LỖI: Kiểm tra xem models có phải là tuple không
            if isinstance(models, tuple):
                # Nếu là tuple, extract models từ tuple
                extracted_models = models[0]
                model_results[timeframe] = extracted_models
                self._add_log(f"⚠️ Đã xử lý kết quả tuple từ train_all_models")
            else:
                # Nếu không phải tuple, sử dụng kết quả trực tiếp
                model_results[timeframe] = models
                
            self._add_log(f"✅ Đã huấn luyện thành công mô hình cho {timeframe}")
            
            # Ghi log số lượng mô hình đã huấn luyện
            if model_results[timeframe]:
                model_count = len(model_results[timeframe])
                self._add_log(f"✅ Đã huấn luyện thành công {model_count} mô hình cho {timeframe}")
                logger.info(f"Trained {model_count} models for {timeframe} with {len(combined_data)} data points")
                
        except Exception as e:
            self._add_log(f"❌ Lỗi trong quá trình huấn luyện: {str(e)}")
            logger.error(f"Error in training process: {e}")
            model_results[timeframe] = {}
    else:
        self._add_log(f"❌ Không có dữ liệu khả dụng cho {timeframe} sau khi xử lý tất cả các đoạn")
        logger.error(f"No processed data available for {timeframe} after processing all chunks")