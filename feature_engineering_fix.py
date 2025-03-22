"""
Sửa lỗi cho module feature_engineering

Giải quyết vấn đề: "Empty dataset received for normalization. Skipping normalization."
"""

import sys
import logging
import pandas as pd
import numpy as np

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='feature_engineering_fix.log'
)

logger = logging.getLogger("FeatureEngineeringFix")

def fix_normalize_features():
    """
    Sửa lỗi trong hàm normalize_features của module feature_engineering
    """
    try:
        # Import module gốc
        from utils.feature_engineering import FeatureEngineer
        
        # Backup phương thức gốc
        original_normalize = FeatureEngineer.normalize_features
        
        # Định nghĩa phiên bản đã sửa
        def fixed_normalize_features(self, df, columns=None):
            """
            Chuẩn hóa các tính năng trong DataFrame bằng MinMaxScaler.
            
            Args:
                df (pd.DataFrame): DataFrame cần chuẩn hóa
                columns (list, optional): Danh sách các cột cần chuẩn hóa. Nếu None, chuẩn hóa các cột có kiểu dữ liệu số
                
            Returns:
                pd.DataFrame: DataFrame đã chuẩn hóa
            """
            if df is None or df.empty:
                # Bổ sung xử lý cho dataset rỗng
                logger.warning("Empty dataset received for normalization. Returning empty DataFrame.")
                return pd.DataFrame()
                
            try:
                # Sao chép DataFrame để tránh thay đổi dữ liệu gốc
                result = df.copy()
                
                # Xác định các cột cần chuẩn hóa
                if columns is None:
                    # Nếu không có danh sách cột, chọn tất cả các cột kiểu số
                    columns = list(df.select_dtypes(include=['number']).columns)
                    
                    # Loại bỏ các cột không muốn chuẩn hóa
                    exclude_columns = ['timestamp', 'date', 'time', 'datetime', 'target', 'label']
                    columns = [col for col in columns if col.lower() not in [ex.lower() for ex in exclude_columns]]
                
                # Kiểm tra các cột tồn tại trong DataFrame
                columns = [col for col in columns if col in df.columns]
                
                if not columns:
                    logger.warning("No valid columns for normalization. Returning original DataFrame.")
                    return result
                    
                # Kiểm tra và xử lý giá trị null
                for col in columns:
                    null_count = result[col].isnull().sum()
                    if null_count > 0:
                        logger.warning(f"Column {col} has {null_count} null values. Filling with interpolation.")
                        result[col] = result[col].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
                        
                    # Kiểm tra giá trị vô cùng
                    inf_count = np.isinf(result[col]).sum()
                    if inf_count > 0:
                        logger.warning(f"Column {col} has {inf_count} infinite values. Replacing with NaN and filling.")
                        result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                        result[col] = result[col].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
                
                # Chuẩn hóa dữ liệu
                from sklearn.preprocessing import MinMaxScaler
                
                # Tạo scaler
                scaler = MinMaxScaler()
                
                # Áp dụng cho từng cột thay vì toàn bộ DataFrame để tránh lỗi
                for col in columns:
                    # Tránh cột không đủ dữ liệu
                    if len(result[col].unique()) <= 1:
                        logger.warning(f"Column {col} has only one unique value. Skipping normalization.")
                        continue
                        
                    try:
                        # Reshape dữ liệu thành mảng 2D cho scaler
                        col_reshaped = result[col].values.reshape(-1, 1)
                        
                        # Thực hiện chuẩn hóa
                        result[col] = scaler.fit_transform(col_reshaped).flatten()
                    except Exception as col_err:
                        logger.error(f"Error normalizing column {col}: {str(col_err)}")
                        # Nếu lỗi, để nguyên cột
                
                return result
                
            except Exception as e:
                logger.error(f"Normalization error: {str(e)}")
                # Trong trường hợp lỗi, trả về DataFrame gốc
                return df
        
        # Thay thế phương thức gốc bằng phiên bản đã sửa
        FeatureEngineer.normalize_features = fixed_normalize_features
        
        logger.info("Successfully patched normalize_features method")
        print("Successfully patched normalize_features method to handle empty datasets")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch normalize_features: {str(e)}")
        print(f"Error: Failed to patch normalize_features - {str(e)}")
        return False

def fix_empty_dataset_issue():
    """
    Sửa lỗi khi nhận được dataset rỗng trong feature engineering
    """
    try:
        # Import module gốc
        from utils.feature_engineering import FeatureEngineer
        
        # Backup phương thức gốc
        original_process = FeatureEngineer.process_features
        
        # Định nghĩa phiên bản đã sửa
        def fixed_process_features(self, df, add_basic=True, add_technical=True, add_pattern=False, 
                             add_labels=True, normalize=True, for_training=True):
            """
            Xử lý và tạo các tính năng cho bộ dữ liệu.
            
            Args:
                df (pd.DataFrame): DataFrame cần xử lý
                add_basic (bool): Thêm các tính năng cơ bản
                add_technical (bool): Thêm các chỉ báo kỹ thuật
                add_pattern (bool): Thêm nhận diện mẫu hình nến
                add_labels (bool): Thêm nhãn mục tiêu
                normalize (bool): Chuẩn hóa dữ liệu
                for_training (bool): Chuẩn bị dữ liệu cho huấn luyện
                
            Returns:
                pd.DataFrame: DataFrame với các tính năng đã thêm
            """
            # Kiểm tra đầu vào
            if df is None or df.empty:
                logger.warning("Empty dataset received for processing. Returning empty DataFrame.")
                return pd.DataFrame()
            
            try:
                # Sao chép DataFrame để tránh thay đổi dữ liệu gốc
                result = df.copy()
                
                # Đảm bảo các cột OHLCV tồn tại
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in result.columns]
                
                if missing_columns:
                    # Nếu thiếu cột cần thiết, ghi log và trả về DataFrame rỗng
                    logger.error(f"Missing required columns: {missing_columns}")
                    return pd.DataFrame()
                
                # Thêm các tính năng cơ bản
                if add_basic:
                    try:
                        result = self.add_basic_features(result)
                    except Exception as e:
                        logger.error(f"Error adding basic features: {str(e)}")
                
                # Thêm các chỉ báo kỹ thuật
                if add_technical:
                    try:
                        result = self.add_technical_indicators(result)
                    except Exception as e:
                        logger.error(f"Error adding technical indicators: {str(e)}")
                
                # Thêm nhận diện mẫu hình nến
                if add_pattern:
                    try:
                        result = self.add_candlestick_patterns(result)
                    except Exception as e:
                        logger.error(f"Error adding candlestick patterns: {str(e)}")
                
                # Thêm nhãn mục tiêu
                if add_labels and for_training:
                    try:
                        result = self.add_target_labels(result)
                    except Exception as e:
                        logger.error(f"Error adding target labels: {str(e)}")
                
                # Kiểm tra xem đã có cột khác ngoài OHLCV chưa
                feature_columns = [col for col in result.columns if col not in required_columns + ['timestamp', 'date', 'time']]
                
                if not feature_columns:
                    logger.warning("No feature columns were added. Check feature engineering pipeline.")
                
                # Xóa hàng có giá trị null
                null_rows_before = result.isnull().any(axis=1).sum()
                if null_rows_before > 0:
                    # Thử nội suy dữ liệu trước khi xóa
                    numeric_cols = result.select_dtypes(include=['number']).columns
                    for col in numeric_cols:
                        result[col] = result[col].interpolate(method='linear')
                    
                    # Điền các giá trị null còn lại
                    result = result.fillna(method='ffill').fillna(method='bfill')
                    
                    # Kiểm tra lại xem còn null không
                    null_rows_after = result.isnull().any(axis=1).sum()
                    if null_rows_after > 0:
                        logger.warning(f"Removed {null_rows_after} rows with null values after interpolation")
                        result = result.dropna()
                
                # Chuẩn hóa dữ liệu nếu yêu cầu
                if normalize and not result.empty:
                    try:
                        result = self.normalize_features(result)
                    except Exception as e:
                        logger.error(f"Error normalizing features: {str(e)}")
                
                # Kiểm tra kết quả cuối cùng
                if result.empty:
                    logger.warning("Processed dataset is empty!")
                else:
                    logger.info(f"Successfully processed dataset with {len(result)} rows and {len(result.columns)} columns")
                
                return result
                
            except Exception as e:
                logger.error(f"Feature processing error: {str(e)}")
                # Trong trường hợp lỗi, trả về DataFrame rỗng
                return pd.DataFrame()
        
        # Thay thế phương thức gốc bằng phiên bản đã sửa
        FeatureEngineer.process_features = fixed_process_features
        
        logger.info("Successfully patched process_features method")
        print("Successfully patched process_features method to handle empty datasets")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch process_features: {str(e)}")
        print(f"Error: Failed to patch process_features - {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting to apply fixes to feature_engineering module...")
    
    # Áp dụng các bản vá
    normalize_fixed = fix_normalize_features()
    process_fixed = fix_empty_dataset_issue()
    
    if normalize_fixed and process_fixed:
        print("All fixes successfully applied!")
        sys.exit(0)
    else:
        print("Some fixes failed. Check logs for details.")
        sys.exit(1)