"""
Module sửa lỗi dữ liệu - Giải quyết vấn đề "Empty dataset received for normalization"
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_fix.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DataFix")

class DataFixTool:
    """Công cụ sửa lỗi dữ liệu mô hình AI"""
    
    def __init__(self, data_dir="data", cache_dir="data/cache"):
        """
        Khởi tạo công cụ sửa lỗi dữ liệu
        
        Args:
            data_dir (str): Thư mục dữ liệu
            cache_dir (str): Thư mục cache dữ liệu
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        
        # Đảm bảo các thư mục tồn tại
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def verify_dataset_integrity(self, timeframe="1m"):
        """
        Kiểm tra tính toàn vẹn của bộ dữ liệu
        
        Args:
            timeframe (str): Khung thời gian của dữ liệu
            
        Returns:
            dict: Thông tin về tính toàn vẹn của dữ liệu
        """
        logger.info(f"Kiểm tra tính toàn vẹn của dữ liệu {timeframe}")
        
        # Tìm tất cả các tập tin dữ liệu gốc
        data_files = list(self.data_dir.glob(f"*{timeframe}*.csv"))
        
        # Tìm tất cả các tập tin cache
        cache_files = list(self.cache_dir.glob(f"*{timeframe}*.pkl"))
        
        # Thống kê
        results = {
            "data_files": {
                "count": len(data_files),
                "files": [f.name for f in data_files],
                "total_size_mb": sum(f.stat().st_size for f in data_files) / (1024 * 1024)
            },
            "cache_files": {
                "count": len(cache_files),
                "files": [f.name for f in cache_files],
                "total_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
            },
            "problems": []
        }
        
        # Kiểm tra từng tập tin dữ liệu
        for data_file in data_files:
            try:
                # Đọc dữ liệu
                df = pd.read_csv(data_file)
                
                # Kiểm tra các cột bắt buộc
                required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    results["problems"].append({
                        "file": data_file.name,
                        "problem": f"Thiếu các cột bắt buộc: {missing_columns}"
                    })
                    continue
                
                # Kiểm tra giá trị null
                null_counts = df[required_columns].isnull().sum()
                if null_counts.sum() > 0:
                    nulls = {col: count for col, count in null_counts.items() if count > 0}
                    results["problems"].append({
                        "file": data_file.name,
                        "problem": f"Có giá trị null trong dữ liệu: {nulls}"
                    })
                
                # Kiểm tra giá trị âm
                for col in ["open", "high", "low", "close", "volume"]:
                    if (df[col] < 0).any():
                        results["problems"].append({
                            "file": data_file.name,
                            "problem": f"Có giá trị âm trong cột {col}"
                        })
                
            except Exception as e:
                results["problems"].append({
                    "file": data_file.name,
                    "problem": f"Lỗi khi đọc tập tin: {str(e)}"
                })
        
        # Kiểm tra các tập tin cache
        for cache_file in cache_files:
            try:
                # Đọc cache
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Kiểm tra cấu trúc
                if not isinstance(data, dict):
                    results["problems"].append({
                        "file": cache_file.name,
                        "problem": "Cache không phải là dict"
                    })
                    continue
                
                # Kiểm tra dữ liệu trống
                if not data:
                    results["problems"].append({
                        "file": cache_file.name,
                        "problem": "Cache rỗng"
                    })
                    continue
                
                # Kiểm tra cấu trúc dữ liệu
                required_keys = ["data", "metadata"]
                missing_keys = [key for key in required_keys if key not in data]
                
                if missing_keys:
                    results["problems"].append({
                        "file": cache_file.name,
                        "problem": f"Thiếu các khóa bắt buộc: {missing_keys}"
                    })
                    continue
                
                # Kiểm tra dữ liệu
                if "data" in data and isinstance(data["data"], pd.DataFrame):
                    df = data["data"]
                    if df.empty:
                        results["problems"].append({
                            "file": cache_file.name,
                            "problem": "DataFrame rỗng"
                        })
                
            except Exception as e:
                results["problems"].append({
                    "file": cache_file.name,
                    "problem": f"Lỗi khi đọc cache: {str(e)}"
                })
        
        logger.info(f"Đã phát hiện {len(results['problems'])} vấn đề")
        return results
    
    def fix_dataset_problems(self, timeframe="1m"):
        """
        Sửa các vấn đề trong bộ dữ liệu
        
        Args:
            timeframe (str): Khung thời gian của dữ liệu
            
        Returns:
            dict: Kết quả sửa chữa
        """
        logger.info(f"Bắt đầu sửa chữa dữ liệu {timeframe}")
        
        # Kiểm tra dữ liệu trước khi sửa
        before = self.verify_dataset_integrity(timeframe)
        
        # Tạo thư mục dự phòng
        backup_dir = self.data_dir / "backup" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(exist_ok=True, parents=True)
        
        # Thực hiện sửa chữa
        fixed_problems = []
        
        # 1. Sửa các tập tin dữ liệu
        for problem in before["problems"]:
            file_name = problem["file"]
            problem_desc = problem["problem"]
            
            # Xác định loại tập tin
            is_data_file = any(file_name == data_file for data_file in before["data_files"]["files"])
            is_cache_file = any(file_name == cache_file for cache_file in before["cache_files"]["files"])
            
            if is_data_file:
                fixed = self._fix_data_file(self.data_dir / file_name, backup_dir, problem_desc)
                if fixed:
                    fixed_problems.append({
                        "file": file_name,
                        "problem": problem_desc,
                        "action": "fixed"
                    })
                
            elif is_cache_file:
                fixed = self._fix_cache_file(self.cache_dir / file_name, backup_dir, problem_desc)
                if fixed:
                    fixed_problems.append({
                        "file": file_name,
                        "problem": problem_desc,
                        "action": "fixed"
                    })
        
        # Điền vào khoảng trống trong dữ liệu (nếu cần)
        self._fill_data_gaps(timeframe)
        
        # Tạo lại cache nếu cần
        self._regenerate_cache(timeframe)
        
        # Kiểm tra lại sau khi sửa
        after = self.verify_dataset_integrity(timeframe)
        
        return {
            "before": before,
            "after": after,
            "fixed": fixed_problems
        }
    
    def _fix_data_file(self, file_path, backup_dir, problem_desc):
        """
        Sửa tập tin dữ liệu
        
        Args:
            file_path (Path): Đường dẫn tập tin
            backup_dir (Path): Thư mục dự phòng
            problem_desc (str): Mô tả vấn đề
            
        Returns:
            bool: True nếu sửa thành công, False nếu không
        """
        try:
            # Sao lưu tập tin
            backup_path = backup_dir / file_path.name
            if file_path.exists():
                import shutil
                shutil.copy2(file_path, backup_path)
            
            # Đọc dữ liệu
            df = pd.read_csv(file_path) if file_path.exists() else None
            
            if df is None or df.empty:
                logger.warning(f"Không thể đọc hoặc tập tin rỗng: {file_path}")
                return False
            
            # Giải quyết vấn đề dựa trên mô tả
            fixed = False
            
            # Thiếu cột
            if "Thiếu các cột bắt buộc" in problem_desc:
                required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
                for col in required_columns:
                    if col not in df.columns:
                        # Tạo cột giả
                        if col == "timestamp":
                            df[col] = pd.date_range(start="2023-01-01", periods=len(df), freq="1min")
                        elif col == "volume":
                            df[col] = np.abs(np.random.normal(1000, 500, len(df)))
                        else:
                            # Với giá, sử dụng giá trị từ cột khác nếu có
                            if "close" in df.columns:
                                df[col] = df["close"]
                            else:
                                df[col] = np.abs(np.random.normal(1000, 50, len(df)))
                fixed = True
            
            # Giá trị null
            if "giá trị null" in problem_desc:
                # Điền giá trị null
                numeric_columns = ["open", "high", "low", "close", "volume"]
                for col in numeric_columns:
                    if col in df.columns:
                        # Sử dụng nội suy tuyến tính cho giá trị null
                        df[col] = df[col].interpolate(method='linear')
                
                        # Nếu vẫn còn giá trị null ở đầu/cuối, sử dụng giá trị gần nhất
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                if "timestamp" in df.columns and df["timestamp"].isnull().any():
                    # Với timestamp, tạo giá trị mới
                    missing_count = df["timestamp"].isnull().sum()
                    if missing_count > 0:
                        last_valid = df["timestamp"].dropna().iloc[-1]
                        if isinstance(last_valid, str):
                            last_valid = pd.to_datetime(last_valid)
                        
                        new_timestamps = pd.date_range(
                            start=last_valid, 
                            periods=missing_count+1, 
                            freq="1min"
                        )[1:]
                        
                        df.loc[df["timestamp"].isnull(), "timestamp"] = new_timestamps
                fixed = True
            
            # Giá trị âm
            if "giá trị âm" in problem_desc:
                # Chuyển đổi giá trị âm thành dương
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = df[col].abs()
                fixed = True
            
            # Lưu lại dữ liệu đã sửa
            if fixed:
                df.to_csv(file_path, index=False)
                logger.info(f"Đã sửa tập tin: {file_path}")
                
            return fixed
            
        except Exception as e:
            logger.error(f"Lỗi khi sửa tập tin {file_path}: {str(e)}")
            return False
    
    def _fix_cache_file(self, file_path, backup_dir, problem_desc):
        """
        Sửa tập tin cache
        
        Args:
            file_path (Path): Đường dẫn tập tin
            backup_dir (Path): Thư mục dự phòng
            problem_desc (str): Mô tả vấn đề
            
        Returns:
            bool: True nếu sửa thành công, False nếu không
        """
        try:
            # Sao lưu tập tin
            backup_path = backup_dir / file_path.name
            if file_path.exists():
                import shutil
                shutil.copy2(file_path, backup_path)
            
            # Trong hầu hết các trường hợp, tốt nhất là xóa cache để tạo lại
            if "Cache rỗng" in problem_desc or "DataFrame rỗng" in problem_desc:
                if file_path.exists():
                    file_path.unlink()
                logger.info(f"Đã xóa cache bị lỗi: {file_path}")
                return True
                
            # Đọc cache
            if not file_path.exists():
                return False
                
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Sửa các vấn đề cụ thể
            fixed = False
            
            # Cache không phải dict
            if "Cache không phải là dict" in problem_desc:
                # Xóa và để tạo lại
                file_path.unlink()
                logger.info(f"Đã xóa cache không hợp lệ: {file_path}")
                return True
            
            # Thiếu khóa
            if "Thiếu các khóa bắt buộc" in problem_desc:
                required_keys = ["data", "metadata"]
                for key in required_keys:
                    if key not in data:
                        if key == "data":
                            data[key] = pd.DataFrame()
                        elif key == "metadata":
                            data[key] = {
                                "created_at": datetime.datetime.now().isoformat(),
                                "version": "1.0"
                            }
                fixed = True
            
            # Lưu lại cache đã sửa
            if fixed:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Đã sửa cache: {file_path}")
                
            return fixed
            
        except Exception as e:
            logger.error(f"Lỗi khi sửa cache {file_path}: {str(e)}")
            return False
    
    def _fill_data_gaps(self, timeframe):
        """
        Điền vào khoảng trống trong dữ liệu
        
        Args:
            timeframe (str): Khung thời gian
        """
        logger.info(f"Kiểm tra và điền khoảng trống dữ liệu {timeframe}")
        
        # Lấy tất cả tập tin dữ liệu theo timeframe
        data_files = sorted(self.data_dir.glob(f"*{timeframe}*.csv"))
        
        if not data_files:
            logger.warning(f"Không tìm thấy tập tin dữ liệu cho {timeframe}")
            return
        
        # Đọc và kết hợp tất cả dữ liệu
        all_data = []
        for file in data_files:
            try:
                df = pd.read_csv(file)
                if "timestamp" in df.columns and not df.empty:
                    # Chuyển đổi timestamp sang datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Lỗi khi đọc {file}: {str(e)}")
        
        if not all_data:
            logger.warning(f"Không có dữ liệu hợp lệ cho {timeframe}")
            return
        
        # Kết hợp dữ liệu
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="first")
        combined_df = combined_df.sort_values("timestamp")
        
        # Tính toán khoảng thời gian dựa trên timeframe
        freq = ""
        if timeframe == "1m":
            freq = "1min"
        elif timeframe == "5m":
            freq = "5min"
        elif timeframe == "15m":
            freq = "15min"
        elif timeframe == "1h":
            freq = "1H"
        elif timeframe == "4h":
            freq = "4H"
        elif timeframe == "1d":
            freq = "1D"
        
        if not freq:
            logger.warning(f"Không thể xác định tần suất cho timeframe {timeframe}")
            return
        
        # Tạo dãy thời gian hoàn chỉnh
        start_time = combined_df["timestamp"].min()
        end_time = combined_df["timestamp"].max()
        
        full_range = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # Kiểm tra các giá trị bị thiếu
        missing_times = set(full_range) - set(combined_df["timestamp"])
        
        if missing_times:
            logger.info(f"Phát hiện {len(missing_times)} giá trị bị thiếu cho {timeframe}")
            
            # Tạo DataFrame cho các giá trị bị thiếu
            missing_df = pd.DataFrame({"timestamp": list(missing_times)})
            
            # Thêm các cột khác bằng nội suy
            full_df = pd.concat([combined_df, missing_df], ignore_index=True)
            full_df = full_df.sort_values("timestamp")
            
            # Nội suy dữ liệu
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in full_df.columns:
                    full_df[col] = full_df[col].interpolate(method='linear')
            
            # Điền giá trị còn thiếu ở đầu và cuối
            full_df = full_df.fillna(method='ffill').fillna(method='bfill')
            
            # Tạo tập tin mới với dữ liệu đã điền
            output_file = self.data_dir / f"combined_{timeframe}_filled.csv"
            full_df.to_csv(output_file, index=False)
            
            logger.info(f"Đã tạo tập tin dữ liệu đã điền: {output_file}")
        else:
            logger.info(f"Không phát hiện giá trị bị thiếu cho {timeframe}")
    
    def _regenerate_cache(self, timeframe):
        """
        Tạo lại các tập tin cache
        
        Args:
            timeframe (str): Khung thời gian
        """
        logger.info(f"Tạo lại cache cho {timeframe}")
        
        # Xóa các cache hiện tại
        cache_files = list(self.cache_dir.glob(f"*{timeframe}*.pkl"))
        for file in cache_files:
            try:
                file.unlink()
                logger.info(f"Đã xóa cache cũ: {file}")
            except Exception as e:
                logger.error(f"Lỗi khi xóa cache {file}: {str(e)}")
        
        # Đọc dữ liệu đã kết hợp nếu có
        combined_file = self.data_dir / f"combined_{timeframe}_filled.csv"
        if combined_file.exists():
            df = pd.read_csv(combined_file)
            
            # Chuyển đổi timestamp sang datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Chuẩn bị metadata
            metadata = {
                "created_at": datetime.datetime.now().isoformat(),
                "source": str(combined_file),
                "timeframe": timeframe,
                "rows": len(df),
                "version": "1.0"
            }
            
            # Tạo cache mới
            cache_data = {
                "data": df,
                "metadata": metadata
            }
            
            # Lưu cache
            cache_file = self.cache_dir / f"combined_{timeframe}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Đã tạo cache mới: {cache_file}")
            
def run_data_fix():
    """Chạy sửa chữa dữ liệu tự động"""
    tool = DataFixTool()
    
    # Sửa dữ liệu cho cả hai timeframe
    results = {}
    for timeframe in ["1m", "5m"]:
        results[timeframe] = tool.fix_dataset_problems(timeframe)
    
    # Lưu báo cáo
    with open("data_fix_report.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # In kết quả
    for timeframe, result in results.items():
        print(f"=== Kết quả sửa chữa cho {timeframe} ===")
        print(f"Trước: {len(result['before']['problems'])} vấn đề")
        print(f"Sau: {len(result['after']['problems'])} vấn đề")
        print(f"Đã sửa: {len(result['fixed'])} vấn đề")
        
    return results
    
if __name__ == "__main__":
    run_data_fix()