"""
Script khởi động hệ thống ETHUSDT Dashboard với giám sát và xử lý lỗi tự động.

Mục đích: Đảm bảo ứng dụng chạy ổn định và khắc phục các lỗi phổ biến như:
- Empty dataset received for normalization
- Tự động khởi động lại khi gặp lỗi
- Duy trì kết nối liên tục đến Binance API
"""

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
from pathlib import Path

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_log.txt"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Runner")

def apply_feature_engineering_fix():
    """Áp dụng bản vá cho feature_engineering"""
    logger.info("Áp dụng bản vá cho module feature_engineering...")
    try:
        import feature_engineering_fix
        result = feature_engineering_fix.fix_normalize_features() and feature_engineering_fix.fix_empty_dataset_issue()
        if result:
            logger.info("Áp dụng bản vá feature_engineering thành công!")
            return True
        else:
            logger.error("Không thể áp dụng bản vá feature_engineering.")
            return False
    except Exception as e:
        logger.error(f"Lỗi khi áp dụng bản vá feature_engineering: {str(e)}")
        return False

def fix_data_issues():
    """Sửa lỗi dữ liệu"""
    logger.info("Sửa lỗi dữ liệu...")
    try:
        from utils.data_fix import run_data_fix
        results = run_data_fix()
        
        # Kiểm tra kết quả
        success = True
        for timeframe, result in results.items():
            if len(result["after"]["problems"]) > 0:
                logger.warning(f"Vẫn còn {len(result['after']['problems'])} vấn đề với dữ liệu {timeframe}.")
                success = False
                
        return success
    except Exception as e:
        logger.error(f"Lỗi khi sửa dữ liệu: {str(e)}")
        return False

def start_deployment_service():
    """Khởi động dịch vụ triển khai"""
    logger.info("Khởi động dịch vụ triển khai...")
    try:
        from deployment.deploy_service import DeploymentService
        service = DeploymentService()
        service.start()
        return service
    except Exception as e:
        logger.error(f"Lỗi khi khởi động dịch vụ triển khai: {str(e)}")
        return None

def run_with_startup_script():
    """Chạy ứng dụng sử dụng startup script"""
    logger.info("Chạy ứng dụng sử dụng startup script...")
    try:
        # Đường dẫn đến startup script
        startup_script = Path("deployment/startup.sh")
        
        # Đảm bảo script có quyền thực thi
        if not os.access(startup_script, os.X_OK):
            os.chmod(startup_script, 0o755)
        
        # Chạy script
        process = subprocess.Popen(
            [str(startup_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Đọc và ghi log output
        for line in process.stdout:
            logger.info(f"[startup] {line.strip()}")
            
        # Đọc và ghi log error
        for line in process.stderr:
            logger.error(f"[startup] {line.strip()}")
            
        # Đợi process kết thúc
        returncode = process.wait()
        
        if returncode != 0:
            logger.error(f"Startup script kết thúc với mã lỗi: {returncode}")
            return False
        else:
            logger.info("Startup script chạy thành công!")
            return True
    except Exception as e:
        logger.error(f"Lỗi khi chạy startup script: {str(e)}")
        return False

def run_streamlit_directly():
    """Chạy Streamlit trực tiếp"""
    logger.info("Chạy Streamlit trực tiếp...")
    try:
        # Chạy Streamlit
        process = subprocess.Popen(
            ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0", "--server.headless=true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Đọc và ghi log output
        for line in process.stdout:
            logger.info(f"[streamlit] {line.strip()}")
            
        # Đọc và ghi log error
        for line in process.stderr:
            logger.error(f"[streamlit] {line.strip()}")
            
        # Đợi process kết thúc
        returncode = process.wait()
        
        if returncode != 0:
            logger.error(f"Streamlit kết thúc với mã lỗi: {returncode}")
            return False
        else:
            logger.info("Streamlit chạy thành công!")
            return True
    except Exception as e:
        logger.error(f"Lỗi khi chạy Streamlit: {str(e)}")
        return False

def main():
    """Hàm chính"""
    # Khởi tạo parser
    parser = argparse.ArgumentParser(description="Chạy ETHUSDT Dashboard với giám sát và xử lý lỗi tự động")
    parser.add_argument("--mode", choices=["service", "direct", "script"], default="service", 
                      help="Chế độ chạy: service (dịch vụ triển khai), direct (chạy trực tiếp), script (chạy startup script)")
    parser.add_argument("--skip-fixes", action="store_true", help="Bỏ qua các bản vá lỗi")
    
    # Parse tham số
    args = parser.parse_args()
    
    # Hiển thị thông tin
    logger.info("Bắt đầu chạy ETHUSDT Dashboard với giám sát và xử lý lỗi tự động")
    logger.info(f"Chế độ chạy: {args.mode}")
    
    # Áp dụng các bản vá lỗi
    if not args.skip_fixes:
        feature_fix_result = apply_feature_engineering_fix()
        data_fix_result = fix_data_issues()
        
        if not feature_fix_result or not data_fix_result:
            logger.warning("Có vấn đề khi áp dụng bản vá lỗi. Tiếp tục chạy...")
    else:
        logger.info("Bỏ qua các bản vá lỗi theo yêu cầu")
    
    # Chạy theo chế độ đã chọn
    service = None
    try:
        if args.mode == "service":
            # Sử dụng dịch vụ triển khai
            service = start_deployment_service()
            if service:
                # Giữ cho chương trình chạy
                logger.info("Dịch vụ triển khai đã được khởi động. Nhấn Ctrl+C để dừng.")
                while True:
                    time.sleep(1)
        elif args.mode == "script":
            # Sử dụng startup script
            run_with_startup_script()
        else:  # direct
            # Chạy Streamlit trực tiếp
            run_streamlit_directly()
    except KeyboardInterrupt:
        logger.info("Nhận Ctrl+C, đang dừng...")
    finally:
        # Dừng dịch vụ nếu đã khởi động
        if service:
            service.stop()
    
    logger.info("Kết thúc chương trình")

if __name__ == "__main__":
    main()