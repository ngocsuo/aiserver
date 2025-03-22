"""
Module cập nhật cấu hình cho hệ thống AI Trading
"""
import os
import re
import logging

# Configure logging
logger = logging.getLogger("config_updater")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def update_config_value(key, value):
    """
    Cập nhật giá trị trong tập tin config.py
    
    Args:
        key (str): Khóa cấu hình cần cập nhật
        value: Giá trị mới
    
    Returns:
        bool: True nếu cập nhật thành công, False nếu không
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.py')
    
    # Kiểm tra tập tin tồn tại
    if not os.path.exists(config_path):
        logger.error(f"Không tìm thấy tập tin config.py tại {config_path}")
        return False
    
    try:
        # Đọc nội dung tập tin
        with open(config_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Chuẩn bị giá trị để thay thế
        if isinstance(value, str):
            # Nếu giá trị là chuỗi, thêm dấu ngoặc kép
            formatted_value = f'"{value}"'
        else:
            # Nếu không phải chuỗi, chuyển thành dạng chuỗi bình thường
            formatted_value = str(value)
        
        # Tạo pattern tìm kiếm
        pattern = rf'{key}\s*=\s*.*'
        replacement = f'{key} = {formatted_value}'
        
        # Thay thế giá trị
        new_content = re.sub(pattern, replacement, content)
        
        # Kiểm tra xem có thay đổi không
        if new_content == content:
            logger.warning(f"Không tìm thấy hoặc không thể thay đổi cấu hình cho {key}")
            return False
        
        # Ghi lại tập tin
        with open(config_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        
        logger.info(f"Đã cập nhật {key} = {formatted_value} trong config.py")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật config.py: {str(e)}")
        return False