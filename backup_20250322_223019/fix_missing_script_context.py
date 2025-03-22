"""
Sửa lỗi 'Thread missing ScriptRunContext' trong Streamlit

Lỗi này xảy ra khi bạn cố gắng truy cập Streamlit API từ một thread khác với thread chính
của ứng dụng Streamlit. Streamlit API không thread-safe và chỉ nên được gọi từ thread chính.

Để khắc phục lỗi này, bạn cần:
1. Sử dụng thread_safe_logging để ghi log thay vì cập nhật trực tiếp st.session_state
2. Đảm bảo các hàm update_status và update_log KHÔNG sử dụng st.session_state

Sửa lỗi cho hàm update_status:
"""

# ======= CÁCH SỬA 1: Sử dụng thread_safe_logging ===========

def update_status():
    """Cập nhật trạng thái huấn luyện trong thread riêng biệt"""
    from utils.thread_safe_logging import thread_safe_log
    import time
    
    while True:
        try:
            # Thay vì cập nhật trực tiếp st.session_state
            thread_safe_log(f"[STATUS] Đang huấn luyện - {time.strftime('%H:%M:%S')}")
            time.sleep(2)
        except Exception as e:
            thread_safe_log(f"[ERROR] Lỗi khi cập nhật trạng thái: {str(e)}")
            # Ngủ lâu hơn nếu gặp lỗi để tránh quá nhiều lỗi
            time.sleep(10)

# ======= CÁCH SỬA 2: Thay đổi thiết kế để tránh sử dụng thread riêng cho update_status ===========

def fix_design_pattern():
    """
    Thay đổi mẫu thiết kế để tránh cập nhật UI từ thread riêng
    
    Thay vì tạo một thread riêng để cập nhật UI, hãy sử dụng cơ chế ghi log -> đọc log
    và để Streamlit tự động refresh theo chu kỳ để hiển thị thông tin mới nhất
    """
    import streamlit as st
    import time
    import threading
    from utils.thread_safe_logging import thread_safe_log, read_logs_from_file
    
    # 1. Thiết lập hàm huấn luyện không sử dụng st.session_state
    def training_process():
        """Hàm huấn luyện chính, chạy trong thread riêng"""
        thread_safe_log("Bắt đầu quá trình huấn luyện...")
        
        for i in range(10):
            # Thực hiện các bước huấn luyện
            thread_safe_log(f"Đang huấn luyện - Bước {i+1}/10")
            time.sleep(2)  # Mô phỏng quá trình huấn luyện
            
        thread_safe_log("Huấn luyện hoàn tất!")
    
    # 2. Sử dụng giao diện Streamlit để hiển thị và khởi động quá trình huấn luyện
    st.title("Huấn luyện mô hình AI")
    
    # Kiểm tra xem có đang huấn luyện không
    if "training_in_progress" not in st.session_state:
        st.session_state.training_in_progress = False
    
    # Nút bắt đầu huấn luyện
    if st.button("Bắt đầu huấn luyện") and not st.session_state.training_in_progress:
        st.session_state.training_in_progress = True
        
        # Khởi động thread huấn luyện
        training_thread = threading.Thread(target=training_process)
        training_thread.daemon = True
        training_thread.start()
        
        st.success("Đã bắt đầu quá trình huấn luyện!")
    
    # 3. Hiển thị logs - tự động cập nhật khi Streamlit rerun
    if st.session_state.training_in_progress:
        st.subheader("Nhật ký huấn luyện")
        
        # Đọc logs từ file - không sử dụng thread riêng để cập nhật UI
        logs = read_logs_from_file()
        
        # Hiển thị logs
        for log in logs:
            st.text(log)
        
        # Thêm nút tải lại thủ công nếu cần
        if st.button("Tải lại nhật ký"):
            st.experimental_rerun()
        
        # Thiết lập tự động tải lại
        st.empty()
        time.sleep(2)  # Đợi 2 giây
        st.experimental_rerun()  # Tải lại trang để cập nhật logs