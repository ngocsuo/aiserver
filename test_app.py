"""
Ứng dụng Streamlit đơn giản để kiểm tra kết nối
"""
import streamlit as st

st.title("Test Application")
st.write("Nếu bạn thấy thông báo này, Streamlit đang hoạt động bình thường!")

# Hiển thị thông tin cơ bản
st.subheader("Thông tin hệ thống")
st.write(f"Streamlit version: {st.__version__}")

# Tạo một nút interactivity đơn giản
if st.button("Click tôi"):
    st.success("Nút đã được nhấp!")