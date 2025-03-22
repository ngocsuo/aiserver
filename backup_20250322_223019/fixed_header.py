"""
Main Streamlit application for ETHUSDT prediction dashboard.
Enhanced with improved UI, advanced technical analysis, and multi-source data integration.
"""
# Code khi render giao diện, thay thế hàm create_header bằng markdown trực tiếp

def render_main_interface():
    # Load custom CSS
    load_custom_css()
    
    # Thay thế hàm create_header bằng markdown trực tiếp
    st.markdown("# AI TRADING ORACLE")
    st.markdown("### Hệ Thống Dự Đoán ETHUSDT Tự Động")
    
    # Sidebar navigation
    section = st.sidebar.selectbox("Chuyển hướng", ["Bảng điều khiển", "Kiểm soát hệ thống", "Giao dịch tự động", "Huấn luyện & API", "Về chúng tôi"])
    
    # Handle navigation
    if section == "Bảng điều khiển":
        # Main dashboard section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Dự đoán và phân tích ETHUSDT")
            
            # Display the latest prediction if available
            if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
                latest_prediction = st.session_state.predictions[-1]
                display_current_prediction(latest_prediction)
            else:
                st.warning("Chưa có dữ liệu dự đoán. Hãy tạo dự đoán mới.")
            
            # Add buttons for prediction and reload data
            pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 2])
            with pred_col1:
                if st.button("🧠 Tạo dự đoán", use_container_width=True):
                    make_prediction()
            
            with pred_col2:
                if st.button("🔄 Tải lại dữ liệu", use_container_width=True):
                    fetch_realtime_data()
                    st.rerun()
            
            with pred_col3:
                # Display data source information
                if hasattr(st.session_state, 'data_source'):
                    if hasattr(st.session_state, 'api_status') and not st.session_state.api_status.get('connected', False):
                        st.markdown(f"📊 Nguồn dữ liệu: <span style='color: orange;'>{st.session_state.data_source}</span> - <span style='color: red;'>{st.session_state.api_status.get('message', 'Kết nối thất bại')}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"📊 Nguồn dữ liệu: <span style='color: {st.session_state.data_source_color};'>{st.session_state.data_source}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("📊 Nguồn dữ liệu: Chưa khởi tạo")
    
    # Initialize if not already done
    if not st.session_state.initialized and not st.session_state.auto_initialize_triggered:
        st.session_state.auto_initialize_triggered = True
        initialize_system()