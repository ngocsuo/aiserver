"""
Sửa lỗi pandas style.map trong phiên bản pandas mới
"""

# Tìm dòng code này trong app.py (khoảng dòng 2194):
# styled_df = recent_preds.style.map(style_trend, subset=['trend'])

# Thay thế bằng code tương thích với nhiều phiên bản pandas:
def fix_pandas_style():
    # Cách 1: Sử dụng applymap (phiên bản pandas cũ)
    try:
        styled_df = recent_preds.style.applymap(style_trend, subset=['trend'])
    except AttributeError:
        # Cách 2: Sử dụng map (phiên bản pandas mới)
        try:
            styled_df = recent_preds.style.map(style_trend, subset=['trend'])
        except AttributeError:
            # Cách 3: Nếu cả hai đều không hoạt động, sử dụng phương thức khác
            def highlight_trend(s):
                return ['background-color: green; color: white' if x == 'LONG' 
                        else 'background-color: red; color: white' if x == 'SHORT'
                        else 'background-color: gray; color: white' for x in s]
            
            styled_df = recent_preds.style.apply(highlight_trend, subset=['trend'])
    
    return styled_df

# Để sửa lỗi, thay thế dòng gây lỗi bằng:
# try:
#     # Thử cách 1: sử dụng style.applymap (pandas cũ)
#     styled_df = recent_preds.style.applymap(style_trend, subset=['trend'])
# except AttributeError:
#     # Thử cách 2: sử dụng style.apply với hàm khác
#     def highlight_trend(s):
#         return ['background-color: green; color: white' if x == 'LONG' 
#                 else 'background-color: red; color: white' if x == 'SHORT'
#                 else 'background-color: gray; color: white' for x in s]
#     
#     styled_df = recent_preds.style.apply(highlight_trend, subset=['trend'])