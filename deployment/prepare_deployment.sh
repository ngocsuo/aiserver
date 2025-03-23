#!/bin/bash
# Script chuẩn bị triển khai ETHUSDT Dashboard

# Màu sắc đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thư mục hiện tại
CURRENT_DIR=$(pwd)
OUTPUT_DIR="$CURRENT_DIR/deployment"
PACKAGE_NAME="ethusdt_dashboard_$(date +%Y%m%d_%H%M%S).zip"
OUTPUT_PATH="$OUTPUT_DIR/$PACKAGE_NAME"

echo -e "${YELLOW}=== CHUẨN BỊ TRIỂN KHAI ETHUSDT DASHBOARD ===${NC}"
echo "Thời gian: $(date)"

# Danh sách thư mục cần đóng gói
DIRS_TO_INCLUDE=(
    "dashboard"
    "models"
    "prediction"
    "utils"
    "data"
    "logs"
    "automation_scripts"
)

# Danh sách file cần đóng gói
FILES_TO_INCLUDE=(
    "app.py"
    "config.py"
    "requirements_server.txt"
    "server_setup.sh"
    "api.py"
    "README.md"
    "thread_safe_logging.py"
    "continuous_trainer_fix.py"
    "continuous_trainer_fixed.py"
    "enhanced_data_collector.py"
    "enhanced_proxy_config.py"
)

# Đảm bảo thư mục đầu ra tồn tại
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Tạo thư mục tạm để chứa các file
TEMP_DIR="$OUTPUT_DIR/temp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DIR"

# Sao chép các thư mục
echo -e "${BLUE}Sao chép các thư mục...${NC}"
for DIR in "${DIRS_TO_INCLUDE[@]}"; do
    if [ -d "$CURRENT_DIR/$DIR" ]; then
        echo "Sao chép thư mục: $DIR"
        mkdir -p "$TEMP_DIR/$DIR"
        cp -r "$CURRENT_DIR/$DIR"/* "$TEMP_DIR/$DIR"/ 2>/dev/null || true
    else
        echo -e "${YELLOW}Thư mục $DIR không tồn tại, tạo mới.${NC}"
        mkdir -p "$TEMP_DIR/$DIR"
    fi
done

# Sao chép các file
echo -e "${BLUE}Sao chép các file...${NC}"
for FILE in "${FILES_TO_INCLUDE[@]}"; do
    if [ -f "$CURRENT_DIR/$FILE" ]; then
        echo "Sao chép file: $FILE"
        cp "$CURRENT_DIR/$FILE" "$TEMP_DIR/"
    else
        echo -e "${YELLOW}File $FILE không tồn tại, bỏ qua.${NC}"
    fi
done

# Tạo file requirements_server.txt nếu chưa tồn tại
if [ ! -f "$CURRENT_DIR/requirements_server.txt" ]; then
    echo -e "${YELLOW}File requirements_server.txt không tồn tại, tạo mới...${NC}"
    cat > "$TEMP_DIR/requirements_server.txt" << EOF
streamlit>=1.31.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.14.0
python-binance>=1.0.16
scikit-learn>=1.2.0
tensorflow>=2.12.0
requests>=2.28.0
websocket-client>=1.5.0
python-dotenv>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
psutil>=5.9.0
EOF
else
    cp "$CURRENT_DIR/requirements_server.txt" "$TEMP_DIR/"
fi

# Đảm bảo các script có quyền thực thi
find "$TEMP_DIR" -name "*.sh" -exec chmod +x {} \;

# Tạo script khởi động lại
echo -e "${BLUE}Tạo script khởi động lại...${NC}"
cat > "$TEMP_DIR/restart.sh" << EOF
#!/bin/bash
systemctl restart ethusdt-dashboard
echo "Đã khởi động lại dịch vụ ETHUSDT Dashboard!"
EOF
chmod +x "$TEMP_DIR/restart.sh"

# Tạo script trạng thái
echo -e "${BLUE}Tạo script kiểm tra trạng thái...${NC}"
cat > "$TEMP_DIR/status.sh" << EOF
#!/bin/bash
systemctl status ethusdt-dashboard
echo ""
echo "THÔNG TIN BỔ SUNG:"
echo "===================="
echo "Sử dụng CPU:"
top -bn1 | head -20
echo ""
echo "Sử dụng RAM:"
free -m
echo ""
echo "Sử dụng ổ đĩa:"
df -h
EOF
chmod +x "$TEMP_DIR/status.sh"

# Đóng gói
echo -e "${BLUE}Đóng gói thành file zip...${NC}"
cd "$TEMP_DIR"
zip -r "$OUTPUT_PATH" ./* > /dev/null
cd "$CURRENT_DIR"

# Xóa thư mục tạm
rm -rf "$TEMP_DIR"

# Hiển thị kết quả
if [ -f "$OUTPUT_PATH" ]; then
    FILESIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
    echo -e "${GREEN}Đóng gói thành công!${NC}"
    echo "File: $OUTPUT_PATH"
    echo "Kích thước: $FILESIZE"
    echo ""
    echo -e "${BLUE}Hướng dẫn sử dụng:${NC}"
    echo "1. Sao chép file này lên server:"
    echo "   scp $OUTPUT_PATH root@45.76.196.13:/root/"
    echo ""
    echo "2. Đăng nhập vào server:"
    echo "   ssh root@45.76.196.13"
    echo ""
    echo "3. Giải nén file trên server:"
    echo "   mkdir -p /root/ethusdt_dashboard"
    echo "   unzip /root/$PACKAGE_NAME -d /root/ethusdt_dashboard"
    echo ""
    echo "4. Chạy script cài đặt:"
    echo "   cd /root/ethusdt_dashboard"
    echo "   chmod +x server_setup.sh"
    echo "   ./server_setup.sh"
else
    echo -e "${RED}Đóng gói thất bại!${NC}"
fi

echo -e "${YELLOW}=== KẾT THÚC CHUẨN BỊ TRIỂN KHAI ===${NC}"