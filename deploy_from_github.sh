#!/bin/bash
# Script triển khai ETHUSDT Dashboard từ Github
# Cách sử dụng: ./deploy_from_github.sh [GitHub_URL] [user@server_ip] [password]

# Thiết lập màu sắc
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Cấu hình mặc định
REPO_URL=${1:-"https://github.com/yourusername/ethusdt-dashboard.git"}
SERVER_ADDRESS=${2:-""}
SSH_PASSWORD=${3:-""}
DEPLOY_DIR="/opt/ethusdt-dashboard"
PORT=5000

# Kiểm tra tham số
if [ -z "$SERVER_ADDRESS" ]; then
    echo -e "${RED}Lỗi: Thiếu địa chỉ server. Sử dụng: $0 [repo_url] user@server_ip [password]${NC}"
    exit 1
fi

if [ -z "$SSH_PASSWORD" ]; then
    echo -e "${YELLOW}Chú ý: Không có mật khẩu SSH được cung cấp. Sẽ sử dụng SSH key hoặc hỏi mật khẩu.${NC}"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     ETHUSDT Dashboard Deployment      ${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Repository: ${REPO_URL}"
echo -e "Server: ${SERVER_ADDRESS}"
echo -e "Deploy directory: ${DEPLOY_DIR}"

# Tạo SSH command tự động nhập mật khẩu
SSH_CMD="ssh"
SCP_CMD="scp"
if [ ! -z "$SSH_PASSWORD" ]; then
    # Kiểm tra sshpass đã được cài đặt chưa
    if ! command -v sshpass &> /dev/null; then
        echo -e "${YELLOW}Warning: sshpass không được cài đặt. Sử dụng phương thức SSH thông thường.${NC}"
    else
        SSH_CMD="sshpass -p ${SSH_PASSWORD} ssh"
        SCP_CMD="sshpass -p ${SSH_PASSWORD} scp"
    fi
fi

# 1. Tạo file cài đặt prepare_server_env.sh
echo -e "\n${GREEN}[1/7] Tạo file prepare_server_env.sh...${NC}"

cat > prepare_server_env.sh << 'EOF'
#!/bin/bash
# Script cài đặt môi trường cho ETHUSDT Dashboard trên server mới
# Sử dụng: sudo ./prepare_server_env.sh

# Kiểm tra quyền root
if [ "$(id -u)" != "0" ]; then
   echo "Script này phải được chạy với quyền root" 1>&2
   exit 1
fi

# Thông tin cấu hình
APP_DIR="/opt/ethusdt-dashboard"
APP_USER="ethusdt"
APP_PORT=5000
LOG_DIR="$APP_DIR/logs"
DATA_DIR="$APP_DIR/data"
MODELS_DIR="$APP_DIR/saved_models"

# Thiết lập màu
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Cài đặt ETHUSDT Dashboard ===${NC}"
echo -e "${YELLOW}Bắt đầu cài đặt môi trường...${NC}"

# 1. Cập nhật hệ thống
echo -e "${GREEN}[1/9] Đang cập nhật hệ thống...${NC}"
apt update && apt upgrade -y

# 2. Cài đặt các gói phụ thuộc
echo -e "${GREEN}[2/9] Đang cài đặt các gói phụ thuộc...${NC}"
apt install -y python3 python3-pip python3-venv git build-essential libssl-dev libffi-dev python3-dev curl

# 3. Tạo thư mục ứng dụng
echo -e "${GREEN}[3/9] Đang tạo thư mục ứng dụng...${NC}"
mkdir -p $APP_DIR
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR
mkdir -p $MODELS_DIR

# 4. Tạo người dùng hệ thống
echo -e "${GREEN}[4/9] Đang tạo người dùng hệ thống...${NC}"
id -u $APP_USER &>/dev/null || useradd -r -d $APP_DIR -s /bin/bash $APP_USER

# 5. Thiết lập môi trường Python
echo -e "${GREEN}[5/9] Đang thiết lập môi trường Python...${NC}"
cd $APP_DIR
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 6. Tạo service systemd
echo -e "${GREEN}[6/9] Đang tạo service systemd...${NC}"
cat > /etc/systemd/system/ethusdt-dashboard.service << _EOF
[Unit]
Description=ETHUSDT Dashboard
After=network.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/streamlit run app.py --server.port=$APP_PORT --server.address=0.0.0.0 --server.headless=true
Restart=always
StandardOutput=append:$LOG_DIR/service.log
StandardError=append:$LOG_DIR/service_error.log
Environment="PYTHONPATH=$APP_DIR"
Environment="PATH=$APP_DIR/venv/bin:$PATH"
Environment="LOG_DIR=$LOG_DIR"
Environment="DATA_DIR=$DATA_DIR"
Environment="MODELS_DIR=$MODELS_DIR"

[Install]
WantedBy=multi-user.target
_EOF

# 7. Cấu hình thư mục Streamlit
echo -e "${GREEN}[7/9] Đang cấu hình Streamlit...${NC}"
mkdir -p $APP_DIR/.streamlit
cat > $APP_DIR/.streamlit/config.toml << _EOF
[server]
headless = true
address = "0.0.0.0"
port = $APP_PORT
maxUploadSize = 5
maxMessageSize = 500

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans-serif"
_EOF

# 8. Thiết lập quyền
echo -e "${GREEN}[8/9] Đang thiết lập quyền...${NC}"
chown -R $APP_USER:$APP_USER $APP_DIR
chmod -R 755 $APP_DIR
chmod +x $APP_DIR/venv/bin/streamlit

# 9. Tạo tệp đánh dấu hoàn thành
echo -e "${GREEN}[9/9] Hoàn thành cài đặt môi trường!${NC}"
touch $APP_DIR/.env_setup_complete

# Thông báo hoàn thành
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}Cài đặt môi trường đã hoàn tất!${NC}"
echo -e "${YELLOW}Tiếp theo:${NC}"
echo -e "1. Sao chép mã nguồn vào $APP_DIR"
echo -e "2. Cài đặt các gói phụ thuộc: cd $APP_DIR && source venv/bin/activate && pip install -r requirements_server.txt"
echo -e "3. Cấu hình API key Binance: chỉnh sửa file $APP_DIR/.env"
echo -e "4. Khởi động dịch vụ: sudo systemctl enable ethusdt-dashboard && sudo systemctl start ethusdt-dashboard"
echo -e "5. Kiểm tra trạng thái: sudo systemctl status ethusdt-dashboard"
echo -e "${GREEN}============================================${NC}"

exit 0
EOF

chmod +x prepare_server_env.sh

# 2. Tạo file kiểm tra sức khỏe server_health_check.sh
echo -e "\n${GREEN}[2/7] Tạo file server_health_check.sh...${NC}"

cat > server_health_check.sh << 'EOF'
#!/bin/bash
# Script kiểm tra sức khỏe hệ thống ETHUSDT Dashboard trên server
# Sử dụng: ./server_health_check.sh [--detailed]

# Thiết lập màu
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thông tin cấu hình
APP_DIR="/opt/ethusdt-dashboard"
APP_USER="ethusdt"
APP_PORT=5000
LOG_DIR="$APP_DIR/logs"
DATA_DIR="$APP_DIR/data"
MODELS_DIR="$APP_DIR/saved_models"
SERVICE_NAME="ethusdt-dashboard"

# Định dạng tiêu đề và kết quả
function print_header() {
    echo -e "\n${BLUE}===== $1 =====${NC}"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Kiểm tra chi tiết hay không
DETAILED=false
if [ "$1" == "--detailed" ]; then
    DETAILED=true
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     ETHUSDT Dashboard Health Check     ${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Thời gian: $(date)"
echo -e "Server: $(hostname)"

# 1. Kiểm tra hệ thống
print_header "Kiểm tra tài nguyên hệ thống"

# Kiểm tra CPU
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
echo -e "CPU Usage: ${CPU_USAGE}%"
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    print_error "CPU đang quá tải (${CPU_USAGE}%)"
elif (( $(echo "$CPU_USAGE > 50" | bc -l) )); then
    print_warning "CPU đang cao (${CPU_USAGE}%)"
else
    print_success "CPU bình thường (${CPU_USAGE}%)"
fi

# Kiểm tra RAM
MEM_TOTAL=$(free -m | grep Mem | awk '{print $2}')
MEM_USED=$(free -m | grep Mem | awk '{print $3}')
MEM_PERCENT=$(echo "scale=2; $MEM_USED*100/$MEM_TOTAL" | bc)
echo -e "Memory Usage: ${MEM_USED}MB / ${MEM_TOTAL}MB (${MEM_PERCENT}%)"
if (( $(echo "$MEM_PERCENT > 90" | bc -l) )); then
    print_error "RAM đang quá tải (${MEM_PERCENT}%)"
elif (( $(echo "$MEM_PERCENT > 70" | bc -l) )); then
    print_warning "RAM đang cao (${MEM_PERCENT}%)"
else
    print_success "RAM bình thường (${MEM_PERCENT}%)"
fi

# Kiểm tra disk
DISK_TOTAL=$(df -h / | awk 'NR==2 {print $2}')
DISK_USED=$(df -h / | awk 'NR==2 {print $3}')
DISK_PERCENT=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
echo -e "Disk Usage: ${DISK_USED} / ${DISK_TOTAL} (${DISK_PERCENT}%)"
if (( DISK_PERCENT > 90 )); then
    print_error "Disk gần hết (${DISK_PERCENT}%)"
elif (( DISK_PERCENT > 75 )); then
    print_warning "Disk đang cao (${DISK_PERCENT}%)"
else
    print_success "Disk bình thường (${DISK_PERCENT}%)"
fi

# 2. Kiểm tra dịch vụ
print_header "Kiểm tra dịch vụ"

# Kiểm tra dịch vụ systemd
if systemctl is-active --quiet $SERVICE_NAME; then
    print_success "Dịch vụ $SERVICE_NAME đang chạy"
else
    print_error "Dịch vụ $SERVICE_NAME không chạy"
    echo -e "Khởi động dịch vụ: sudo systemctl start $SERVICE_NAME"
fi

# Kiểm tra port
if netstat -tuln | grep -q ":$APP_PORT"; then
    print_success "Port $APP_PORT đang mở và có dịch vụ đang lắng nghe"
else
    print_error "Port $APP_PORT không có dịch vụ nào đang lắng nghe"
    echo -e "Kiểm tra trạng thái dịch vụ: sudo systemctl status $SERVICE_NAME"
fi

# 3. Kiểm tra kết nối mạng
print_header "Kiểm tra kết nối mạng"

# Kiểm tra kết nối internet
if ping -c 1 google.com &> /dev/null; then
    print_success "Kết nối internet hoạt động"
else
    print_error "Không thể kết nối internet"
fi

# Kiểm tra kết nối Binance API
if curl -s "https://api.binance.com/api/v3/ping" &> /dev/null; then
    print_success "Kết nối Binance API hoạt động"
else
    print_error "Không thể kết nối Binance API"
    echo -e "Kiểm tra proxy hoặc cấu hình mạng"
fi

# 4. Kiểm tra dữ liệu và logs
print_header "Kiểm tra dữ liệu và logs"

# Kiểm tra thư mục dữ liệu
if [ -d "$DATA_DIR" ]; then
    DATA_FILES=$(find $DATA_DIR -type f | wc -l)
    print_success "Thư mục dữ liệu tồn tại ($DATA_FILES files)"
else
    print_error "Thư mục dữ liệu không tồn tại"
fi

# Kiểm tra thư mục mô hình
if [ -d "$MODELS_DIR" ]; then
    MODEL_FILES=$(find $MODELS_DIR -type f | wc -l)
    print_success "Thư mục mô hình tồn tại ($MODEL_FILES files)"
    
    # Kiểm tra thời gian cập nhật mô hình
    if [ $MODEL_FILES -gt 0 ]; then
        LATEST_MODEL=$(find $MODELS_DIR -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        LATEST_MODEL_TIME=$(stat -c %y "$LATEST_MODEL" | cut -d'.' -f1)
        echo -e "Mô hình mới nhất: $(basename "$LATEST_MODEL") (${LATEST_MODEL_TIME})"
    fi
else
    print_error "Thư mục mô hình không tồn tại"
fi

# Kiểm tra logs
if [ -d "$LOG_DIR" ]; then
    LOG_FILES=$(find $LOG_DIR -type f | wc -l)
    print_success "Thư mục logs tồn tại ($LOG_FILES files)"
    
    # Kiểm tra log lỗi gần đây
    if [ -f "$LOG_DIR/service_error.log" ]; then
        ERROR_COUNT=$(grep -i "error\|exception\|fail" $LOG_DIR/service_error.log | wc -l)
        if [ $ERROR_COUNT -gt 0 ]; then
            print_warning "Tìm thấy $ERROR_COUNT lỗi trong log"
            if [ "$DETAILED" = true ]; then
                echo -e "\nLỗi gần đây:"
                grep -i "error\|exception\|fail" $LOG_DIR/service_error.log | tail -5
            fi
        else
            print_success "Không tìm thấy lỗi trong log"
        fi
    fi
else
    print_error "Thư mục logs không tồn tại"
fi

# 5. Kiểm tra ứng dụng web
print_header "Kiểm tra ứng dụng web"

# Kiểm tra response của ứng dụng web
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$APP_PORT)
if [ "$HTTP_CODE" = "200" ]; then
    print_success "Ứng dụng web trả về HTTP 200 OK"
else
    print_error "Ứng dụng web trả về HTTP $HTTP_CODE"
    echo -e "Kiểm tra log lỗi: sudo tail -f $LOG_DIR/service_error.log"
fi

# Xuất thông tin chi tiết nếu có yêu cầu
if [ "$DETAILED" = true ]; then
    print_header "Thông tin chi tiết"
    
    echo -e "\n${YELLOW}Tiến trình Streamlit:${NC}"
    ps -ef | grep streamlit | grep -v grep
    
    echo -e "\n${YELLOW}Logs gần đây:${NC}"
    if [ -f "$LOG_DIR/service.log" ]; then
        tail -n 10 $LOG_DIR/service.log
    else
        echo "Không tìm thấy file log"
    fi
    
    echo -e "\n${YELLOW}Thông tin phiên bản:${NC}"
    if [ -f "$APP_DIR/venv/bin/python" ]; then
        echo -e "Python: $($APP_DIR/venv/bin/python --version 2>&1)"
        echo -e "Streamlit: $($APP_DIR/venv/bin/pip show streamlit | grep Version | awk '{print $2}')"
        echo -e "TensorFlow: $($APP_DIR/venv/bin/pip show tensorflow | grep Version | awk '{print $2}')"
    fi
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}     Kết thúc kiểm tra sức khỏe         ${NC}"
echo -e "${BLUE}========================================${NC}"

# Hướng dẫn cách khắc phục
print_header "Hướng dẫn khắc phục"
echo -e "- Khởi động lại dịch vụ: ${YELLOW}sudo systemctl restart $SERVICE_NAME${NC}"
echo -e "- Xem log ứng dụng: ${YELLOW}sudo tail -f $LOG_DIR/service.log${NC}"
echo -e "- Xem log lỗi: ${YELLOW}sudo tail -f $LOG_DIR/service_error.log${NC}"
echo -e "- Kiểm tra trạng thái dịch vụ: ${YELLOW}sudo systemctl status $SERVICE_NAME${NC}"
echo -e "- Kiểm tra chi tiết sức khỏe: ${YELLOW}$0 --detailed${NC}"

exit 0
EOF

chmod +x server_health_check.sh

# 3. Tạo file requirements_server.txt
echo -e "\n${GREEN}[3/7] Tạo file requirements_server.txt...${NC}"

cat > requirements_server.txt << 'EOF'
binance==1.0.19
python-binance==1.0.19
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.2
plotly==5.18.0
scikit-learn==1.3.2
tensorflow==2.15.0
keras==2.15.0
requests==2.31.0
python-dotenv==1.0.0
pytz==2023.3
matplotlib==3.8.2
flask==3.0.0
twilio==8.10.0
pysocks==1.7.1
python-socks==2.3.0
psutil==5.9.6
EOF

# 4. Tạo script triển khai tổng thể
echo -e "\n${GREEN}[4/7] Tạo script triển khai...${NC}"

cat > deploy.sh << 'EOF'
#!/bin/bash
# Script triển khai ETHUSDT Dashboard

# Thiết lập màu
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thông tin cấu hình
DEPLOY_DIR="/opt/ethusdt-dashboard"
BINANCE_API_KEY=${BINANCE_API_KEY:-""}
BINANCE_API_SECRET=${BINANCE_API_SECRET:-""}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     ETHUSDT Dashboard Deployment      ${NC}"
echo -e "${BLUE}========================================${NC}"

# 1. Cài đặt môi trường
echo -e "\n${GREEN}[1/5] Cài đặt môi trường server...${NC}"
if [ -f ./prepare_server_env.sh ]; then
    sudo ./prepare_server_env.sh
else
    echo -e "${RED}Lỗi: Không tìm thấy file prepare_server_env.sh${NC}"
    exit 1
fi

# 2. Cài đặt dependencies
echo -e "\n${GREEN}[2/5] Cài đặt các gói phụ thuộc...${NC}"
cd $DEPLOY_DIR
source venv/bin/activate
if [ -f requirements_server.txt ]; then
    pip install -r requirements_server.txt
else
    echo -e "${YELLOW}Cảnh báo: Không tìm thấy file requirements_server.txt, sử dụng requirements.txt...${NC}"
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    else
        echo -e "${RED}Lỗi: Không tìm thấy file requirements.txt${NC}"
        exit 1
    fi
fi

# 3. Cấu hình Binance API
echo -e "\n${GREEN}[3/5] Cấu hình Binance API...${NC}"
if [ ! -z "$BINANCE_API_KEY" ] && [ ! -z "$BINANCE_API_SECRET" ]; then
    echo -e "BINANCE_API_KEY=$BINANCE_API_KEY" > .env
    echo -e "BINANCE_API_SECRET=$BINANCE_API_SECRET" >> .env
    echo -e "${GREEN}API keys đã được cấu hình.${NC}"
else
    echo -e "${YELLOW}Chú ý: API keys chưa được cấu hình, bạn sẽ cần cấu hình thủ công.${NC}"
    echo -e "Chỉnh sửa file: $DEPLOY_DIR/.env"
    touch .env
fi

# 4. Thiết lập quyền
echo -e "\n${GREEN}[4/5] Thiết lập quyền...${NC}"
sudo chown -R ethusdt:ethusdt $DEPLOY_DIR
sudo chmod -R 755 $DEPLOY_DIR
sudo chmod 600 $DEPLOY_DIR/.env
sudo systemctl enable ethusdt-dashboard

# 5. Khởi động dịch vụ
echo -e "\n${GREEN}[5/5] Khởi động dịch vụ...${NC}"
sudo systemctl restart ethusdt-dashboard
sleep 3
sudo systemctl status ethusdt-dashboard

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}ETHUSDT Dashboard đã được triển khai!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Truy cập dashboard: http://$(hostname -I | awk '{print $1}'):5000"
echo -e "Kiểm tra sức khỏe hệ thống: ${YELLOW}./server_health_check.sh${NC}"
echo -e "Xem logs: ${YELLOW}sudo journalctl -fu ethusdt-dashboard${NC}"

exit 0
EOF

chmod +x deploy.sh

# 5. Truyền file lên server
echo -e "\n${GREEN}[5/7] Truyền file lên server...${NC}"
$SCP_CMD prepare_server_env.sh server_health_check.sh requirements_server.txt deploy.sh $SERVER_ADDRESS:/tmp/

# 6. Kiểm tra thư mục Git trên server
echo -e "\n${GREEN}[6/7] Chuẩn bị triển khai trên server...${NC}"
$SSH_CMD $SERVER_ADDRESS << EOF
    echo -e "${BLUE}Đã kết nối đến server ${SERVER_ADDRESS}${NC}"
    sudo mkdir -p $DEPLOY_DIR
    cd $DEPLOY_DIR
    
    # Di chuyển file cài đặt
    sudo cp /tmp/prepare_server_env.sh /tmp/server_health_check.sh /tmp/requirements_server.txt /tmp/deploy.sh .
    sudo chmod +x prepare_server_env.sh server_health_check.sh deploy.sh
    
    # Clone repository từ GitHub
    echo -e "${GREEN}Cloning repository từ ${REPO_URL}...${NC}"
    if [ -d ".git" ]; then
        echo -e "${YELLOW}Git repository đã tồn tại. Đang cập nhật...${NC}"
        sudo git pull
    else
        sudo git clone $REPO_URL .
    fi
    
    # Kiểm tra port 5000
    echo -e "${GREEN}Kiểm tra port 5000...${NC}"
    if netstat -tuln | grep ":5000 "; then
        echo -e "${YELLOW}Port 5000 đã được sử dụng. Kiểm tra dịch vụ:${NC}"
        netstat -tuln | grep ":5000 "
        ps -ef | grep streamlit
    else
        echo -e "${GREEN}Port 5000 khả dụng.${NC}"
    fi
EOF

# 7. Bắt đầu quá trình triển khai
echo -e "\n${GREEN}[7/7] Tiến hành triển khai...${NC}"
$SSH_CMD $SERVER_ADDRESS << EOF
    cd $DEPLOY_DIR
    echo -e "${BLUE}Bắt đầu triển khai ETHUSDT Dashboard...${NC}"
    sudo ./deploy.sh

    # Kiểm tra kết quả
    echo -e "\n${GREEN}Kiểm tra kết quả triển khai:${NC}"
    if systemctl is-active --quiet ethusdt-dashboard; then
        echo -e "${GREEN}✅ Dịch vụ đã được triển khai thành công và đang chạy.${NC}"
        
        # Kiểm tra port 5000
        if netstat -tuln | grep -q ":5000 "; then
            echo -e "${GREEN}✅ Port 5000 đang lắng nghe.${NC}"
            echo -e "${GREEN}✅ Dashboard có thể truy cập tại: http://$(hostname -I | awk '{print $1}'):5000${NC}"
        else
            echo -e "${RED}❌ Port 5000 không được mở.${NC}"
        fi
    else
        echo -e "${RED}❌ Dịch vụ không chạy. Kiểm tra logs:${NC}"
        sudo journalctl -u ethusdt-dashboard -n 20
    fi
EOF

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Quá trình triển khai đã hoàn tất!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Để kiểm tra trạng thái hệ thống, hãy sử dụng lệnh:"
echo -e "${YELLOW}$SSH_CMD $SERVER_ADDRESS 'cd $DEPLOY_DIR && sudo ./server_health_check.sh'${NC}"
echo -e "\nĐể truy cập dashboard, mở trình duyệt và truy cập URL:"
IP_ADDRESS=$($SSH_CMD $SERVER_ADDRESS "hostname -I | awk '{print \$1}'")
echo -e "${YELLOW}http://$IP_ADDRESS:5000${NC}"

exit 0