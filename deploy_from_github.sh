#!/bin/bash
# Script triển khai ETHUSDT Dashboard từ GitHub đến máy chủ

# Màu sắc cho đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Cấu hình
SERVER_IP="" # Nhập IP máy chủ của bạn
SERVER_USER="root"
SSH_PORT="22"
REMOTE_DIR="/root/ethusdt_dashboard"
GITHUB_REPO="https://github.com/ngocsuo/aiserver.git"
BRANCH="main"
SSH_PASSWORD="" # Không lưu mật khẩu trong tập tin

echo -e "${YELLOW}=== TRIỂN KHAI ETHUSDT DASHBOARD TỪ GITHUB ===${NC}"
echo "Thời gian: $(date)"

# Yêu cầu thông tin nếu chưa cấu hình
if [ -z "$SERVER_IP" ]; then
    read -p "Nhập địa chỉ IP máy chủ: " SERVER_IP
fi

if [ -z "$SSH_PORT" ]; then
    read -p "Nhập cổng SSH (mặc định: 22): " SSH_PORT
    SSH_PORT=${SSH_PORT:-22}
fi

if [ -z "$SERVER_USER" ]; then
    read -p "Nhập tên người dùng SSH (mặc định: root): " SERVER_USER
    SERVER_USER=${SERVER_USER:-root}
fi

# Yêu cầu nhập mật khẩu SSH
read -sp "Nhập mật khẩu SSH: " SSH_PASSWORD
echo ""

# Kiểm tra nếu mật khẩu trống
if [ -z "$SSH_PASSWORD" ]; then
    echo -e "${RED}Lỗi: Mật khẩu SSH không được để trống${NC}"
    exit 1
fi

if [ -z "$GITHUB_REPO" ]; then
    read -p "Nhập URL repository GitHub (ví dụ: https://github.com/username/ethusdt-dashboard.git): " GITHUB_REPO
fi

echo -e "${BLUE}Thông tin kết nối:${NC}"
echo "Server: $SERVER_USER@$SERVER_IP:$SSH_PORT"
echo "Thư mục từ xa: $REMOTE_DIR"
echo "GitHub Repository: $GITHUB_REPO"
echo "Nhánh: $BRANCH"

# Kiểm tra tính khả dụng của địa chỉ IP máy chủ
echo -e "${BLUE}Kiểm tra địa chỉ IP...${NC}"
if [[ ! "$SERVER_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}Địa chỉ IP không hợp lệ: $SERVER_IP${NC}"
    echo -e "${YELLOW}Vui lòng nhập địa chỉ IP đúng định dạng (ví dụ: 192.168.1.1)${NC}"
    exit 1
fi

# Kiểm tra cổng SSH
echo -e "${BLUE}Kiểm tra cổng SSH...${NC}"
if ! [[ "$SSH_PORT" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Cổng SSH không hợp lệ: $SSH_PORT${NC}"
    echo -e "${YELLOW}Vui lòng nhập cổng SSH dạng số (ví dụ: 22)${NC}"
    exit 1
fi

# Kiểm tra xem sshpass đã được cài đặt chưa
if ! command -v sshpass &> /dev/null; then
    echo -e "${BLUE}Cài đặt sshpass...${NC}"
    sudo apt-get update && sudo apt-get install -y sshpass
fi

# Định nghĩa lệnh SSH với mật khẩu
SSH_CMD="sshpass -p '$SSH_PASSWORD' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT $SERVER_USER@$SERVER_IP"

# Kiểm tra kết nối SSH
echo -e "${BLUE}Kiểm tra kết nối đến máy chủ...${NC}"
echo -e "${BLUE}Thử kết nối tới: $SERVER_USER@$SERVER_IP:$SSH_PORT${NC}"

# Thêm tùy chọn verbose cho SSH để hiển thị nhiều thông tin hơn
VERBOSE_SSH_CMD="sshpass -p '$SSH_PASSWORD' ssh -v -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT $SERVER_USER@$SERVER_IP"
echo -e "${BLUE}Đang thử kết nối với timeout 10 giây...${NC}"

# Thay thế ping bằng timeout để kiểm tra kết nối
echo -e "${BLUE}Kiểm tra kết nối mạng đến $SERVER_IP:$SSH_PORT...${NC}"
timeout 5 bash -c "</dev/tcp/$SERVER_IP/$SSH_PORT" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Cảnh báo: Không thể kết nối trực tiếp đến $SERVER_IP:$SSH_PORT${NC}"
    echo -e "${BLUE}Tiếp tục với kết nối SSH đầy đủ (có thể bị firewall hoặc cấu hình bảo mật chặn kiểm tra TCP đơn giản)...${NC}"
else
    echo -e "${GREEN}Cổng $SSH_PORT trên $SERVER_IP đang mở và có thể kết nối!${NC}"
fi

if ! eval "$VERBOSE_SSH_CMD \"echo 'Kết nối thành công'\""; then
    echo -e "${RED}Không thể kết nối đến máy chủ. Vui lòng kiểm tra thông tin và quyền truy cập.${NC}"
    echo -e "${YELLOW}Hãy kiểm tra các điểm sau:${NC}"
    echo -e "1. Địa chỉ IP máy chủ ($SERVER_IP) chính xác"
    echo -e "2. Cổng SSH ($SSH_PORT) đang mở"
    echo -e "3. Tên người dùng ($SERVER_USER) và mật khẩu chính xác"
    echo -e "4. Dịch vụ SSH đang chạy trên máy chủ"
    echo -e "5. Firewall không chặn kết nối SSH"
    exit 1
fi
echo -e "${GREEN}Kết nối SSH thành công!${NC}"

# Kiểm tra Git trên máy chủ
echo -e "${BLUE}Kiểm tra Git trên máy chủ...${NC}"
if ! eval "$SSH_CMD \"command -v git\"" > /dev/null 2>&1; then
    echo -e "${BLUE}Cài đặt Git trên máy chủ...${NC}"
    eval "$SSH_CMD \"apt-get update && apt-get install -y git\""
fi

# Sao lưu dữ liệu và cấu hình quan trọng trên máy chủ
echo -e "${BLUE}Sao lưu dữ liệu quan trọng trên máy chủ...${NC}"
BACKUP_DIR="$REMOTE_DIR/backup_$(date +%Y%m%d_%H%M%S)"
eval "$SSH_CMD \"mkdir -p $BACKUP_DIR && \
    if [ -d \\\"$REMOTE_DIR\\\" ]; then \
        cp -r $REMOTE_DIR/data $BACKUP_DIR/ 2>/dev/null || true && \
        cp -r $REMOTE_DIR/saved_models $BACKUP_DIR/ 2>/dev/null || true && \
        cp $REMOTE_DIR/.env $BACKUP_DIR/ 2>/dev/null || true && \
        cp $REMOTE_DIR/binance_time.json $BACKUP_DIR/ 2>/dev/null || true && \
        cp -r $REMOTE_DIR/logs $BACKUP_DIR/ 2>/dev/null || true; \
    fi\""

# Triển khai từ GitHub
echo -e "${BLUE}Triển khai mã nguồn từ GitHub...${NC}"
eval "$SSH_CMD \"if [ -d \\\"$REMOTE_DIR/.git\\\" ]; then \
    cd $REMOTE_DIR && git fetch --all && git reset --hard origin/$BRANCH && git pull; \
else \
    rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR && \
    git clone --depth 1 -b $BRANCH $GITHUB_REPO $REMOTE_DIR; \
fi\""

# Khôi phục dữ liệu và cấu hình quan trọng
echo -e "${BLUE}Khôi phục dữ liệu và cấu hình quan trọng...${NC}"
eval "$SSH_CMD \"if [ -d \\\"$BACKUP_DIR\\\" ]; then \
    mkdir -p $REMOTE_DIR/data $REMOTE_DIR/saved_models $REMOTE_DIR/logs && \
    cp -r $BACKUP_DIR/data/* $REMOTE_DIR/data/ 2>/dev/null || true && \
    cp -r $BACKUP_DIR/saved_models/* $REMOTE_DIR/saved_models/ 2>/dev/null || true && \
    cp $BACKUP_DIR/.env $REMOTE_DIR/ 2>/dev/null || true && \
    cp $BACKUP_DIR/binance_time.json $REMOTE_DIR/ 2>/dev/null || true && \
    cp -r $BACKUP_DIR/logs/* $REMOTE_DIR/logs/ 2>/dev/null || true; \
fi\""

# Chạy thiết lập server
echo -e "${BLUE}Chạy script thiết lập server...${NC}"
eval "$SSH_CMD \"cd $REMOTE_DIR && (chmod +x server_setup.sh && ./server_setup.sh || systemctl restart ethusdt-dashboard)\""

# Kiểm tra trạng thái dịch vụ
echo -e "${BLUE}Kiểm tra trạng thái dịch vụ...${NC}"
eval "$SSH_CMD \"systemctl status ethusdt-dashboard | grep Active || echo 'Dịch vụ không tìm thấy. Đảm bảo server_setup.sh đã được chạy.'\""

# Kiểm tra trạng thái cổng 5000
echo -e "${BLUE}Kiểm tra trạng thái cổng 5000 (dashboard)...${NC}"
eval "$SSH_CMD \"timeout 5 bash -c '</dev/tcp/localhost/5000' && echo -e '${GREEN}Cổng 5000 đang chạy!${NC}' || echo -e '${YELLOW}Cổng 5000 chưa hoạt động, vui lòng chờ thêm...${NC}'\""

# Tạo tập lệnh kiểm tra hoàn chỉnh
echo -e "${BLUE}Tạo tập lệnh kiểm tra hoàn chỉnh...${NC}"
CHECK_SCRIPT="\
#!/bin/bash\n\
echo 'Kiểm tra tệp cấu hình...'\n\
if [ -f '/etc/systemd/system/ethusdt-dashboard.service' ]; then\n\
  echo '✅ Tệp cấu hình systemd đã tồn tại'\n\
else\n\
  echo '❌ Tệp cấu hình systemd không tồn tại'\n\
fi\n\
\n\
echo 'Kiểm tra thư mục dữ liệu...'\n\
if [ -d '$REMOTE_DIR/data' ] && [ -d '$REMOTE_DIR/saved_models' ]; then\n\
  echo '✅ Thư mục dữ liệu đã tồn tại'\n\
else\n\
  echo '❌ Thiếu thư mục dữ liệu'\n\
  mkdir -p $REMOTE_DIR/data $REMOTE_DIR/saved_models $REMOTE_DIR/logs\n\
fi\n\
\n\
echo 'Kiểm tra trạng thái dịch vụ...'\n\
systemctl status ethusdt-dashboard | grep Active\n\
\n\
echo 'Kiểm tra trạng thái cổng 5000...'\n\
if timeout 5 bash -c '</dev/tcp/localhost/5000'; then\n\
  echo '✅ Cổng 5000 đang chạy!'\n\
else\n\
  echo '❌ Cổng 5000 chưa hoạt động - khởi động lại dịch vụ...'\n\
  systemctl restart ethusdt-dashboard\n\
  echo 'Đợi 10 giây để dịch vụ khởi động...'\n\
  sleep 10\n\
  if timeout 5 bash -c '</dev/tcp/localhost/5000'; then\n\
    echo '✅ Cổng 5000 đang chạy sau khi khởi động lại!'\n\
  else\n\
    echo '❌ Cổng 5000 vẫn không hoạt động sau khi khởi động lại.'\n\
    echo 'Lỗi từ log:'\n\
    journalctl -u ethusdt-dashboard -n 20 --no-pager\n\
  fi\n\
fi\n\
"

eval "$SSH_CMD \"echo -e \\\"$CHECK_SCRIPT\\\" > $REMOTE_DIR/check_deployment.sh && chmod +x $REMOTE_DIR/check_deployment.sh\""
eval "$SSH_CMD \"cd $REMOTE_DIR && ./check_deployment.sh\""

echo -e "${GREEN}=== TRIỂN KHAI HOÀN TẤT ===${NC}"
echo "Bạn có thể truy cập dashboard tại: http://$SERVER_IP:5000"
echo "Các bước tiếp theo:"
echo "1. Kiểm tra ứng dụng hoạt động"
echo "2. Kiểm tra logs bằng lệnh: sshpass -p '$SSH_PASSWORD' ssh $SERVER_USER@$SERVER_IP -p $SSH_PORT 'journalctl -fu ethusdt-dashboard'"
echo "3. Chạy kiểm tra lại: sshpass -p '$SSH_PASSWORD' ssh $SERVER_USER@$SERVER_IP -p $SSH_PORT 'cd $REMOTE_DIR && ./check_deployment.sh'"
