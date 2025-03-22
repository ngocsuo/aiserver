#!/bin/bash
# Script đồng bộ code từ Replit sang server

# SSH password sẽ được nhập thủ công khi script chạy
SERVER="45.76.196.13"
USER="root"
REMOTE_DIR="/root/ethusdt_dashboard"
LOCAL_DIR="."

echo "Đồng bộ code từ Replit sang server $SERVER..."

# Đồng bộ các file Python và thư mục
rsync -avz -e "ssh -o StrictHostKeyChecking=no" \
  --include="*.py" \
  --include="*.html" \
  --include="*.css" \
  --include="*.js" \
  --include="*.json" \
  --include="*.md" \
  --include="*.txt" \
  --include="*.sh" \
  --include="utils/" \
  --include="utils/**" \
  --include="models/" \
  --include="models/**" \
  --include="prediction/" \
  --include="prediction/**" \
  --include="dashboard/" \
  --include="dashboard/**" \
  --include="deployment/" \
  --include="deployment/**" \
  --exclude=".git/" \
  --exclude="venv/" \
  --exclude="__pycache__/" \
  --exclude="*.pyc" \
  --exclude=".streamlit/" \
  --exclude="logs/" \
  --exclude="data/" \
  --exclude="saved_models/" \
  $LOCAL_DIR/* $USER@$SERVER:$REMOTE_DIR/

# Chuyển API keys từ Replit sang server
echo "Cấu hình API keys trên server..."
ssh -o StrictHostKeyChecking=no $USER@$SERVER "
cat > $REMOTE_DIR/setup_api_keys.sh << 'EOF'
#!/bin/bash
# Script thiết lập API keys

# Cập nhật .bashrc
grep -q 'BINANCE_API_KEY' /root/.bashrc || cat >> /root/.bashrc << 'ENVVARS'

# Binance API Credentials
export BINANCE_API_KEY=\"$BINANCE_API_KEY\"
export BINANCE_API_SECRET=\"$BINANCE_API_SECRET\"
ENVVARS

# Cập nhật biến môi trường tạm thời
export BINANCE_API_KEY=\"$BINANCE_API_KEY\"
export BINANCE_API_SECRET=\"$BINANCE_API_SECRET\"

# Cập nhật script restart.sh
if grep -q 'BINANCE_API_KEY' $REMOTE_DIR/restart.sh; then
  # API key đã tồn tại trong restart.sh, cập nhật chúng
  sed -i \"s|export BINANCE_API_KEY=.*|export BINANCE_API_KEY=\\\"$BINANCE_API_KEY\\\"|g\" $REMOTE_DIR/restart.sh
  sed -i \"s|export BINANCE_API_SECRET=.*|export BINANCE_API_SECRET=\\\"$BINANCE_API_SECRET\\\"|g\" $REMOTE_DIR/restart.sh
else
  # Thêm API key vào script restart.sh
  sed -i '/# Khởi động ứng dụng/i # Cài đặt biến môi trường\nexport BINANCE_API_KEY=\\\"$BINANCE_API_KEY\\\"\nexport BINANCE_API_SECRET=\\\"$BINANCE_API_SECRET\\\"\n' $REMOTE_DIR/restart.sh
fi

echo \"API keys đã được thiết lập thành công!\"
EOF

chmod +x $REMOTE_DIR/setup_api_keys.sh
export BINANCE_API_KEY=\"$BINANCE_API_KEY\"
export BINANCE_API_SECRET=\"$BINANCE_API_SECRET\"
$REMOTE_DIR/setup_api_keys.sh
"

# Khởi động lại ứng dụng trên server
echo "Khởi động lại ứng dụng..."
ssh -o StrictHostKeyChecking=no $USER@$SERVER "cd $REMOTE_DIR && ./restart.sh"

echo "Đồng bộ hoàn tất và ứng dụng đã được khởi động lại!"