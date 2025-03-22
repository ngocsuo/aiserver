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

# Khởi động lại ứng dụng trên server
ssh -o StrictHostKeyChecking=no $USER@$SERVER "cd $REMOTE_DIR && ./restart.sh"

echo "Đồng bộ hoàn tất và ứng dụng đã được khởi động lại!"