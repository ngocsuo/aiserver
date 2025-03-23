#!/bin/bash
# Script thiết lập đồng bộ GitHub cho ETHUSDT Dashboard

# Màu sắc cho đầu ra
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== THIẾT LẬP ĐỒNG BỘ GITHUB CHO ETHUSDT DASHBOARD ===${NC}"
echo "Thời gian: $(date)"

# Kiểm tra Git đã được cài đặt chưa
if ! command -v git &> /dev/null; then
    echo -e "${RED}Git chưa được cài đặt. Vui lòng cài đặt Git trước khi tiếp tục.${NC}"
    exit 1
fi

# Kiểm tra repository đã được khởi tạo chưa
if [ -d ".git" ]; then
    echo -e "${BLUE}Repository Git đã tồn tại.${NC}"
else
    echo -e "${BLUE}Khởi tạo repository Git...${NC}"
    git init
fi

# Kiểm tra và thêm remote GitHub
read -p "Nhập URL repository GitHub của bạn (ví dụ: https://github.com/username/ethusdt-dashboard.git): " GITHUB_REPO

if [ -z "$GITHUB_REPO" ]; then
    echo -e "${RED}URL repository không được để trống.${NC}"
    exit 1
fi

# Kiểm tra remote origin đã tồn tại chưa
if git remote | grep -q "origin"; then
    echo -e "${BLUE}Remote 'origin' đã tồn tại. Cập nhật URL...${NC}"
    git remote set-url origin $GITHUB_REPO
else
    echo -e "${BLUE}Thêm remote 'origin'...${NC}"
    git remote add origin $GITHUB_REPO
fi

# Thêm tất cả file vào staging area (trừ các file đã loại trừ trong .gitignore)
echo -e "${BLUE}Thêm các file vào staging area...${NC}"
git add .

# Tạo commit ban đầu
echo -e "${BLUE}Tạo commit đầu tiên...${NC}"
git commit -m "Initial commit: ETHUSDT Dashboard setup"

# Tạo và chuyển đến nhánh main nếu cần
if ! git rev-parse --verify main &> /dev/null; then
    echo -e "${BLUE}Tạo và chuyển đến nhánh 'main'...${NC}"
    git branch -M main
fi

# Push lên GitHub
echo -e "${BLUE}Đẩy code lên GitHub. Bạn có thể được yêu cầu nhập thông tin đăng nhập GitHub...${NC}"
git push -u origin main

echo -e "${GREEN}=== THIẾT LẬP HOÀN TẤT ===${NC}"
echo "Repository của bạn đã được khởi tạo và đẩy lên GitHub."