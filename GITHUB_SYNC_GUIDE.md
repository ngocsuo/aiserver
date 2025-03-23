# Hướng dẫn đồng bộ ETHUSDT Dashboard qua GitHub

Tài liệu này hướng dẫn cách thiết lập và sử dụng GitHub để đồng bộ và triển khai dự án ETHUSDT Dashboard một cách hiệu quả và dễ dàng.

## Mục lục

1. [Lợi ích của đồng bộ qua GitHub](#lợi-ích-của-đồng-bộ-qua-github)
2. [Thiết lập ban đầu](#thiết-lập-ban-đầu)
3. [Quy trình làm việc hàng ngày](#quy-trình-làm-việc-hàng-ngày)
4. [Triển khai lên máy chủ](#triển-khai-lên-máy-chủ)
5. [Thiết lập triển khai tự động](#thiết-lập-triển-khai-tự-động)
6. [Xử lý sự cố thường gặp](#xử-lý-sự-cố-thường-gặp)

## Lợi ích của đồng bộ qua GitHub

- **Quản lý phiên bản**: Theo dõi thay đổi, dễ dàng quay lại phiên bản trước nếu có lỗi
- **Đồng bộ đa thiết bị**: Làm việc từ nhiều máy tính khác nhau mà không lo mất code
- **Triển khai tự động**: Tự động triển khai lên máy chủ khi có thay đổi
- **Cộng tác**: Dễ dàng làm việc cùng nhiều người (nếu cần)
- **Sao lưu**: Dữ liệu code luôn được sao lưu trên GitHub

## Thiết lập ban đầu

### 1. Tạo repository trên GitHub

1. Đăng nhập vào GitHub và tạo repository mới
2. Đặt tên repository (ví dụ: "ethusdt-dashboard")
3. Chọn "Private" nếu muốn giữ code riêng tư

### 2. Thiết lập local repository và đẩy lên GitHub

Sử dụng script `setup_github_sync.sh` đã được tạo sẵn:

```bash
chmod +x setup_github_sync.sh
./setup_github_sync.sh
```

Khi chạy script, bạn sẽ được yêu cầu nhập URL repository GitHub đã tạo.

### 3. Kiểm tra kết quả

Sau khi chạy script, kiểm tra trên GitHub để đảm bảo code đã được đẩy lên thành công.

## Quy trình làm việc hàng ngày

### 1. Cập nhật thay đổi từ GitHub về local

```bash
git pull origin main
```

### 2. Thực hiện các thay đổi trên code

### 3. Kiểm tra trạng thái thay đổi

```bash
git status
```

### 4. Thêm các file đã thay đổi vào staging area

```bash
git add .
```

### 5. Tạo commit cho các thay đổi

```bash
git commit -m "Mô tả những thay đổi đã thực hiện"
```

### 6. Đẩy thay đổi lên GitHub

```bash
git push origin main
```

## Triển khai lên máy chủ

### Triển khai thủ công

Sử dụng script `deploy_from_github.sh`:

```bash
chmod +x deploy_from_github.sh
./deploy_from_github.sh
```

Khi chạy script, bạn sẽ cần cung cấp:
- Địa chỉ IP máy chủ
- Cổng SSH (mặc định là 22)
- URL repository GitHub

## Thiết lập triển khai tự động

### 1. Thêm Secrets vào GitHub repository

1. Truy cập GitHub repository > Settings > Secrets and variables > Actions
2. Thêm các secret sau:
   - `SSH_PRIVATE_KEY`: Nội dung của private key SSH
   - `KNOWN_HOSTS`: Nội dung file known_hosts hoặc kết quả của lệnh `ssh-keyscan <địa_chỉ_ip_server>`
   - `SERVER_IP`: Địa chỉ IP của máy chủ
   - `SERVER_USER`: Tên người dùng SSH (thường là "root")
   - `SSH_PORT`: Cổng SSH (mặc định là 22)
   - `REMOTE_DIR`: Thư mục từ xa (ví dụ: "/root/ethusdt_dashboard")

### 2. Kích hoạt GitHub Actions

Workflow GitHub Actions đã được thiết lập trong file `.github/workflows/deploy.yml`. Mỗi khi bạn đẩy code lên nhánh `main`, code sẽ tự động được triển khai lên máy chủ.

Bạn cũng có thể kích hoạt triển khai thủ công bằng cách:
1. Truy cập tab "Actions" trong GitHub repository
2. Chọn workflow "Deploy ETHUSDT Dashboard"
3. Nhấn "Run workflow"

## Xử lý sự cố thường gặp

### Lỗi kết nối SSH

- Kiểm tra thông tin kết nối (địa chỉ IP, cổng SSH, tên người dùng)
- Đảm bảo SSH key đã được thêm vào máy chủ
- Kiểm tra firewall của máy chủ có cho phép kết nối SSH không

### Lỗi triển khai

- Kiểm tra quyền truy cập thư mục trên máy chủ
- Kiểm tra logs triển khai trong GitHub Actions
- Kiểm tra logs trên máy chủ: `journalctl -fu ethusdt-dashboard`

### Lỗi conflict khi đẩy code

```bash
# Lấy thay đổi mới nhất từ GitHub
git pull origin main

# Giải quyết conflict nếu có
# Sau đó commit và push lại
git add .
git commit -m "Fix conflicts"
git push origin main
```