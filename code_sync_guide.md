# Hướng dẫn đồng bộ code giữa Replit và Server

## Phương pháp 1: Sử dụng GitHub làm trung gian

Đây là phương pháp được khuyến nghị nhất vì tính linh hoạt và độ an toàn.

### Bước 1: Tạo repository GitHub

1. Đăng nhập GitHub và tạo repository mới (private là tốt nhất để bảo mật)
2. Lấy URL của repository (dạng `https://github.com/username/repository.git`)

### Bước 2: Thiết lập Git trên Replit

```bash
# Trong terminal của Replit
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/username/repository.git
git push -u origin main
```

### Bước 3: Thiết lập Git trên server

```bash
# SSH vào server
ssh user@your-server-ip

# Di chuyển đến thư mục dự án
cd /đường/dẫn/đến/dự/án

# Clone repository từ GitHub
git clone https://github.com/username/repository.git

# Nếu thư mục đã có dữ liệu, bạn cần sao lưu nó trước
mv /đường/dẫn/đến/dự/án /đường/dẫn/đến/dự/án_backup
git clone https://github.com/username/repository.git /đường/dẫn/đến/dự/án

# Thiết lập thông tin đăng nhập Git (nếu cần)
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
```

### Bước 4: Quá trình làm việc đồng bộ

#### Khi sửa code trên Replit:

```bash
git add .
git commit -m "Fix error: too many values to unpack"
git push origin main
```

#### Để cập nhật code trên server:

```bash
cd /đường/dẫn/đến/dự/án
git pull origin main
```

#### Khi sửa code trên server (nếu cần):

```bash
git add .
git commit -m "Server-specific fixes"
git push origin main
```

## Phương pháp 2: Sử dụng rsync để đồng bộ trực tiếp

Nếu bạn không muốn sử dụng Git, bạn có thể dùng rsync để đồng bộ trực tiếp.

### Trên Replit:

1. Xuất code từ Replit (Downloads > Download as zip)
2. Giải nén trên máy tính của bạn

### Từ máy tính của bạn đến server:

```bash
rsync -avz --exclude='.git' --exclude='__pycache__' /đường/dẫn/local/code/ user@your-server-ip:/đường/dẫn/đến/dự/án/
```

## Phương pháp 3: Sử dụng GitHub Actions cho CI/CD tự động

Phương pháp nâng cao này tự động triển khai code lên server khi bạn push lên GitHub.

### Bước 1: Thiết lập SSH key trên GitHub

1. Tạo SSH key trên server:
   ```bash
   ssh-keygen -t rsa -b 4096 -C "github-actions-deploy"
   ```
2. Thêm public key vào `~/.ssh/authorized_keys` trên server
3. Thêm private key vào GitHub repository secrets với tên `SSH_PRIVATE_KEY`

### Bước 2: Tạo file workflow GitHub Actions

Tạo file `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Server

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up SSH
      uses: webfactory/ssh-agent@v0.5.3
      with:
        ssh-private-key: ${{ secrets.SSH_PRIVATE_KEY }}
    
    - name: Deploy to server
      run: |
        ssh -o StrictHostKeyChecking=no user@your-server-ip "cd /đường/dẫn/đến/dự/án && git pull origin main"
        ssh -o StrictHostKeyChecking=no user@your-server-ip "systemctl restart your_app_name || supervisorctl restart your_app_name"
```

## Phương pháp 4: Sử dụng VSCode Remote-SSH

Nếu bạn muốn sửa code trực tiếp trên server qua giao diện VSCode:

1. Cài đặt Visual Studio Code trên máy tính của bạn
2. Cài đặt extension "Remote - SSH"
3. Nhấn F1, chọn "Remote-SSH: Connect to Host..." và nhập thông tin server
4. Duyệt đến thư mục dự án và chỉnh sửa trực tiếp

## Phương pháp 5: Sử dụng Replit Transfer API

Replit cung cấp API để đồng bộ code, nhưng bạn cần viết script để sử dụng tính năng này:

```python
import requests
import os
import json

def download_repl(repl_id, output_dir):
    """Download a repl from Replit."""
    api_url = f"https://replit.com/data/repls/{repl_id}"
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json",
        "Cookie": "connect.sid=YOUR_CONNECT_SID"  # Lấy từ cookies của trình duyệt
    }
    response = requests.get(api_url, headers=headers)
    data = response.json()
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu thông tin repl
    with open(os.path.join(output_dir, "repl_info.json"), "w") as f:
        json.dump(data, f, indent=2)
    
    # Tải các file
    for file in data["files"]:
        file_path = os.path.join(output_dir, file["path"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        content_response = requests.get(file["url"])
        with open(file_path, "wb") as f:
            f.write(content_response.content)

# Sử dụng:
download_repl("your-repl-id", "/path/to/output")
```

## Lời khuyên

1. **Sử dụng Git** là cách hiệu quả nhất để quản lý code và đồng bộ giữa nhiều môi trường
2. **Tự động hóa quy trình triển khai** với GitHub Actions hoặc các công cụ CI/CD để giảm thiểu lỗi thủ công
3. **Tách biệt cấu hình môi trường** bằng cách sử dụng các file .env khác nhau cho Replit và server
4. **Luôn sao lưu trước khi đồng bộ** để tránh mất mát dữ liệu nếu có lỗi
5. **Kiểm tra kỹ lưỡng** sau mỗi lần đồng bộ để đảm bảo ứng dụng vẫn hoạt động bình thường