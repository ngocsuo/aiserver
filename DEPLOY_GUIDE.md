# Hướng dẫn đồng bộ code lên server

Chào mừng đến với hướng dẫn triển khai ETHUSDT Dashboard. Tài liệu này sẽ hướng dẫn bạn các phương pháp để đồng bộ code từ Replit lên server sản phẩm.

## Phương pháp 1: Đồng bộ nhanh với 1 click

Đây là cách đơn giản nhất để đồng bộ tất cả các thay đổi lên server:

```bash
./deploy.sh
```

Script này sẽ tự động:
- Đồng bộ tất cả các file và thư mục
- Khởi động lại dịch vụ trên server
- Kiểm tra trạng thái hoạt động

## Phương pháp 2: Đồng bộ liên tục (theo dõi thay đổi)

Nếu bạn muốn làm việc lâu dài và cần tự động đồng bộ khi có thay đổi:

```bash
./automation_scripts/continuous_sync.sh
```

Script này sẽ:
- Chạy trong nền và theo dõi các thay đổi trên các file
- Tự động đồng bộ khi phát hiện thay đổi
- Khởi động lại dịch vụ trên server khi cần

Để dừng, nhấn `Ctrl+C`.

## Phương pháp 3: Đồng bộ thủ công khi cần

Nếu bạn muốn kiểm soát khi nào đồng bộ:

```bash
./automation_scripts/post_update_hook.sh
```

Script này sẽ đồng bộ tất cả các thay đổi một lần.

## Phương pháp 4: Đóng gói và triển khai

Đây là cách tạo một package hoàn chỉnh:

```bash
./deployment/prepare_deployment.sh
```

Sau đó làm theo hướng dẫn được hiển thị để chuyển file zip lên server và giải nén.

## Kiểm tra sau khi đồng bộ

Sau khi đồng bộ, kiểm tra trạng thái của server:

```bash
./automation_scripts/check_server_status.sh
```

## Sửa lỗi thường gặp

### Lỗi kết nối SSH
- Kiểm tra địa chỉ IP server, tên người dùng
- Kiểm tra kết nối mạng
- Đảm bảo SSH key đã được thiết lập đúng

### Dịch vụ không khởi động
- Kiểm tra logs: `ssh root@45.76.196.13 "journalctl -u ethusdt-dashboard -n 50"`
- Chạy setup script: `ssh root@45.76.196.13 "cd /root/ethusdt_dashboard && ./server_setup.sh"`

### Không thể truy cập dashboard
- Kiểm tra firewall: `ssh root@45.76.196.13 "ufw status"`
- Đảm bảo port 5000 được mở: `ssh root@45.76.196.13 "netstat -tuln | grep 5000"`