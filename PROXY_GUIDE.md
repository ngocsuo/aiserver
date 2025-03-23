# Hướng dẫn cấu hình Proxy cấp hệ thống cho ETHUSDT Dashboard

Phiên bản mới của ETHUSDT Dashboard đã loại bỏ hoàn toàn mã proxy khỏi ứng dụng để tránh vấn đề bảo mật và làm cho mã nguồn dễ bảo trì hơn. Thay vào đó, proxy được xử lý ở cấp hệ thống/server.

## Lý do sử dụng Proxy cấp hệ thống

- **Bảo mật hơn**: Không cần lưu trữ thông tin proxy trong mã nguồn
- **Hiệu suất cao hơn**: Tất cả các kết nối đi qua proxy cấp hệ thống, không chỉ ứng dụng Python
- **Độ tin cậy cao hơn**: Không bị ảnh hưởng bởi các thay đổi trong thư viện Python
- **Dễ quản lý**: Thay đổi cấu hình proxy không yêu cầu sửa đổi mã nguồn

## Cách cấu hình Proxy cấp hệ thống trên Linux

### 1. Sử dụng biến môi trường

Bạn có thể cấu hình proxy cho toàn hệ thống bằng cách thêm các biến môi trường sau vào file `/etc/environment`:

```bash
http_proxy="http://proxy_user:proxy_password@proxy_host:proxy_port"
https_proxy="http://proxy_user:proxy_password@proxy_host:proxy_port"
no_proxy="localhost,127.0.0.1,::1"
HTTP_PROXY="http://proxy_user:proxy_password@proxy_host:proxy_port"
HTTPS_PROXY="http://proxy_user:proxy_password@proxy_host:proxy_port"
NO_PROXY="localhost,127.0.0.1,::1"
```

Sau khi thêm, khởi động lại hệ thống hoặc chạy lệnh sau để áp dụng thay đổi:

```bash
source /etc/environment
```

### 2. Sử dụng proxy toàn cầu với Systemd

Bạn có thể cấu hình proxy toàn cầu cho các dịch vụ systemd bằng cách tạo file `/etc/systemd/system.conf.d/proxy.conf`:

```ini
[Manager]
DefaultEnvironment="http_proxy=http://proxy_user:proxy_password@proxy_host:proxy_port" "https_proxy=http://proxy_user:proxy_password@proxy_host:proxy_port" "no_proxy=localhost,127.0.0.1,::1"
```

Sau đó chạy:

```bash
sudo systemctl daemon-reload
```

### 3. Cấu hình Proxy cho Supervisor

Nếu bạn sử dụng Supervisor để chạy ứng dụng, thêm biến môi trường vào cấu hình dịch vụ:

```ini
[program:ethusdt_dashboard]
command=/path/to/python /opt/ethusdt-dashboard/app.py
directory=/opt/ethusdt-dashboard
user=yourusername
environment=http_proxy="http://proxy_user:proxy_password@proxy_host:proxy_port",https_proxy="http://proxy_user:proxy_password@proxy_host:proxy_port"
```

### 4. Sử dụng proxy SOCKS với Dante

Bạn có thể cài đặt Dante server để cung cấp proxy SOCKS5 cục bộ:

```bash
sudo apt install dante-server
```

Cấu hình `/etc/dante.conf` để chuyển tiếp kết nối qua proxy của bạn:

```
logoutput: /var/log/dante.log

internal: 127.0.0.1 port=1080
external: eth0

socksmethod: username none
clientmethod: none

user.privileged: root
user.unprivileged: nobody

client pass {
    from: 0.0.0.0/0 to: 0.0.0.0/0
    log: error connect disconnect
}

pass {
    from: 0.0.0.0/0 to: 0.0.0.0/0 port=1-65535
    command: connect
    log: error connect disconnect
    protocol: tcp udp
}
```

Khởi động dịch vụ:

```bash
sudo service dante-server restart
```

## Cấu hình Proxy cho Python Requests/Binance API

Sau khi đã cấu hình proxy cấp hệ thống, các thư viện Python như `requests` và `python-binance` sẽ tự động sử dụng proxy từ biến môi trường.

Bạn không cần thực hiện bất kỳ thay đổi nào trong mã nguồn ứng dụng.

## Xác minh Kết nối

Để xác minh rằng cấu hình proxy hoạt động, hãy thử chạy lệnh sau:

```bash
curl -v https://api.binance.com/api/v3/ping
```

Nếu cấu hình proxy chính xác, bạn sẽ thấy kết nối thông qua proxy trong đầu ra và nhận được phản hồi thành công từ Binance API.

## Xử lý sự cố

1. Kiểm tra xem proxy có hoạt động không với lệnh `curl`:
   ```bash
   curl -x http://proxy_host:proxy_port -U proxy_user:proxy_password https://api.binance.com/api/v3/ping
   ```

2. Kiểm tra log của ứng dụng để tìm các lỗi liên quan đến kết nối:
   ```bash
   tail -f /path/to/app.log
   ```

3. Nếu sử dụng SOCKS5, đảm bảo rằng bạn đã cài đặt thư viện hỗ trợ cho Python:
   ```bash
   pip install pysocks requests[socks]
   ```

Nếu bạn cần hỗ trợ thêm, vui lòng tham khảo tài liệu của hệ điều hành hoặc liên hệ với quản trị viên hệ thống.