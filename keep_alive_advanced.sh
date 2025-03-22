#!/bin/bash
# Script nâng cao để giữ Replit hoạt động trong quá trình huấn luyện

# Tạo file log
LOGFILE="keep_alive_advanced.log"
echo "$(date) - Starting advanced keep-alive service" > $LOGFILE

# Định nghĩa hàm ping
ping_app() {
  # Gửi request HTTP đến ứng dụng chính
  curl -s http://0.0.0.0:5000/ > /dev/null 2>&1
  
  # Cập nhật timestamp
  date +%s > .last_ping
  
  # Log thành công
  echo "$(date) - Keep-alive ping sent" >> $LOGFILE
}

# Chạy ping định kỳ
while true; do
  # Kiểm tra trạng thái bộ nhớ và CPU
  MEM_USAGE=$(free -m | grep Mem | awk '{print $3}')
  
  # Ghi log
  echo "$(date) - Memory: ${MEM_USAGE}MB" >> $LOGFILE
  
  # Gửi ping
  ping_app
  
  # Đợi 10 giây
  sleep 10
done
