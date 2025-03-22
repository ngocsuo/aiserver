#!/bin/bash
# Script để giữ Replit hoạt động trong quá trình huấn luyện dài
# Chạy trong nền và ghi log định kỳ

echo "$(date) - Starting keep-alive service" >> keep_alive.log

# Tạo một file request.txt để gửi request định kỳ
cat << 'EOT' > request.txt
GET / HTTP/1.1
Host: 0.0.0.0:5000
User-Agent: keep-alive-bot
Accept: */*
Connection: keep-alive

EOT

# Vô hạn gửi request và ghi log định kỳ
while true; do
  # Tạo request HTTP để giữ server hoạt động
  cat request.txt | nc -w 1 0.0.0.0 5000 > /dev/null 2>&1
  
  # Ghi log thời gian hoạt động
  echo "$(date) - Keep-alive ping sent, memory usage: $(free -m | grep Mem | awk '{print $3}')MB" >> keep_alive.log
  
  # Cập nhật file timestamp
  date +%s > .last_ping
  
  # Đợi 30 giây
  sleep 30
done
