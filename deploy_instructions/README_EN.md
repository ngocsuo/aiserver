# ETHUSDT Prediction System Deployment Guide

This document provides detailed instructions for deploying the ETHUSDT prediction system on your own server or VPS.

## System Requirements

- **VPS/Server**: Ubuntu 20.04 LTS or newer
- **RAM**: Minimum 4GB (8GB recommended for optimal performance)
- **CPU**: 2 cores or higher
- **Storage**: Minimum 20GB SSD
- **Internet Connection**: Stable with no geographic restrictions to Binance API
- **Python**: Version 3.9 or higher

## Environment Setup

### 1. Install Required Software

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git supervisor nginx
```

### 2. Download Source Code

```bash
# Create application directory
mkdir -p /opt/ethusdt_predictor
cd /opt/ethusdt_predictor

# Copy source code from backup or Git (if you store it on Git)
# Example using Git:
# git clone https://your-repository-url.git .

# Or copy manually using tools like SCP, SFTP, etc.
```

### 3. Create Python Virtual Environment

```bash
cd /opt/ethusdt_predictor
python3 -m venv venv
source venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

If requirements.txt doesn't exist, you can install the following libraries:

```bash
pip install streamlit pandas numpy plotly python-binance scikit-learn tensorflow
```

## System Configuration

### 1. Setup Binance API

To connect to the Binance API, you need to provide your Binance API key and secret. You can set them as environment variables:

```bash
# Create .env file in /opt/ethusdt_predictor
cat > /opt/ethusdt_predictor/.env << EOF
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
EOF

# Ensure proper permissions
chmod 600 /opt/ethusdt_predictor/.env
```

### 2. Configure Automatic System Startup with Supervisor

```bash
# Create supervisor configuration
sudo tee /etc/supervisor/conf.d/ethusdt_predictor.conf > /dev/null << EOF
[program:ethusdt_predictor]
command=/opt/ethusdt_predictor/venv/bin/streamlit run /opt/ethusdt_predictor/app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true
directory=/opt/ethusdt_predictor
user=root
autostart=true
autorestart=true
startretries=10
stderr_logfile=/var/log/ethusdt_predictor.err.log
stdout_logfile=/var/log/ethusdt_predictor.out.log
environment=BINANCE_API_KEY="%(ENV_BINANCE_API_KEY)s",BINANCE_API_SECRET="%(ENV_BINANCE_API_SECRET)s"
EOF

# Update and start service
sudo supervisorctl reread
sudo supervisorctl update
```

### 3. Configure Nginx (Optional)

If you want to access the application via the web with a domain and HTTPS, you can configure Nginx:

```bash
# Create Nginx configuration
sudo tee /etc/nginx/sites-available/ethusdt_predictor > /dev/null << EOF
server {
    listen 80;
    server_name your-domain.com; # Replace with your domain

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Enable configuration
sudo ln -s /etc/nginx/sites-available/ethusdt_predictor /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Configure HTTPS with Certbot (Optional)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Start and Check System

```bash
# Restart supervisord to ensure the app is running
sudo supervisorctl restart ethusdt_predictor

# Check logs
sudo tail -f /var/log/ethusdt_predictor.out.log
```

After completion, you can access the system via:
- http://your-server-ip:5000 (if not using Nginx)
- https://your-domain.com (if configured with Nginx and HTTPS)

## Troubleshooting

### Binance API Connection Issues

If you encounter errors connecting to the Binance API, check:
1. API key and secret are correctly configured
2. API key has the necessary access permissions (read futures market data)
3. Your VPS/server is not geographically restricted from accessing Binance

### Application Startup Errors

If the application doesn't start, check the logs:

```bash
sudo tail -f /var/log/ethusdt_predictor.err.log
```

### Performance Optimization

If the system runs slowly:
1. Increase RAM and CPU for your VPS/server
2. Adjust parameters in config.py (reduce LOOKBACK_PERIODS, increase UPDATE_INTERVAL)
3. Disable unnecessary features (e.g., complex indicators)

## Backup and Restore

### Data Backup

```bash
# Backup data directory and models
cd /opt
tar -czf ethusdt_backup_$(date +%Y%m%d).tar.gz ethusdt_predictor/saved_models
```

### Data Restoration

```bash
# Restore from backup
cd /opt
tar -xzf ethusdt_backup_YYYYMMDD.tar.gz
```

## Security

- **API Keys**: Always ensure API keys have read-only permissions unless you need automated trading functionality
- **Firewall**: Configure ufw to only open necessary ports (SSH, HTTP/HTTPS)
- **Accounts**: Don't run the application with root privileges in a production environment (adjust supervisor file accordingly)

## System Updates

To update the source code:

```bash
cd /opt/ethusdt_predictor
# Backup current configuration
cp config.py config.py.backup

# Download new source code (from Git or copy manually)
# git pull

# Restore personal configuration if needed
# cp config.py.backup config.py

# Restart application
sudo supervisorctl restart ethusdt_predictor
```

## Support Contact

If you encounter issues during deployment, please contact:
- Email: your-email@example.com
- Telegram: @your_telegram_username