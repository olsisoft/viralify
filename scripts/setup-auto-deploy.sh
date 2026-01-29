#!/bin/bash
#
# Setup script for Viralify auto-deployment
#
# This script:
# 1. Installs the webhook service
# 2. Configures GitHub webhook
# 3. Starts the service
#
# Usage:
#   ./setup-auto-deploy.sh
#
# Prerequisites:
#   - Root access
#   - Python 3 installed
#   - rebuild.sh exists in repo root
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  Viralify Auto-Deploy Setup"
echo "========================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./setup-auto-deploy.sh)"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required. Installing..."
    apt-get update && apt-get install -y python3
fi

# Generate webhook secret if not set
if [ -z "$WEBHOOK_SECRET" ]; then
    WEBHOOK_SECRET=$(openssl rand -hex 20)
    echo "Generated webhook secret: $WEBHOOK_SECRET"
    echo ""
    echo "IMPORTANT: Save this secret! You'll need it for GitHub webhook config."
    echo ""
fi

# Get Discord webhook URL (optional)
read -p "Discord webhook URL (press Enter to skip): " DISCORD_URL

# Get Telegram config (optional)
read -p "Telegram bot token (press Enter to skip): " TELEGRAM_TOKEN
if [ -n "$TELEGRAM_TOKEN" ]; then
    read -p "Telegram chat ID: " TELEGRAM_CHAT_ID
fi

# Create service file with configuration
echo "Creating systemd service..."
cat > /etc/systemd/system/viralify-deploy.service << EOF
[Unit]
Description=Viralify Auto-Deploy Webhook Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${REPO_DIR}
ExecStart=/usr/bin/python3 ${REPO_DIR}/scripts/deploy-webhook.py

Environment="WEBHOOK_SECRET=${WEBHOOK_SECRET}"
Environment="WEBHOOK_PORT=9000"
Environment="REPO_PATH=${REPO_DIR}"
Environment="REBUILD_SCRIPT=./rebuild.sh"
EOF

if [ -n "$DISCORD_URL" ]; then
    echo "Environment=\"DISCORD_WEBHOOK_URL=${DISCORD_URL}\"" >> /etc/systemd/system/viralify-deploy.service
fi

if [ -n "$TELEGRAM_TOKEN" ]; then
    echo "Environment=\"TELEGRAM_BOT_TOKEN=${TELEGRAM_TOKEN}\"" >> /etc/systemd/system/viralify-deploy.service
    echo "Environment=\"TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}\"" >> /etc/systemd/system/viralify-deploy.service
fi

cat >> /etc/systemd/system/viralify-deploy.service << EOF

Restart=always
RestartSec=10

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
echo "Starting service..."
systemctl daemon-reload
systemctl enable viralify-deploy
systemctl start viralify-deploy

# Check status
sleep 2
if systemctl is-active --quiet viralify-deploy; then
    echo ""
    echo "========================================"
    echo "  Setup Complete!"
    echo "========================================"
    echo ""
    echo "Webhook server is running on port 9000"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Go to your GitHub repo settings:"
    echo "   https://github.com/olsisoft/viralify/settings/hooks/new"
    echo ""
    echo "2. Configure the webhook:"
    echo "   - Payload URL: http://YOUR_SERVER_IP:9000/webhook"
    echo "   - Content type: application/json"
    echo "   - Secret: ${WEBHOOK_SECRET}"
    echo "   - Events: Just the push event"
    echo ""
    echo "3. Make sure port 9000 is open in your firewall"
    echo ""
    echo "To check logs:"
    echo "   journalctl -u viralify-deploy -f"
    echo ""
else
    echo "ERROR: Service failed to start!"
    echo "Check logs with: journalctl -u viralify-deploy -n 50"
    exit 1
fi
