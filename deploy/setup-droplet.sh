#!/bin/bash
# ===========================================
# Viralify - DigitalOcean Droplet Setup Script
# ===========================================
# Run this on a fresh Ubuntu 22.04 Droplet
# Usage: curl -sSL https://raw.githubusercontent.com/YOUR_REPO/deploy/setup-droplet.sh | bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}===========================================
 Viralify - DigitalOcean Droplet Setup
===========================================${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

# Get domain from user
read -p "Enter your domain (e.g., viralify.app): " DOMAIN
read -p "Enter your email for SSL certificates: " EMAIL

echo -e "${YELLOW}[1/7] Updating system...${NC}"
apt-get update && apt-get upgrade -y

echo -e "${YELLOW}[2/7] Installing dependencies...${NC}"
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    ufw \
    fail2ban

echo -e "${YELLOW}[3/7] Installing Docker...${NC}"
# Remove old versions
apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker
systemctl start docker
systemctl enable docker

echo -e "${YELLOW}[4/7] Configuring firewall...${NC}"
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow http
ufw allow https
ufw --force enable

echo -e "${YELLOW}[5/7] Configuring fail2ban...${NC}"
systemctl start fail2ban
systemctl enable fail2ban

echo -e "${YELLOW}[6/7] Creating app directory...${NC}"
mkdir -p /opt/viralify
cd /opt/viralify

# Create directory structure
mkdir -p nginx/conf.d certbot/conf certbot/www init-scripts

echo -e "${YELLOW}[7/7] Creating swap space (for 8GB droplets)...${NC}"
if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab
fi

echo -e "${GREEN}===========================================
 Setup Complete!
===========================================${NC}"

echo -e "
${YELLOW}Next steps:${NC}

1. Clone your repository:
   cd /opt/viralify
   git clone https://github.com/YOUR_USERNAME/viralify.git repo

2. Copy deployment files:
   cp repo/deploy/docker-compose.prod.yml .
   cp repo/deploy/nginx/nginx.conf nginx/
   cp repo/deploy/nginx/conf.d/app.conf.initial nginx/conf.d/app.conf
   cp repo/infrastructure/docker/init.sql init-scripts/
   cp repo/infrastructure/docker/init-pgvector.sql init-scripts/

3. Create .env file:
   cp repo/deploy/.env.prod.example .env
   nano .env  # Edit with your secrets

4. Replace YOUR_DOMAIN.com in nginx config:
   sed -i 's/YOUR_DOMAIN.com/${DOMAIN}/g' nginx/conf.d/app.conf

5. Start services (without SSL first):
   docker compose -f docker-compose.prod.yml up -d

6. Get SSL certificate:
   docker compose run --rm certbot certonly --webroot -w /var/www/certbot \\
     --email ${EMAIL} -d ${DOMAIN} -d www.${DOMAIN} --agree-tos

7. Switch to SSL config:
   cp repo/deploy/nginx/conf.d/app.conf nginx/conf.d/app.conf
   sed -i 's/YOUR_DOMAIN.com/${DOMAIN}/g' nginx/conf.d/app.conf
   docker compose restart nginx

Your domain: ${DOMAIN}
Your email: ${EMAIL}
App directory: /opt/viralify
"
