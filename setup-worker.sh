#!/bin/bash
# ===========================================
# VIRALIFY WORKER SERVER SETUP
# ===========================================
# Run this on the NEW worker server
#
# Usage:
#   1. Copy this script to the worker server
#   2. chmod +x setup-worker.sh
#   3. ./setup-worker.sh
# ===========================================

set -e

echo "=========================================="
echo "VIRALIFY WORKER SETUP"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    apt-get update && apt-get install -y docker-compose-plugin
    ln -sf /usr/libexec/docker/cli-plugins/docker-compose /usr/local/bin/docker-compose 2>/dev/null || true
fi

# Create directory
mkdir -p /opt/viralify
cd /opt/viralify

# Check if files exist
if [ ! -f "docker-compose.workers.yml" ]; then
    echo ""
    echo "ERROR: Missing files!"
    echo "Please copy these files from the main server:"
    echo "  - docker-compose.workers.yml"
    echo "  - .env.workers"
    echo "  - services/ directory (or clone the repo)"
    echo ""
    exit 1
fi

if [ ! -f ".env.workers" ]; then
    echo ""
    echo "ERROR: Missing .env.workers file!"
    echo "Copy it from the main server: /opt/viralify/.env.workers"
    echo ""
    exit 1
fi

# Test connection to main server
MAIN_IP=$(grep MAIN_SERVER_HOST .env.workers | cut -d= -f2)
echo ""
echo "Testing connection to main server ($MAIN_IP)..."

echo -n "  PostgreSQL (5432): "
timeout 3 bash -c "echo > /dev/tcp/$MAIN_IP/5432" 2>/dev/null && echo "OK" || echo "FAILED"

echo -n "  RabbitMQ (5672):   "
timeout 3 bash -c "echo > /dev/tcp/$MAIN_IP/5672" 2>/dev/null && echo "OK" || echo "FAILED"

echo -n "  Redis (6379):      "
timeout 3 bash -c "echo > /dev/tcp/$MAIN_IP/6379" 2>/dev/null && echo "OK" || echo "FAILED"

echo ""
echo "If any test failed, ensure:"
echo "  1. Ports are open on main server firewall"
echo "  2. Docker services are running on main server"
echo ""

read -p "Continue with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Start workers
echo ""
echo "Starting workers..."
docker-compose -f docker-compose.workers.yml --env-file .env.workers up -d --build

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Check status:  docker-compose -f docker-compose.workers.yml ps"
echo "View logs:     docker-compose -f docker-compose.workers.yml logs -f course-worker"
echo "Scale workers: docker-compose -f docker-compose.workers.yml up -d --scale course-worker=3"
echo ""
