#!/bin/bash
# ===========================================
# VIRALIFY WORKER SERVER SETUP
# ===========================================
# Run this on the NEW worker server
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/olsisoft/viralify/master/setup-worker.sh | sudo bash
#
# Or manually:
#   1. wget https://raw.githubusercontent.com/olsisoft/viralify/master/setup-worker.sh
#   2. chmod +x setup-worker.sh
#   3. sudo ./setup-worker.sh
# ===========================================

set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo ./setup-worker.sh"
    exit 1
fi

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

# Ensure docker compose is available
if ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose plugin..."
    apt-get update && apt-get install -y docker-compose-plugin
fi

# Create directory and clone repo
INSTALL_DIR="/opt/viralify"
if [ ! -d "$INSTALL_DIR/.git" ]; then
    echo "Cloning Viralify repository..."
    rm -rf "$INSTALL_DIR"
    git clone https://github.com/olsisoft/viralify.git "$INSTALL_DIR"
else
    echo "Updating Viralify repository..."
    cd "$INSTALL_DIR"
    git pull origin master
fi

cd "$INSTALL_DIR"

# Check/create .env.workers
if [ ! -f ".env.workers" ]; then
    echo ""
    echo "Creating .env.workers from template..."
    cp .env.workers.example .env.workers

    echo ""
    echo "=========================================="
    echo "CONFIGURATION REQUIRED"
    echo "=========================================="
    echo ""
    echo "Please edit /opt/viralify/.env.workers with your settings:"
    echo ""
    echo "  nano /opt/viralify/.env.workers"
    echo ""
    echo "Required settings:"
    echo "  - MAIN_SERVER_HOST=<IP of your main server>"
    echo "  - DB_PASSWORD=<your database password>"
    echo "  - RABBITMQ_PASSWORD=<your rabbitmq password>"
    echo "  - REDIS_PASSWORD=<your redis password>"
    echo "  - OPENAI_API_KEY=<your OpenAI API key>"
    echo ""
    echo "Then run this script again."
    echo ""
    exit 0
fi

# Test connection to main server
MAIN_IP=$(grep "^MAIN_SERVER_HOST=" .env.workers | cut -d= -f2)

if [ -z "$MAIN_IP" ] || [ "$MAIN_IP" = "10.0.0.1" ]; then
    echo ""
    echo "ERROR: MAIN_SERVER_HOST not configured!"
    echo "Edit /opt/viralify/.env.workers and set the main server IP"
    echo ""
    exit 1
fi

echo ""
echo "Testing connection to main server ($MAIN_IP)..."

CONN_OK=true

echo -n "  PostgreSQL (5432): "
if timeout 3 bash -c "echo > /dev/tcp/$MAIN_IP/5432" 2>/dev/null; then
    echo "OK"
else
    echo "FAILED"
    CONN_OK=false
fi

echo -n "  RabbitMQ (5672):   "
if timeout 3 bash -c "echo > /dev/tcp/$MAIN_IP/5672" 2>/dev/null; then
    echo "OK"
else
    echo "FAILED"
    CONN_OK=false
fi

echo -n "  Redis (6379):      "
if timeout 3 bash -c "echo > /dev/tcp/$MAIN_IP/6379" 2>/dev/null; then
    echo "OK"
else
    echo "FAILED"
    CONN_OK=false
fi

echo ""

if [ "$CONN_OK" = false ]; then
    echo "WARNING: Some connections failed!"
    echo ""
    echo "Ensure on the main server:"
    echo "  1. Ports 5432, 5672, 6379 are open in firewall"
    echo "  2. Docker services are running"
    echo "  3. Services are bound to 0.0.0.0 (not just localhost)"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start workers
echo ""
echo "Building and starting workers..."
docker compose -f docker-compose.workers.yml --env-file .env.workers up -d --build

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  cd /opt/viralify"
echo ""
echo "  # Check status"
echo "  docker compose -f docker-compose.workers.yml ps"
echo ""
echo "  # View logs"
echo "  docker compose -f docker-compose.workers.yml logs -f"
echo ""
echo "  # Scale workers (e.g., 3 replicas)"
echo "  docker compose -f docker-compose.workers.yml up -d --scale course-worker=3"
echo ""
echo "  # Stop workers"
echo "  docker compose -f docker-compose.workers.yml down"
echo ""
