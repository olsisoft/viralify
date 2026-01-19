#!/bin/bash
# ===========================================
# Viralify - Deployment Script
# ===========================================
# Run this to deploy updates
# Usage: ./deploy.sh [service]
# Examples:
#   ./deploy.sh           # Deploy all services
#   ./deploy.sh frontend  # Deploy only frontend
#   ./deploy.sh backend   # Deploy all backend services

set -e

SERVICE=${1:-"all"}
DEPLOY_DIR="/opt/viralify"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd $DEPLOY_DIR

echo -e "${GREEN}===========================================
 Viralify - Deploying: $SERVICE
===========================================${NC}"

# Pull latest code
echo -e "${YELLOW}[1/4] Pulling latest code...${NC}"
cd repo
git pull origin main
cd ..

# Copy any updated config files
echo -e "${YELLOW}[2/4] Updating configuration...${NC}"
cp repo/deploy/docker-compose.prod.yml .
cp repo/infrastructure/docker/init.sql init-scripts/ 2>/dev/null || true
cp repo/infrastructure/docker/init-pgvector.sql init-scripts/ 2>/dev/null || true

# Build and deploy
echo -e "${YELLOW}[3/4] Building and deploying...${NC}"

case "$SERVICE" in
    "all")
        docker compose -f docker-compose.prod.yml build
        docker compose -f docker-compose.prod.yml up -d
        ;;
    "frontend")
        docker compose -f docker-compose.prod.yml build frontend
        docker compose -f docker-compose.prod.yml up -d frontend
        ;;
    "backend")
        docker compose -f docker-compose.prod.yml build \
            api-gateway auth-service scheduler-service \
            trend-analyzer content-generator analytics-service \
            media-generator presentation-generator course-generator
        docker compose -f docker-compose.prod.yml up -d \
            api-gateway auth-service scheduler-service \
            trend-analyzer content-generator analytics-service \
            media-generator presentation-generator course-generator
        ;;
    *)
        docker compose -f docker-compose.prod.yml build $SERVICE
        docker compose -f docker-compose.prod.yml up -d $SERVICE
        ;;
esac

# Cleanup
echo -e "${YELLOW}[4/4] Cleaning up...${NC}"
docker image prune -f

echo -e "${GREEN}===========================================
 Deployment Complete!
===========================================${NC}"

# Show status
docker compose -f docker-compose.prod.yml ps
