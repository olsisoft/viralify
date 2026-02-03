#!/bin/bash
set -e

echo "=== VIRALIFY REBUILD ==="
echo ""

# Stop and remove ALL viralify-related containers (from any docker-compose project)
echo "[1/4] Stopping all Viralify containers..."
docker ps -a --filter "name=viralify" -q | xargs -r docker rm -f 2>/dev/null || true

# Also stop containers from the 'repo' project (tiktok-* legacy names)
echo "[2/4] Stopping legacy containers..."
docker ps -a --filter "name=tiktok-" -q | xargs -r docker rm -f 2>/dev/null || true
docker ps -a --filter "label=com.docker.compose.project=repo" -q | xargs -r docker rm -f 2>/dev/null || true

# Prune orphaned networks
echo "[3/4] Cleaning up networks..."
docker network prune -f 2>/dev/null || true

# Start all services from the single source of truth
echo "[4/4] Starting all services..."
cd /opt/viralify
docker compose -f docker-compose.prod.yml up -d --build

echo ""
echo "=== REBUILD COMPLETE ==="
echo ""
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "viralify|NAMES"
