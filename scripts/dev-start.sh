#!/bin/bash
# ===========================================
# Viralify - Local Development Startup Script
# ===========================================
# Usage: ./scripts/dev-start.sh [service]
# Examples:
#   ./scripts/dev-start.sh           # Start all services
#   ./scripts/dev-start.sh frontend  # Start only frontend
#   ./scripts/dev-start.sh backend   # Start all backend services
#   ./scripts/dev-start.sh core      # Start core services only

set -e

SERVICE=${1:-"all"}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info() { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Load environment variables from .env.local if exists
if [ -f "$PROJECT_ROOT/.env.local" ]; then
    info "Loading environment from .env.local"
    export $(grep -v '^#' "$PROJECT_ROOT/.env.local" | xargs)
else
    warn ".env.local not found. Copy .env.local.example to .env.local and configure it."
fi

# Infrastructure URLs for local development
export DATABASE_URL="postgresql://tiktok_user:tiktok_secure_pass_2024@localhost:5432/tiktok_platform"
export REDIS_URL="redis://:redis_secure_2024@localhost:6379"
export RABBITMQ_URL="amqp://tiktok:rabbitmq_secure_2024@localhost:5672/"
export ELASTICSEARCH_URL="http://localhost:9200"

# Service URLs for local development
export AUTH_SERVICE_URL="http://localhost:8081"
export TREND_SERVICE_URL="http://localhost:8000"
export CONTENT_SERVICE_URL="http://localhost:8001"
export ANALYTICS_SERVICE_URL="http://localhost:8002"
export MEDIA_GENERATOR_URL="http://localhost:8004"
export PRESENTATION_GENERATOR_URL="http://localhost:8006"
export COURSE_GENERATOR_URL="http://localhost:8007"

# Check if infrastructure is running
info "Checking infrastructure containers..."
if ! docker-compose -f "$PROJECT_ROOT/docker-compose.dev.yml" ps --services --filter "status=running" | grep -q postgres; then
    warn "Infrastructure not running. Starting..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.dev.yml" up -d
    info "Waiting for infrastructure to be ready..."
    sleep 10
fi
success "Infrastructure is running"

# Function to start a service in a new terminal (macOS/Linux)
start_service() {
    local name=$1
    local path=$2
    local command=$3
    local port=$4

    info "Starting $name on port $port..."
    local service_path="$PROJECT_ROOT/$path"

    if [ ! -d "$service_path" ]; then
        error "Service path not found: $service_path"
        return
    fi

    # Detect terminal emulator
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal --title="$name" -- bash -c "cd '$service_path' && echo 'Starting $name...' && $command; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -title "$name" -e "cd '$service_path' && echo 'Starting $name...' && $command; exec bash" &
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        osascript -e "tell app \"Terminal\" to do script \"cd '$service_path' && echo 'Starting $name...' && $command\""
    else
        # Fallback: run in background
        warn "No terminal emulator found. Running $name in background..."
        (cd "$service_path" && $command > "/tmp/viralify-$name.log" 2>&1 &)
    fi
}

# Service definitions
declare -A SERVICES
SERVICES["frontend"]="frontend|npm run dev|3000"
SERVICES["api-gateway"]="services/api-gateway|./mvnw spring-boot:run|8080"
SERVICES["auth-service"]="services/auth-service|./mvnw spring-boot:run|8081"
SERVICES["trend-analyzer"]="services/trend-analyzer|uvicorn main:app --reload --host 0.0.0.0 --port 8000|8000"
SERVICES["content-generator"]="services/content-generator|uvicorn main:app --reload --host 0.0.0.0 --port 8001|8001"
SERVICES["analytics-service"]="services/analytics-service|uvicorn main:app --reload --host 0.0.0.0 --port 8002|8002"
SERVICES["notification-service"]="services/notification-service|uvicorn main:app --reload --host 0.0.0.0 --port 8003|8003"
SERVICES["media-generator"]="services/media-generator|uvicorn main:app --reload --host 0.0.0.0 --port 8004|8004"
SERVICES["coaching-service"]="services/coaching-service|uvicorn main:app --reload --host 0.0.0.0 --port 8005|8005"
SERVICES["presentation-generator"]="services/presentation-generator|python main.py|8006"
SERVICES["course-generator"]="services/course-generator|python main.py|8007"
SERVICES["practice-agent"]="services/practice-agent|uvicorn main:app --reload --host 0.0.0.0 --port 8008|8008"
SERVICES["scheduler-service"]="services/scheduler-service|./mvnw spring-boot:run|8082"
SERVICES["tiktok-connector"]="services/tiktok-connector|./mvnw spring-boot:run|8083"

# Service groups
BACKEND_SERVICES="api-gateway auth-service trend-analyzer content-generator analytics-service media-generator presentation-generator course-generator practice-agent"
CORE_SERVICES="api-gateway auth-service media-generator presentation-generator course-generator frontend"

start_service_by_name() {
    local name=$1
    local config=${SERVICES[$name]}
    if [ -z "$config" ]; then
        error "Unknown service: $name"
        return 1
    fi
    IFS='|' read -r path command port <<< "$config"
    start_service "$name" "$path" "$command" "$port"
    sleep 0.5
}

case "$SERVICE" in
    "all")
        info "Starting all services..."
        for svc in "${!SERVICES[@]}"; do
            start_service_by_name "$svc"
        done
        ;;
    "backend")
        info "Starting backend services..."
        for svc in $BACKEND_SERVICES; do
            start_service_by_name "$svc"
        done
        ;;
    "core")
        info "Starting core services (minimal for course generation)..."
        for svc in $CORE_SERVICES; do
            start_service_by_name "$svc"
        done
        ;;
    "frontend")
        start_service_by_name "frontend"
        ;;
    *)
        if [ -n "${SERVICES[$SERVICE]}" ]; then
            start_service_by_name "$SERVICE"
        else
            error "Unknown service: $SERVICE"
            info "Available services: ${!SERVICES[*]}"
            info "Groups: all, backend, core, frontend"
            exit 1
        fi
        ;;
esac

echo ""
success "Services started!"
echo ""
echo -e "${YELLOW}Service URLs:${NC}"
echo "  Frontend:              http://localhost:3000"
echo "  API Gateway:           http://localhost:8080"
echo "  Auth Service:          http://localhost:8081"
echo "  Media Generator:       http://localhost:8004"
echo "  Presentation Gen:      http://localhost:8006"
echo "  Course Generator:      http://localhost:8007"
echo ""
echo -e "${YELLOW}Infrastructure:${NC}"
echo "  PostgreSQL:            localhost:5432"
echo "  Redis:                 localhost:6379"
echo "  RabbitMQ:              localhost:5672 (Management: http://localhost:15672)"
echo "  Elasticsearch:         http://localhost:9200"
