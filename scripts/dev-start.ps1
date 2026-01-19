# ===========================================
# Viralify - Local Development Startup Script
# ===========================================
# Usage: .\scripts\dev-start.ps1 [service]
# Examples:
#   .\scripts\dev-start.ps1           # Start all services
#   .\scripts\dev-start.ps1 frontend  # Start only frontend
#   .\scripts\dev-start.ps1 backend   # Start all backend services

param(
    [string]$Service = "all"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Colors for output
function Write-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Load environment variables from .env.local if exists
$EnvFile = Join-Path $ProjectRoot ".env.local"
if (Test-Path $EnvFile) {
    Write-Info "Loading environment from .env.local"
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
} else {
    Write-Warn ".env.local not found. Copy .env.local.example to .env.local and configure it."
}

# Infrastructure URLs for local development
$env:DATABASE_URL = "postgresql://tiktok_user:tiktok_secure_pass_2024@localhost:5432/tiktok_platform"
$env:REDIS_URL = "redis://:redis_secure_2024@localhost:6379"
$env:RABBITMQ_URL = "amqp://tiktok:rabbitmq_secure_2024@localhost:5672/"
$env:ELASTICSEARCH_URL = "http://localhost:9200"

# Service URLs for local development (services talk to each other via localhost)
$env:AUTH_SERVICE_URL = "http://localhost:8081"
$env:TREND_SERVICE_URL = "http://localhost:8000"
$env:CONTENT_SERVICE_URL = "http://localhost:8001"
$env:ANALYTICS_SERVICE_URL = "http://localhost:8002"
$env:MEDIA_GENERATOR_URL = "http://localhost:8004"
$env:PRESENTATION_GENERATOR_URL = "http://localhost:8006"
$env:COURSE_GENERATOR_URL = "http://localhost:8007"

# Check if infrastructure is running
Write-Info "Checking infrastructure containers..."
$infraRunning = docker-compose -f "$ProjectRoot\docker-compose.dev.yml" ps --services --filter "status=running" 2>$null
if (-not $infraRunning) {
    Write-Warn "Infrastructure not running. Starting..."
    docker-compose -f "$ProjectRoot\docker-compose.dev.yml" up -d
    Write-Info "Waiting for infrastructure to be ready..."
    Start-Sleep -Seconds 10
}
Write-Success "Infrastructure is running"

# Function to start a service in a new terminal
function Start-Service {
    param(
        [string]$Name,
        [string]$Path,
        [string]$Command,
        [int]$Port
    )

    Write-Info "Starting $Name on port $Port..."
    $servicePath = Join-Path $ProjectRoot $Path

    if (-not (Test-Path $servicePath)) {
        Write-Err "Service path not found: $servicePath"
        return
    }

    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$servicePath'; Write-Host 'Starting $Name...' -ForegroundColor Green; $Command"
}

# Service definitions
$Services = @{
    "frontend" = @{
        Path = "frontend"
        Command = "npm run dev"
        Port = 3000
    }
    "api-gateway" = @{
        Path = "services\api-gateway"
        Command = ".\mvnw spring-boot:run"
        Port = 8080
    }
    "auth-service" = @{
        Path = "services\auth-service"
        Command = ".\mvnw spring-boot:run"
        Port = 8081
    }
    "trend-analyzer" = @{
        Path = "services\trend-analyzer"
        Command = "uvicorn main:app --reload --host 0.0.0.0 --port 8000"
        Port = 8000
    }
    "content-generator" = @{
        Path = "services\content-generator"
        Command = "uvicorn main:app --reload --host 0.0.0.0 --port 8001"
        Port = 8001
    }
    "analytics-service" = @{
        Path = "services\analytics-service"
        Command = "uvicorn main:app --reload --host 0.0.0.0 --port 8002"
        Port = 8002
    }
    "notification-service" = @{
        Path = "services\notification-service"
        Command = "uvicorn main:app --reload --host 0.0.0.0 --port 8003"
        Port = 8003
    }
    "media-generator" = @{
        Path = "services\media-generator"
        Command = "uvicorn main:app --reload --host 0.0.0.0 --port 8004"
        Port = 8004
    }
    "coaching-service" = @{
        Path = "services\coaching-service"
        Command = "uvicorn main:app --reload --host 0.0.0.0 --port 8005"
        Port = 8005
    }
    "presentation-generator" = @{
        Path = "services\presentation-generator"
        Command = "python main.py"
        Port = 8006
    }
    "course-generator" = @{
        Path = "services\course-generator"
        Command = "python main.py"
        Port = 8007
    }
    "practice-agent" = @{
        Path = "services\practice-agent"
        Command = "uvicorn main:app --reload --host 0.0.0.0 --port 8008"
        Port = 8008
    }
    "scheduler-service" = @{
        Path = "services\scheduler-service"
        Command = ".\mvnw spring-boot:run"
        Port = 8082
    }
    "tiktok-connector" = @{
        Path = "services\tiktok-connector"
        Command = ".\mvnw spring-boot:run"
        Port = 8083
    }
}

# Backend services (without frontend)
$BackendServices = @(
    "api-gateway", "auth-service", "trend-analyzer", "content-generator",
    "analytics-service", "media-generator", "presentation-generator",
    "course-generator", "practice-agent"
)

# Core services (minimal set for course generation)
$CoreServices = @(
    "api-gateway", "auth-service", "media-generator",
    "presentation-generator", "course-generator"
)

switch ($Service) {
    "all" {
        Write-Info "Starting all services..."
        foreach ($svc in $Services.Keys) {
            $s = $Services[$svc]
            Start-Service -Name $svc -Path $s.Path -Command $s.Command -Port $s.Port
            Start-Sleep -Milliseconds 500
        }
    }
    "backend" {
        Write-Info "Starting backend services..."
        foreach ($svc in $BackendServices) {
            $s = $Services[$svc]
            Start-Service -Name $svc -Path $s.Path -Command $s.Command -Port $s.Port
            Start-Sleep -Milliseconds 500
        }
    }
    "core" {
        Write-Info "Starting core services (minimal for course generation)..."
        foreach ($svc in $CoreServices) {
            $s = $Services[$svc]
            Start-Service -Name $svc -Path $s.Path -Command $s.Command -Port $s.Port
            Start-Sleep -Milliseconds 500
        }
        # Also start frontend
        $s = $Services["frontend"]
        Start-Service -Name "frontend" -Path $s.Path -Command $s.Command -Port $s.Port
    }
    "frontend" {
        $s = $Services["frontend"]
        Start-Service -Name "frontend" -Path $s.Path -Command $s.Command -Port $s.Port
    }
    default {
        if ($Services.ContainsKey($Service)) {
            $s = $Services[$Service]
            Start-Service -Name $Service -Path $s.Path -Command $s.Command -Port $s.Port
        } else {
            Write-Err "Unknown service: $Service"
            Write-Info "Available services: $($Services.Keys -join ', ')"
            Write-Info "Groups: all, backend, core, frontend"
            exit 1
        }
    }
}

Write-Host ""
Write-Success "Services started! Check the new terminal windows."
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Yellow
Write-Host "  Frontend:              http://localhost:3000"
Write-Host "  API Gateway:           http://localhost:8080"
Write-Host "  Auth Service:          http://localhost:8081"
Write-Host "  Media Generator:       http://localhost:8004"
Write-Host "  Presentation Gen:      http://localhost:8006"
Write-Host "  Course Generator:      http://localhost:8007"
Write-Host ""
Write-Host "Infrastructure:" -ForegroundColor Yellow
Write-Host "  PostgreSQL:            localhost:5432"
Write-Host "  Redis:                 localhost:6379"
Write-Host "  RabbitMQ:              localhost:5672 (Management: http://localhost:15672)"
Write-Host "  Elasticsearch:         http://localhost:9200"
