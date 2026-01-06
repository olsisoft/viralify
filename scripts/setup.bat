@echo off
echo ========================================
echo Viralify Platform Setup Script
echo ========================================
echo.

:: Check for Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop.
    exit /b 1
)
echo [OK] Docker is installed

:: Check for Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed. Please install Node.js 20+.
    exit /b 1
)
echo [OK] Node.js is installed

:: Check for Java
java --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Java is not installed. Required for building Java services locally.
)else (
    echo [OK] Java is installed
)

:: Setup Gradle wrapper for each Java service
echo.
echo Setting up Gradle wrappers for Java services...

set JAVA_SERVICES=api-gateway auth-service scheduler-service tiktok-connector
set PROJECT_ROOT=%~dp0..

for %%s in (%JAVA_SERVICES%) do (
    echo Setting up %%s...
    cd "%PROJECT_ROOT%\services\%%s"
    if not exist gradle\wrapper (
        mkdir gradle\wrapper 2>nul
    )
)

:: Install frontend dependencies
echo.
echo Installing frontend dependencies...
cd "%PROJECT_ROOT%\frontend"
call npm install

:: Create .env from example if it doesn't exist
if not exist "%PROJECT_ROOT%\.env" (
    echo Creating .env from .env.example...
    copy "%PROJECT_ROOT%\.env.example" "%PROJECT_ROOT%\.env"
    echo [WARN] Please edit .env file with your API keys!
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env file with your API keys (TikTok, OpenAI, etc.)
echo 2. Run: docker-compose up -d postgres redis rabbitmq elasticsearch
echo 3. Wait for services to start, then run: docker-compose up
echo 4. Open http://localhost:3000 in your browser
echo.
