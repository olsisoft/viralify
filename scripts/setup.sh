#!/bin/bash

echo "========================================"
echo "Viralify Platform Setup Script"
echo "========================================"
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed. Please install Docker."
    exit 1
fi
echo "[OK] Docker is installed"

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed. Please install Node.js 20+."
    exit 1
fi
echo "[OK] Node.js is installed"

# Check for Java
if command -v java &> /dev/null; then
    echo "[OK] Java is installed"
else
    echo "[WARN] Java is not installed. Required for building Java services locally."
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Setup Gradle wrapper for each Java service
echo ""
echo "Setting up Gradle wrappers for Java services..."

JAVA_SERVICES="api-gateway auth-service scheduler-service tiktok-connector"

for service in $JAVA_SERVICES; do
    echo "Setting up $service..."
    cd "$PROJECT_ROOT/services/$service"
    mkdir -p gradle/wrapper

    # Create gradlew if not exists
    if [ ! -f gradlew ]; then
        cat > gradlew << 'GRADLEW'
#!/bin/sh
echo "Downloading Gradle wrapper..."
GRADLE_VERSION=8.5
GRADLE_URL="https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip"
mkdir -p gradle/wrapper
curl -sL "$GRADLE_URL" -o gradle/wrapper/gradle.zip
cd gradle/wrapper && unzip -q gradle.zip && rm gradle.zip
cd ../..
./gradle/wrapper/gradle-${GRADLE_VERSION}/bin/gradle wrapper
./gradlew "$@"
GRADLEW
        chmod +x gradlew
    fi
done

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
cd "$PROJECT_ROOT/frontend"
npm install

# Create .env from example if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Creating .env from .env.example..."
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo "[WARN] Please edit .env file with your API keys!"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys (TikTok, OpenAI, etc.)"
echo "2. Run: docker-compose up -d postgres redis rabbitmq elasticsearch"
echo "3. Wait for services to start, then run: docker-compose up"
echo "4. Open http://localhost:3000 in your browser"
echo ""
