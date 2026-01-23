# Visual Generator Service Dockerfile
# Isolated service for diagram generation with heavy dependencies

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - graphviz: Required by Python Diagrams library
# - fonts: For proper text rendering in diagrams
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    fonts-liberation \
    fonts-dejavu-core \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI and Uvicorn (not in original requirements)
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /tmp/viralify/diagrams

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/tmp/viralify/diagrams

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
