# Presentation Generator Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Pillow, fonts, and FFmpeg
RUN apt-get update && apt-get install -y \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fontconfig \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Update font cache
RUN fc-cache -f -v

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create fonts directory and copy system fonts
RUN mkdir -p /app/fonts && \
    cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf /app/fonts/ && \
    cp /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf /app/fonts/ && \
    cp /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf /app/fonts/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8006

# Expose port
EXPOSE 8006

# Health check - tolerant settings for long-running operations
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8006/health || exit 1

# Run the application
CMD ["python", "main.py"]
