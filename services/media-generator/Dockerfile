FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for video/image processing, fonts, diagrams and Node.js for Mermaid
# Added: git, wget for model downloads; libopenblas for numpy/torch; libportaudio for audio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    fonts-dejavu-core \
    fonts-liberation \
    fonts-noto \
    fontconfig \
    curl \
    gnupg \
    chromium \
    graphviz \
    libgraphviz-dev \
    git \
    wget \
    libopenblas-dev \
    libportaudio2 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f -v

# Install Node.js for Mermaid CLI
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Mermaid CLI globally
RUN npm install -g @mermaid-js/mermaid-cli

# Set Puppeteer to use system Chromium
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for temp files and models
RUN mkdir -p /app/temp /app/assets /tmp/mermaid /tmp/diagrams /tmp/avatars /tmp/visuals \
    /app/models/wav2lip /app/models/fomm /app/models/fomm/driving_videos \
    /tmp/viralify/avatars /tmp/viralify/wav2lip /tmp/viralify/fomm

# Make setup script executable
RUN chmod +x /app/scripts/setup_local_avatar.sh 2>/dev/null || true

EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
