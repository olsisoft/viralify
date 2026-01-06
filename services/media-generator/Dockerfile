FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for video/image processing, fonts, diagrams and Node.js for Mermaid
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

# Create directories for temp files
RUN mkdir -p /app/temp /app/assets /tmp/mermaid /tmp/diagrams /tmp/avatars /tmp/visuals

EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
