# Presentation Generator Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Pillow, fonts, FFmpeg, Graphviz, and CairoSVG
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fontconfig \
    ffmpeg \
    curl \
    graphviz \
    graphviz-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Update font cache
RUN fc-cache -f -v

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Patch viralify-diagrams bugs:
# 1. Set -> set for Python 3.9+ compatibility
RUN sed -i 's/-> Set\[/-> set[/g' /usr/local/lib/python3.11/site-packages/viralify_diagrams/layouts/edge_styler.py 2>/dev/null || true

# 2. Fix missing exports and add aliases in taxonomy/__init__.py
RUN echo "" >> /usr/local/lib/python3.11/site-packages/viralify_diagrams/taxonomy/__init__.py && \
    echo "# Additional exports from submodules" >> /usr/local/lib/python3.11/site-packages/viralify_diagrams/taxonomy/__init__.py && \
    echo "from viralify_diagrams.taxonomy.router import SlideConfig" >> /usr/local/lib/python3.11/site-packages/viralify_diagrams/taxonomy/__init__.py && \
    echo "from viralify_diagrams.taxonomy.slide_optimizer import OptimizationResult" >> /usr/local/lib/python3.11/site-packages/viralify_diagrams/taxonomy/__init__.py && \
    echo "# Backwards compatibility aliases" >> /usr/local/lib/python3.11/site-packages/viralify_diagrams/taxonomy/__init__.py && \
    echo "TargetAudience = AudienceType" >> /usr/local/lib/python3.11/site-packages/viralify_diagrams/taxonomy/__init__.py && \
    echo "DiagramComplexity = ComplexityLevel" >> /usr/local/lib/python3.11/site-packages/viralify_diagrams/taxonomy/__init__.py

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
