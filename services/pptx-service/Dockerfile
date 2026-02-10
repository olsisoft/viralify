# ===========================================
# Viralify PPTX Service - Dockerfile
# ===========================================
# Production-ready Node.js service for PPTX generation
# with LibreOffice for PNG export

# Stage 1: Build
FROM node:20-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci --only=production=false

# Copy source code
COPY src/ ./src/

# Build TypeScript
RUN npm run build

# Stage 2: Production
FROM node:20-slim

LABEL maintainer="Viralify Team"
LABEL description="PPTX Generation Service using PptxGenJS"
LABEL version="1.0.0"

WORKDIR /app

# Install LibreOffice and utilities for PNG conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    # LibreOffice for PPTX to PDF conversion
    libreoffice-impress \
    libreoffice-common \
    # Poppler utilities for PDF to PNG
    poppler-utils \
    # Fonts for better rendering
    fonts-liberation \
    fonts-dejavu \
    fonts-noto \
    fonts-roboto \
    fonts-open-sans \
    # ImageMagick as fallback
    imagemagick \
    # Utilities
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure ImageMagick policy for PDF
RUN sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>/<policy domain="coder" rights="read|write" pattern="PDF" \/>/g' /etc/ImageMagick-6/policy.xml || true

# Create non-root user
RUN groupadd -r viralify && useradd -r -g viralify viralify

# Create directories
RUN mkdir -p /tmp/viralify/pptx /tmp/viralify/pptx-images /var/log/viralify \
    && chown -R viralify:viralify /tmp/viralify /var/log/viralify

# Copy built application
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./

# Set ownership
RUN chown -R viralify:viralify /app

# Switch to non-root user
USER viralify

# Environment variables
ENV NODE_ENV=production
ENV PORT=8013
ENV PPTX_OUTPUT_DIR=/tmp/viralify/pptx
ENV PPTX_IMAGES_DIR=/tmp/viralify/pptx-images
ENV LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8013/health || exit 1

# Expose port
EXPOSE 8013

# Start the service
CMD ["node", "dist/index.js"]
