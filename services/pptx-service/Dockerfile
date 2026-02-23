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

# Install all dependencies (including devDependencies for TypeScript build)
RUN npm ci

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
    # Java runtime required by LibreOffice for javaldx
    default-jre-headless \
    libreoffice-java-common \
    # Poppler utilities for PDF to PNG
    poppler-utils \
    # Fonts for better rendering
    fonts-liberation \
    fonts-dejavu \
    fonts-noto \
    fonts-noto-cjk \
    fonts-roboto \
    fonts-open-sans \
    # Additional fonts for themes (wget needed to download Google Fonts)
    wget \
    unzip \
    fontconfig \
    # ImageMagick as fallback
    imagemagick \
    # Utilities
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Google Fonts used by themes: Poppins, Inter, JetBrains Mono, Montserrat, Raleway, Orbitron, Space Grotesk, Fira Code
RUN mkdir -p /usr/share/fonts/google-fonts \
    && cd /tmp \
    # Poppins (heading font for DARK, LIGHT, GRADIENT themes)
    && wget -q "https://fonts.google.com/download?family=Poppins" -O poppins.zip \
    && unzip -qo poppins.zip -d /usr/share/fonts/google-fonts/ 2>/dev/null || true \
    # Inter (body font for DARK, LIGHT themes)
    && wget -q "https://fonts.google.com/download?family=Inter" -O inter.zip \
    && unzip -qo inter.zip -d /usr/share/fonts/google-fonts/ 2>/dev/null || true \
    # JetBrains Mono (code font for DARK, LIGHT, GRADIENT themes)
    && wget -q "https://fonts.google.com/download?family=JetBrains+Mono" -O jetbrains-mono.zip \
    && unzip -qo jetbrains-mono.zip -d /usr/share/fonts/google-fonts/ 2>/dev/null || true \
    # Montserrat (heading font for GRADIENT theme)
    && wget -q "https://fonts.google.com/download?family=Montserrat" -O montserrat.zip \
    && unzip -qo montserrat.zip -d /usr/share/fonts/google-fonts/ 2>/dev/null || true \
    # Raleway (heading font for OCEAN theme)
    && wget -q "https://fonts.google.com/download?family=Raleway" -O raleway.zip \
    && unzip -qo raleway.zip -d /usr/share/fonts/google-fonts/ 2>/dev/null || true \
    # Orbitron (heading font for NEON theme)
    && wget -q "https://fonts.google.com/download?family=Orbitron" -O orbitron.zip \
    && unzip -qo orbitron.zip -d /usr/share/fonts/google-fonts/ 2>/dev/null || true \
    # Space Grotesk (body font for NEON theme)
    && wget -q "https://fonts.google.com/download?family=Space+Grotesk" -O space-grotesk.zip \
    && unzip -qo space-grotesk.zip -d /usr/share/fonts/google-fonts/ 2>/dev/null || true \
    # Fira Code (code font for CORPORATE, NEON themes)
    && wget -q "https://fonts.google.com/download?family=Fira+Code" -O fira-code.zip \
    && unzip -qo fira-code.zip -d /usr/share/fonts/google-fonts/ 2>/dev/null || true \
    # Source Code Pro (code font for OCEAN theme)
    && wget -q "https://fonts.google.com/download?family=Source+Code+Pro" -O source-code-pro.zip \
    && unzip -qo source-code-pro.zip -d /usr/share/fonts/google-fonts/ 2>/dev/null || true \
    # Roboto Slab (heading font for CORPORATE theme)
    && wget -q "https://fonts.google.com/download?family=Roboto+Slab" -O roboto-slab.zip \
    && unzip -qo roboto-slab.zip -d /usr/share/fonts/google-fonts/ 2>/dev/null || true \
    # Refresh font cache
    && fc-cache -f -v \
    && rm -f /tmp/*.zip

# Configure ImageMagick policy for PDF
RUN sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>/<policy domain="coder" rights="read|write" pattern="PDF" \/>/g' /etc/ImageMagick-6/policy.xml || true

# Create non-root user with home directory
RUN groupadd -r viralify && useradd -r -g viralify -m -d /home/viralify viralify

# Create directories and LibreOffice profile
RUN mkdir -p /tmp/viralify/pptx /tmp/viralify/pptx-images /var/log/viralify \
    # Create LibreOffice user directories to avoid permission errors
    && mkdir -p /home/viralify/.cache/dconf \
    && mkdir -p /home/viralify/.config/libreoffice \
    && chown -R viralify:viralify /tmp/viralify /var/log/viralify /home/viralify

# Copy built application
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./

# Set ownership
RUN chown -R viralify:viralify /app

# Switch to non-root user
USER viralify

# Environment variables
# Set JAVA_HOME for LibreOffice Java integration
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
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
