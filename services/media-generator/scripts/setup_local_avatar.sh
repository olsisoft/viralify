#!/bin/bash
# Setup script for local avatar processing (Wav2Lip + FOMM)
# Run this inside the container after build

set -e

echo "=========================================="
echo "Setting up Local Avatar Processing"
echo "=========================================="

# Create directories
mkdir -p /app/models/wav2lip
mkdir -p /app/models/fomm
mkdir -p /app/models/fomm/driving_videos
mkdir -p /tmp/viralify/avatars
mkdir -p /tmp/viralify/wav2lip
mkdir -p /tmp/viralify/fomm

# Clone Wav2Lip repository
if [ ! -d "/app/wav2lip" ]; then
    echo ""
    echo "Cloning Wav2Lip repository..."
    git clone https://github.com/Rudrabha/Wav2Lip.git /app/wav2lip
    cd /app/wav2lip
    pip install -r requirements.txt --quiet
    echo "Wav2Lip setup complete"
else
    echo "Wav2Lip already exists"
fi

# Clone FOMM repository
if [ ! -d "/app/fomm" ]; then
    echo ""
    echo "Cloning First Order Motion Model repository..."
    git clone https://github.com/AliaksandrSiarohin/first-order-model.git /app/fomm
    cd /app/fomm
    pip install -r requirements.txt --quiet 2>/dev/null || true
    echo "FOMM setup complete"
else
    echo "FOMM already exists"
fi

# Download face detection model for Wav2Lip
FACE_DETECT_MODEL="/app/wav2lip/face_detection/detection/sfd/s3fd.pth"
if [ ! -f "$FACE_DETECT_MODEL" ]; then
    echo ""
    echo "Downloading face detection model for Wav2Lip..."
    mkdir -p /app/wav2lip/face_detection/detection/sfd
    wget -q --show-progress -O "$FACE_DETECT_MODEL" \
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" || \
    echo "Warning: Face detection model download failed (optional)"
fi

# Download models
echo ""
echo "Downloading AI models..."
cd /app
python scripts/download_models.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Model locations:"
echo "  Wav2Lip: /app/models/wav2lip/"
echo "  FOMM:    /app/models/fomm/"
echo ""
echo "Note: Add driving videos to /app/models/fomm/driving_videos/"
echo "=========================================="
