# HunyuanVideo-Avatar RunPod Serverless Dockerfile
# Optimized for RTX 4090 / A100 (24GB+ VRAM)

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install ninja packaging wheel

# Clone HunyuanVideo-Avatar
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar.git /app/hunyuan-avatar

WORKDIR /app/hunyuan-avatar

# Install requirements
RUN pip install -r requirements.txt

# Install Flash Attention
RUN pip install flash-attn==2.6.3 --no-build-isolation

# Install RunPod SDK and Hugging Face Hub
RUN pip install runpod requests boto3 huggingface_hub

# Download model weights (this will be cached in the Docker image)
# For faster builds, you can mount weights as a network volume instead
RUN mkdir -p weights && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='tencent/HunyuanVideo-Avatar', local_dir='./weights', local_dir_use_symlinks=False)"

# Copy our handler
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app

# Expose port for health checks
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Start the RunPod handler
CMD ["python", "-u", "/app/handler.py"]
