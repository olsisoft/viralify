#!/bin/bash

echo "Starting HunyuanVideo-Avatar RunPod Handler..."

# Set environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Check GPU
nvidia-smi

# Start handler
python -u /app/handler.py
