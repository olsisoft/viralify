# HunyuanVideo-Avatar RunPod Deployment

Deploy Tencent's HunyuanVideo-Avatar on RunPod Serverless for cost-effective full-body avatar animation.

## Cost Comparison

| Provider | Cost/15s Video | Full-Body | Quality |
|----------|---------------|-----------|---------|
| Replicate OmniHuman | $2.80 | ✅ | Best |
| **HunyuanVideo-Avatar** | **$0.10-0.20** | ✅ | Very Good |
| SadTalker | $0.002 | ❌ | Good |

**Savings: 93-96% vs Replicate OmniHuman**

## Deployment Steps

### 1. Build Docker Image

```bash
cd services/media-generator/runpod/hunyuan-avatar

# Build image
docker build -t viralify/hunyuan-avatar:latest .

# Push to Docker Hub (or RunPod registry)
docker push viralify/hunyuan-avatar:latest
```

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: `hunyuan-avatar`
   - **Docker Image**: `viralify/hunyuan-avatar:latest`
   - **GPU Type**: RTX 4090 (24GB) or A100 (40GB)
   - **Max Workers**: 3
   - **Idle Timeout**: 60 seconds
   - **Execution Timeout**: 600 seconds

4. Copy the **Endpoint ID** for configuration

### 3. Configure Viralify

Add to your `.env` or docker-compose environment:

```env
RUNPOD_API_KEY=your_runpod_api_key
HUNYUAN_ENDPOINT_ID=your_endpoint_id
```

### 4. Test the Endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_url": "https://example.com/avatar.png",
      "audio_url": "https://example.com/speech.wav",
      "settings": {
        "infer_steps": 50,
        "cfg_scale": 7.5
      }
    }
  }'
```

## API Reference

### Input

```json
{
  "input": {
    "image_url": "https://...",      // OR image_base64
    "audio_url": "https://...",      // OR audio_base64
    "settings": {
      "image_size": 704,             // Output resolution (704x768)
      "sample_n_frames": 129,        // ~5 seconds at 25fps
      "cfg_scale": 7.5,              // Guidance scale
      "infer_steps": 50,             // Inference steps (30=fast, 50=balanced, 75=quality)
      "use_deepcache": true,         // Enable DeepCache acceleration
      "use_fp8": true                // Enable FP8 for lower VRAM
    }
  }
}
```

### Output

```json
{
  "video_url": "data:video/mp4;base64,...",  // Base64 encoded video
  "duration": 5.16,                           // Video duration in seconds
  "inference_time": 120.5,                    // Processing time in seconds
  "status": "success"
}
```

## Quality Presets

| Preset | Steps | Cost Estimate | Use Case |
|--------|-------|---------------|----------|
| Draft | 30 | ~$0.05 | Quick preview |
| Standard | 50 | ~$0.10 | Production |
| High | 75 | ~$0.20 | Best quality |

## GPU Recommendations

| GPU | VRAM | Cost/hour | Recommended For |
|-----|------|-----------|-----------------|
| RTX 4090 | 24GB | $0.44 | Best value |
| A100 40GB | 40GB | $1.29 | Faster processing |
| A100 80GB | 80GB | $1.89 | Multiple concurrent jobs |

## Troubleshooting

### Out of Memory (OOM)
- Enable `use_fp8: true` in settings
- Reduce `image_size` to 512
- Use GPU with more VRAM

### Slow Generation
- Enable `use_deepcache: true`
- Reduce `infer_steps` to 30
- Use faster GPU (A100)

### Poor Quality
- Increase `infer_steps` to 75
- Increase `cfg_scale` to 8.0
- Use higher resolution input image
