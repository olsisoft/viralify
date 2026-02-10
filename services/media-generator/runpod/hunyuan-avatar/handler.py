"""
HunyuanVideo-Avatar RunPod Serverless Handler

Generates full-body avatar videos from image + audio input.
Optimized for 24GB+ VRAM GPUs (RTX 4090, A100, etc.)
"""

import os
import sys
import uuid
import time
import base64
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import runpod
import requests

# Add HunyuanVideo-Avatar to path
HUNYUAN_PATH = "/app/hunyuan-avatar"
sys.path.insert(0, HUNYUAN_PATH)

# Output directory
OUTPUT_DIR = Path("/tmp/hunyuan_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Inference settings
DEFAULT_SETTINGS = {
    "image_size": 704,           # 704x768 resolution
    "sample_n_frames": 129,      # ~5 seconds at 25fps
    "cfg_scale": 7.5,
    "infer_steps": 50,
    "use_deepcache": True,
    "use_fp8": True,             # Enable FP8 for lower VRAM
}


def download_file(url: str, output_path: str) -> bool:
    """Download file from URL."""
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return False


def upload_to_storage(file_path: str, job_id: str) -> Optional[str]:
    """
    Upload result to cloud storage and return URL.
    For now, we return base64 encoded video.
    In production, upload to S3/GCS and return signed URL.
    """
    try:
        with open(file_path, 'rb') as f:
            video_data = f.read()

        # For small files, return base64
        if len(video_data) < 50 * 1024 * 1024:  # < 50MB
            return f"data:video/mp4;base64,{base64.b64encode(video_data).decode()}"

        # For larger files, you'd upload to S3/GCS here
        # For now, just return the local path (RunPod will handle cleanup)
        return file_path

    except Exception as e:
        print(f"[ERROR] Failed to upload: {e}")
        return None


def create_input_csv(image_path: str, audio_path: str, output_path: str) -> str:
    """Create input CSV for HunyuanVideo-Avatar."""
    csv_path = OUTPUT_DIR / f"input_{uuid.uuid4().hex[:8]}.csv"

    # CSV format expected by HunyuanVideo-Avatar
    csv_content = f"""image_path,audio_path,save_path
{image_path},{audio_path},{output_path}
"""

    with open(csv_path, 'w') as f:
        f.write(csv_content)

    return str(csv_path)


def run_inference(
    image_path: str,
    audio_path: str,
    settings: Dict[str, Any]
) -> Optional[str]:
    """
    Run HunyuanVideo-Avatar inference.
    Returns path to generated video.
    """
    job_id = uuid.uuid4().hex[:8]
    output_video = OUTPUT_DIR / f"avatar_{job_id}.mp4"

    # Create input CSV
    csv_path = create_input_csv(image_path, audio_path, str(output_video))

    # Build command
    cmd = [
        "python3",
        f"{HUNYUAN_PATH}/hymm_sp/sample_gpu_poor.py",
        "--input", csv_path,
        "--save-path", str(OUTPUT_DIR),
        "--image-size", str(settings.get("image_size", 704)),
        "--sample-n-frames", str(settings.get("sample_n_frames", 129)),
        "--cfg-scale", str(settings.get("cfg_scale", 7.5)),
        "--infer-steps", str(settings.get("infer_steps", 50)),
    ]

    # Add optimization flags
    if settings.get("use_fp8", True):
        cmd.append("--use-fp8")

    if settings.get("use_deepcache", True):
        cmd.extend(["--use-deepcache", "1"])

    if settings.get("cpu_offload", False):
        cmd.append("--cpu-offload")

    print(f"[INFO] Running inference: {' '.join(cmd)}")
    start_time = time.time()

    try:
        # Run inference
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        result = subprocess.run(
            cmd,
            cwd=HUNYUAN_PATH,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        elapsed = time.time() - start_time
        print(f"[INFO] Inference completed in {elapsed:.1f}s")

        if result.returncode != 0:
            print(f"[ERROR] Inference failed: {result.stderr}")
            return None

        # Find output video
        if output_video.exists():
            return str(output_video)

        # Search for any generated video
        for video_file in OUTPUT_DIR.glob("*.mp4"):
            return str(video_file)

        print("[ERROR] No output video found")
        return None

    except subprocess.TimeoutExpired:
        print("[ERROR] Inference timed out after 30 minutes")
        return None
    except Exception as e:
        print(f"[ERROR] Inference error: {e}")
        return None


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for HunyuanVideo-Avatar.

    Input:
    {
        "input": {
            "image_url": "https://...",      # Avatar image URL
            "audio_url": "https://...",      # Audio file URL
            "image_base64": "...",           # OR base64 image
            "audio_base64": "...",           # OR base64 audio
            "settings": {                    # Optional settings
                "image_size": 704,
                "sample_n_frames": 129,
                "cfg_scale": 7.5,
                "infer_steps": 50,
                "use_fp8": true
            }
        }
    }

    Output:
    {
        "video_url": "...",                  # Generated video URL/base64
        "duration": 5.16,                    # Video duration in seconds
        "inference_time": 120.5,             # Inference time in seconds
        "status": "success"
    }
    """
    job_id = event.get("id", uuid.uuid4().hex[:8])
    print(f"[INFO] Starting job {job_id}")

    try:
        input_data = event.get("input", {})

        # Get settings
        settings = {**DEFAULT_SETTINGS, **input_data.get("settings", {})}

        # Download/decode image
        image_path = OUTPUT_DIR / f"image_{job_id}.png"
        if "image_url" in input_data:
            if not download_file(input_data["image_url"], str(image_path)):
                return {"error": "Failed to download image", "status": "failed"}
        elif "image_base64" in input_data:
            img_data = base64.b64decode(input_data["image_base64"])
            with open(image_path, 'wb') as f:
                f.write(img_data)
        else:
            return {"error": "No image provided (image_url or image_base64)", "status": "failed"}

        # Download/decode audio
        audio_path = OUTPUT_DIR / f"audio_{job_id}.wav"
        if "audio_url" in input_data:
            if not download_file(input_data["audio_url"], str(audio_path)):
                return {"error": "Failed to download audio", "status": "failed"}
        elif "audio_base64" in input_data:
            audio_data = base64.b64decode(input_data["audio_base64"])
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
        else:
            return {"error": "No audio provided (audio_url or audio_base64)", "status": "failed"}

        print(f"[INFO] Image: {image_path}, Audio: {audio_path}")

        # Run inference
        start_time = time.time()
        video_path = run_inference(str(image_path), str(audio_path), settings)
        inference_time = time.time() - start_time

        if not video_path:
            return {"error": "Inference failed", "status": "failed"}

        # Get video duration
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True
            )
            duration = float(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError, OSError):
            duration = settings["sample_n_frames"] / 25.0  # Estimate

        # Upload/encode result
        video_url = upload_to_storage(video_path, job_id)

        if not video_url:
            return {"error": "Failed to upload video", "status": "failed"}

        # Cleanup temp files
        try:
            os.remove(image_path)
            os.remove(audio_path)
        except OSError:
            pass

        return {
            "video_url": video_url,
            "duration": duration,
            "inference_time": inference_time,
            "settings": settings,
            "status": "success"
        }

    except Exception as e:
        print(f"[ERROR] Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


# RunPod serverless entrypoint
if __name__ == "__main__":
    print("[INFO] Starting HunyuanVideo-Avatar RunPod Handler")
    runpod.serverless.start({"handler": handler})
