"""
HunyuanVideo-Avatar Service

Full-body avatar animation using Tencent's HunyuanVideo-Avatar model
deployed on RunPod Serverless.

Cost: ~$0.10-0.20 per 15s video (vs $2.80 on Replicate OmniHuman)
"""

import os
import asyncio
import logging
import base64
import time
import httpx
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class HunyuanQuality(str, Enum):
    """Quality presets for HunyuanVideo-Avatar."""
    DRAFT = "draft"       # Fast, lower quality (30 steps)
    STANDARD = "standard" # Balanced (50 steps)
    HIGH = "high"         # Best quality (75 steps)


# Quality preset configurations
QUALITY_PRESETS = {
    HunyuanQuality.DRAFT: {
        "infer_steps": 30,
        "cfg_scale": 6.0,
        "use_deepcache": True,
        "use_fp8": True,
    },
    HunyuanQuality.STANDARD: {
        "infer_steps": 50,
        "cfg_scale": 7.5,
        "use_deepcache": True,
        "use_fp8": True,
    },
    HunyuanQuality.HIGH: {
        "infer_steps": 75,
        "cfg_scale": 8.0,
        "use_deepcache": False,
        "use_fp8": False,
    },
}

# Cost estimates per 15s video (RunPod RTX 4090)
COST_ESTIMATES = {
    HunyuanQuality.DRAFT: 0.05,
    HunyuanQuality.STANDARD: 0.10,
    HunyuanQuality.HIGH: 0.20,
}


class HunyuanAvatarService:
    """
    Service for generating full-body avatar videos using HunyuanVideo-Avatar
    deployed on RunPod Serverless.
    """

    def __init__(
        self,
        runpod_api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        output_dir: str = "/tmp/viralify/hunyuan"
    ):
        self.api_key = runpod_api_key or os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.getenv("HUNYUAN_ENDPOINT_ID")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # RunPod API endpoints
        self.base_url = "https://api.runpod.ai/v2"

    def is_available(self) -> bool:
        """Check if service is configured."""
        return bool(self.api_key and self.endpoint_id)

    async def generate_avatar_video(
        self,
        image_path: str,
        audio_path: str,
        quality: HunyuanQuality = HunyuanQuality.STANDARD,
        output_path: Optional[str] = None,
        timeout: int = 600
    ) -> Dict[str, Any]:
        """
        Generate full-body avatar video with lip-sync.

        Args:
            image_path: Path to avatar image (full-body recommended)
            audio_path: Path to audio file for lip-sync
            quality: Quality preset (draft/standard/high)
            output_path: Optional output video path
            timeout: Max wait time in seconds

        Returns:
            Dict with video_url, duration, cost_estimate, status
        """
        result = {
            "video_url": None,
            "duration": 0,
            "cost_estimate": COST_ESTIMATES.get(quality, 0.10),
            "inference_time": 0,
            "status": "pending",
            "error": None,
            "provider": "hunyuan-avatar"
        }

        if not self.is_available():
            result["status"] = "failed"
            result["error"] = "HunyuanVideo-Avatar service not configured"
            return result

        try:
            # Read and encode files
            image_base64 = await self._encode_file(image_path)
            audio_base64 = await self._encode_file(audio_path)

            if not image_base64 or not audio_base64:
                raise ValueError("Failed to encode input files")

            # Get quality settings
            settings = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[HunyuanQuality.STANDARD])

            # Submit job to RunPod
            logger.info(f"[HunyuanAvatar] Submitting job (quality={quality.value})...")
            job_id = await self._submit_job(image_base64, audio_base64, settings)

            if not job_id:
                raise RuntimeError("Failed to submit job to RunPod")

            logger.info(f"[HunyuanAvatar] Job submitted: {job_id}")

            # Poll for completion
            start_time = time.time()
            job_result = await self._poll_job(job_id, timeout)
            elapsed = time.time() - start_time

            if not job_result:
                raise RuntimeError("Job timed out or failed")

            if job_result.get("status") == "failed":
                raise RuntimeError(job_result.get("error", "Unknown error"))

            # Get video from result
            video_data = job_result.get("video_url")
            if not video_data:
                raise RuntimeError("No video in job result")

            # Save video
            if not output_path:
                import uuid
                output_path = str(self.output_dir / f"hunyuan_{uuid.uuid4().hex[:8]}.mp4")

            await self._save_video(video_data, output_path)

            result["video_url"] = output_path
            result["duration"] = job_result.get("duration", 0)
            result["inference_time"] = job_result.get("inference_time", elapsed)
            result["status"] = "completed"

            logger.info(f"[HunyuanAvatar] Video generated in {elapsed:.1f}s: {output_path}")

        except Exception as e:
            logger.error(f"[HunyuanAvatar] Error: {e}")
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    async def _encode_file(self, file_path: str) -> Optional[str]:
        """Read and base64 encode a file."""
        try:
            # Handle URL
            if file_path.startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.get(file_path)
                    response.raise_for_status()
                    return base64.b64encode(response.content).decode()

            # Handle local file
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        except Exception as e:
            logger.error(f"[HunyuanAvatar] Failed to encode {file_path}: {e}")
            return None

    async def _submit_job(
        self,
        image_base64: str,
        audio_base64: str,
        settings: Dict[str, Any]
    ) -> Optional[str]:
        """Submit job to RunPod endpoint."""
        url = f"{self.base_url}/{self.endpoint_id}/run"

        payload = {
            "input": {
                "image_base64": image_base64,
                "audio_base64": audio_base64,
                "settings": settings
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data.get("id")

        except Exception as e:
            logger.error(f"[HunyuanAvatar] Failed to submit job: {e}")
            return None

    async def _poll_job(
        self,
        job_id: str,
        timeout: int = 600
    ) -> Optional[Dict[str, Any]]:
        """Poll RunPod job until completion."""
        url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        start_time = time.time()
        poll_interval = 5  # seconds

        async with httpx.AsyncClient(timeout=30) as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                    status = data.get("status")
                    logger.debug(f"[HunyuanAvatar] Job {job_id} status: {status}")

                    if status == "COMPLETED":
                        return data.get("output", {})

                    if status in ["FAILED", "CANCELLED"]:
                        error = data.get("error", "Job failed")
                        return {"status": "failed", "error": error}

                    # Still running, wait and poll again
                    await asyncio.sleep(poll_interval)

                except Exception as e:
                    logger.warning(f"[HunyuanAvatar] Poll error: {e}")
                    await asyncio.sleep(poll_interval)

        return None

    async def _save_video(self, video_data: str, output_path: str):
        """Save video from base64 or URL."""
        try:
            if video_data.startswith("data:video"):
                # Base64 encoded
                base64_data = video_data.split(",", 1)[1]
                video_bytes = base64.b64decode(base64_data)
                with open(output_path, "wb") as f:
                    f.write(video_bytes)

            elif video_data.startswith(("http://", "https://")):
                # URL - download
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.get(video_data)
                    response.raise_for_status()
                    with open(output_path, "wb") as f:
                        f.write(response.content)

            else:
                # Assume it's a local path or raw base64
                video_bytes = base64.b64decode(video_data)
                with open(output_path, "wb") as f:
                    f.write(video_bytes)

        except Exception as e:
            logger.error(f"[HunyuanAvatar] Failed to save video: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check RunPod endpoint health."""
        if not self.is_available():
            return {"status": "not_configured"}

        url = f"{self.base_url}/{self.endpoint_id}/health"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, headers=headers)
                data = response.json()
                return {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "workers": data.get("workers", {}),
                    "jobs_in_queue": data.get("jobsInQueue", 0)
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton
_hunyuan_service = None


def get_hunyuan_avatar_service() -> HunyuanAvatarService:
    global _hunyuan_service
    if _hunyuan_service is None:
        _hunyuan_service = HunyuanAvatarService()
    return _hunyuan_service
