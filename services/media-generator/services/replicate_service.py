"""
Replicate Service - Serverless GPU API for avatar animation.

Uses pre-deployed models on Replicate for cost-effective lip-sync:
- SadTalker: High-quality talking head generation
- Wav2Lip: Accurate lip synchronization

Pricing: ~$0.0023/sec (much cheaper than D-ID)
"""

import os
import asyncio
import logging
import httpx
import time
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class ReplicateModel(str, Enum):
    """Available models on Replicate."""
    SADTALKER = "sadtalker"      # Head motion + lip-sync
    WAV2LIP = "wav2lip"          # Accurate lip-sync only
    OMNI_HUMAN = "omni-human"    # Full body animation + lip-sync (best quality)


# Model versions on Replicate (update if newer versions available)
MODEL_VERSIONS = {
    ReplicateModel.SADTALKER: "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376",
    ReplicateModel.WAV2LIP: "devxpy/cog-wav2lip:8d65e3f4f4298520e079198b493c25adfc43c058ffec924f2aefc8010ed25eef",
    ReplicateModel.OMNI_HUMAN: "bytedance/omni-human:566f1b03016969ac39e242c1ae4a39034686ca8850fc3dba83dceaceb96f74b2",
}


class ReplicateService:
    """
    Replicate API service for avatar animation.

    Uses serverless GPU for processing - no local GPU required.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("REPLICATE_API_KEY")
        self.base_url = "https://api.replicate.com/v1"
        self.output_dir = Path("/tmp/viralify/replicate")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """Check if Replicate API is configured."""
        return self.api_key is not None and len(self.api_key) > 0

    def _get_headers(self) -> Dict[str, str]:
        """Get API headers."""
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }

    async def _upload_file_to_url(self, file_path: str) -> Optional[str]:
        """
        Upload a file and get a public URL.
        Uses Replicate's file upload API.
        """
        try:
            # First, create an upload URL
            async with httpx.AsyncClient(timeout=60) as client:
                # Create upload
                create_response = await client.post(
                    f"{self.base_url}/files",
                    headers=self._get_headers(),
                    json={
                        "filename": Path(file_path).name,
                        "content_type": self._get_content_type(file_path)
                    }
                )

                if create_response.status_code != 201:
                    logger.error(f"[Replicate] Failed to create upload: {create_response.text}")
                    return None

                upload_data = create_response.json()
                upload_url = upload_data.get("upload_url")
                file_url = upload_data.get("urls", {}).get("get")

                # Upload the file
                with open(file_path, "rb") as f:
                    file_content = f.read()

                upload_response = await client.put(
                    upload_url,
                    content=file_content,
                    headers={"Content-Type": self._get_content_type(file_path)}
                )

                if upload_response.status_code in [200, 201]:
                    logger.info(f"[Replicate] File uploaded: {file_url}")
                    return file_url
                else:
                    logger.error(f"[Replicate] Upload failed: {upload_response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"[Replicate] Upload error: {e}")
            return None

    def _get_content_type(self, file_path: str) -> str:
        """Get content type from file extension."""
        ext = Path(file_path).suffix.lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".mp4": "video/mp4",
        }
        return content_types.get(ext, "application/octet-stream")

    async def generate_sadtalker(
        self,
        source_image: str,
        audio_path: str,
        output_path: Optional[str] = None,
        preprocess: str = "crop",
        still_mode: bool = False,
        expression_scale: float = 1.0
    ) -> Optional[str]:
        """
        Generate talking head video using SadTalker.

        Args:
            source_image: Path to source face image
            audio_path: Path to audio file
            output_path: Optional output video path
            preprocess: Preprocessing mode ('crop', 'resize', 'full')
            still_mode: If True, minimal head motion
            expression_scale: Expression intensity (0.0-1.5)

        Returns:
            Path to output video or None if failed
        """
        if not self.is_available():
            logger.error("[Replicate] API key not configured")
            return None

        try:
            logger.info("[Replicate] Starting SadTalker generation...")

            # Upload files to get URLs
            image_url = await self._upload_file_to_url(source_image)
            audio_url = await self._upload_file_to_url(audio_path)

            if not image_url or not audio_url:
                # Try using local file paths with data URIs as fallback
                logger.info("[Replicate] Using data URI fallback...")
                image_url = self._file_to_data_uri(source_image)
                audio_url = self._file_to_data_uri(audio_path)

            # Create prediction
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    f"{self.base_url}/predictions",
                    headers=self._get_headers(),
                    json={
                        "version": MODEL_VERSIONS[ReplicateModel.SADTALKER].split(":")[-1],
                        "input": {
                            "source_image": image_url,
                            "driven_audio": audio_url,
                            "preprocess": preprocess,
                            "still_mode": still_mode,
                            "expression_scale": expression_scale,
                            "pose_style": 0,
                            "batch_size": 2,
                            "enhancer": "gfpgan"  # Face enhancement
                        }
                    }
                )

                if response.status_code != 201:
                    logger.error(f"[Replicate] Prediction failed: {response.text}")
                    return None

                prediction = response.json()
                prediction_id = prediction["id"]
                logger.info(f"[Replicate] Prediction started: {prediction_id}")

                # Poll for completion
                result = await self._poll_prediction(prediction_id)

                if result and result.get("output"):
                    video_url = result["output"]
                    # Download the result
                    return await self._download_result(video_url, output_path)

                return None

        except Exception as e:
            logger.error(f"[Replicate] SadTalker error: {e}")
            return None

    async def generate_wav2lip(
        self,
        source_image: str,
        audio_path: str,
        output_path: Optional[str] = None,
        fps: int = 25,
        pads: str = "0 10 0 0"
    ) -> Optional[str]:
        """
        Generate lip-synced video using Wav2Lip.

        Args:
            source_image: Path to source face image/video
            audio_path: Path to audio file
            output_path: Optional output video path
            fps: Output video FPS
            pads: Padding around face (top right bottom left)

        Returns:
            Path to output video or None if failed
        """
        if not self.is_available():
            logger.error("[Replicate] API key not configured")
            return None

        try:
            logger.info("[Replicate] Starting Wav2Lip generation...")

            # Upload files
            image_url = await self._upload_file_to_url(source_image)
            audio_url = await self._upload_file_to_url(audio_path)

            if not image_url or not audio_url:
                logger.info("[Replicate] Using data URI fallback...")
                image_url = self._file_to_data_uri(source_image)
                audio_url = self._file_to_data_uri(audio_path)

            # Create prediction
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    f"{self.base_url}/predictions",
                    headers=self._get_headers(),
                    json={
                        "version": MODEL_VERSIONS[ReplicateModel.WAV2LIP].split(":")[-1],
                        "input": {
                            "face": image_url,
                            "audio": audio_url,
                            "fps": fps,
                            "pads": pads
                        }
                    }
                )

                if response.status_code != 201:
                    logger.error(f"[Replicate] Prediction failed: {response.text}")
                    return None

                prediction = response.json()
                prediction_id = prediction["id"]
                logger.info(f"[Replicate] Prediction started: {prediction_id}")

                # Poll for completion
                result = await self._poll_prediction(prediction_id)

                if result and result.get("output"):
                    video_url = result["output"]
                    return await self._download_result(video_url, output_path)

                return None

        except Exception as e:
            logger.error(f"[Replicate] Wav2Lip error: {e}")
            return None

    async def generate_omni_human(
        self,
        source_image: str,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate full-body animated video using ByteDance OmniHuman.

        This model provides:
        - Natural lip synchronization
        - Full body movement with arm gestures
        - Realistic facial expressions
        - High-quality output

        Note: Audio should be max 15 seconds for best quality.
        Pricing: ~$0.14/sec of output video

        Args:
            source_image: Path/URL to source image (full body preferred)
            audio_path: Path to audio file
            output_path: Optional output video path

        Returns:
            Path to output video or None if failed
        """
        if not self.is_available():
            logger.error("[Replicate] API key not configured")
            return None

        try:
            logger.info("[Replicate] Starting OmniHuman full-body generation...")
            logger.info(f"[Replicate] Source image: {source_image[:80]}...")
            logger.info(f"[Replicate] Audio path: {audio_path}")

            # Handle image - could be URL or local path
            if source_image.startswith(("http://", "https://")):
                image_url = source_image
                logger.info("[Replicate] Using image URL directly")
            else:
                # Local file - use data URI
                logger.info("[Replicate] Converting local image to data URI...")
                image_url = self._file_to_data_uri(source_image)

            # Handle audio - always local path, use data URI
            logger.info("[Replicate] Converting audio to data URI...")
            audio_url = self._file_to_data_uri(audio_path)

            logger.info(f"[Replicate] Image URL: {image_url[:80] if image_url else 'None'}...")
            logger.info(f"[Replicate] Audio URL: {audio_url[:80] if audio_url else 'None'}...")

            # Create prediction
            async with httpx.AsyncClient(timeout=600) as client:  # Longer timeout for full body
                response = await client.post(
                    f"{self.base_url}/predictions",
                    headers=self._get_headers(),
                    json={
                        "version": MODEL_VERSIONS[ReplicateModel.OMNI_HUMAN].split(":")[-1],
                        "input": {
                            "image": image_url,
                            "audio": audio_url
                        }
                    }
                )

                if response.status_code != 201:
                    logger.error(f"[Replicate] OmniHuman prediction failed: {response.text}")
                    return None

                prediction = response.json()
                prediction_id = prediction["id"]
                logger.info(f"[Replicate] OmniHuman prediction started: {prediction_id}")

                # Poll for completion (longer timeout for full body processing)
                result = await self._poll_prediction(prediction_id, timeout=600, interval=3)

                if result and result.get("output"):
                    video_url = result["output"]
                    logger.info(f"[Replicate] OmniHuman succeeded! Downloading...")
                    return await self._download_result(video_url, output_path)

                return None

        except Exception as e:
            logger.error(f"[Replicate] OmniHuman error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def generate_avatar(
        self,
        source_image: str,
        audio_path: str,
        output_path: Optional[str] = None,
        model: ReplicateModel = ReplicateModel.OMNI_HUMAN,  # Default to full body
        **kwargs
    ) -> Optional[str]:
        """
        Generate avatar video using specified model.

        Args:
            source_image: Path to source image
            audio_path: Path to audio file
            output_path: Optional output path
            model: Which model to use:
                - OMNI_HUMAN: Full body animation with arms (best quality)
                - SADTALKER: Head motion + lip-sync
                - WAV2LIP: Lip-sync only
            **kwargs: Additional model-specific parameters

        Returns:
            Path to output video or None if failed
        """
        if model == ReplicateModel.OMNI_HUMAN:
            return await self.generate_omni_human(
                source_image, audio_path, output_path
            )
        elif model == ReplicateModel.SADTALKER:
            return await self.generate_sadtalker(
                source_image, audio_path, output_path, **kwargs
            )
        elif model == ReplicateModel.WAV2LIP:
            return await self.generate_wav2lip(
                source_image, audio_path, output_path, **kwargs
            )
        else:
            logger.error(f"[Replicate] Unknown model: {model}")
            return None

    async def _poll_prediction(
        self,
        prediction_id: str,
        timeout: int = 300,
        interval: int = 2
    ) -> Optional[Dict[str, Any]]:
        """Poll prediction until complete or timeout."""
        start_time = time.time()

        async with httpx.AsyncClient(timeout=30) as client:
            while time.time() - start_time < timeout:
                response = await client.get(
                    f"{self.base_url}/predictions/{prediction_id}",
                    headers=self._get_headers()
                )

                if response.status_code != 200:
                    logger.error(f"[Replicate] Poll failed: {response.text}")
                    return None

                prediction = response.json()
                status = prediction.get("status")

                logger.info(f"[Replicate] Status: {status}")

                if status == "succeeded":
                    return prediction
                elif status == "failed":
                    error = prediction.get("error", "Unknown error")
                    logger.error(f"[Replicate] Prediction failed: {error}")
                    return None
                elif status == "canceled":
                    logger.warning("[Replicate] Prediction was canceled")
                    return None

                await asyncio.sleep(interval)

        logger.error("[Replicate] Prediction timed out")
        return None

    async def _download_result(
        self,
        url: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Download result video from URL."""
        try:
            import uuid

            if not output_path:
                output_path = str(self.output_dir / f"replicate_{uuid.uuid4().hex[:8]}.mp4")

            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.get(url, follow_redirects=True)

                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"[Replicate] Downloaded result: {output_path}")
                    return output_path
                else:
                    logger.error(f"[Replicate] Download failed: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"[Replicate] Download error: {e}")
            return None

    def _file_to_data_uri(self, file_path: str) -> str:
        """Convert file to data URI (fallback for upload failures)."""
        import base64

        content_type = self._get_content_type(file_path)
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()

        return f"data:{content_type};base64,{data}"

    async def check_balance(self) -> Optional[Dict[str, Any]]:
        """Check Replicate account balance/status."""
        if not self.is_available():
            return None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.base_url}/account",
                    headers=self._get_headers()
                )

                if response.status_code == 200:
                    return response.json()

                return None

        except Exception as e:
            logger.error(f"[Replicate] Balance check error: {e}")
            return None


# Singleton
_replicate_service = None

def get_replicate_service() -> ReplicateService:
    global _replicate_service
    if _replicate_service is None:
        _replicate_service = ReplicateService()
    return _replicate_service
