"""
First Order Motion Model (FOMM) Service - Local body animation.
Animates a source image using motion from a driving video.

Model: https://github.com/AliaksandrSiarohin/first-order-model
"""

import os
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional, List
import httpx

logger = logging.getLogger(__name__)


class FOMMService:
    """
    Local body/gesture animation using First Order Motion Model.

    FOMM transfers motion from a driving video to a source image:
    - Source: Static image of person/avatar
    - Driver: Video with desired movements
    - Output: Animated video of source with driver's movements

    Great for adding natural body gestures and movements.
    """

    MODEL_DIR = "/app/models/fomm"
    CHECKPOINT_FILE = "vox-cpk.pth.tar"

    # Pre-recorded driving videos for common gestures
    DRIVING_VIDEOS = {
        "talking": "driving_talking.mp4",
        "presenting": "driving_presenting.mp4",
        "nodding": "driving_nodding.mp4",
        "gesturing": "driving_gesturing.mp4",
        "neutral": "driving_neutral.mp4"
    }

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or self.MODEL_DIR)
        self.checkpoint_path = self.model_dir / self.CHECKPOINT_FILE
        self.config_path = self.model_dir / "vox-256.yaml"
        self.driving_dir = self.model_dir / "driving_videos"
        self.output_dir = Path("/tmp/viralify/fomm")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """Check if FOMM model is available."""
        return self.checkpoint_path.exists() and self.config_path.exists()

    def get_driving_video(self, gesture_type: str = "talking") -> Optional[str]:
        """Get path to a pre-recorded driving video for the given gesture type."""
        video_name = self.DRIVING_VIDEOS.get(gesture_type, self.DRIVING_VIDEOS["talking"])
        video_path = self.driving_dir / video_name

        if video_path.exists():
            return str(video_path)

        # Try to find any available driving video
        if self.driving_dir.exists():
            for video in self.driving_dir.glob("*.mp4"):
                return str(video)

        return None

    async def download_models(self) -> bool:
        """Download FOMM model and config if not present."""
        if self.is_available():
            logger.info("FOMM model already exists")
            return True

        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # Model URLs
            checkpoint_url = "https://github.com/AliaksandrSiarohin/first-order-model/releases/download/v1.0.0/vox-cpk.pth.tar"
            config_url = "https://raw.githubusercontent.com/AliaksandrSiarohin/first-order-model/master/config/vox-256.yaml"

            async with httpx.AsyncClient(timeout=600) as client:
                # Download checkpoint (large file ~700MB)
                if not self.checkpoint_path.exists():
                    logger.info(f"Downloading FOMM checkpoint...")
                    response = await client.get(checkpoint_url, follow_redirects=True)
                    if response.status_code == 200:
                        with open(self.checkpoint_path, "wb") as f:
                            f.write(response.content)
                        logger.info(f"FOMM checkpoint downloaded")
                    else:
                        logger.error(f"Failed to download checkpoint: {response.status_code}")
                        return False

                # Download config
                if not self.config_path.exists():
                    logger.info(f"Downloading FOMM config...")
                    response = await client.get(config_url)
                    if response.status_code == 200:
                        with open(self.config_path, "w") as f:
                            f.write(response.text)
                        logger.info(f"FOMM config downloaded")

            return self.is_available()

        except Exception as e:
            logger.error(f"Error downloading FOMM model: {e}")
            return False

    async def animate(
        self,
        source_image: str,
        driving_video: Optional[str] = None,
        gesture_type: str = "talking",
        output_path: Optional[str] = None,
        relative: bool = True,
        adapt_scale: bool = True
    ) -> Optional[str]:
        """
        Animate a source image using motion from a driving video.

        Args:
            source_image: Path to source image (person/avatar)
            driving_video: Path to driving video (or use built-in gesture)
            gesture_type: Type of gesture if no driving_video provided
            output_path: Optional output path
            relative: Use relative keypoint displacement
            adapt_scale: Adapt movement scale to source

        Returns:
            Path to animated video, or None if failed
        """
        if not self.is_available():
            logger.warning("FOMM model not available")
            return None

        try:
            import uuid

            # Get driving video
            if not driving_video:
                driving_video = self.get_driving_video(gesture_type)
                if not driving_video:
                    logger.error("No driving video available")
                    return None

            # Generate output path
            if not output_path:
                output_filename = f"fomm_{uuid.uuid4().hex[:8]}.mp4"
                output_path = str(self.output_dir / output_filename)

            logger.info(f"[FOMM] Animating source image...")
            logger.info(f"[FOMM] Source: {source_image}")
            logger.info(f"[FOMM] Driver: {driving_video}")

            # Run FOMM inference
            cmd = [
                "python", "/app/fomm/demo.py",
                "--config", str(self.config_path),
                "--checkpoint", str(self.checkpoint_path),
                "--source_image", source_image,
                "--driving_video", driving_video,
                "--result_video", output_path
            ]

            if relative:
                cmd.append("--relative")
            if adapt_scale:
                cmd.append("--adapt_scale")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/app/fomm" if os.path.exists("/app/fomm") else None
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and os.path.exists(output_path):
                logger.info(f"[FOMM] Success: {output_path}")
                return output_path
            else:
                logger.error(f"[FOMM] Failed: {stderr.decode()}")
                return None

        except Exception as e:
            logger.error(f"[FOMM] Error: {e}")
            return None

    async def create_driving_video_from_audio(
        self,
        audio_path: str,
        duration: Optional[float] = None
    ) -> Optional[str]:
        """
        Create a synthetic driving video based on audio intensity.
        This generates subtle head movements synchronized with speech.

        For real body movements, use pre-recorded driving videos.
        """
        try:
            import uuid

            output_path = self.output_dir / f"driver_{uuid.uuid4().hex[:8]}.mp4"

            # Get audio duration if not provided
            if not duration:
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path
                ]
                process = await asyncio.create_subprocess_exec(
                    *probe_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await process.communicate()
                duration = float(stdout.decode().strip()) if stdout else 10

            # Create a simple driving video with subtle movements
            # This uses FFmpeg to generate a moving pattern that FOMM can use
            # In production, you'd want pre-recorded driving videos

            logger.info(f"[FOMM] Creating synthetic driver video ({duration}s)")

            # For now, return None - use pre-recorded drivers instead
            # Synthetic drivers don't produce good results
            return None

        except Exception as e:
            logger.error(f"[FOMM] Driver creation error: {e}")
            return None


# Singleton
_fomm_service = None

def get_fomm_service() -> FOMMService:
    global _fomm_service
    if _fomm_service is None:
        _fomm_service = FOMMService()
    return _fomm_service
