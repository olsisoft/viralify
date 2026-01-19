"""
Wav2Lip Service - Local lip-sync generation using Wav2Lip model.
Generates realistic lip movements synchronized with audio.

Model: https://github.com/Rudrabha/Wav2Lip
"""

import os
import asyncio
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import httpx

logger = logging.getLogger(__name__)


class Wav2LipService:
    """
    Local lip-sync generation using Wav2Lip.

    Wav2Lip generates accurate lip movements from:
    - A face image or video
    - An audio file

    Output: Video with synchronized lip movements
    """

    # Model paths
    MODEL_DIR = "/app/models/wav2lip"
    WAV2LIP_MODEL = "wav2lip_gan.pth"

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or self.MODEL_DIR)
        self.model_path = self.model_dir / self.WAV2LIP_MODEL
        self.output_dir = Path("/tmp/viralify/wav2lip")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model_loaded = False

    def is_available(self) -> bool:
        """Check if Wav2Lip model is available."""
        return self.model_path.exists()

    async def download_model(self) -> bool:
        """Download Wav2Lip model if not present."""
        if self.is_available():
            logger.info("Wav2Lip model already exists")
            return True

        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # Model URL (from official repo releases)
            model_url = "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth"

            logger.info(f"Downloading Wav2Lip model from {model_url}...")

            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.get(model_url, follow_redirects=True)
                if response.status_code == 200:
                    with open(self.model_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Wav2Lip model downloaded to {self.model_path}")
                    return True
                else:
                    logger.error(f"Failed to download model: {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Error downloading Wav2Lip model: {e}")
            return False

    async def generate_lipsync(
        self,
        face_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        resize_factor: int = 1,
        fps: int = 25
    ) -> Optional[str]:
        """
        Generate lip-synced video from face image/video and audio.

        Args:
            face_path: Path to face image or video
            audio_path: Path to audio file (wav, mp3)
            output_path: Optional output path (auto-generated if None)
            resize_factor: Resize factor for faster processing
            fps: Output video FPS

        Returns:
            Path to generated video, or None if failed
        """
        if not self.is_available():
            logger.warning("Wav2Lip model not available, attempting download...")
            if not await self.download_model():
                logger.error("Cannot generate lip-sync: model not available")
                return None

        try:
            import uuid

            # Generate output path if not provided
            if not output_path:
                output_filename = f"wav2lip_{uuid.uuid4().hex[:8]}.mp4"
                output_path = str(self.output_dir / output_filename)

            # Prepare paths
            face_path = str(face_path)
            audio_path = str(audio_path)

            logger.info(f"[Wav2Lip] Generating lip-sync...")
            logger.info(f"[Wav2Lip] Face: {face_path}")
            logger.info(f"[Wav2Lip] Audio: {audio_path}")

            # Run Wav2Lip inference
            # Using subprocess to call the inference script
            cmd = [
                "python", "/app/wav2lip/inference.py",
                "--checkpoint_path", str(self.model_path),
                "--face", face_path,
                "--audio", audio_path,
                "--outfile", output_path,
                "--resize_factor", str(resize_factor),
                "--fps", str(fps),
                "--nosmooth"  # Disable smoothing for faster processing
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/app/wav2lip" if os.path.exists("/app/wav2lip") else None
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0 and os.path.exists(output_path):
                logger.info(f"[Wav2Lip] Success: {output_path}")
                return output_path
            else:
                logger.error(f"[Wav2Lip] Failed: {stderr.decode()}")
                return None

        except Exception as e:
            logger.error(f"[Wav2Lip] Error: {e}")
            return None

    async def generate_lipsync_simple(
        self,
        face_path: str,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Simplified lip-sync using FFmpeg-based approach when Wav2Lip model unavailable.
        This creates a basic talking effect by overlaying audio on a static/animated face.

        For better results, use generate_lipsync() with the actual model.
        """
        try:
            import uuid

            if not output_path:
                output_filename = f"lipsync_simple_{uuid.uuid4().hex[:8]}.mp4"
                output_path = str(self.output_dir / output_filename)

            # Get audio duration
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

            # Check if face is image or video
            is_image = face_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))

            if is_image:
                # Create video from image with Ken Burns effect (slight zoom/pan for life)
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", face_path,
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-tune", "stillimage",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-vf", f"scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.0005,1.2)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(duration*25)}:s=1080x1920:fps=25",
                    "-shortest",
                    "-t", str(duration),
                    output_path
                ]
            else:
                # Combine existing video with new audio
                cmd = [
                    "ffmpeg", "-y",
                    "-i", face_path,
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    output_path
                ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if process.returncode == 0 and os.path.exists(output_path):
                logger.info(f"[Wav2Lip-Simple] Created: {output_path}")
                return output_path

            return None

        except Exception as e:
            logger.error(f"[Wav2Lip-Simple] Error: {e}")
            return None


# Singleton
_wav2lip_service = None

def get_wav2lip_service() -> Wav2LipService:
    global _wav2lip_service
    if _wav2lip_service is None:
        _wav2lip_service = Wav2LipService()
    return _wav2lip_service
