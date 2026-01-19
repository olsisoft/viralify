"""
Background Remover Service - AI-based background removal for avatar images.
Uses rembg for high-quality background removal before sending to D-ID.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class BackgroundRemoverService:
    """Service for removing backgrounds from avatar images using AI."""

    def __init__(self, output_dir: str = "/tmp/viralify/avatars"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rembg_session = None

    def _get_rembg_session(self):
        """Lazy load rembg session to avoid import overhead."""
        if self._rembg_session is None:
            try:
                from rembg import new_session
                # Use u2net model for best quality
                self._rembg_session = new_session("u2net")
                logger.info("Initialized rembg session with u2net model")
            except Exception as e:
                logger.warning(f"Failed to initialize rembg: {e}")
                self._rembg_session = "failed"
        return self._rembg_session if self._rembg_session != "failed" else None

    async def remove_background(
        self,
        image_source: str,
        output_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Remove background from an image (URL or local path) and return path to transparent PNG.

        Args:
            image_source: URL or local path to the source image
            output_filename: Optional filename for output (auto-generated if None)

        Returns:
            Path to the transparent PNG, or None if failed
        """
        try:
            from rembg import remove
            from PIL import Image
            import io
            import hashlib

            # Generate output filename from source hash if not provided
            if not output_filename:
                source_hash = hashlib.md5(image_source.encode()).hexdigest()[:12]
                output_filename = f"avatar_nobg_{source_hash}.png"

            output_path = self.output_dir / output_filename

            # Check if already processed
            if output_path.exists():
                logger.info(f"Using cached background-removed image: {output_path}")
                return str(output_path)

            # Get image data - handle both URL and local path
            if image_source.startswith(("http://", "https://")):
                # Download from URL
                logger.info(f"Downloading avatar image from: {image_source[:80]}...")
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.get(image_source)
                    if response.status_code != 200:
                        logger.error(f"Failed to download image: {response.status_code}")
                        return None
                    image_data = response.content
            else:
                # Read from local file
                logger.info(f"Reading local image: {image_source}")
                if not os.path.exists(image_source):
                    logger.error(f"Local image file not found: {image_source}")
                    return None
                with open(image_source, "rb") as f:
                    image_data = f.read()

            # Process in thread pool to avoid blocking
            def process_image():
                # Open image
                input_image = Image.open(io.BytesIO(image_data))

                # Get rembg session for better performance
                session = self._get_rembg_session()

                # Remove background
                if session:
                    output_image = remove(input_image, session=session)
                else:
                    output_image = remove(input_image)

                # Save as PNG with transparency
                output_image.save(output_path, "PNG")
                return str(output_path)

            # Run in executor to not block event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, process_image)

            logger.info(f"Background removed successfully: {result}")
            return result

        except ImportError as e:
            logger.error(f"rembg not installed: {e}")
            return None
        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            return None

    async def process_avatar_for_did(
        self,
        avatar_url: str,
        upload_to_did: bool = True,
        did_api_key: Optional[str] = None
    ) -> str:
        """
        Process avatar image for D-ID: remove background and optionally upload.

        Args:
            avatar_url: Original avatar image URL
            upload_to_did: Whether to upload to D-ID after processing
            did_api_key: D-ID API key for upload

        Returns:
            URL to use with D-ID (either local path or D-ID uploaded URL)
        """
        # Remove background
        processed_path = await self.remove_background(avatar_url)

        if not processed_path:
            logger.warning("Background removal failed, using original image")
            return avatar_url

        # If upload requested and API key provided, upload to D-ID
        if upload_to_did and did_api_key:
            try:
                uploaded_url = await self._upload_to_did(processed_path, did_api_key)
                if uploaded_url:
                    return uploaded_url
            except Exception as e:
                logger.error(f"Failed to upload to D-ID: {e}")

        # Return local path (D-ID can't use local paths, so return original if upload failed)
        # For now, return the processed path - the caller should handle upload
        return processed_path

    async def _upload_to_did(self, image_path: str, api_key: str) -> Optional[str]:
        """Upload processed image to D-ID."""
        async with httpx.AsyncClient(timeout=120) as client:
            with open(image_path, "rb") as f:
                files = {"image": (os.path.basename(image_path), f, "image/png")}
                headers = {
                    "Authorization": f"Basic {api_key}",
                    "Accept": "application/json"
                }

                response = await client.post(
                    "https://api.d-id.com/images",
                    headers=headers,
                    files=files
                )

            if response.status_code == 201:
                data = response.json()
                url = data.get("url")
                logger.info(f"Uploaded processed avatar to D-ID: {url}")
                return url
            else:
                logger.error(f"D-ID upload failed: {response.status_code} - {response.text}")
                return None


# Singleton instance
_bg_remover = None


def get_background_remover() -> BackgroundRemoverService:
    """Get the background remover singleton."""
    global _bg_remover
    if _bg_remover is None:
        _bg_remover = BackgroundRemoverService()
    return _bg_remover
