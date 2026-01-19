"""
Face Swap Service - Replace faces using Replicate easel/advanced-face-swap model.

Features:
- Swap user's face onto avatar images
- Preserve user's hair or avatar's hair
- Support gender-aware swapping
- High-quality commercial-grade results

Pricing: ~$0.014 per swap
"""

import os
import asyncio
import logging
import httpx
import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class HairSource(str, Enum):
    """Hair preservation option."""
    USER = "user"      # Keep user's hair
    TARGET = "target"  # Keep avatar's hair


class FaceSwapService:
    """
    Face swap service using Replicate API.

    Uses codeplugtech/face-swap model for reliable face replacement.
    Fallback to easel/advanced-face-swap if needed.
    """

    # Primary model - more reliable, CPU-based, ~30 sec, $0.003/run
    MODEL_VERSION = "codeplugtech/face-swap:278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34"
    # Fallback model - higher quality but less reliable
    FALLBACK_MODEL = "easel/advanced-face-swap:602d8c526aca9e5081f0515649ff8998e058cf7e6b9ff32717d25327f18c5145"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("REPLICATE_API_KEY")
        self.base_url = "https://api.replicate.com/v1"
        self.output_dir = Path("/tmp/viralify/face_swap")
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

    def _get_content_type(self, file_path: str) -> str:
        """Get content type from file extension."""
        ext = Path(file_path).suffix.lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        return content_types.get(ext, "image/jpeg")

    def _file_to_data_uri(self, file_path: str) -> str:
        """Convert local file to data URI."""
        content_type = self._get_content_type(file_path)
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:{content_type};base64,{data}"

    async def _prepare_image_url(self, image: str) -> str:
        """
        Prepare image for API - downloads and converts to data URI.

        Args:
            image: Can be URL, local path, or base64 data

        Returns:
            Data URI ready for API (Replicate works best with data URIs)
        """
        # Already a data URI
        if image.startswith("data:"):
            return image

        # Base64 string without data URI prefix
        if len(image) > 500 and "/" not in image and "\\" not in image:
            # Assume it's base64 image data
            return f"data:image/jpeg;base64,{image}"

        # URL - download and convert to data URI (more reliable than passing URLs)
        if image.startswith(("http://", "https://")):
            try:
                import uuid
                temp_path = self.output_dir / f"temp_{uuid.uuid4().hex[:8]}.jpg"

                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.get(image, follow_redirects=True)
                    if response.status_code == 200:
                        with open(temp_path, "wb") as f:
                            f.write(response.content)
                        data_uri = self._file_to_data_uri(str(temp_path))
                        # Clean up temp file
                        temp_path.unlink(missing_ok=True)
                        return data_uri
                    else:
                        logger.warning(f"[FaceSwap] Failed to download image: {response.status_code}")
                        return image  # Fall back to URL
            except Exception as e:
                logger.warning(f"[FaceSwap] Image download error: {e}, using URL directly")
                return image

        # Local file path
        if os.path.exists(image):
            return self._file_to_data_uri(image)

        # Unknown format, return as-is
        logger.warning(f"[FaceSwap] Unknown image format: {image[:50]}...")
        return image

    async def swap_face(
        self,
        target_image: str,
        swap_image: str,
        gender: str = "auto",
        hair_source: HairSource = HairSource.USER,
        upscale: bool = True,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Swap a face onto a target image.

        Args:
            target_image: Avatar image to swap face onto (URL, path, or base64)
            swap_image: User's face image to use (URL, path, or base64)
            gender: "male", "female", or "auto" for automatic detection
            hair_source: HairSource.USER to keep user's hair, HairSource.TARGET for avatar's
            upscale: Apply 2x upscale for better quality
            output_path: Optional path to save result

        Returns:
            Path to face-swapped image or None if failed
        """
        if not self.is_available():
            logger.error("[FaceSwap] Replicate API key not configured")
            return None

        try:
            print("[FaceSwap] Starting face swap...", flush=True)
            print(f"[FaceSwap] Hair source: {hair_source.value}", flush=True)

            # Prepare image URLs
            target_url = await self._prepare_image_url(target_image)
            swap_url = await self._prepare_image_url(swap_image)

            print(f"[FaceSwap] Target: {target_url[:80]}...", flush=True)
            print(f"[FaceSwap] Swap face: {swap_url[:80]}...", flush=True)

            # Build API input for codeplugtech/face-swap model
            # Note: This model uses input_image and swap_image
            # It doesn't support hair_source, gender, or upscale
            api_input = {
                "input_image": target_url,  # Target/destination image
                "swap_image": swap_url       # Source face image
            }

            print(f"[FaceSwap] Using model: codeplugtech/face-swap", flush=True)
            print(f"[FaceSwap] Input keys: {list(api_input.keys())}", flush=True)

            # Create prediction
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.base_url}/predictions",
                    headers=self._get_headers(),
                    json={
                        "version": self.MODEL_VERSION.split(":")[-1],
                        "input": api_input
                    }
                )

                print(f"[FaceSwap] API response status: {response.status_code}", flush=True)
                if response.status_code != 201:
                    print(f"[FaceSwap] Prediction creation failed: {response.text}", flush=True)
                    return None

                prediction = response.json()
                prediction_id = prediction["id"]
                print(f"[FaceSwap] Prediction started: {prediction_id}", flush=True)

                # Poll for completion
                result = await self._poll_prediction(prediction_id)

                if result:
                    print(f"[FaceSwap] Poll result keys: {list(result.keys())}", flush=True)
                    if result.get("output"):
                        output_url = result["output"]
                        print(f"[FaceSwap] Success! Output URL: {output_url[:80]}...", flush=True)
                        return await self._download_result(output_url, output_path)
                    else:
                        print(f"[FaceSwap] No output in result: {result.get('error', 'No error info')}", flush=True)
                else:
                    print("[FaceSwap] No result from poll", flush=True)

                return None

        except Exception as e:
            logger.error(f"[FaceSwap] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _poll_prediction(
        self,
        prediction_id: str,
        timeout: int = 120,
        interval: int = 2
    ) -> Optional[Dict[str, Any]]:
        """Poll prediction until complete or timeout with retry on transient errors."""
        start_time = time.time()
        retry_count = 0
        max_retries = 3

        async with httpx.AsyncClient(timeout=30) as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(
                        f"{self.base_url}/predictions/{prediction_id}",
                        headers=self._get_headers()
                    )

                    if response.status_code != 200:
                        logger.error(f"[FaceSwap] Poll failed: {response.text}")
                        return None

                    prediction = response.json()
                    status = prediction.get("status")
                    retry_count = 0  # Reset on success

                    print(f"[FaceSwap] Status: {status}", flush=True)

                    if status == "succeeded":
                        return prediction
                    elif status == "failed":
                        error = prediction.get("error", "Unknown error")
                        print(f"[FaceSwap] Prediction FAILED: {error}", flush=True)
                        print(f"[FaceSwap] Full prediction: {prediction}", flush=True)
                        return None
                    elif status == "canceled":
                        logger.warning("[FaceSwap] Prediction was canceled")
                        return None

                except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"[FaceSwap] Max retries reached: {e}")
                        return None
                    wait_time = interval * (2 ** retry_count)  # Exponential backoff
                    logger.warning(f"[FaceSwap] Connection error, retry {retry_count}/{max_retries} in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue

                await asyncio.sleep(interval)

        logger.error("[FaceSwap] Prediction timed out")
        return None

    async def _download_result(
        self,
        url: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Download result image from URL."""
        try:
            import uuid

            if not output_path:
                output_path = str(self.output_dir / f"faceswap_{uuid.uuid4().hex[:8]}.png")

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(url, follow_redirects=True)

                if response.status_code == 200:
                    # Ensure directory exists
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"[FaceSwap] Downloaded result: {output_path}")
                    return output_path
                else:
                    logger.error(f"[FaceSwap] Download failed: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"[FaceSwap] Download error: {e}")
            return None

    async def create_custom_avatar(
        self,
        base_avatar_image: str,
        user_face_image: str,
        avatar_name: str,
        user_id: str = "default",
        gender: str = "auto",
        hair_source: HairSource = HairSource.USER
    ) -> Optional[Dict[str, Any]]:
        """
        Create a reusable custom avatar by face-swapping.

        Args:
            base_avatar_image: Base avatar image to use as template
            user_face_image: User's face to swap onto avatar
            avatar_name: Name for the new custom avatar
            user_id: User identifier for storage
            gender: Gender for face swap
            hair_source: Hair preservation option

        Returns:
            Dict with custom_avatar_id, preview_url, and image_path
        """
        import uuid

        # Create output path for custom avatar
        custom_avatar_dir = Path(f"/tmp/viralify/custom_avatars/{user_id}")
        custom_avatar_dir.mkdir(parents=True, exist_ok=True)

        avatar_id = f"custom-{uuid.uuid4().hex[:8]}"
        output_path = str(custom_avatar_dir / f"{avatar_id}.png")

        # Perform face swap
        result_path = await self.swap_face(
            target_image=base_avatar_image,
            swap_image=user_face_image,
            gender=gender,
            hair_source=hair_source,
            upscale=True,
            output_path=output_path
        )

        if not result_path:
            return None

        return {
            "custom_avatar_id": avatar_id,
            "name": avatar_name,
            "image_path": result_path,
            "preview_url": result_path,  # Local path, can be served via API
            "user_id": user_id,
            "base_avatar": base_avatar_image,
            "hair_source": hair_source.value
        }


# Singleton
_face_swap_service = None

def get_face_swap_service() -> FaceSwapService:
    """Get singleton instance of FaceSwapService."""
    global _face_swap_service
    if _face_swap_service is None:
        _face_swap_service = FaceSwapService()
    return _face_swap_service
