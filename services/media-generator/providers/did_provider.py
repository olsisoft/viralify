"""
D-ID Provider - Generate avatar videos with lip-sync using D-ID API.

D-ID creates realistic talking head videos from:
- Pre-built presenters
- Custom uploaded photos
- Audio input (URL or generated TTS)
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)


class DIDProvider:
    """Provider for D-ID avatar video generation with lip-sync."""

    BASE_URL = "https://api.d-id.com"

    # Status polling configuration
    POLL_INTERVAL = 2.0  # seconds
    MAX_POLL_ATTEMPTS = 150  # 5 minutes max wait

    def __init__(self, api_key: str, output_dir: str = "/tmp/did"):
        """
        Initialize D-ID provider.

        Args:
            api_key: D-ID API key (Basic auth format)
            output_dir: Directory to store downloaded videos
        """
        self.api_key = api_key
        self.output_dir = output_dir
        self.headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        os.makedirs(output_dir, exist_ok=True)

    async def create_talk(
        self,
        source_url: str,
        audio_url: str,
        driver_type: str = "microsoft",
        expression: str = "neutral",
        stitch: bool = True,
        result_format: str = "mp4"
    ) -> str:
        """
        Create a talking head video with lip-sync.

        Args:
            source_url: Presenter image URL or D-ID presenter ID
            audio_url: URL to audio file for lip-sync
            driver_type: Lip-sync driver (microsoft, wav2lip)
            expression: Avatar expression (neutral, happy, serious)
            stitch: Whether to stitch video seamlessly
            result_format: Output format (mp4, webm)

        Returns:
            Talk job ID for status polling
        """
        async with httpx.AsyncClient(timeout=60) as client:
            payload = {
                "source_url": source_url,
                "script": {
                    "type": "audio",
                    "audio_url": audio_url
                },
                "config": {
                    "stitch": stitch,
                    "result_format": result_format,
                    "driver_type": driver_type
                }
            }

            # Add expression if supported
            if expression != "neutral":
                payload["config"]["expression"] = expression

            logger.info(f"Creating D-ID talk with source: {source_url[:50]}...")

            response = await client.post(
                f"{self.BASE_URL}/talks",
                headers=self.headers,
                json=payload
            )

            if response.status_code == 201:
                data = response.json()
                talk_id = data.get("id")
                logger.info(f"D-ID talk created: {talk_id}")
                return talk_id
            else:
                error_msg = response.text
                logger.error(f"D-ID create_talk failed: {response.status_code} - {error_msg}")
                raise RuntimeError(f"D-ID API error: {error_msg}")

    async def create_talk_with_text(
        self,
        source_url: str,
        script_text: str,
        voice_id: str = "en-US-JennyNeural",
        driver_type: str = "microsoft",
        expression: str = "neutral"
    ) -> str:
        """
        Create talk with text-to-speech (D-ID generates audio).

        Args:
            source_url: Presenter image URL or ID
            script_text: Text script for TTS
            voice_id: Microsoft voice ID for TTS
            driver_type: Lip-sync driver type
            expression: Avatar expression

        Returns:
            Talk job ID
        """
        async with httpx.AsyncClient(timeout=60) as client:
            payload = {
                "source_url": source_url,
                "script": {
                    "type": "text",
                    "input": script_text,
                    "provider": {
                        "type": "microsoft",
                        "voice_id": voice_id
                    }
                },
                "config": {
                    "stitch": True,
                    "result_format": "mp4",
                    "driver_type": driver_type
                }
            }

            if expression != "neutral":
                payload["config"]["expression"] = expression

            response = await client.post(
                f"{self.BASE_URL}/talks",
                headers=self.headers,
                json=payload
            )

            if response.status_code == 201:
                return response.json().get("id")
            else:
                raise RuntimeError(f"D-ID API error: {response.text}")

    async def get_talk_status(self, talk_id: str) -> Dict[str, Any]:
        """
        Get the status of a talk generation job.

        Args:
            talk_id: The talk job ID

        Returns:
            Status dict with 'status', 'result_url', etc.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.BASE_URL}/talks/{talk_id}",
                headers=self.headers
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(f"Failed to get talk status: {response.text}")

    async def poll_until_complete(
        self,
        talk_id: str,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Poll for talk completion with status updates.

        Args:
            talk_id: The talk job ID
            callback: Optional callback(status_dict) for progress updates

        Returns:
            Final status dict with result_url
        """
        attempts = 0

        while attempts < self.MAX_POLL_ATTEMPTS:
            status = await self.get_talk_status(talk_id)
            current_status = status.get("status", "unknown")

            logger.debug(f"D-ID talk {talk_id} status: {current_status}")

            if callback:
                callback(status)

            if current_status == "done":
                logger.info(f"D-ID talk completed: {talk_id}")
                return status

            elif current_status == "error":
                error = status.get("error", {})
                error_msg = error.get("description", "Unknown error")
                logger.error(f"D-ID talk failed: {error_msg}")
                raise RuntimeError(f"D-ID generation failed: {error_msg}")

            elif current_status in ["created", "started", "pending"]:
                await asyncio.sleep(self.POLL_INTERVAL)
                attempts += 1

            else:
                logger.warning(f"Unknown D-ID status: {current_status}")
                await asyncio.sleep(self.POLL_INTERVAL)
                attempts += 1

        raise TimeoutError(f"D-ID talk {talk_id} timed out after {attempts} attempts")

    async def generate_avatar_video(
        self,
        source_url: str,
        audio_url: str,
        driver_type: str = "microsoft",
        expression: str = "neutral"
    ) -> Dict[str, Any]:
        """
        Full workflow: create talk and poll until complete.

        Args:
            source_url: Presenter image/ID
            audio_url: Audio URL for lip-sync
            driver_type: Lip-sync driver
            expression: Avatar expression

        Returns:
            Dict with video_url, duration, thumbnail_url
        """
        # Create the talk
        talk_id = await self.create_talk(
            source_url=source_url,
            audio_url=audio_url,
            driver_type=driver_type,
            expression=expression
        )

        # Poll until complete
        result = await self.poll_until_complete(talk_id)

        # Download video locally
        video_url = result.get("result_url")
        if video_url:
            local_path = await self._download_video(video_url, talk_id)
        else:
            local_path = None

        return {
            "job_id": talk_id,
            "video_url": local_path or video_url,
            "remote_url": video_url,
            "duration": result.get("duration", 0),
            "thumbnail_url": result.get("thumbnail_url"),
            "status": "completed"
        }

    async def upload_source_image(self, image_path: str) -> str:
        """
        Upload a custom image to D-ID for use as avatar source.

        Args:
            image_path: Local path to image file

        Returns:
            D-ID source URL for the uploaded image
        """
        async with httpx.AsyncClient(timeout=120) as client:
            with open(image_path, "rb") as f:
                files = {"image": (os.path.basename(image_path), f, "image/png")}

                # Remove Content-Type for multipart
                headers = {k: v for k, v in self.headers.items() if k != "Content-Type"}

                response = await client.post(
                    f"{self.BASE_URL}/images",
                    headers=headers,
                    files=files
                )

            if response.status_code == 201:
                data = response.json()
                source_url = data.get("url")
                logger.info(f"Uploaded custom image: {source_url}")
                return source_url
            else:
                raise RuntimeError(f"Failed to upload image: {response.text}")

    async def list_presenters(self) -> list:
        """
        List available D-ID presenters (pre-built avatars).

        Returns:
            List of presenter dicts with id, preview_url, etc.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.BASE_URL}/clips/presenters",
                headers=self.headers
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("presenters", [])
            else:
                logger.warning(f"Failed to list presenters: {response.text}")
                return []

    async def get_credits(self) -> Dict[str, Any]:
        """Get remaining API credits."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.BASE_URL}/credits",
                headers=self.headers
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.text}

    async def _download_video(self, url: str, talk_id: str) -> str:
        """Download video from D-ID to local storage."""
        import uuid

        filename = f"avatar_{talk_id}_{uuid.uuid4().hex[:6]}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.get(url)

            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                logger.info(f"Downloaded avatar video: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to download video: {response.status_code}")
                return url  # Return remote URL as fallback

    async def delete_talk(self, talk_id: str) -> bool:
        """Delete a talk to free up storage."""
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(
                f"{self.BASE_URL}/talks/{talk_id}",
                headers=self.headers
            )
            return response.status_code == 200
