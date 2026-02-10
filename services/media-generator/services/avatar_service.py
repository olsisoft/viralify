"""
Avatar Service - Manages avatar video generation with Replicate and D-ID fallback.

Features:
- Predefined avatar gallery
- Custom photo upload for personalized avatars
- HYBRID approach: Replicate (cheap) → D-ID (fallback)
- ElevenLabs TTS for voiceover generation
- Lip-sync with pre-generated voiceovers
"""

import os
import json
import logging
import tempfile
from typing import Optional, List, Dict, Any
from pathlib import Path
import httpx

from models.avatar_models import (
    AvatarStyle,
    AvatarGender,
    AvatarProvider,
    PredefinedAvatar,
    AvatarVideoRequest,
    AvatarVideoResult,
    CustomAvatarRequest,
    CustomAvatarResult,
    AvatarGalleryResponse,
)
from providers.did_provider import DIDProvider

logger = logging.getLogger(__name__)


class AvatarService:
    """
    Service for managing avatar video generation.
    Uses HYBRID approach: Replicate (cheap) → D-ID (fallback).
    TTS: ElevenLabs for high-quality voiceovers.
    """

    def __init__(
        self,
        did_api_key: str,
        heygen_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        replicate_api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        output_dir: str = "/tmp/avatars"
    ):
        """
        Initialize avatar service.

        Args:
            did_api_key: D-ID API key
            heygen_api_key: Optional HeyGen API key for fallback
            elevenlabs_api_key: ElevenLabs API key for TTS
            replicate_api_key: Replicate API key for serverless GPU
            config_path: Path to avatars.json config
            output_dir: Directory for generated videos
        """
        self.did = DIDProvider(api_key=did_api_key, output_dir=output_dir)
        self.heygen_api_key = heygen_api_key
        self.elevenlabs_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.replicate_key = replicate_api_key or os.getenv("REPLICATE_API_KEY", "")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Lazy-loaded services
        self._local_avatar_service = None

        # Load avatar gallery
        if config_path:
            self.config_path = config_path
        else:
            # Default to config/avatars.json relative to this file
            self.config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config",
                "avatars.json"
            )

        self._avatars_cache = None
        self._custom_avatars: Dict[str, Dict[str, PredefinedAvatar]] = {}  # user_id -> {avatar_id -> avatar}

    @property
    def local_avatar_service(self):
        """Lazy load LocalAvatarService for Replicate/D-ID hybrid processing."""
        if self._local_avatar_service is None:
            from services.local_avatar_service import LocalAvatarService
            self._local_avatar_service = LocalAvatarService(
                did_api_key=self.did.api_key,
                replicate_api_key=self.replicate_key,
                output_dir=self.output_dir
            )
        return self._local_avatar_service

    def _load_config(self) -> Dict[str, Any]:
        """Load avatar configuration from JSON file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Avatar config not found: {self.config_path}")
            return {"avatars": [], "default_avatar_id": None}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid avatar config JSON: {e}")
            return {"avatars": [], "default_avatar_id": None}

    def get_predefined_avatars(
        self,
        style: Optional[AvatarStyle] = None,
        gender: Optional[AvatarGender] = None,
        include_premium: bool = True
    ) -> List[PredefinedAvatar]:
        """
        Get list of predefined avatars with optional filtering.

        Args:
            style: Filter by style (professional, casual, creative)
            gender: Filter by gender
            include_premium: Whether to include premium avatars

        Returns:
            List of matching PredefinedAvatar objects
        """
        if self._avatars_cache is None:
            config = self._load_config()
            self._avatars_cache = [
                PredefinedAvatar(**avatar_data)
                for avatar_data in config.get("avatars", [])
            ]

        avatars = self._avatars_cache

        # Apply filters
        if style:
            avatars = [a for a in avatars if a.style == style]

        if gender:
            avatars = [a for a in avatars if a.gender == gender]

        if not include_premium:
            avatars = [a for a in avatars if not a.is_premium]

        return avatars

    def get_avatar_gallery(
        self,
        user_id: Optional[str] = None,
        style: Optional[AvatarStyle] = None,
        gender: Optional[AvatarGender] = None
    ) -> AvatarGalleryResponse:
        """
        Get the full avatar gallery response with user's custom avatars.

        Args:
            user_id: User ID to include their custom avatars
            style: Optional style filter
            gender: Optional gender filter

        Returns:
            AvatarGalleryResponse with all available avatars
        """
        # Get predefined avatars
        avatars = self.get_predefined_avatars(style=style, gender=gender)

        # Add user's custom avatars
        if user_id and user_id in self._custom_avatars:
            custom = list(self._custom_avatars[user_id].values())
            if style:
                custom = [a for a in custom if a.style == style]
            if gender:
                custom = [a for a in custom if a.gender == gender]
            avatars = custom + avatars  # Custom avatars first

        return AvatarGalleryResponse(
            avatars=avatars,
            total_count=len(avatars),
            styles=[s.value for s in AvatarStyle],
            genders=[g.value for g in AvatarGender]
        )

    def get_avatar_by_id(self, avatar_id: str, user_id: Optional[str] = None) -> Optional[PredefinedAvatar]:
        """Get a specific avatar by ID or name (flexible matching)."""
        # Check custom avatars first
        if user_id and user_id in self._custom_avatars:
            if avatar_id in self._custom_avatars[user_id]:
                return self._custom_avatars[user_id][avatar_id]

        # Check predefined avatars
        avatars = self.get_predefined_avatars()
        avatar_id_lower = avatar_id.lower()

        # 1. Exact ID match
        for avatar in avatars:
            if avatar.id == avatar_id:
                return avatar

        # 2. Partial ID match (e.g., "sarah" matches "avatar-pro-female-sarah")
        for avatar in avatars:
            if avatar_id_lower in avatar.id.lower():
                logger.info(f"Matched avatar '{avatar.id}' via partial ID match for '{avatar_id}'")
                return avatar

        # 3. Name match (e.g., "emma" matches "Emma - Friendly Guide")
        for avatar in avatars:
            name_lower = avatar.name.lower()
            # Check if search term is in the name
            if avatar_id_lower in name_lower:
                logger.info(f"Matched avatar '{avatar.id}' via name match for '{avatar_id}'")
                return avatar

        # 4. No match found - log available avatars
        available_ids = [a.id for a in avatars]
        logger.warning(f"Avatar '{avatar_id}' not found. Available avatars: {available_ids[:5]}...")
        return None

    def get_default_avatar(self) -> PredefinedAvatar:
        """Get the default avatar."""
        config = self._load_config()
        default_id = config.get("default_avatar_id")

        if default_id:
            avatar = self.get_avatar_by_id(default_id)
            if avatar:
                return avatar

        # Fallback to first avatar
        avatars = self.get_predefined_avatars()
        if avatars:
            return avatars[0]

        # Last resort - create a basic avatar
        return PredefinedAvatar(
            id="default-avatar",
            name="Default Presenter",
            preview_url="",
            did_presenter_id="amy-jcwCkr1grs",
            style=AvatarStyle.PROFESSIONAL,
            gender=AvatarGender.FEMALE
        )

    async def generate_avatar_video(
        self,
        request: AvatarVideoRequest,
        user_id: Optional[str] = None
    ) -> AvatarVideoResult:
        """
        Generate an avatar video with lip-sync.

        Args:
            request: Avatar video generation request
            user_id: User ID for custom avatar lookup

        Returns:
            AvatarVideoResult with video URL and metadata
        """
        # Get avatar details
        avatar = self.get_avatar_by_id(request.avatar_id, user_id)
        if not avatar:
            avatar = self.get_default_avatar()
            logger.warning(f"Avatar {request.avatar_id} not found, using default")

        # Determine audio source
        if request.voiceover_url:
            # Need to upload audio to D-ID if it's a local/internal URL
            audio_url = await self._prepare_audio_for_did(request.voiceover_url)
        elif request.script_text and request.voice_id:
            # D-ID will generate TTS internally
            return await self._generate_with_tts(avatar, request)
        else:
            raise ValueError("Either voiceover_url or (script_text + voice_id) required")

        # Try D-ID first
        try:
            result = await self.did.generate_avatar_video(
                source_url=avatar.did_presenter_id,
                audio_url=audio_url,
                driver_type=request.driver_type or "microsoft",
                expression=request.expression or "neutral"
            )

            return AvatarVideoResult(
                video_url=result["video_url"],
                provider=AvatarProvider.DID,
                duration=result.get("duration", 0),
                job_id=result["job_id"],
                status="completed",
                thumbnail_url=result.get("thumbnail_url")
            )

        except Exception as e:
            logger.error(f"D-ID generation failed: {e}")

            # Try HeyGen fallback if available
            if self.heygen_api_key and avatar.heygen_avatar_id:
                try:
                    return await self._fallback_heygen(avatar, audio_url, request)
                except Exception as he:
                    logger.error(f"HeyGen fallback also failed: {he}")

            # Re-raise original error
            raise RuntimeError(f"Avatar generation failed: {e}")

    async def _prepare_audio_for_did(self, audio_url: str) -> str:
        """
        Prepare audio for D-ID by uploading if it's a local/internal URL.
        D-ID requires publicly accessible HTTPS URLs.

        Args:
            audio_url: The audio URL (could be local file path or internal URL)

        Returns:
            D-ID compatible audio URL
        """
        # If it's already a public HTTPS URL, return as is
        if audio_url.startswith("https://"):
            return audio_url

        # If it's a local file path, upload directly
        if audio_url.startswith("/") and os.path.exists(audio_url):
            logger.info(f"Uploading local audio file to D-ID: {audio_url}")
            return await self.did.upload_audio(audio_url)

        # If it's an internal HTTP URL (like localhost), download then upload
        if audio_url.startswith("http://"):
            logger.info(f"Downloading audio from internal URL: {audio_url}")
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.get(audio_url)
                    if response.status_code != 200:
                        raise RuntimeError(f"Failed to download audio: {response.status_code}")

                    # Save to temp file
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        f.write(response.content)
                        temp_path = f.name

                    # Upload to D-ID
                    logger.info(f"Uploading downloaded audio to D-ID: {temp_path}")
                    did_url = await self.did.upload_audio(temp_path)

                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

                    return did_url

            except Exception as e:
                logger.error(f"Failed to prepare audio for D-ID: {e}")
                raise RuntimeError(f"Failed to prepare audio: {e}")

        # Unknown format, return as is and hope for the best
        logger.warning(f"Unknown audio URL format, passing to D-ID as-is: {audio_url}")
        return audio_url

    async def _generate_with_tts(
        self,
        avatar: PredefinedAvatar,
        request: AvatarVideoRequest
    ) -> AvatarVideoResult:
        """
        Generate avatar video using ElevenLabs TTS + Replicate/D-ID hybrid lip-sync.

        Flow:
        1. Generate voiceover with ElevenLabs
        2. Use LocalAvatarService (Replicate → D-ID fallback) for lip-sync
        """
        import uuid
        job_id = f"avatar-{uuid.uuid4().hex[:8]}"

        try:
            # Step 1: Generate voiceover with ElevenLabs
            logger.info(f"[AvatarService] Generating TTS with ElevenLabs for voice {request.voice_id}...")
            audio_path = await self._generate_elevenlabs_tts(
                text=request.script_text,
                voice_id=request.voice_id or "21m00Tcm4TlvDq8ikWAM"  # Default: Rachel
            )

            if not audio_path:
                raise RuntimeError("Failed to generate voiceover with ElevenLabs")

            logger.info(f"[AvatarService] TTS generated: {audio_path}")

            # Step 2: Use LocalAvatarService (Replicate → D-ID) for lip-sync
            from services.local_avatar_service import AnimationProvider

            # Get avatar provider preference from environment
            avatar_provider_env = os.getenv("AVATAR_PROVIDER", "hybrid").lower()
            provider_map = {
                "hybrid": AnimationProvider.HYBRID,
                "replicate": AnimationProvider.REPLICATE,
                "d-id": AnimationProvider.DID,
            }
            selected_provider = provider_map.get(avatar_provider_env, AnimationProvider.HYBRID)

            # Get quality mode from request (default to 'final' for best quality)
            quality_mode = getattr(request, 'quality', 'final')
            logger.info(f"[AvatarService] Generating lip-sync with {selected_provider.value} provider (quality={quality_mode})...")

            result = await self.local_avatar_service.generate_avatar_video(
                source_image=avatar.did_presenter_id,  # Avatar image URL
                audio_path=audio_path,
                provider=selected_provider,
                gesture_type="talking",
                remove_background=False,  # Avatar images already have proper background
                quality=quality_mode
            )

            if result["status"] != "completed":
                raise RuntimeError(f"Lip-sync generation failed: {result.get('error', 'Unknown error')}")

            # Map provider to AvatarProvider enum
            provider_used = result.get("provider_used", "replicate")
            avatar_provider = AvatarProvider.DID if provider_used == "d-id" else AvatarProvider.DID  # Use DID as placeholder

            return AvatarVideoResult(
                video_url=result["video_url"],
                provider=avatar_provider,
                duration=result.get("duration", 0),
                job_id=job_id,
                status="completed",
                thumbnail_url=None
            )

        except Exception as e:
            logger.error(f"[AvatarService] Avatar generation failed: {e}")
            raise

    async def _generate_elevenlabs_tts(
        self,
        text: str,
        voice_id: str
    ) -> Optional[str]:
        """
        Generate voiceover using ElevenLabs API.

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID

        Returns:
            Path to generated audio file, or None if failed
        """
        if not self.elevenlabs_key:
            logger.error("[AvatarService] ElevenLabs API key not configured")
            return None

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": self.elevenlabs_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75
                        }
                    }
                )

                if response.status_code == 200:
                    import uuid
                    audio_dir = Path(self.output_dir) / "audio"
                    audio_dir.mkdir(parents=True, exist_ok=True)

                    audio_path = audio_dir / f"tts_{uuid.uuid4().hex[:8]}.mp3"
                    with open(audio_path, "wb") as f:
                        f.write(response.content)

                    logger.info(f"[AvatarService] ElevenLabs TTS saved to {audio_path}")
                    return str(audio_path)
                else:
                    logger.error(f"[AvatarService] ElevenLabs error: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"[AvatarService] ElevenLabs TTS failed: {e}")
            return None

    async def _fallback_heygen(
        self,
        avatar: PredefinedAvatar,
        audio_url: str,
        request: AvatarVideoRequest
    ) -> AvatarVideoResult:
        """Fallback to HeyGen for avatar generation."""
        # HeyGen implementation would go here
        # For now, raise an error as it's not fully implemented
        raise NotImplementedError("HeyGen fallback not yet implemented")

    async def create_custom_avatar(
        self,
        request: CustomAvatarRequest
    ) -> CustomAvatarResult:
        """
        Create a custom avatar from user's uploaded photo.

        Args:
            request: Custom avatar creation request

        Returns:
            CustomAvatarResult with the new avatar
        """
        try:
            # Upload image to D-ID
            source_url = await self.did.upload_source_image(request.photo_url)

            # Create avatar object
            import uuid
            avatar_id = f"custom-{request.user_id}-{uuid.uuid4().hex[:8]}"

            avatar = PredefinedAvatar(
                id=avatar_id,
                name=request.name,
                preview_url=request.photo_url,
                did_presenter_id=source_url,
                style=request.style,
                gender=AvatarGender.NEUTRAL,  # User can update if needed
                description=f"Custom avatar for user {request.user_id}",
                is_premium=False
            )

            # Store in cache
            if request.user_id not in self._custom_avatars:
                self._custom_avatars[request.user_id] = {}
            self._custom_avatars[request.user_id][avatar_id] = avatar

            return CustomAvatarResult(
                avatar=avatar,
                provider_source_id=source_url,
                processing_status="ready"
            )

        except Exception as e:
            logger.error(f"Custom avatar creation failed: {e}")
            raise

    async def delete_custom_avatar(self, avatar_id: str, user_id: str) -> bool:
        """Delete a user's custom avatar."""
        if user_id in self._custom_avatars:
            if avatar_id in self._custom_avatars[user_id]:
                del self._custom_avatars[user_id][avatar_id]
                return True
        return False

    async def get_generation_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of an avatar generation job."""
        try:
            status = await self.did.get_talk_status(job_id)
            return {
                "job_id": job_id,
                "status": status.get("status", "unknown"),
                "progress": self._estimate_progress(status.get("status", "")),
                "result_url": status.get("result_url"),
                "error": status.get("error")
            }
        except Exception as e:
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }

    def _estimate_progress(self, status: str) -> int:
        """Estimate progress percentage from status."""
        progress_map = {
            "created": 10,
            "pending": 20,
            "started": 40,
            "processing": 60,
            "rendering": 80,
            "done": 100,
            "error": 0
        }
        return progress_map.get(status, 50)

    async def check_credits(self) -> Dict[str, Any]:
        """Check remaining API credits."""
        return await self.did.get_credits()
