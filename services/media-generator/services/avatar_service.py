"""
Avatar Service - Manages avatar video generation with D-ID and HeyGen fallback.

Features:
- Predefined avatar gallery
- Custom photo upload for personalized avatars
- Automatic fallback between providers
- Lip-sync with pre-generated voiceovers
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

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
    Uses D-ID as primary provider with HeyGen as fallback.
    """

    def __init__(
        self,
        did_api_key: str,
        heygen_api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        output_dir: str = "/tmp/avatars"
    ):
        """
        Initialize avatar service.

        Args:
            did_api_key: D-ID API key
            heygen_api_key: Optional HeyGen API key for fallback
            config_path: Path to avatars.json config
            output_dir: Directory for generated videos
        """
        self.did = DIDProvider(api_key=did_api_key, output_dir=output_dir)
        self.heygen_api_key = heygen_api_key
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

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
        """Get a specific avatar by ID."""
        # Check custom avatars first
        if user_id and user_id in self._custom_avatars:
            if avatar_id in self._custom_avatars[user_id]:
                return self._custom_avatars[user_id][avatar_id]

        # Check predefined avatars
        avatars = self.get_predefined_avatars()
        for avatar in avatars:
            if avatar.id == avatar_id:
                return avatar

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
            audio_url = request.voiceover_url
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

    async def _generate_with_tts(
        self,
        avatar: PredefinedAvatar,
        request: AvatarVideoRequest
    ) -> AvatarVideoResult:
        """Generate avatar video using D-ID's built-in TTS."""
        try:
            talk_id = await self.did.create_talk_with_text(
                source_url=avatar.did_presenter_id,
                script_text=request.script_text,
                voice_id=request.voice_id or "en-US-JennyNeural",
                driver_type=request.driver_type or "microsoft",
                expression=request.expression or "neutral"
            )

            result = await self.did.poll_until_complete(talk_id)

            return AvatarVideoResult(
                video_url=result.get("result_url", ""),
                provider=AvatarProvider.DID,
                duration=result.get("duration", 0),
                job_id=talk_id,
                status="completed",
                thumbnail_url=result.get("thumbnail_url")
            )

        except Exception as e:
            logger.error(f"D-ID TTS generation failed: {e}")
            raise

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
