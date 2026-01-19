"""
Avatar models for D-ID and HeyGen avatar video generation.
"""

from enum import Enum
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List


class AvatarStyle(str, Enum):
    """Visual styles for avatar presenters."""
    PROFESSIONAL = "professional"  # Business, corporate look
    CASUAL = "casual"              # Relaxed, friendly appearance
    CREATIVE = "creative"          # Artistic, unique styles


class AvatarGender(str, Enum):
    """Gender options for avatar filtering."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class AvatarProvider(str, Enum):
    """Supported avatar video providers."""
    DID = "d-id"         # Primary provider
    HEYGEN = "heygen"    # Fallback provider


class PredefinedAvatar(BaseModel):
    """A predefined avatar from the gallery."""
    id: str = Field(description="Unique avatar identifier")
    name: str = Field(description="Display name for the avatar")
    preview_url: str = Field(description="URL to avatar preview image")
    did_presenter_id: str = Field(
        description="D-ID presenter/source ID for Talks API (face animation only)"
    )
    did_clip_presenter_id: Optional[str] = Field(
        default=None,
        description="D-ID presenter ID for Clips API (full body movement)"
    )
    heygen_avatar_id: Optional[str] = Field(
        default=None,
        description="HeyGen avatar ID for fallback"
    )
    style: AvatarStyle = Field(description="Avatar visual style")
    gender: AvatarGender = Field(description="Avatar gender")
    description: Optional[str] = Field(
        default=None,
        description="Brief description of the avatar"
    )
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Languages this avatar supports well"
    )
    is_premium: bool = Field(
        default=False,
        description="Whether this is a premium avatar"
    )


class AvatarVideoRequest(BaseModel):
    """Request to generate an avatar video with lip-sync."""
    avatar_id: str = Field(description="Avatar ID from gallery or custom upload")
    script_text: Optional[str] = Field(
        default=None,
        description="Text script for TTS (if no voiceover provided)"
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="Voice ID for TTS generation"
    )
    voiceover_url: Optional[str] = Field(
        default=None,
        description="Pre-generated voiceover audio URL"
    )
    output_format: str = Field(
        default="9:16",
        description="Output aspect ratio (9:16, 16:9, 1:1)"
    )
    background_color: Optional[str] = Field(
        default=None,
        description="Background color (hex) or 'transparent'"
    )
    expression: Optional[str] = Field(
        default="neutral",
        description="Avatar expression: neutral, happy, serious"
    )
    driver_type: Optional[str] = Field(
        default="microsoft",
        description="D-ID driver type for lip-sync"
    )
    quality: str = Field(
        default="final",
        description="Quality mode: 'draft' (~$0.002, SadTalker), 'preview' (~$0.01), 'final' (~$2.80, OmniHuman)"
    )


class AvatarVideoResult(BaseModel):
    """Result of avatar video generation."""
    video_url: str = Field(description="URL to the generated avatar video")
    provider: AvatarProvider = Field(description="Provider that generated the video")
    duration: float = Field(description="Video duration in seconds")
    job_id: str = Field(description="Provider job ID for status tracking")
    status: str = Field(
        default="completed",
        description="Generation status: pending, processing, completed, failed"
    )
    thumbnail_url: Optional[str] = Field(
        default=None,
        description="URL to video thumbnail"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if generation failed"
    )
    cost_estimate: Optional[float] = Field(
        default=None,
        description="Estimated cost in USD for this generation"
    )
    quality: Optional[str] = Field(
        default=None,
        description="Quality mode used: draft, preview, or final"
    )


class CustomAvatarRequest(BaseModel):
    """Request to create a custom avatar from user photo."""
    photo_url: str = Field(description="URL to the uploaded user photo")
    user_id: str = Field(description="User ID for ownership tracking")
    name: str = Field(description="Display name for the custom avatar")
    style: AvatarStyle = Field(
        default=AvatarStyle.PROFESSIONAL,
        description="Desired avatar style"
    )


class CustomAvatarResult(BaseModel):
    """Result of custom avatar creation."""
    avatar: PredefinedAvatar = Field(description="The created avatar object")
    provider_source_id: str = Field(
        description="D-ID source ID for the custom photo"
    )
    processing_status: str = Field(
        default="ready",
        description="Processing status: processing, ready, failed"
    )


class AvatarGalleryResponse(BaseModel):
    """Response containing the avatar gallery."""
    avatars: List[PredefinedAvatar] = Field(
        description="List of available avatars"
    )
    total_count: int = Field(description="Total number of avatars")
    styles: List[str] = Field(
        default_factory=lambda: [s.value for s in AvatarStyle],
        description="Available styles for filtering"
    )
    genders: List[str] = Field(
        default_factory=lambda: [g.value for g in AvatarGender],
        description="Available genders for filtering"
    )
