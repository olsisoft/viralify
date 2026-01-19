"""
Voice Cloning Models

Data models for voice cloning, sample management, and voice profile handling.
Phase 4: Voice Cloning feature.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class VoiceProvider(str, Enum):
    """Supported voice cloning providers"""
    ELEVENLABS = "elevenlabs"
    RESEMBLE = "resemble"
    COQUI = "coqui"


class SampleStatus(str, Enum):
    """Voice sample processing status"""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ERROR = "error"


class VoiceProfileStatus(str, Enum):
    """Voice profile status"""
    DRAFT = "draft"           # Collecting samples
    TRAINING = "training"     # Model being trained
    READY = "ready"           # Ready for use
    FAILED = "failed"         # Training failed
    SUSPENDED = "suspended"   # Suspended for policy violation


class VoiceGender(str, Enum):
    """Voice gender classification"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceAge(str, Enum):
    """Voice age classification"""
    YOUNG = "young"       # 18-30
    MIDDLE = "middle"     # 30-50
    MATURE = "mature"     # 50+


class VoiceAccent(str, Enum):
    """Common voice accents"""
    AMERICAN = "american"
    BRITISH = "british"
    AUSTRALIAN = "australian"
    INDIAN = "indian"
    FRENCH = "french"
    SPANISH = "spanish"
    GERMAN = "german"
    OTHER = "other"


# ========================================
# Core Models
# ========================================

class VoiceSample(BaseModel):
    """
    A voice sample uploaded by the user for cloning.
    Multiple samples improve voice quality.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str = Field(..., description="Parent voice profile ID")
    user_id: str = Field(..., description="Owner user ID")

    # File info
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Storage path")
    file_size_bytes: int = Field(default=0)
    duration_seconds: float = Field(default=0.0, description="Audio duration")
    format: str = Field(default="mp3", description="Audio format")

    # Quality metrics
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI-assessed quality")
    noise_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Background noise level")
    clarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Speech clarity")

    # Validation
    status: SampleStatus = Field(default=SampleStatus.PENDING)
    rejection_reason: Optional[str] = Field(None)

    # Transcript (for training)
    transcript: Optional[str] = Field(None, description="What was said in the sample")
    is_transcript_verified: bool = Field(default=False)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(None)


class VoiceProfile(BaseModel):
    """
    A user's cloned voice profile.
    Contains metadata and references to the trained model.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="Owner user ID")

    # Profile info
    name: str = Field(..., min_length=1, max_length=100, description="Voice profile name")
    description: Optional[str] = Field(None, max_length=500)

    # Voice characteristics
    gender: VoiceGender = Field(default=VoiceGender.NEUTRAL)
    age: VoiceAge = Field(default=VoiceAge.MIDDLE)
    accent: VoiceAccent = Field(default=VoiceAccent.AMERICAN)
    language: str = Field(default="en", description="Primary language code")

    # Samples
    samples: List[VoiceSample] = Field(default_factory=list)
    total_sample_duration: float = Field(default=0.0, description="Total duration of all samples")

    # Provider info
    provider: VoiceProvider = Field(default=VoiceProvider.ELEVENLABS)
    provider_voice_id: Optional[str] = Field(None, description="Voice ID from provider")
    provider_model_id: Optional[str] = Field(None, description="Model ID if applicable")

    # Status
    status: VoiceProfileStatus = Field(default=VoiceProfileStatus.DRAFT)
    training_progress: float = Field(default=0.0, ge=0.0, le=100.0)
    error_message: Optional[str] = Field(None)

    # Settings
    default_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="Voice stability")
    default_similarity: float = Field(default=0.75, ge=0.0, le=1.0, description="Similarity boost")
    default_style: float = Field(default=0.0, ge=0.0, le=1.0, description="Style exaggeration")

    # Consent & Legal
    consent_given: bool = Field(default=False, description="User confirmed voice ownership")
    consent_timestamp: Optional[datetime] = Field(None)
    consent_ip_address: Optional[str] = Field(None)

    # Usage tracking
    total_characters_generated: int = Field(default=0)
    total_generations: int = Field(default=0)
    last_used_at: Optional[datetime] = Field(None)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    trained_at: Optional[datetime] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "voice-001",
                "user_id": "user-123",
                "name": "My Professional Voice",
                "gender": "male",
                "status": "ready",
                "provider": "elevenlabs",
                "provider_voice_id": "abc123xyz"
            }
        }


class VoiceGenerationSettings(BaseModel):
    """Settings for generating speech with a cloned voice"""
    stability: float = Field(default=0.5, ge=0.0, le=1.0, description="Voice stability (higher = more consistent)")
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0, description="How similar to original voice")
    style: float = Field(default=0.0, ge=0.0, le=1.0, description="Style exaggeration")
    use_speaker_boost: bool = Field(default=True, description="Enhance speaker clarity")

    # Speed & emotion
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    emotion: Optional[str] = Field(None, description="Emotion hint: neutral, happy, sad, angry, excited")

    # Output format
    output_format: str = Field(default="mp3_44100_128", description="Output audio format")


# ========================================
# Request/Response Models
# ========================================

class CreateVoiceProfileRequest(BaseModel):
    """Request to create a new voice profile"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    gender: VoiceGender = Field(default=VoiceGender.NEUTRAL)
    age: VoiceAge = Field(default=VoiceAge.MIDDLE)
    accent: VoiceAccent = Field(default=VoiceAccent.AMERICAN)
    language: str = Field(default="en")
    provider: VoiceProvider = Field(default=VoiceProvider.ELEVENLABS)


class CreateVoiceProfileResponse(BaseModel):
    """Response after creating a voice profile"""
    profile_id: str
    name: str
    status: VoiceProfileStatus
    message: str
    min_samples_required: int = Field(default=1)
    min_duration_seconds: int = Field(default=30)


class UploadSampleResponse(BaseModel):
    """Response after uploading a voice sample"""
    sample_id: str
    profile_id: str
    duration_seconds: float
    quality_score: Optional[float]
    status: SampleStatus
    message: str
    total_duration: float = Field(description="Total sample duration for profile")
    can_start_training: bool = Field(description="Whether enough samples for training")


class StartTrainingRequest(BaseModel):
    """Request to start voice model training"""
    profile_id: str
    consent_confirmed: bool = Field(..., description="User confirms voice ownership")


class StartTrainingResponse(BaseModel):
    """Response after starting training"""
    profile_id: str
    status: VoiceProfileStatus
    estimated_time_seconds: int
    message: str


class GenerateClonedSpeechRequest(BaseModel):
    """Request to generate speech with cloned voice"""
    profile_id: str
    text: str = Field(..., min_length=1, max_length=5000)
    settings: Optional[VoiceGenerationSettings] = Field(default_factory=VoiceGenerationSettings)


class GenerateClonedSpeechResponse(BaseModel):
    """Response after generating cloned speech"""
    audio_url: str
    duration_seconds: float
    characters_used: int
    profile_id: str


class VoiceProfileListResponse(BaseModel):
    """Response listing voice profiles"""
    profiles: List[VoiceProfile]
    total: int


class VoiceProfileDetailResponse(BaseModel):
    """Detailed voice profile response"""
    profile: VoiceProfile
    samples: List[VoiceSample]
    can_train: bool
    training_requirements: Dict[str, Any]


class PreviewVoiceRequest(BaseModel):
    """Request to preview a voice with sample text"""
    profile_id: str
    text: str = Field(
        default="Hello! This is a preview of my cloned voice. How does it sound?",
        min_length=10,
        max_length=500
    )
    settings: Optional[VoiceGenerationSettings] = Field(default_factory=VoiceGenerationSettings)


class VoiceSampleRequirements(BaseModel):
    """Requirements for voice samples"""
    min_samples: int = Field(default=1, description="Minimum number of samples")
    max_samples: int = Field(default=25, description="Maximum number of samples")
    min_duration_seconds: int = Field(default=30, description="Minimum total duration")
    max_duration_seconds: int = Field(default=180, description="Maximum total duration for best results")
    ideal_duration_seconds: int = Field(default=60, description="Ideal duration for good quality")
    supported_formats: List[str] = Field(default=["mp3", "wav", "m4a", "ogg", "webm"])
    max_file_size_mb: int = Field(default=50)
    sample_rate_hz: int = Field(default=44100, description="Recommended sample rate")
    tips: List[str] = Field(default=[
        "Record in a quiet environment",
        "Speak clearly and naturally",
        "Avoid background music or noise",
        "Use a good quality microphone",
        "Read diverse content (not just one phrase)",
        "Include emotional range if possible"
    ])


# ========================================
# Consent Model
# ========================================

class VoiceConsentRecord(BaseModel):
    """Record of user consent for voice cloning"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    profile_id: str

    # Consent details
    consent_text: str = Field(
        default="I confirm that I am the owner of this voice or have explicit permission "
                "from the voice owner to create a cloned voice. I understand that this "
                "voice clone will be used for content generation on the Viralify platform."
    )
    consent_given: bool = Field(default=False)

    # Verification
    ip_address: Optional[str] = Field(None)
    user_agent: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Legal
    terms_version: str = Field(default="1.0")
    accepted_terms: bool = Field(default=False)
