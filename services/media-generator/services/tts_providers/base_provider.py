"""
Base TTS Provider

Abstract base class for all TTS providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import os


class TTSProviderType(str, Enum):
    """Available TTS providers"""
    CHATTERBOX = "chatterbox"
    KOKORO = "kokoro"
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    PIPER = "piper"


class VoiceGender(str, Enum):
    """Voice gender options"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


@dataclass
class TTSConfig:
    """Configuration for TTS generation"""
    text: str
    language: str = "en"
    voice_id: Optional[str] = None
    voice_gender: VoiceGender = VoiceGender.NEUTRAL
    speed: float = 1.0
    # Voice cloning options
    clone_audio_path: Optional[str] = None
    clone_audio_bytes: Optional[bytes] = None
    # Provider-specific options
    emotion: Optional[str] = None
    exaggeration: float = 0.5
    temperature: float = 0.8
    # Output options
    output_format: str = "mp3"
    sample_rate: int = 24000


@dataclass
class TTSResult:
    """Result from TTS generation"""
    success: bool
    audio_data: Optional[bytes] = None
    duration_seconds: float = 0.0
    sample_rate: int = 24000
    provider_used: Optional[TTSProviderType] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceInfo:
    """Information about an available voice"""
    voice_id: str
    name: str
    provider: TTSProviderType
    language: str
    gender: VoiceGender
    supports_cloning: bool = False
    description: Optional[str] = None
    preview_url: Optional[str] = None


class BaseTTSProvider(ABC):
    """Abstract base class for TTS providers"""

    def __init__(self, provider_type: TTSProviderType):
        self.provider_type = provider_type
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return self.provider_type.value

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and properly configured"""
        pass

    @abstractmethod
    async def generate(self, config: TTSConfig) -> TTSResult:
        """Generate TTS audio from text"""
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        pass

    @abstractmethod
    def get_available_voices(self, language: Optional[str] = None) -> List[VoiceInfo]:
        """Get list of available voices, optionally filtered by language"""
        pass

    @abstractmethod
    def supports_voice_cloning(self) -> bool:
        """Check if this provider supports voice cloning"""
        pass

    def get_default_voice(self, language: str, gender: VoiceGender = VoiceGender.NEUTRAL) -> Optional[str]:
        """Get default voice ID for a language and gender"""
        voices = self.get_available_voices(language)
        if not voices:
            return None

        # Try to find matching gender
        for voice in voices:
            if voice.gender == gender:
                return voice.voice_id

        # Return first available
        return voices[0].voice_id if voices else None

    def _log(self, message: str):
        """Log a message with provider prefix"""
        print(f"[TTS-{self.name.upper()}] {message}", flush=True)
