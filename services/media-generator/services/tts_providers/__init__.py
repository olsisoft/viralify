"""
TTS Providers Package

Hybrid TTS architecture supporting multiple providers:
- Chatterbox: Voice cloning + Premium multilingual (GPU)
- Kokoro: Fast standard voices (CPU/GPU)
- ElevenLabs: Premium API fallback
- OpenAI: Standard API fallback
"""

from .base_provider import BaseTTSProvider, TTSResult, TTSConfig
from .provider_service import TTSProviderService, get_tts_service

__all__ = [
    "BaseTTSProvider",
    "TTSResult",
    "TTSConfig",
    "TTSProviderService",
    "get_tts_service",
]
