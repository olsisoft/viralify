"""
TTS Provider Service

Orchestrates multiple TTS providers with intelligent routing:
1. Voice cloning → Chatterbox (self-hosted) or ElevenLabs (API)
2. Standard voices non-English → Kokoro (fast) or Chatterbox (quality)
3. Standard voices English → Kokoro (fast) or OpenAI (quality)
4. Fallback chain: Kokoro → Chatterbox → ElevenLabs → OpenAI
"""

import os
from typing import List, Optional, Dict, Any
from enum import Enum

from .base_provider import (
    BaseTTSProvider,
    TTSProviderType,
    TTSConfig,
    TTSResult,
    VoiceInfo,
    VoiceGender,
)
from .chatterbox_provider import ChatterboxProvider
from .kokoro_provider import KokoroProvider
from .elevenlabs_provider import ElevenLabsProvider
from .openai_provider import OpenAIProvider


class TTSQuality(str, Enum):
    """TTS quality levels"""
    DRAFT = "draft"      # Fast, lower quality (Kokoro)
    STANDARD = "standard"  # Good balance (Kokoro/Chatterbox)
    PREMIUM = "premium"   # Best quality (Chatterbox/ElevenLabs)


class TTSProviderService:
    """
    Hybrid TTS service with intelligent provider routing.

    Routing logic:
    - Voice cloning: Chatterbox (GPU) → ElevenLabs (API fallback)
    - Premium quality: Chatterbox → ElevenLabs
    - Standard quality: Kokoro (fast) → Chatterbox
    - Draft/Preview: Kokoro only (fastest)
    """

    def __init__(self):
        self._providers: Dict[TTSProviderType, BaseTTSProvider] = {}
        self._initialized = False
        self._available_providers: List[TTSProviderType] = []

    async def initialize(self):
        """Initialize all available providers"""
        if self._initialized:
            return

        print("[TTS-SERVICE] Initializing TTS providers...", flush=True)

        # Initialize providers
        providers = [
            KokoroProvider(),      # Fast, CPU-friendly
            ChatterboxProvider(),  # High quality, GPU
            ElevenLabsProvider(),  # API fallback
            OpenAIProvider(),      # API fallback
        ]

        for provider in providers:
            try:
                if await provider.is_available():
                    self._providers[provider.provider_type] = provider
                    self._available_providers.append(provider.provider_type)
                    print(f"[TTS-SERVICE] ✓ {provider.name} available", flush=True)
                else:
                    print(f"[TTS-SERVICE] ✗ {provider.name} not available", flush=True)
            except Exception as e:
                print(f"[TTS-SERVICE] ✗ {provider.name} error: {e}", flush=True)

        self._initialized = True
        print(f"[TTS-SERVICE] Initialized with {len(self._available_providers)} providers", flush=True)

    def _get_provider(self, provider_type: TTSProviderType) -> Optional[BaseTTSProvider]:
        """Get a specific provider if available"""
        return self._providers.get(provider_type)

    def _select_provider(
        self,
        config: TTSConfig,
        quality: TTSQuality = TTSQuality.STANDARD,
        prefer_self_hosted: bool = True,
    ) -> List[BaseTTSProvider]:
        """
        Select providers based on requirements.
        Returns ordered list of providers to try.
        """
        providers = []
        needs_cloning = config.clone_audio_path or config.clone_audio_bytes

        # Voice cloning required
        if needs_cloning:
            # Prefer Chatterbox for self-hosted cloning
            if TTSProviderType.CHATTERBOX in self._providers:
                providers.append(self._providers[TTSProviderType.CHATTERBOX])
            # Fallback to ElevenLabs API
            if TTSProviderType.ELEVENLABS in self._providers:
                providers.append(self._providers[TTSProviderType.ELEVENLABS])
            return providers

        # Draft quality - fastest option
        if quality == TTSQuality.DRAFT:
            if TTSProviderType.KOKORO in self._providers:
                providers.append(self._providers[TTSProviderType.KOKORO])
            if TTSProviderType.OPENAI in self._providers:
                providers.append(self._providers[TTSProviderType.OPENAI])
            return providers

        # Premium quality
        if quality == TTSQuality.PREMIUM:
            if prefer_self_hosted and TTSProviderType.CHATTERBOX in self._providers:
                providers.append(self._providers[TTSProviderType.CHATTERBOX])
            if TTSProviderType.ELEVENLABS in self._providers:
                providers.append(self._providers[TTSProviderType.ELEVENLABS])
            if TTSProviderType.KOKORO in self._providers:
                providers.append(self._providers[TTSProviderType.KOKORO])
            if TTSProviderType.OPENAI in self._providers:
                providers.append(self._providers[TTSProviderType.OPENAI])
            return providers

        # Standard quality - balance speed and quality
        # For non-English, prefer providers with good multilingual support
        if config.language != "en":
            if TTSProviderType.KOKORO in self._providers:
                kokoro = self._providers[TTSProviderType.KOKORO]
                if config.language in kokoro.get_supported_languages():
                    providers.append(kokoro)

            if prefer_self_hosted and TTSProviderType.CHATTERBOX in self._providers:
                providers.append(self._providers[TTSProviderType.CHATTERBOX])

            if TTSProviderType.ELEVENLABS in self._providers:
                providers.append(self._providers[TTSProviderType.ELEVENLABS])
        else:
            # English - Kokoro is fast and good
            if TTSProviderType.KOKORO in self._providers:
                providers.append(self._providers[TTSProviderType.KOKORO])

            if TTSProviderType.OPENAI in self._providers:
                providers.append(self._providers[TTSProviderType.OPENAI])

            if prefer_self_hosted and TTSProviderType.CHATTERBOX in self._providers:
                providers.append(self._providers[TTSProviderType.CHATTERBOX])

        # Always add remaining providers as fallbacks
        for pt in [TTSProviderType.KOKORO, TTSProviderType.OPENAI,
                   TTSProviderType.ELEVENLABS, TTSProviderType.CHATTERBOX]:
            if pt in self._providers and self._providers[pt] not in providers:
                providers.append(self._providers[pt])

        return providers

    async def generate(
        self,
        text: str,
        language: str = "en",
        voice_id: Optional[str] = None,
        voice_gender: VoiceGender = VoiceGender.NEUTRAL,
        speed: float = 1.0,
        quality: TTSQuality = TTSQuality.STANDARD,
        clone_audio_path: Optional[str] = None,
        clone_audio_bytes: Optional[bytes] = None,
        preferred_provider: Optional[TTSProviderType] = None,
        prefer_self_hosted: bool = True,
    ) -> TTSResult:
        """
        Generate TTS audio with automatic provider selection.

        Args:
            text: Text to synthesize
            language: Language code (en, fr, es, etc.)
            voice_id: Optional specific voice ID
            voice_gender: Preferred voice gender
            speed: Speech speed multiplier
            quality: Quality level (draft, standard, premium)
            clone_audio_path: Path to audio file for voice cloning
            clone_audio_bytes: Audio bytes for voice cloning
            preferred_provider: Force a specific provider
            prefer_self_hosted: Prefer self-hosted over API

        Returns:
            TTSResult with audio data or error
        """
        await self.initialize()

        config = TTSConfig(
            text=text,
            language=language,
            voice_id=voice_id,
            voice_gender=voice_gender,
            speed=speed,
            clone_audio_path=clone_audio_path,
            clone_audio_bytes=clone_audio_bytes,
        )

        # If specific provider requested
        if preferred_provider and preferred_provider in self._providers:
            provider = self._providers[preferred_provider]
            result = await provider.generate(config)
            if result.success:
                return result
            print(f"[TTS-SERVICE] Preferred provider {preferred_provider} failed, trying fallbacks", flush=True)

        # Select providers based on requirements
        providers = self._select_provider(config, quality, prefer_self_hosted)

        if not providers:
            return TTSResult(
                success=False,
                error="No TTS providers available",
            )

        # Try each provider in order
        last_error = None
        for provider in providers:
            print(f"[TTS-SERVICE] Trying {provider.name}...", flush=True)
            result = await provider.generate(config)

            if result.success:
                print(f"[TTS-SERVICE] Success with {provider.name}", flush=True)
                return result

            last_error = result.error
            print(f"[TTS-SERVICE] {provider.name} failed: {result.error}", flush=True)

        return TTSResult(
            success=False,
            error=f"All providers failed. Last error: {last_error}",
        )

    async def generate_with_cloning(
        self,
        text: str,
        clone_audio_path: str,
        language: str = "en",
        speed: float = 1.0,
    ) -> TTSResult:
        """Generate TTS with voice cloning"""
        return await self.generate(
            text=text,
            language=language,
            speed=speed,
            quality=TTSQuality.PREMIUM,
            clone_audio_path=clone_audio_path,
        )

    def get_available_providers(self) -> List[TTSProviderType]:
        """Get list of available provider types"""
        return self._available_providers.copy()

    def get_all_voices(self, language: Optional[str] = None) -> List[VoiceInfo]:
        """Get all available voices from all providers"""
        voices = []
        for provider in self._providers.values():
            voices.extend(provider.get_available_voices(language))
        return voices

    def get_supported_languages(self) -> List[str]:
        """Get union of all supported languages"""
        languages = set()
        for provider in self._providers.values():
            languages.update(provider.get_supported_languages())
        return sorted(list(languages))

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers"""
        return {
            "available_providers": [p.value for p in self._available_providers],
            "supports_voice_cloning": any(
                p.supports_voice_cloning() for p in self._providers.values()
            ),
            "supported_languages": self.get_supported_languages(),
            "providers": {
                pt.value: {
                    "available": pt in self._providers,
                    "supports_cloning": self._providers[pt].supports_voice_cloning()
                    if pt in self._providers
                    else False,
                    "languages": self._providers[pt].get_supported_languages()
                    if pt in self._providers
                    else [],
                }
                for pt in TTSProviderType
            },
        }


# Singleton instance
_tts_service: Optional[TTSProviderService] = None


def get_tts_service() -> TTSProviderService:
    """Get the singleton TTS service instance"""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSProviderService()
    return _tts_service
