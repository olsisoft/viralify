"""
ElevenLabs TTS Provider

Premium API-based TTS with excellent voice cloning.
"""

import os
import httpx
from typing import List, Optional

from .base_provider import (
    BaseTTSProvider,
    TTSProviderType,
    TTSConfig,
    TTSResult,
    VoiceInfo,
    VoiceGender,
)


# ElevenLabs supported languages
ELEVENLABS_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "ru": "Russian",
    "nl": "Dutch",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "sv": "Swedish",
    "cs": "Czech",
}

# Default voices by language
ELEVENLABS_VOICES = {
    "en": [
        {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "gender": VoiceGender.FEMALE},
        {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "gender": VoiceGender.MALE},
        {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "gender": VoiceGender.FEMALE},
        {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni", "gender": VoiceGender.MALE},
        {"id": "onwK4e9ZLuTAKqWW03F9", "name": "Daniel", "gender": VoiceGender.MALE},
    ],
    "fr": [
        {"id": "IKne3meq5aSn9XLyUdCD", "name": "Thomas", "gender": VoiceGender.MALE},
        {"id": "XB0fDUnXU5powFXDhCwa", "name": "Charlotte", "gender": VoiceGender.FEMALE},
    ],
    "es": [
        {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "gender": VoiceGender.MALE},
    ],
    "de": [
        {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "gender": VoiceGender.MALE},
    ],
}


class ElevenLabsProvider(BaseTTSProvider):
    """ElevenLabs TTS provider"""

    def __init__(self):
        super().__init__(TTSProviderType.ELEVENLABS)
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.base_url = "https://api.elevenlabs.io/v1"

    async def is_available(self) -> bool:
        """Check if ElevenLabs is available"""
        if self._available is not None:
            return self._available

        if not self.api_key:
            self._log("No API key configured")
            self._available = False
            return False

        # Test API connection
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/user",
                    headers={"xi-api-key": self.api_key},
                    timeout=10.0,
                )
                self._available = response.status_code == 200
                return self._available
        except Exception as e:
            self._log(f"API check failed: {e}")
            self._available = False
            return False

    async def generate(self, config: TTSConfig) -> TTSResult:
        """Generate speech using ElevenLabs"""
        try:
            # Get voice ID
            voice_id = config.voice_id
            if not voice_id:
                voice_id = self._get_default_voice_id(
                    config.language, config.voice_gender
                )

            # Choose model based on language
            model_id = (
                "eleven_multilingual_v2"
                if config.language != "en"
                else "eleven_monolingual_v1"
            )

            self._log(
                f"Generating: lang={config.language}, voice={voice_id}, model={model_id}"
            )

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": config.text,
                        "model_id": model_id,
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75,
                        },
                    },
                    timeout=120.0,
                )

                if response.status_code != 200:
                    return TTSResult(
                        success=False,
                        error=f"API error: {response.status_code} - {response.text}",
                        provider_used=self.provider_type,
                    )

                audio_data = response.content

                # Estimate duration (ElevenLabs doesn't return it directly)
                # Rough estimate: ~150 words per minute
                word_count = len(config.text.split())
                duration = (word_count / 150) * 60

                return TTSResult(
                    success=True,
                    audio_data=audio_data,
                    duration_seconds=duration,
                    sample_rate=44100,
                    provider_used=self.provider_type,
                    metadata={
                        "voice_id": voice_id,
                        "model_id": model_id,
                        "language": config.language,
                    },
                )

        except Exception as e:
            self._log(f"Generation failed: {e}")
            return TTSResult(
                success=False,
                error=str(e),
                provider_used=self.provider_type,
            )

    def _get_default_voice_id(self, language: str, gender: VoiceGender) -> str:
        """Get default voice ID for language and gender"""
        lang_voices = ELEVENLABS_VOICES.get(language, ELEVENLABS_VOICES["en"])

        for voice in lang_voices:
            if voice["gender"] == gender:
                return voice["id"]

        return lang_voices[0]["id"]

    def get_supported_languages(self) -> List[str]:
        """Get supported language codes"""
        return list(ELEVENLABS_LANGUAGES.keys())

    def get_available_voices(self, language: Optional[str] = None) -> List[VoiceInfo]:
        """Get available voices"""
        voices = []

        for lang, lang_voices in ELEVENLABS_VOICES.items():
            if language and lang != language:
                continue

            for voice in lang_voices:
                voices.append(
                    VoiceInfo(
                        voice_id=voice["id"],
                        name=voice["name"],
                        provider=self.provider_type,
                        language=lang,
                        gender=voice["gender"],
                        supports_cloning=True,
                        description=f"ElevenLabs {voice['name']}",
                    )
                )

        return voices

    def supports_voice_cloning(self) -> bool:
        """ElevenLabs supports voice cloning via their API"""
        return True
