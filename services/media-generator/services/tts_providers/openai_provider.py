"""
OpenAI TTS Provider

API-based TTS with good quality voices.
Best for English, supports other languages but quality may vary.
"""

import os
from typing import List, Optional
from openai import AsyncOpenAI

from .base_provider import (
    BaseTTSProvider,
    TTSProviderType,
    TTSConfig,
    TTSResult,
    VoiceInfo,
    VoiceGender,
)


# OpenAI TTS voices
OPENAI_VOICES = [
    {"id": "alloy", "name": "Alloy", "gender": VoiceGender.NEUTRAL},
    {"id": "echo", "name": "Echo", "gender": VoiceGender.MALE},
    {"id": "fable", "name": "Fable", "gender": VoiceGender.NEUTRAL},
    {"id": "onyx", "name": "Onyx", "gender": VoiceGender.MALE},
    {"id": "nova", "name": "Nova", "gender": VoiceGender.FEMALE},
    {"id": "shimmer", "name": "Shimmer", "gender": VoiceGender.FEMALE},
]

# OpenAI TTS technically supports many languages but is optimized for English
OPENAI_LANGUAGES = {
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
}


class OpenAIProvider(BaseTTSProvider):
    """OpenAI TTS provider"""

    def __init__(self):
        super().__init__(TTSProviderType.OPENAI)
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = None
        self.model = "tts-1-hd"

    async def _get_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client"""
        if self.client is None:
            self.client = AsyncOpenAI(api_key=self.api_key)
        return self.client

    async def is_available(self) -> bool:
        """Check if OpenAI is available"""
        if self._available is not None:
            return self._available

        if not self.api_key:
            self._log("No API key configured")
            self._available = False
            return False

        self._available = True
        return True

    async def generate(self, config: TTSConfig) -> TTSResult:
        """Generate speech using OpenAI"""
        try:
            client = await self._get_client()

            # Get voice
            voice = config.voice_id
            if not voice or voice not in [v["id"] for v in OPENAI_VOICES]:
                voice = self._get_default_voice_id(config.voice_gender)

            self._log(f"Generating: voice={voice}, model={self.model}")

            response = await client.audio.speech.create(
                model=self.model,
                voice=voice,
                input=config.text,
                response_format="mp3" if config.output_format == "mp3" else "wav",
                speed=config.speed,
            )

            audio_data = response.content

            # Estimate duration (~150 words per minute)
            word_count = len(config.text.split())
            duration = (word_count / 150) * 60 / config.speed

            return TTSResult(
                success=True,
                audio_data=audio_data,
                duration_seconds=duration,
                sample_rate=24000,
                provider_used=self.provider_type,
                metadata={
                    "voice": voice,
                    "model": self.model,
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

    def _get_default_voice_id(self, gender: VoiceGender) -> str:
        """Get default voice ID for gender"""
        if gender == VoiceGender.FEMALE:
            return "nova"
        elif gender == VoiceGender.MALE:
            return "onyx"
        return "alloy"

    def get_supported_languages(self) -> List[str]:
        """Get supported language codes"""
        return list(OPENAI_LANGUAGES.keys())

    def get_available_voices(self, language: Optional[str] = None) -> List[VoiceInfo]:
        """Get available voices - same for all languages"""
        voices = []

        for voice in OPENAI_VOICES:
            voices.append(
                VoiceInfo(
                    voice_id=voice["id"],
                    name=voice["name"],
                    provider=self.provider_type,
                    language="en",  # Optimized for English
                    gender=voice["gender"],
                    supports_cloning=False,
                    description=f"OpenAI {voice['name']}",
                )
            )

        return voices

    def supports_voice_cloning(self) -> bool:
        """OpenAI does not support voice cloning"""
        return False
