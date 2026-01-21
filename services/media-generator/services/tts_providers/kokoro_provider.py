"""
Kokoro TTS Provider

Fast, lightweight TTS with good quality.
Can run on CPU with near real-time performance.
License: Apache 2.0
"""

import os
import io
import tempfile
from typing import List, Optional, Dict

from .base_provider import (
    BaseTTSProvider,
    TTSProviderType,
    TTSConfig,
    TTSResult,
    VoiceInfo,
    VoiceGender,
)


# Kokoro language codes and voices
KOKORO_LANGUAGES = {
    "en": {"code": "a", "name": "American English"},
    "en-gb": {"code": "b", "name": "British English"},
    "es": {"code": "e", "name": "Spanish"},
    "fr": {"code": "f", "name": "French"},
    "hi": {"code": "h", "name": "Hindi"},
    "it": {"code": "i", "name": "Italian"},
    "ja": {"code": "j", "name": "Japanese"},
    "pt": {"code": "p", "name": "Portuguese (Brazilian)"},
    "zh": {"code": "z", "name": "Mandarin Chinese"},
}

# Available voices per language
KOKORO_VOICES = {
    "en": [
        {"id": "af_heart", "name": "Heart", "gender": VoiceGender.FEMALE},
        {"id": "af_bella", "name": "Bella", "gender": VoiceGender.FEMALE},
        {"id": "af_sarah", "name": "Sarah", "gender": VoiceGender.FEMALE},
        {"id": "af_nicole", "name": "Nicole", "gender": VoiceGender.FEMALE},
        {"id": "af_sky", "name": "Sky", "gender": VoiceGender.FEMALE},
        {"id": "am_adam", "name": "Adam", "gender": VoiceGender.MALE},
        {"id": "am_michael", "name": "Michael", "gender": VoiceGender.MALE},
    ],
    "en-gb": [
        {"id": "bf_emma", "name": "Emma", "gender": VoiceGender.FEMALE},
        {"id": "bf_isabella", "name": "Isabella", "gender": VoiceGender.FEMALE},
        {"id": "bm_george", "name": "George", "gender": VoiceGender.MALE},
        {"id": "bm_lewis", "name": "Lewis", "gender": VoiceGender.MALE},
    ],
    "fr": [
        {"id": "ff_siwis", "name": "Siwis", "gender": VoiceGender.FEMALE},
    ],
    "es": [
        {"id": "ef_dora", "name": "Dora", "gender": VoiceGender.FEMALE},
        {"id": "em_alex", "name": "Alex", "gender": VoiceGender.MALE},
    ],
    "it": [
        {"id": "if_sara", "name": "Sara", "gender": VoiceGender.FEMALE},
        {"id": "im_nicola", "name": "Nicola", "gender": VoiceGender.MALE},
    ],
    "pt": [
        {"id": "pf_dora", "name": "Dora", "gender": VoiceGender.FEMALE},
        {"id": "pm_alex", "name": "Alex", "gender": VoiceGender.MALE},
    ],
    "ja": [
        {"id": "jf_alpha", "name": "Alpha", "gender": VoiceGender.FEMALE},
        {"id": "jf_gongitsune", "name": "Gongitsune", "gender": VoiceGender.FEMALE},
        {"id": "jm_kumo", "name": "Kumo", "gender": VoiceGender.MALE},
    ],
    "zh": [
        {"id": "zf_xiaobei", "name": "Xiaobei", "gender": VoiceGender.FEMALE},
        {"id": "zf_xiaoni", "name": "Xiaoni", "gender": VoiceGender.FEMALE},
        {"id": "zf_xiaoxiao", "name": "Xiaoxiao", "gender": VoiceGender.FEMALE},
        {"id": "zm_yunjian", "name": "Yunjian", "gender": VoiceGender.MALE},
    ],
    "hi": [
        {"id": "hf_alpha", "name": "Alpha", "gender": VoiceGender.FEMALE},
        {"id": "hm_omega", "name": "Omega", "gender": VoiceGender.MALE},
    ],
}


class KokoroProvider(BaseTTSProvider):
    """Kokoro TTS provider - fast and lightweight"""

    def __init__(self):
        super().__init__(TTSProviderType.KOKORO)
        self._pipeline = None
        self._pipelines: Dict[str, any] = {}  # Cache pipelines per language
        self._initialized = False

    async def _initialize(self, lang_code: str = "a"):
        """Lazy initialization of Kokoro pipeline"""
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]

        try:
            from kokoro import KPipeline

            self._log(f"Loading Kokoro pipeline for language code: {lang_code}")
            pipeline = KPipeline(lang_code=lang_code)
            self._pipelines[lang_code] = pipeline
            self._initialized = True

            return pipeline

        except ImportError as e:
            self._log(f"Kokoro not installed: {e}")
            raise
        except Exception as e:
            self._log(f"Failed to initialize Kokoro: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if Kokoro is available"""
        if self._available is not None:
            return self._available

        try:
            import kokoro

            self._available = True
            return True

        except ImportError:
            self._log("Kokoro package not installed")
            self._available = False
            return False

    def _get_lang_code(self, language: str) -> str:
        """Convert language to Kokoro language code"""
        if language in KOKORO_LANGUAGES:
            return KOKORO_LANGUAGES[language]["code"]
        # Default to American English
        return "a"

    def _get_default_voice(self, language: str, gender: VoiceGender) -> str:
        """Get default voice for language and gender"""
        lang_key = language if language in KOKORO_VOICES else "en"
        voices = KOKORO_VOICES.get(lang_key, KOKORO_VOICES["en"])

        # Find matching gender
        for voice in voices:
            if voice["gender"] == gender:
                return voice["id"]

        # Return first voice
        return voices[0]["id"]

    async def generate(self, config: TTSConfig) -> TTSResult:
        """Generate speech using Kokoro"""
        try:
            import soundfile as sf
            import numpy as np

            # Get language code
            lang_code = self._get_lang_code(config.language)
            pipeline = await self._initialize(lang_code)

            # Get voice
            voice = config.voice_id
            if not voice or voice.startswith("kokoro_"):
                voice = self._get_default_voice(config.language, config.voice_gender)

            self._log(f"Generating: lang={config.language}, voice={voice}")

            # Generate audio
            audio_chunks = []
            for graphemes, phonemes, audio in pipeline(
                config.text, voice=voice, speed=config.speed
            ):
                audio_chunks.append(audio)

            # Concatenate audio
            if not audio_chunks:
                return TTSResult(
                    success=False,
                    error="No audio generated",
                    provider_used=self.provider_type,
                )

            full_audio = np.concatenate(audio_chunks)
            sample_rate = 24000  # Kokoro outputs at 24kHz

            # Convert to bytes
            buffer = io.BytesIO()

            if config.output_format == "mp3":
                # Save as WAV first
                sf.write(buffer, full_audio, sample_rate, format="WAV")
                buffer.seek(0)
                audio_data = await self._convert_to_mp3(buffer.read(), sample_rate)
            else:
                sf.write(buffer, full_audio, sample_rate, format="WAV")
                audio_data = buffer.getvalue()

            # Calculate duration
            duration = len(full_audio) / sample_rate

            return TTSResult(
                success=True,
                audio_data=audio_data,
                duration_seconds=duration,
                sample_rate=sample_rate,
                provider_used=self.provider_type,
                metadata={
                    "language": config.language,
                    "voice": voice,
                },
            )

        except Exception as e:
            self._log(f"Generation failed: {e}")
            return TTSResult(
                success=False,
                error=str(e),
                provider_used=self.provider_type,
            )

    async def _convert_to_mp3(self, wav_data: bytes, sample_rate: int) -> bytes:
        """Convert WAV to MP3 using ffmpeg"""
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_file.write(wav_data)
            wav_path = wav_file.name

        mp3_path = wav_path.replace(".wav", ".mp3")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    wav_path,
                    "-acodec",
                    "libmp3lame",
                    "-ab",
                    "192k",
                    mp3_path,
                ],
                check=True,
                capture_output=True,
            )

            with open(mp3_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
            if os.path.exists(mp3_path):
                os.unlink(mp3_path)

    def get_supported_languages(self) -> List[str]:
        """Get supported language codes"""
        return list(KOKORO_LANGUAGES.keys())

    def get_available_voices(self, language: Optional[str] = None) -> List[VoiceInfo]:
        """Get available voices"""
        voices = []

        for lang, lang_voices in KOKORO_VOICES.items():
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
                        supports_cloning=False,
                        description=f"Kokoro {voice['name']} ({lang})",
                    )
                )

        return voices

    def supports_voice_cloning(self) -> bool:
        """Kokoro does not support voice cloning"""
        return False

    def get_default_voice(
        self, language: str, gender: VoiceGender = VoiceGender.NEUTRAL
    ) -> Optional[str]:
        """Get default voice for language"""
        return self._get_default_voice(language, gender)
