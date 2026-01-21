"""
Chatterbox TTS Provider

High-quality multilingual TTS with voice cloning support.
Requires GPU (4-8GB VRAM).
License: MIT
"""

import os
import io
import tempfile
from typing import List, Optional

from .base_provider import (
    BaseTTSProvider,
    TTSProviderType,
    TTSConfig,
    TTSResult,
    VoiceInfo,
    VoiceGender,
)


# Supported languages by Chatterbox Multilingual
CHATTERBOX_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "tr": "Turkish",
    "hi": "Hindi",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "ms": "Malay",
    "no": "Norwegian",
    "sw": "Swahili",
}


class ChatterboxProvider(BaseTTSProvider):
    """Chatterbox TTS provider with voice cloning"""

    def __init__(self):
        super().__init__(TTSProviderType.CHATTERBOX)
        self._model = None
        self._model_multilingual = None
        self._device = None
        self._initialized = False

    async def _initialize(self):
        """Lazy initialization of Chatterbox models"""
        if self._initialized:
            return

        try:
            import torch
            import torchaudio

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            if self._device == "cpu":
                self._log("WARNING: Running on CPU - performance will be poor")

            # Import Chatterbox models
            from chatterbox.tts import ChatterboxTTS
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self._log(f"Loading Chatterbox models on {self._device}...")

            # Load English model for voice cloning
            self._model = ChatterboxTTS.from_pretrained(device=self._device)

            # Load multilingual model
            self._model_multilingual = ChatterboxMultilingualTTS.from_pretrained(
                device=self._device
            )

            self._initialized = True
            self._log("Chatterbox models loaded successfully")

        except ImportError as e:
            self._log(f"Chatterbox not installed: {e}")
            raise
        except Exception as e:
            self._log(f"Failed to initialize Chatterbox: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if Chatterbox is available"""
        if self._available is not None:
            return self._available

        try:
            import torch

            # Check for GPU
            if not torch.cuda.is_available():
                self._log("No GPU available - Chatterbox disabled")
                self._available = False
                return False

            # Check if chatterbox is installed
            import chatterbox

            self._available = True
            return True

        except ImportError:
            self._log("Chatterbox package not installed")
            self._available = False
            return False

    async def generate(self, config: TTSConfig) -> TTSResult:
        """Generate speech using Chatterbox"""
        try:
            await self._initialize()

            import torch
            import torchaudio
            import io

            # Determine if we need voice cloning
            use_cloning = config.clone_audio_path or config.clone_audio_bytes

            # Use multilingual model for non-English
            if config.language != "en" and not use_cloning:
                wav = self._model_multilingual.generate(
                    config.text,
                    language_id=config.language,
                )
            else:
                # English or voice cloning
                kwargs = {
                    "exaggeration": config.exaggeration,
                    "temperature": config.temperature,
                }

                # Handle voice cloning audio
                if config.clone_audio_path:
                    kwargs["audio_prompt_path"] = config.clone_audio_path
                elif config.clone_audio_bytes:
                    # Save bytes to temp file
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as f:
                        f.write(config.clone_audio_bytes)
                        kwargs["audio_prompt_path"] = f.name

                wav = self._model.generate(config.text, **kwargs)

            # Get sample rate from model
            sample_rate = (
                self._model.sr
                if config.language == "en" or use_cloning
                else self._model_multilingual.sr
            )

            # Convert to bytes
            buffer = io.BytesIO()

            if config.output_format == "mp3":
                # Save as WAV first, then convert
                torchaudio.save(buffer, wav, sample_rate, format="wav")
                buffer.seek(0)
                audio_data = await self._convert_to_mp3(buffer.read(), sample_rate)
            else:
                torchaudio.save(buffer, wav, sample_rate, format="wav")
                audio_data = buffer.getvalue()

            # Calculate duration
            duration = wav.shape[1] / sample_rate

            return TTSResult(
                success=True,
                audio_data=audio_data,
                duration_seconds=duration,
                sample_rate=sample_rate,
                provider_used=self.provider_type,
                metadata={
                    "language": config.language,
                    "voice_cloning": use_cloning,
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
        import tempfile

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
        return list(CHATTERBOX_LANGUAGES.keys())

    def get_available_voices(self, language: Optional[str] = None) -> List[VoiceInfo]:
        """Get available voices - Chatterbox uses voice cloning"""
        # Chatterbox doesn't have predefined voices - it uses cloning
        voices = [
            VoiceInfo(
                voice_id="chatterbox_default",
                name="Chatterbox Default",
                provider=self.provider_type,
                language="en",
                gender=VoiceGender.NEUTRAL,
                supports_cloning=True,
                description="Default voice - use voice cloning for custom voices",
            )
        ]

        # Add multilingual voices
        for lang_code, lang_name in CHATTERBOX_LANGUAGES.items():
            if language and lang_code != language:
                continue
            voices.append(
                VoiceInfo(
                    voice_id=f"chatterbox_ml_{lang_code}",
                    name=f"Chatterbox {lang_name}",
                    provider=self.provider_type,
                    language=lang_code,
                    gender=VoiceGender.NEUTRAL,
                    supports_cloning=True,
                    description=f"Multilingual voice for {lang_name}",
                )
            )

        return voices

    def supports_voice_cloning(self) -> bool:
        """Chatterbox supports zero-shot voice cloning"""
        return True
