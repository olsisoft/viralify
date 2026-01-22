"""
Audio Agent

Generates TTS audio with precise word-level timestamps.
Supports multiple TTS backends:
1. Hybrid TTS Service (Kokoro/Chatterbox - self-hosted)
2. ElevenLabs API (multilingual)
3. OpenAI TTS API (fallback)

Uses Whisper for timestamp extraction.
"""

import os
import json
import tempfile
import subprocess
import httpx
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI

from .base_agent import BaseAgent, AgentResult, WordTimestamp


class AudioAgent(BaseAgent):
    """Generates TTS audio with word-level timestamps for precise sync"""

    def __init__(self):
        super().__init__("AUDIO_AGENT")
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.media_service_url = os.getenv("MEDIA_SERVICE_URL", "http://media-generator:8004")
        self.voice = os.getenv("TTS_VOICE", "onyx")
        self.model = "tts-1-hd"
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        # Use hybrid TTS service when available (Kokoro/Chatterbox)
        self.use_hybrid_tts = os.getenv("USE_HYBRID_TTS", "true").lower() == "true"
        self.tts_quality = os.getenv("TTS_QUALITY", "standard")  # draft, standard, premium

    async def execute(self, state: Dict[str, Any]) -> AgentResult:
        """Generate audio with word timestamps for a scene"""
        voiceover_text = state.get("voiceover_text", "")
        scene_index = state.get("scene_index", 0)
        job_id = state.get("job_id", "unknown")
        content_language = state.get("content_language", "en")

        if not voiceover_text:
            self.log(f"Scene {scene_index}: No voiceover text provided")
            return AgentResult(
                success=False,
                errors=["No voiceover text provided"]
            )

        self.log(f"Scene {scene_index}: Generating audio for {len(voiceover_text.split())} words (language: {content_language})")

        try:
            # Step 1: Generate TTS audio (use ElevenLabs for non-English)
            audio_data = await self._generate_tts(voiceover_text, content_language)

            if not audio_data:
                raise Exception("TTS generation returned no audio data")

            # Step 2: Save audio temporarily and upload
            audio_result = await self._upload_audio(audio_data, job_id, scene_index)

            # Step 3: Get actual audio duration using ffprobe (most reliable)
            actual_duration = await self._get_audio_duration(audio_data)

            # Step 4: Extract word timestamps using Whisper
            word_timestamps = await self._extract_timestamps(audio_data, voiceover_text, content_language)

            # Use actual audio duration from ffprobe (most accurate)
            # Only add small buffer for video composition
            if actual_duration and actual_duration > 0:
                duration = actual_duration + 0.3
                self.log(f"Scene {scene_index}: Using actual audio duration: {actual_duration:.2f}s")
            elif word_timestamps:
                # Fallback to Whisper timestamps
                duration = word_timestamps[-1].end + 0.3
            else:
                # Last resort: estimate
                wps = {"fr": 3.0, "es": 3.2, "de": 2.4, "it": 3.0}.get(content_language, 2.5)
                duration = (len(voiceover_text.split()) / wps) + 0.3

            self.log(f"Scene {scene_index}: Audio generated - {duration:.2f}s, {len(word_timestamps)} word timestamps")

            return AgentResult(
                success=True,
                data={
                    "audio_url": audio_result["url"],
                    "audio_duration": duration,
                    "word_timestamps": [
                        {
                            "word": wt.word,
                            "start": wt.start,
                            "end": wt.end
                        }
                        for wt in word_timestamps
                    ],
                    "transcript": voiceover_text
                }
            )

        except Exception as e:
            self.log(f"Scene {scene_index}: Audio generation failed - {e}")
            return AgentResult(
                success=False,
                errors=[str(e)]
            )

    async def _generate_tts(self, text: str, language: str = "en") -> bytes:
        """
        Generate TTS audio with intelligent provider selection.

        Priority:
        1. Hybrid TTS Service (Kokoro/Chatterbox) - self-hosted, cost-effective
        2. ElevenLabs API - for non-English or premium quality
        3. OpenAI TTS API - fallback for English
        """
        # Try hybrid TTS service first (Kokoro/Chatterbox)
        if self.use_hybrid_tts:
            try:
                audio_data = await self._generate_tts_hybrid(text, language)
                if audio_data:
                    return audio_data
            except Exception as e:
                self.log(f"Hybrid TTS failed: {e}, trying fallback providers")

        # Fallback: ElevenLabs for non-English, OpenAI for English
        if language != "en" and self.elevenlabs_api_key:
            return await self._generate_tts_elevenlabs(text, language)
        else:
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="mp3"
            )
            return response.content

    async def _generate_tts_hybrid(self, text: str, language: str) -> Optional[bytes]:
        """Generate TTS using the hybrid TTS service (Kokoro/Chatterbox)"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.media_service_url}/api/v1/tts/generate",
                    json={
                        "text": text,
                        "language": language,
                        "quality": self.tts_quality,
                        "prefer_self_hosted": True,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and data.get("audio_url"):
                        # Download the audio file
                        audio_response = await client.get(data["audio_url"])
                        if audio_response.status_code == 200:
                            self.log(f"Hybrid TTS success: provider={data.get('provider_used', 'unknown')}")
                            return audio_response.content

                self.log(f"Hybrid TTS returned status {response.status_code}")
                return None

        except Exception as e:
            self.log(f"Hybrid TTS error: {e}")
            return None

    async def _generate_tts_elevenlabs(self, text: str, language: str) -> bytes:
        """Generate TTS audio using ElevenLabs multilingual model"""
        # Default multilingual voices per language (ElevenLabs voice IDs)
        language_voices = {
            "fr": "IKne3meq5aSn9XLyUdCD",  # French male voice
            "es": "pNInz6obpgDQGcFmaJgB",  # Spanish
            "de": "pNInz6obpgDQGcFmaJgB",  # German
            "pt": "pNInz6obpgDQGcFmaJgB",  # Portuguese
            "it": "pNInz6obpgDQGcFmaJgB",  # Italian
            "nl": "pNInz6obpgDQGcFmaJgB",  # Dutch
            "pl": "pNInz6obpgDQGcFmaJgB",  # Polish
            "ru": "pNInz6obpgDQGcFmaJgB",  # Russian
            "zh": "pNInz6obpgDQGcFmaJgB",  # Chinese
        }

        voice_id = language_voices.get(language, "pNInz6obpgDQGcFmaJgB")

        self.log(f"Using ElevenLabs multilingual TTS for language: {language}, voice: {voice_id}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": self.elevenlabs_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                },
                timeout=120.0
            )

            if response.status_code != 200:
                self.log(f"ElevenLabs TTS error: {response.status_code} - {response.text}")
                # Fallback to OpenAI TTS
                self.log("Falling back to OpenAI TTS")
                openai_response = await self.client.audio.speech.create(
                    model=self.model,
                    voice=self.voice,
                    input=text,
                    response_format="mp3"
                )
                return openai_response.content

            return response.content

    async def _upload_audio(self, audio_data: bytes, job_id: str, scene_index: int) -> Dict[str, Any]:
        """Upload audio to media service and get URL"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                # Upload to media service
                with open(temp_path, "rb") as f:
                    files = {"file": (f"scene_{scene_index}_audio.mp3", f, "audio/mpeg")}
                    response = await client.post(
                        f"{self.media_service_url}/api/v1/media/upload",
                        files=files
                    )

                if response.status_code == 200:
                    return response.json()
                else:
                    # Fallback: return local file path as URL
                    self.log(f"Upload failed ({response.status_code}), using local path")
                    return {"url": f"file://{temp_path}", "local_path": temp_path}

            except Exception as e:
                self.log(f"Upload error: {e}, using local path")
                return {"url": f"file://{temp_path}", "local_path": temp_path}

    async def _get_audio_duration(self, audio_data: bytes) -> Optional[float]:
        """Get actual audio duration using ffprobe (most reliable method)"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            # Use ffprobe to get duration
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    temp_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            os.unlink(temp_path)

            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                return duration

            return None

        except Exception as e:
            self.log(f"Failed to get audio duration via ffprobe: {e}")
            return None

    async def _extract_timestamps(self, audio_data: bytes, original_text: str, language: str = "en") -> List[WordTimestamp]:
        """Extract word-level timestamps using Whisper"""
        try:
            # Save audio to temp file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            # Use Whisper for transcription with timestamps
            with open(temp_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            # Clean up temp file
            os.unlink(temp_path)

            # Parse word timestamps from response
            word_timestamps = []

            if hasattr(transcript, 'words') and transcript.words:
                for word_data in transcript.words:
                    word_timestamps.append(WordTimestamp(
                        word=word_data.get("word", word_data.word if hasattr(word_data, 'word') else ""),
                        start=float(word_data.get("start", word_data.start if hasattr(word_data, 'start') else 0)),
                        end=float(word_data.get("end", word_data.end if hasattr(word_data, 'end') else 0))
                    ))
            else:
                # Fallback: generate estimated timestamps from text
                word_timestamps = self._estimate_timestamps(original_text, language)

            return word_timestamps

        except Exception as e:
            self.log(f"Timestamp extraction failed: {e}, using estimates")
            return self._estimate_timestamps(original_text, language)

    def _estimate_timestamps(self, text: str, language: str = "en") -> List[WordTimestamp]:
        """Estimate word timestamps when Whisper fails"""
        words = text.split()
        timestamps = []

        # Language-specific speech rates (words per second)
        # French/Spanish/Italian are typically spoken faster
        language_wps = {
            "en": 2.5,  # ~150 WPM
            "fr": 3.0,  # ~180 WPM - French is spoken faster
            "es": 3.2,  # ~190 WPM - Spanish is spoken faster
            "de": 2.4,  # ~145 WPM - German is slightly slower
            "it": 3.0,  # ~180 WPM
            "pt": 2.8,  # ~170 WPM
            "nl": 2.6,  # ~155 WPM
            "pl": 2.7,  # ~160 WPM
            "ru": 2.5,  # ~150 WPM
            "zh": 3.5,  # Chinese has shorter "words" (characters)
        }

        words_per_second = language_wps.get(language, 2.5)
        current_time = 0.0

        for word in words:
            # Longer words take slightly longer to say
            # Adjust factor based on language
            char_factor = 0.04 if language in ["fr", "es", "it"] else 0.05
            word_duration = max(0.15, len(word) * char_factor + 0.15)

            timestamps.append(WordTimestamp(
                word=word,
                start=current_time,
                end=current_time + word_duration
            ))

            current_time += word_duration + 0.08  # Smaller gap for faster languages

        return timestamps

    async def regenerate_section(
        self,
        text: str,
        start_word_index: int,
        end_word_index: int,
        job_id: str,
        scene_index: int
    ) -> AgentResult:
        """Regenerate audio for a specific section (for sync fixes)"""
        words = text.split()
        section_text = " ".join(words[start_word_index:end_word_index])

        self.log(f"Regenerating section: words {start_word_index}-{end_word_index}")

        # Generate new audio for section
        audio_data = await self._generate_tts(section_text)
        audio_result = await self._upload_audio(audio_data, job_id, scene_index)
        word_timestamps = await self._extract_timestamps(audio_data, section_text)

        return AgentResult(
            success=True,
            data={
                "section_audio_url": audio_result["url"],
                "section_timestamps": [
                    {"word": wt.word, "start": wt.start, "end": wt.end}
                    for wt in word_timestamps
                ],
                "start_word_index": start_word_index,
                "end_word_index": end_word_index
            }
        )
