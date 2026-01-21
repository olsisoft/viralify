"""
Audio Agent

Generates TTS audio with precise word-level timestamps.
Uses OpenAI TTS and Whisper for timestamp extraction.
"""

import os
import json
import tempfile
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

            # Step 3: Extract word timestamps using Whisper
            word_timestamps = await self._extract_timestamps(audio_data, voiceover_text)

            # Calculate duration from timestamps or estimate
            if word_timestamps:
                # Add 2.0s buffer to prevent audio cutoff at end
                # This accounts for TTS tail-off and compression artifacts
                duration = word_timestamps[-1].end + 2.0
            else:
                # Fallback: estimate ~2.5 words per second + 2s buffer
                duration = (len(voiceover_text.split()) / 2.5) + 2.0

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
        """Generate TTS audio using OpenAI (English) or ElevenLabs (other languages)"""
        if language != "en" and self.elevenlabs_api_key:
            # Use ElevenLabs for non-English with multilingual model
            return await self._generate_tts_elevenlabs(text, language)
        else:
            # Use OpenAI TTS for English
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="mp3"
            )
            return response.content

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

    async def _extract_timestamps(self, audio_data: bytes, original_text: str) -> List[WordTimestamp]:
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
                word_timestamps = self._estimate_timestamps(original_text)

            return word_timestamps

        except Exception as e:
            self.log(f"Timestamp extraction failed: {e}, using estimates")
            return self._estimate_timestamps(original_text)

    def _estimate_timestamps(self, text: str) -> List[WordTimestamp]:
        """Estimate word timestamps when Whisper fails"""
        words = text.split()
        timestamps = []

        # Estimate ~2.5 words per second (150 WPM)
        words_per_second = 2.5
        current_time = 0.0

        for word in words:
            # Longer words take slightly longer to say
            word_duration = max(0.2, len(word) * 0.05 + 0.2)

            timestamps.append(WordTimestamp(
                word=word,
                start=current_time,
                end=current_time + word_duration
            ))

            current_time += word_duration + 0.1  # Add small gap between words

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
