"""
Slide Audio Generator - TTS per slide with parallel execution

This module generates audio for each slide independently, enabling
perfect synchronization by construction (audio duration = slide duration).

Benefits over single TTS + SSVS:
- Perfect sync: No post-hoc matching needed
- Parallel execution: Same or better total time
- Easy debugging: Each audio file is independent
- Reliable: No Whisper timestamp dependency
"""

import asyncio
import os
import tempfile
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import httpx


@dataclass
class SlideAudio:
    """Audio generated for a single slide"""
    slide_index: int
    slide_id: str
    audio_path: str
    audio_url: Optional[str]
    duration: float  # seconds
    text: str  # Original voiceover text
    word_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_index": self.slide_index,
            "slide_id": self.slide_id,
            "audio_path": self.audio_path,
            "audio_url": self.audio_url,
            "duration": round(self.duration, 3),
            "text": self.text,
            "word_count": self.word_count
        }


@dataclass
class SlideAudioBatch:
    """Batch of audio files for all slides"""
    slide_audios: List[SlideAudio]
    total_duration: float
    output_dir: str

    @property
    def timeline(self) -> List[Dict[str, Any]]:
        """Generate direct timeline from audio durations"""
        timeline = []
        current_time = 0.0

        for audio in self.slide_audios:
            timeline.append({
                "slide_index": audio.slide_index,
                "slide_id": audio.slide_id,
                "start": round(current_time, 3),
                "end": round(current_time + audio.duration, 3),
                "duration": round(audio.duration, 3),
                "audio_path": audio.audio_path
            })
            current_time += audio.duration

        return timeline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_audios": [a.to_dict() for a in self.slide_audios],
            "total_duration": round(self.total_duration, 3),
            "timeline": self.timeline,
            "output_dir": self.output_dir
        }


class SlideAudioGenerator:
    """
    Generates TTS audio for each slide independently.

    Uses parallel execution for efficiency - all slides are processed
    concurrently, so total time ≈ longest single slide (not sum of all).

    Usage:
        generator = SlideAudioGenerator(tts_service_url="http://media-generator:8004")
        batch = await generator.generate_batch(slides)

        # Timeline is now perfectly synchronized
        for timing in batch.timeline:
            print(f"Slide {timing['slide_index']}: {timing['start']}s - {timing['end']}s")
    """

    def __init__(
        self,
        tts_service_url: Optional[str] = None,
        output_dir: Optional[str] = None,
        voice_id: str = "alloy",  # OpenAI voice
        speech_rate: float = 1.0,
        max_concurrent: int = 5,  # Limit parallel TTS calls
        padding_before: float = 0.0,  # Silence before each slide
        padding_after: float = 0.1,   # Small pause after each slide
    ):
        self.tts_service_url = tts_service_url or os.getenv(
            "MEDIA_GENERATOR_URL", "http://media-generator:8004"
        )
        # IMPORTANT: Use /tmp/presentations/slide_audio for proper URL serving
        # The /tmp/presentations directory is served by nginx
        self.output_dir = output_dir or os.getenv(
            "SLIDE_AUDIO_DIR", "/tmp/presentations/slide_audio"
        )
        self.voice_id = voice_id
        self.speech_rate = speech_rate
        self.max_concurrent = max_concurrent
        self.padding_before = padding_before
        self.padding_after = padding_after

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(max_concurrent)

        print(f"[SLIDE_AUDIO] Initialized with TTS: {self.tts_service_url}", flush=True)

    async def generate_batch(
        self,
        slides: List[Dict[str, Any]],
        voice_id: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> SlideAudioBatch:
        """
        Generate audio for all slides in parallel.

        Args:
            slides: List of slide dicts with 'voiceover_text' field
            voice_id: Optional voice ID override
            job_id: Optional job ID for file naming

        Returns:
            SlideAudioBatch with all audio files and timeline
        """
        voice = voice_id or self.voice_id
        job_id = job_id or self._generate_job_id(slides)

        print(f"[SLIDE_AUDIO] Generating audio for {len(slides)} slides (parallel, max {self.max_concurrent})", flush=True)

        # Create tasks for parallel execution
        tasks = []
        for i, slide in enumerate(slides):
            task = self._generate_slide_audio(
                slide_index=i,
                slide=slide,
                voice_id=voice,
                job_id=job_id
            )
            tasks.append(task)

        # Execute all in parallel (with semaphore limiting)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        slide_audios = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[SLIDE_AUDIO] ERROR slide {i}: {result}", flush=True)
                # Create placeholder with estimated duration
                slide = slides[i]
                text = slide.get("voiceover_text", "") or ""
                estimated_duration = self._estimate_duration(text)
                slide_audios.append(SlideAudio(
                    slide_index=i,
                    slide_id=slide.get("id", f"slide_{i}"),
                    audio_path="",  # No audio
                    audio_url=None,
                    duration=estimated_duration,
                    text=text,
                    word_count=len(text.split())
                ))
            else:
                slide_audios.append(result)

        # Calculate total duration
        total_duration = sum(a.duration for a in slide_audios)

        batch = SlideAudioBatch(
            slide_audios=slide_audios,
            total_duration=total_duration,
            output_dir=self.output_dir
        )

        print(f"[SLIDE_AUDIO] Batch complete: {len(slide_audios)} audios, {total_duration:.2f}s total", flush=True)

        # Log timeline
        for timing in batch.timeline:
            print(f"[SLIDE_AUDIO] Slide {timing['slide_index']}: {timing['start']:.2f}s - {timing['end']:.2f}s ({timing['duration']:.2f}s)", flush=True)

        return batch

    async def _generate_slide_audio(
        self,
        slide_index: int,
        slide: Dict[str, Any],
        voice_id: str,
        job_id: str
    ) -> SlideAudio:
        """Generate audio for a single slide (with rate limiting)"""
        async with self._semaphore:
            slide_id = slide.get("id", f"slide_{slide_index}")
            text = slide.get("voiceover_text", "") or ""

            if not text.strip():
                # No text - return minimal duration
                return SlideAudio(
                    slide_index=slide_index,
                    slide_id=slide_id,
                    audio_path="",
                    audio_url=None,
                    duration=1.0,  # Minimum 1 second for empty slides
                    text="",
                    word_count=0
                )

            # Generate unique filename
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            filename = f"{job_id}_slide_{slide_index:03d}_{text_hash}.mp3"
            output_path = os.path.join(self.output_dir, filename)

            # Check cache
            if os.path.exists(output_path):
                duration = await self._get_audio_duration(output_path)
                print(f"[SLIDE_AUDIO] Slide {slide_index}: cached ({duration:.2f}s)", flush=True)
                return SlideAudio(
                    slide_index=slide_index,
                    slide_id=slide_id,
                    audio_path=output_path,
                    audio_url=None,
                    duration=duration,
                    text=text,
                    word_count=len(text.split())
                )

            # Call TTS service
            try:
                audio_path, duration = await self._call_tts_service(
                    text=text,
                    voice_id=voice_id,
                    output_path=output_path
                )

                # Add padding if configured
                if self.padding_before > 0 or self.padding_after > 0:
                    audio_path, duration = await self._add_padding(
                        audio_path,
                        self.padding_before,
                        self.padding_after
                    )

                print(f"[SLIDE_AUDIO] Slide {slide_index}: generated ({duration:.2f}s)", flush=True)

                return SlideAudio(
                    slide_index=slide_index,
                    slide_id=slide_id,
                    audio_path=audio_path,
                    audio_url=None,
                    duration=duration,
                    text=text,
                    word_count=len(text.split())
                )

            except Exception as e:
                print(f"[SLIDE_AUDIO] Slide {slide_index} TTS failed: {e}", flush=True)
                raise

    async def _call_tts_service(
        self,
        text: str,
        voice_id: str,
        output_path: str
    ) -> Tuple[str, float]:
        """Call the TTS service to generate audio"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.tts_service_url}/api/v1/tts/generate",
                json={
                    "text": text,
                    "voice_id": voice_id,
                    "speed": self.speech_rate,
                    "output_format": "mp3"
                }
            )

            if response.status_code != 200:
                raise Exception(f"TTS service error: {response.status_code} - {response.text}")

            data = response.json()

            # Download the audio file
            audio_url = data.get("audio_url") or data.get("url")
            if audio_url:
                audio_response = await client.get(audio_url)
                with open(output_path, "wb") as f:
                    f.write(audio_response.content)
            elif data.get("audio_path"):
                # Audio is already on disk (same machine)
                import shutil
                shutil.copy(data["audio_path"], output_path)
            else:
                raise Exception("TTS response missing audio_url or audio_path")

            duration = data.get("duration", 0.0)
            if duration == 0:
                duration = await self._get_audio_duration(output_path)

            return output_path, duration

    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file using ffprobe"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            return float(stdout.decode().strip())
        except Exception:
            # Fallback: estimate from file size (rough approximation for MP3)
            file_size = os.path.getsize(audio_path)
            return file_size / 16000  # ~128kbps MP3

    async def _add_padding(
        self,
        audio_path: str,
        padding_before: float,
        padding_after: float
    ) -> Tuple[str, float]:
        """Add silence padding before/after audio"""
        if padding_before == 0 and padding_after == 0:
            duration = await self._get_audio_duration(audio_path)
            return audio_path, duration

        output_path = audio_path.replace(".mp3", "_padded.mp3")

        # Build FFmpeg filter for padding
        filters = []
        if padding_before > 0:
            filters.append(f"adelay={int(padding_before * 1000)}|{int(padding_before * 1000)}")
        if padding_after > 0:
            filters.append(f"apad=pad_dur={padding_after}")

        filter_str = ",".join(filters)

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", audio_path,
            "-af", filter_str,
            "-acodec", "libmp3lame", "-q:a", "2",
            output_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

        if proc.returncode == 0 and os.path.exists(output_path):
            # Remove original, rename padded
            os.remove(audio_path)
            os.rename(output_path, audio_path)

        duration = await self._get_audio_duration(audio_path)
        return audio_path, duration

    def _estimate_duration(self, text: str) -> float:
        """Estimate audio duration from text (fallback)"""
        words = len(text.split())
        # Average speaking rate: ~150 words per minute
        return max(1.0, words / 2.5)  # words per second ≈ 2.5

    def _generate_job_id(self, slides: List[Dict[str, Any]]) -> str:
        """Generate unique job ID from slides content"""
        content = "".join(s.get("voiceover_text", "") for s in slides)
        return hashlib.md5(content.encode()).hexdigest()[:12]


# Convenience function
async def generate_slide_audio_batch(
    slides: List[Dict[str, Any]],
    voice_id: str = "alloy",
    job_id: Optional[str] = None
) -> SlideAudioBatch:
    """
    Generate audio for all slides with perfect synchronization.

    Example:
        batch = await generate_slide_audio_batch(slides, voice_id="nova")

        # Use the timeline directly - no SSVS needed!
        for timing in batch.timeline:
            print(f"Slide {timing['slide_index']}: {timing['start']}s - {timing['end']}s")
    """
    generator = SlideAudioGenerator(voice_id=voice_id)
    return await generator.generate_batch(slides, job_id=job_id)
