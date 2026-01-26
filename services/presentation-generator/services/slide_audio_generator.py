"""
Slide Audio Generator - TTS per slide with parallel execution

This module generates audio for each slide independently, enabling
perfect synchronization by construction (audio duration = slide duration).

Benefits over single TTS + SSVS:
- Perfect sync: No post-hoc matching needed
- Parallel execution: Same or better total time
- Easy debugging: Each audio file is independent
- Reliable: No Whisper timestamp dependency

VQV-HALLU Integration (Phase 7):
- Validates TTS audio against source text
- Detects hallucinations, distortions, gibberish
- Auto-regenerates if quality score < threshold
- Graceful degradation if service unavailable
"""

import asyncio
import os
import tempfile
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import httpx

# VQV-HALLU client for hallucination detection
from services.vqv_hallu_client import VQVHalluClient, VQVAnalysisResult


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
    # VQV-HALLU validation results
    vqv_validated: bool = False
    vqv_score: Optional[float] = None
    vqv_attempts: int = 0
    vqv_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_index": self.slide_index,
            "slide_id": self.slide_id,
            "audio_path": self.audio_path,
            "audio_url": self.audio_url,
            "duration": round(self.duration, 3),
            "text": self.text,
            "word_count": self.word_count,
            "vqv_validated": self.vqv_validated,
            "vqv_score": self.vqv_score,
            "vqv_attempts": self.vqv_attempts,
            "vqv_issues": self.vqv_issues,
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
        language: str = "en",  # Language code for TTS (CRITICAL for correct pronunciation)
        speech_rate: float = 1.0,
        max_concurrent: int = 5,  # Limit parallel TTS calls
        padding_before: float = 0.0,  # Silence before each slide
        padding_after: float = 0.1,   # Small pause after each slide
        # VQV-HALLU validation settings
        vqv_enabled: Optional[bool] = None,
        vqv_max_attempts: int = 3,
        vqv_min_score: float = 70.0,
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
        self.language = language  # Language for TTS pronunciation
        self.speech_rate = speech_rate
        self.max_concurrent = max_concurrent
        self.padding_before = padding_before
        self.padding_after = padding_after

        # VQV-HALLU validation configuration
        if vqv_enabled is not None:
            self.vqv_enabled = vqv_enabled
        else:
            self.vqv_enabled = os.getenv("VQV_HALLU_ENABLED", "true").lower() == "true"
        self.vqv_max_attempts = vqv_max_attempts
        self.vqv_min_score = vqv_min_score

        # Initialize VQV client (with graceful degradation built-in)
        self._vqv_client = VQVHalluClient(
            enabled=self.vqv_enabled,
            min_acceptable_score=self.vqv_min_score,
        ) if self.vqv_enabled else None

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(max_concurrent)

        vqv_status = "enabled" if self.vqv_enabled else "disabled"
        print(f"[SLIDE_AUDIO] Initialized with TTS: {self.tts_service_url}, language: {self.language}, VQV: {vqv_status}", flush=True)

    async def generate_batch(
        self,
        slides: List[Dict[str, Any]],
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> SlideAudioBatch:
        """
        Generate audio for all slides in parallel.

        Args:
            slides: List of slide dicts with 'voiceover_text' field
            voice_id: Optional voice ID override
            language: Optional language code override (e.g., 'fr', 'en', 'es')
            job_id: Optional job ID for file naming

        Returns:
            SlideAudioBatch with all audio files and timeline
        """
        voice = voice_id or self.voice_id
        lang = language or self.language
        job_id = job_id or self._generate_job_id(slides)

        print(f"[SLIDE_AUDIO] Generating audio for {len(slides)} slides (parallel, max {self.max_concurrent}, lang={lang})", flush=True)

        # Create tasks for parallel execution
        tasks = []
        for i, slide in enumerate(slides):
            task = self._generate_slide_audio(
                slide_index=i,
                slide=slide,
                voice_id=voice,
                language=lang,
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

        # Log VQV validation summary
        if self.vqv_enabled:
            validated_count = sum(1 for a in slide_audios if a.vqv_validated)
            total_attempts = sum(a.vqv_attempts for a in slide_audios)
            avg_score = sum(a.vqv_score or 0 for a in slide_audios if a.vqv_score) / max(1, sum(1 for a in slide_audios if a.vqv_score))
            issues_count = sum(len(a.vqv_issues) for a in slide_audios)
            print(f"[SLIDE_AUDIO] VQV Summary: {validated_count}/{len(slide_audios)} validated, avg_score={avg_score:.1f}, attempts={total_attempts}, issues={issues_count}", flush=True)

        return batch

    async def _generate_slide_audio(
        self,
        slide_index: int,
        slide: Dict[str, Any],
        voice_id: str,
        language: str,
        job_id: str
    ) -> SlideAudio:
        """Generate audio for a single slide (with rate limiting and VQV validation)"""
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

            # Generate unique filename (include language in hash to differentiate)
            text_hash = hashlib.md5(f"{language}:{text}".encode()).hexdigest()[:8]
            base_filename = f"{job_id}_slide_{slide_index:03d}_{text_hash}"

            # VQV validation tracking
            vqv_validated = False
            vqv_score = None
            vqv_attempts = 0
            vqv_issues = []

            # Regeneration loop with VQV validation
            for attempt in range(self.vqv_max_attempts):
                vqv_attempts = attempt + 1

                # Different filename for each attempt to avoid cache issues during regeneration
                if attempt == 0:
                    filename = f"{base_filename}.mp3"
                else:
                    filename = f"{base_filename}_v{attempt + 1}.mp3"
                output_path = os.path.join(self.output_dir, filename)

                # Check cache only on first attempt
                if attempt == 0 and os.path.exists(output_path):
                    duration = await self._get_audio_duration(output_path)
                    print(f"[SLIDE_AUDIO] Slide {slide_index}: cached ({duration:.2f}s)", flush=True)

                    # Validate cached audio with VQV
                    if self._vqv_client and self.vqv_enabled:
                        vqv_result = await self._validate_audio_with_vqv(
                            audio_path=output_path,
                            source_text=text,
                            audio_id=f"slide_{slide_index}",
                            language=language,
                        )
                        vqv_validated = vqv_result.is_acceptable
                        vqv_score = vqv_result.final_score
                        vqv_issues = vqv_result.primary_issues or []

                        if vqv_result.should_regenerate:
                            print(f"[SLIDE_AUDIO] Slide {slide_index}: cached audio failed VQV (score={vqv_score}), regenerating...", flush=True)
                            # Remove cached file and regenerate
                            os.remove(output_path)
                            continue

                    return SlideAudio(
                        slide_index=slide_index,
                        slide_id=slide_id,
                        audio_path=output_path,
                        audio_url=None,
                        duration=duration,
                        text=text,
                        word_count=len(text.split()),
                        vqv_validated=vqv_validated or not self.vqv_enabled,
                        vqv_score=vqv_score,
                        vqv_attempts=vqv_attempts,
                        vqv_issues=vqv_issues,
                    )

                # Generate new audio
                try:
                    audio_path, duration = await self._call_tts_service(
                        text=text,
                        voice_id=voice_id,
                        language=language,
                        output_path=output_path
                    )

                    # Add padding if configured
                    if self.padding_before > 0 or self.padding_after > 0:
                        audio_path, duration = await self._add_padding(
                            audio_path,
                            self.padding_before,
                            self.padding_after
                        )

                    print(f"[SLIDE_AUDIO] Slide {slide_index}: generated attempt {attempt + 1} ({duration:.2f}s)", flush=True)

                    # Validate with VQV-HALLU
                    if self._vqv_client and self.vqv_enabled:
                        vqv_result = await self._validate_audio_with_vqv(
                            audio_path=audio_path,
                            source_text=text,
                            audio_id=f"slide_{slide_index}_attempt_{attempt + 1}",
                            language=language,
                        )
                        vqv_validated = vqv_result.is_acceptable
                        vqv_score = vqv_result.final_score
                        vqv_issues = vqv_result.primary_issues or []

                        if vqv_result.should_regenerate and attempt < self.vqv_max_attempts - 1:
                            print(f"[SLIDE_AUDIO] Slide {slide_index}: VQV failed (score={vqv_score}, issues={vqv_issues}), retrying...", flush=True)
                            # Remove failed audio and try again
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                            continue

                        if vqv_result.should_regenerate:
                            print(f"[SLIDE_AUDIO] Slide {slide_index}: VQV failed after {vqv_attempts} attempts (score={vqv_score}), accepting anyway", flush=True)
                    else:
                        vqv_validated = True  # VQV disabled, auto-accept

                    return SlideAudio(
                        slide_index=slide_index,
                        slide_id=slide_id,
                        audio_path=audio_path,
                        audio_url=None,
                        duration=duration,
                        text=text,
                        word_count=len(text.split()),
                        vqv_validated=vqv_validated,
                        vqv_score=vqv_score,
                        vqv_attempts=vqv_attempts,
                        vqv_issues=vqv_issues,
                    )

                except Exception as e:
                    print(f"[SLIDE_AUDIO] Slide {slide_index} TTS failed (attempt {attempt + 1}): {e}", flush=True)
                    if attempt == self.vqv_max_attempts - 1:
                        raise

            # Should not reach here, but fallback
            raise Exception(f"Failed to generate valid audio for slide {slide_index} after {self.vqv_max_attempts} attempts")

    async def _validate_audio_with_vqv(
        self,
        audio_path: str,
        source_text: str,
        audio_id: str,
        language: str,
    ) -> VQVAnalysisResult:
        """Validate audio with VQV-HALLU service (graceful degradation built-in)"""
        # Fallback score when VQV cannot analyze
        FALLBACK_SCORE = 75.0

        if not self._vqv_client:
            # VQV disabled - return acceptable result with fallback score
            return VQVAnalysisResult(
                audio_id=audio_id,
                status="disabled",
                final_score=FALLBACK_SCORE,
                acoustic_score=FALLBACK_SCORE,
                linguistic_score=FALLBACK_SCORE,
                semantic_score=FALLBACK_SCORE,
                is_acceptable=True,
                recommended_action="accept",
                service_available=False,
                message="VQV validation disabled",
            )

        try:
            result = await self._vqv_client.analyze(
                source_text=source_text,
                audio_path=audio_path,
                audio_id=audio_id,
                content_type="technical_course",
                language=language,
            )

            # Ensure we always have a score (fallback if service returned None)
            if result.final_score is None:
                result.final_score = FALLBACK_SCORE
                result.acoustic_score = result.acoustic_score or FALLBACK_SCORE
                result.linguistic_score = result.linguistic_score or FALLBACK_SCORE
                result.semantic_score = result.semantic_score or FALLBACK_SCORE

            if result.service_available:
                print(f"[SLIDE_AUDIO] VQV {audio_id}: score={result.final_score}, acceptable={result.is_acceptable}", flush=True)
            else:
                print(f"[SLIDE_AUDIO] VQV {audio_id}: service unavailable, using fallback score={FALLBACK_SCORE}", flush=True)

            return result

        except Exception as e:
            print(f"[SLIDE_AUDIO] VQV {audio_id} error: {e}, using fallback score={FALLBACK_SCORE}", flush=True)
            # Graceful degradation - accept audio on VQV error with fallback score
            return VQVAnalysisResult(
                audio_id=audio_id,
                status="error",
                final_score=FALLBACK_SCORE,
                acoustic_score=FALLBACK_SCORE,
                linguistic_score=FALLBACK_SCORE,
                semantic_score=FALLBACK_SCORE,
                is_acceptable=True,
                recommended_action="accept",
                primary_issues=[str(e)],
                service_available=False,
                message=f"VQV error: {e}",
            )

    async def _call_tts_service(
        self,
        text: str,
        voice_id: str,
        language: str,
        output_path: str
    ) -> Tuple[str, float]:
        """Call the TTS service to generate audio"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.tts_service_url}/api/v1/tts/generate",
                json={
                    "text": text,
                    "voice_id": voice_id,
                    "language": language,  # CRITICAL: Pass language to TTS for correct pronunciation
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
    language: str = "en",
    job_id: Optional[str] = None,
    vqv_enabled: Optional[bool] = None,
) -> SlideAudioBatch:
    """
    Generate audio for all slides with perfect synchronization and VQV validation.

    Example:
        batch = await generate_slide_audio_batch(slides, voice_id="nova", language="fr")

        # Use the timeline directly - no SSVS needed!
        for timing in batch.timeline:
            print(f"Slide {timing['slide_index']}: {timing['start']}s - {timing['end']}s")

        # Check VQV validation status
        for audio in batch.slide_audios:
            if not audio.vqv_validated:
                print(f"Warning: Slide {audio.slide_index} has VQV issues: {audio.vqv_issues}")
    """
    generator = SlideAudioGenerator(
        voice_id=voice_id,
        language=language,
        vqv_enabled=vqv_enabled,
    )
    return await generator.generate_batch(slides, language=language, job_id=job_id)
