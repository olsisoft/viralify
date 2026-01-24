"""
Audio Concatenator - Merge slide audio files with crossfade

This module concatenates individual slide audio files into a single
voiceover track, using crossfade to eliminate micro-pauses between slides.

The crossfade creates a seamless transition while maintaining precise
sync points for the timeline.
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .slide_audio_generator import SlideAudio, SlideAudioBatch


@dataclass
class ConcatenatedAudio:
    """Result of audio concatenation"""
    audio_path: str
    audio_url: Optional[str]
    total_duration: float
    slide_count: int
    crossfade_duration: float
    # Timeline adjusted for crossfade overlaps
    timeline: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "audio_url": self.audio_url,
            "total_duration": round(self.total_duration, 3),
            "slide_count": self.slide_count,
            "crossfade_duration": self.crossfade_duration,
            "timeline": self.timeline
        }


class AudioConcatenator:
    """
    Concatenates slide audio files with professional crossfade.

    The crossfade eliminates the "choppy" feel of simple concatenation
    while maintaining precise synchronization.

    Crossfade behavior:
    - Default 100ms crossfade (barely perceptible but smooths transitions)
    - Timeline is adjusted to account for crossfade overlap
    - Each slide's effective start is slightly earlier due to crossfade

    Usage:
        concatenator = AudioConcatenator(crossfade_ms=100)
        result = await concatenator.concatenate(batch, output_path="/tmp/voiceover.mp3")

        # Timeline accounts for crossfade
        for timing in result.timeline:
            print(f"Slide {timing['slide_index']}: {timing['start']}s - {timing['end']}s")
    """

    def __init__(
        self,
        crossfade_ms: float = 100,  # 100ms crossfade (natural, not jarring)
        output_dir: Optional[str] = None,
        normalize_audio: bool = True,  # Normalize volume across slides
        sample_rate: int = 44100,
    ):
        self.crossfade_ms = crossfade_ms
        self.crossfade_seconds = crossfade_ms / 1000.0
        self.output_dir = output_dir or os.getenv(
            "AUDIO_OUTPUT_DIR", "/tmp/viralify/audio"
        )
        self.normalize_audio = normalize_audio
        self.sample_rate = sample_rate

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        print(f"[AUDIO_CONCAT] Initialized with crossfade={crossfade_ms}ms", flush=True)

    async def concatenate(
        self,
        batch: SlideAudioBatch,
        output_path: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> ConcatenatedAudio:
        """
        Concatenate all slide audio files with crossfade.

        Args:
            batch: SlideAudioBatch from SlideAudioGenerator
            output_path: Optional output file path
            job_id: Optional job ID for naming

        Returns:
            ConcatenatedAudio with merged file and adjusted timeline
        """
        audios = [a for a in batch.slide_audios if a.audio_path and os.path.exists(a.audio_path)]

        if not audios:
            raise ValueError("No valid audio files to concatenate")

        if len(audios) == 1:
            # Single file - no concatenation needed
            return ConcatenatedAudio(
                audio_path=audios[0].audio_path,
                audio_url=None,
                total_duration=audios[0].duration,
                slide_count=1,
                crossfade_duration=0,
                timeline=batch.timeline
            )

        # Generate output path
        if not output_path:
            job_id = job_id or f"concat_{len(audios)}"
            output_path = os.path.join(self.output_dir, f"{job_id}_voiceover.mp3")

        print(f"[AUDIO_CONCAT] Concatenating {len(audios)} audio files with {self.crossfade_ms}ms crossfade", flush=True)

        # Build FFmpeg command for crossfade concatenation
        result_path, total_duration = await self._concatenate_with_crossfade(
            audios, output_path
        )

        # Build adjusted timeline accounting for crossfade overlap
        timeline = self._build_crossfade_timeline(audios, total_duration)

        result = ConcatenatedAudio(
            audio_path=result_path,
            audio_url=None,
            total_duration=total_duration,
            slide_count=len(audios),
            crossfade_duration=self.crossfade_seconds,
            timeline=timeline
        )

        print(f"[AUDIO_CONCAT] Complete: {total_duration:.2f}s total", flush=True)

        # Log timeline
        for timing in timeline:
            print(f"[AUDIO_CONCAT] Slide {timing['slide_index']}: {timing['start']:.3f}s - {timing['end']:.3f}s", flush=True)

        return result

    async def _concatenate_with_crossfade(
        self,
        audios: List[SlideAudio],
        output_path: str
    ) -> Tuple[str, float]:
        """
        Concatenate audio files using FFmpeg acrossfade filter.

        For N files, we chain (N-1) crossfades:
        [0][1]acrossfade -> [01]
        [01][2]acrossfade -> [012]
        ...
        """
        if len(audios) == 1:
            return audios[0].audio_path, audios[0].duration

        # Build FFmpeg filter_complex for chained crossfade
        inputs = []
        filter_parts = []

        for i, audio in enumerate(audios):
            inputs.extend(["-i", audio.audio_path])

        # Chain crossfades: [0][1]acrossfade[a1]; [a1][2]acrossfade[a2]; ...
        cf_duration = self.crossfade_seconds

        if len(audios) == 2:
            # Simple case: just one crossfade
            filter_complex = f"[0][1]acrossfade=d={cf_duration}:c1=tri:c2=tri"
            if self.normalize_audio:
                filter_complex += ",loudnorm=I=-16:LRA=11:TP=-1.5"
            filter_complex += "[out]"
        else:
            # Multiple files: chain crossfades
            # First crossfade
            filter_parts.append(f"[0][1]acrossfade=d={cf_duration}:c1=tri:c2=tri[a1]")

            # Middle crossfades
            for i in range(2, len(audios)):
                prev_output = f"a{i-1}"
                curr_output = f"a{i}" if i < len(audios) - 1 else "out"
                filter_parts.append(f"[{prev_output}][{i}]acrossfade=d={cf_duration}:c1=tri:c2=tri[{curr_output}]")

            # Add normalization to final output if needed
            if self.normalize_audio:
                # Replace last output with intermediate
                last_part = filter_parts[-1]
                filter_parts[-1] = last_part.replace("[out]", "[pre_norm]")
                filter_parts.append("[pre_norm]loudnorm=I=-16:LRA=11:TP=-1.5[out]")

            filter_complex = ";".join(filter_parts)

        # Build and run FFmpeg command
        cmd = ["ffmpeg", "-y"]
        cmd.extend(inputs)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-acodec", "libmp3lame",
            "-q:a", "2",  # High quality VBR
            "-ar", str(self.sample_rate),
            output_path
        ])

        print(f"[AUDIO_CONCAT] Running FFmpeg crossfade...", flush=True)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            print(f"[AUDIO_CONCAT] FFmpeg error: {error_msg}", flush=True)

            # Fallback: simple concat without crossfade
            print(f"[AUDIO_CONCAT] Falling back to simple concat", flush=True)
            return await self._simple_concat(audios, output_path)

        # Get actual duration
        total_duration = await self._get_audio_duration(output_path)

        return output_path, total_duration

    async def _simple_concat(
        self,
        audios: List[SlideAudio],
        output_path: str
    ) -> Tuple[str, float]:
        """Fallback: simple concatenation without crossfade"""
        # Create concat file list
        list_path = output_path + ".txt"
        with open(list_path, "w") as f:
            for audio in audios:
                # Escape single quotes in path
                safe_path = audio.audio_path.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-acodec", "libmp3lame",
            "-q:a", "2",
            output_path
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()

        # Cleanup
        if os.path.exists(list_path):
            os.remove(list_path)

        total_duration = await self._get_audio_duration(output_path)
        return output_path, total_duration

    def _build_crossfade_timeline(
        self,
        audios: List[SlideAudio],
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Build timeline adjusted for crossfade overlap.

        With crossfade, each slide starts slightly before the previous one ends.
        The overlap duration is crossfade_seconds / 2 on each side.
        """
        timeline = []
        current_time = 0.0

        for i, audio in enumerate(audios):
            # Calculate effective duration (accounting for crossfade overlap)
            effective_duration = audio.duration

            # Crossfade overlap: slides share crossfade_seconds at boundaries
            # First slide: no overlap at start, half crossfade at end
            # Middle slides: half crossfade at start AND end
            # Last slide: half crossfade at start, no overlap at end
            crossfade_adjustment = 0.0
            if i > 0:
                # Overlap with previous slide
                crossfade_adjustment += self.crossfade_seconds / 2
            if i < len(audios) - 1:
                # Overlap with next slide
                crossfade_adjustment += self.crossfade_seconds / 2

            # Effective duration in final audio
            if i == 0:
                # First slide: full duration minus half crossfade at end
                effective_duration = audio.duration - (self.crossfade_seconds / 2 if len(audios) > 1 else 0)
            elif i == len(audios) - 1:
                # Last slide: full duration minus half crossfade at start
                effective_duration = audio.duration - self.crossfade_seconds / 2
            else:
                # Middle slide: full duration minus crossfade on both sides
                effective_duration = audio.duration - self.crossfade_seconds

            end_time = current_time + max(0.1, effective_duration)

            timeline.append({
                "slide_index": audio.slide_index,
                "slide_id": audio.slide_id,
                "start": round(current_time, 3),
                "end": round(end_time, 3),
                "duration": round(end_time - current_time, 3),
                "original_duration": round(audio.duration, 3),
                "audio_path": audio.audio_path
            })

            current_time = end_time

        # Adjust last slide to match total duration exactly
        if timeline and total_duration > 0:
            timeline[-1]["end"] = round(total_duration, 3)
            timeline[-1]["duration"] = round(total_duration - timeline[-1]["start"], 3)

        return timeline

    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file using ffprobe"""
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()

        try:
            return float(stdout.decode().strip())
        except ValueError:
            return 0.0


# Convenience function
async def concatenate_slide_audio(
    batch: SlideAudioBatch,
    crossfade_ms: float = 100,
    output_path: Optional[str] = None
) -> ConcatenatedAudio:
    """
    Concatenate slide audio files with crossfade.

    Example:
        # Generate per-slide audio
        batch = await generate_slide_audio_batch(slides)

        # Concatenate with crossfade
        result = await concatenate_slide_audio(batch)

        # Use the timeline for video composition
        print(f"Final audio: {result.audio_path}")
        print(f"Total duration: {result.total_duration}s")
    """
    concatenator = AudioConcatenator(crossfade_ms=crossfade_ms)
    return await concatenator.concatenate(batch, output_path=output_path)
