"""
FFmpeg Timeline Compositor

Composes video from a Timeline using FFmpeg filtercomplex.
This provides millisecond-precision synchronization between audio and visuals.

Key Features:
- Uses overlay with enable='between(t,start,end)' for precise timing
- Handles freeze frames for padding
- Supports multiple layers (background, content, overlays)
- Crossfade transitions between slides
- Resource management to prevent OOM (semaphore + memory cleanup)
- Retry with exponential backoff for network downloads
"""

import os
import gc
import asyncio
import subprocess
import tempfile
import shutil
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .timeline_builder import Timeline, VisualEvent, VisualEventType
from .ffmpeg_resource_manager import ffmpeg_manager


# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 10.0  # seconds


async def download_with_retry(
    client,
    url: str,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    log_func=None
) -> Optional[bytes]:
    """
    Download a URL with exponential backoff retry.

    Args:
        client: httpx.AsyncClient instance
        url: URL to download
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be multiplied exponentially)
        log_func: Optional logging function

    Returns:
        bytes content if successful, None if all retries failed
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = await client.get(url)

            if response.status_code == 200:
                return response.content
            elif response.status_code in [429, 500, 502, 503, 504]:
                # Retryable HTTP errors
                last_error = f"HTTP {response.status_code}"
                if log_func:
                    log_func(f"Retry {attempt + 1}/{max_retries} for {url[:60]}: {last_error}")
            else:
                # Non-retryable HTTP error
                if log_func:
                    log_func(f"Non-retryable HTTP {response.status_code} for {url[:60]}")
                return None

        except Exception as e:
            last_error = str(e)
            if log_func:
                log_func(f"Retry {attempt + 1}/{max_retries} for {url[:60]}: {last_error}")

        # Exponential backoff with jitter
        if attempt < max_retries - 1:
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
            await asyncio.sleep(delay)

    if log_func:
        log_func(f"All {max_retries} retries failed for {url[:60]}: {last_error}")
    return None


@dataclass
class CompositionResult:
    """Result of video composition"""
    success: bool
    output_path: Optional[str] = None
    output_url: Optional[str] = None
    duration: float = 0.0
    error: Optional[str] = None


class FFmpegTimelineCompositor:
    """
    Composes video from Timeline using FFmpeg.

    This compositor:
    1. Downloads all assets to local temp directory
    2. Generates FFmpeg filtercomplex for precise timing
    3. Renders final video with audio
    """

    def __init__(self, output_dir: str = "/tmp/presentations/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = None
        self.debug = True

    def log(self, message: str):
        if self.debug:
            print(f"[FFMPEG_COMPOSITOR] {message}", flush=True)

    async def compose(
        self,
        timeline: Timeline,
        output_filename: str,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        quality: str = "medium"
    ) -> CompositionResult:
        """
        Compose video from timeline.

        Args:
            timeline: The Timeline object with all events
            output_filename: Name for output file (without path)
            resolution: Output resolution (width, height)
            fps: Frames per second
            quality: "low", "medium", or "high"

        Returns:
            CompositionResult with output path/url
        """
        self.log(f"Composing video: {len(timeline.visual_events)} events, {timeline.total_duration:.2f}s")

        # Create temp directory for assets
        self.temp_dir = Path(tempfile.mkdtemp(prefix="timeline_compose_"))

        try:
            # Step 1: Download/prepare all assets
            asset_map = await self._prepare_assets(timeline)

            if not asset_map:
                return CompositionResult(
                    success=False,
                    error="Failed to prepare assets"
                )

            # Step 2: Prepare audio
            audio_path = await self._prepare_audio(timeline)

            # Step 3: Generate FFmpeg command
            output_path = self.output_dir / output_filename
            ffmpeg_cmd = self._build_ffmpeg_command(
                timeline, asset_map, audio_path, output_path,
                resolution, fps, quality
            )

            self.log(f"FFmpeg command: {' '.join(ffmpeg_cmd[:10])}...")

            # Step 4: Execute FFmpeg
            success = await self._execute_ffmpeg(ffmpeg_cmd)

            if success and output_path.exists():
                self.log(f"Composition complete: {output_path}")
                return CompositionResult(
                    success=True,
                    output_path=str(output_path),
                    duration=timeline.total_duration
                )
            else:
                return CompositionResult(
                    success=False,
                    error="FFmpeg composition failed"
                )

        except Exception as e:
            self.log(f"Composition error: {e}")
            import traceback
            traceback.print_exc()
            return CompositionResult(
                success=False,
                error=str(e)
            )

        finally:
            # Cleanup temp directory
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def _prepare_assets(self, timeline: Timeline) -> Dict[str, str]:
        """
        Download/copy all assets to temp directory.
        Returns mapping from original path/url to local path.

        Uses retry with exponential backoff for network downloads.
        """
        import httpx

        asset_map = {}
        failed_assets = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for i, event in enumerate(timeline.visual_events):
                source = event.asset_url or event.asset_path

                if not source:
                    continue

                # Determine file extension
                url_path = source.split("?")[0]
                ext = Path(url_path).suffix
                if not ext:
                    if "/audio/" in url_path:
                        ext = ".mp3"
                    elif "/video/" in url_path:
                        ext = ".mp4"
                    else:
                        ext = ".png"

                local_path = self.temp_dir / f"asset_{i}{ext}"

                try:
                    if source.startswith("http"):
                        # Download from URL with retry
                        content = await download_with_retry(
                            client, source,
                            max_retries=MAX_RETRIES,
                            log_func=self.log
                        )
                        if content:
                            local_path.write_bytes(content)
                            asset_map[source] = str(local_path)
                            self.log(f"Downloaded: {source[:60]} -> {local_path.name}")
                        else:
                            failed_assets.append(source)
                            self.log(f"Failed to download after retries: {source[:60]}")
                    elif os.path.exists(source):
                        # Copy local file
                        shutil.copy(source, local_path)
                        asset_map[source] = str(local_path)
                        self.log(f"Copied: {source} -> {local_path.name}")
                    else:
                        failed_assets.append(source)
                        self.log(f"Asset not found: {source}")

                except Exception as e:
                    failed_assets.append(source)
                    self.log(f"Error preparing asset {source[:60]}: {e}")

        self.log(f"Prepared {len(asset_map)} assets, {len(failed_assets)} failed")
        return asset_map

    async def _prepare_audio(self, timeline: Timeline) -> Optional[str]:
        """
        Download/copy audio track to temp directory.

        Uses retry with exponential backoff for network downloads.
        Audio is critical - we use more retries.
        """
        import httpx

        source = timeline.audio_track_url or timeline.audio_track_path

        if not source:
            return None

        url_path = source.split("?")[0]
        ext = Path(url_path).suffix
        if not ext:
            ext = ".mp3"  # Audio default

        local_path = self.temp_dir / f"audio{ext}"

        try:
            if source.startswith("http"):
                # Audio is critical - use more retries
                async with httpx.AsyncClient(timeout=120.0) as client:
                    content = await download_with_retry(
                        client, source,
                        max_retries=5,  # More retries for audio
                        base_delay=2.0,  # Longer base delay
                        log_func=self.log
                    )
                    if content:
                        local_path.write_bytes(content)
                        self.log(f"Downloaded audio: {local_path.name} ({len(content)} bytes)")
                        return str(local_path)
                    else:
                        self.log(f"CRITICAL: Audio download failed after all retries: {source[:60]}")
                        return None
            elif os.path.exists(source):
                shutil.copy(source, local_path)
                self.log(f"Copied audio: {local_path.name}")
                return str(local_path)

        except Exception as e:
            self.log(f"Error preparing audio: {e}")

        return None

    def _build_ffmpeg_command(
        self,
        timeline: Timeline,
        asset_map: Dict[str, str],
        audio_path: Optional[str],
        output_path: Path,
        resolution: Tuple[int, int],
        fps: int,
        quality: str
    ) -> List[str]:
        """
        Build the FFmpeg command with filtercomplex.

        Strategy:
        1. Create a base canvas (color source)
        2. Overlay each visual event with enable='between(t,start,end)'
        3. Add audio track
        4. Encode to output
        """
        width, height = resolution

        # Quality presets - optimized for shared CPU (4 vCPU)
        # veryfast is 2-3x faster than medium with minimal quality loss
        quality_presets = {
            "low": {"crf": 28, "preset": "ultrafast"},
            "medium": {"crf": 23, "preset": "veryfast"},   # Changed from "medium"
            "high": {"crf": 20, "preset": "fast"}          # Changed from "slow"
        }
        preset = quality_presets.get(quality, quality_presets["medium"])

        # Build input list and filtercomplex
        inputs = []
        filter_parts = []

        # Input 0: Base canvas (solid color or first slide as background)
        # We use color source as base
        base_filter = f"color=c=0x1a1a2e:s={width}x{height}:d={timeline.total_duration}:r={fps}[base]"
        filter_parts.append(base_filter)

        current_stream = "base"
        input_idx = 0

        # Sort events by time and layer
        sorted_events = sorted(
            timeline.visual_events,
            key=lambda e: (e.time_start, e.layer)
        )

        for i, event in enumerate(sorted_events):
            source = event.asset_url or event.asset_path
            local_path = asset_map.get(source)

            if not local_path:
                self.log(f"Skipping event {i}: no local asset")
                continue

            inputs.append(local_path)
            input_stream = f"{input_idx}:v"
            input_idx += 1

            # Calculate enable expression
            enable = f"between(t,{event.time_start:.3f},{event.time_end:.3f})"

            # Determine scaling
            # Scale input to fit within the resolution while maintaining aspect ratio
            scale_filter = f"[{input_stream}]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2[scaled{i}]"
            filter_parts.append(scale_filter)

            # Handle different event types
            if event.event_type == VisualEventType.FREEZE_FRAME:
                # For freeze frame, use trim and loop
                freeze_filter = f"[scaled{i}]trim=end_frame=1,loop=-1:1[frozen{i}]"
                filter_parts.append(freeze_filter)
                overlay_input = f"frozen{i}"
            else:
                overlay_input = f"scaled{i}"

            # Overlay with timing
            output_stream = f"out{i}"
            overlay_filter = f"[{current_stream}][{overlay_input}]overlay=enable='{enable}'[{output_stream}]"
            filter_parts.append(overlay_filter)
            current_stream = output_stream

        # Final output formatting
        final_filter = f"[{current_stream}]format=yuv420p[vout]"
        filter_parts.append(final_filter)

        filtercomplex = ";".join(filter_parts)

        # Build command
        cmd = ["ffmpeg", "-y"]

        # Add inputs
        for input_path in inputs:
            cmd.extend(["-i", input_path])

        # Add audio input if available
        if audio_path:
            cmd.extend(["-i", audio_path])

        # Add filtercomplex
        cmd.extend(["-filter_complex", filtercomplex])

        # Map video output
        cmd.extend(["-map", "[vout]"])

        # Map audio if available
        if audio_path:
            audio_input_idx = len(inputs)
            cmd.extend(["-map", f"{audio_input_idx}:a"])

        # Encoding settings
        # Use explicit duration instead of -shortest to prevent sync drift
        cmd.extend([
            "-c:v", "libx264",
            "-crf", str(preset["crf"]),
            "-preset", preset["preset"],
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(round(timeline.total_duration, 3)),  # Explicit duration for sync
            "-async", "1",  # Resync audio if needed
            str(output_path)
        ])

        return cmd

    async def _execute_ffmpeg(self, cmd: List[str]) -> bool:
        """Execute FFmpeg command asynchronously"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=600  # 10 minutes max
            )

            if process.returncode == 0:
                self.log("FFmpeg completed successfully")
                return True
            else:
                self.log(f"FFmpeg failed with code {process.returncode}")
                self.log(f"stderr: {stderr.decode()[-1000:]}")  # Last 1000 chars
                return False

        except asyncio.TimeoutError:
            self.log("FFmpeg timed out")
            process.kill()
            return False
        except Exception as e:
            self.log(f"FFmpeg execution error: {e}")
            return False


class SimpleTimelineCompositor:
    """
    Simpler compositor that uses concat demuxer instead of filtercomplex.
    Better for sequential slides without complex overlays.
    """

    def __init__(self, output_dir: str = "/tmp/presentations/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._asset_cache: Dict[str, str] = {}  # URL -> local path cache

    def log(self, message: str):
        print(f"[SIMPLE_COMPOSITOR] {message}", flush=True)

    async def _download_asset(self, url: str, temp_dir: Path) -> Optional[str]:
        """
        Download a remote asset to local temp directory.

        Uses retry with exponential backoff for reliability.
        """
        import httpx
        import hashlib

        # Check cache first
        if url in self._asset_cache:
            cached = self._asset_cache[url]
            if os.path.exists(cached):
                return cached

        # Determine file extension from URL or content type
        url_path = url.split("?")[0]
        ext = Path(url_path).suffix

        # If no extension, infer from URL path
        if not ext:
            if "/audio/" in url_path or "voiceover" in url_path:
                ext = ".mp3"
            elif "/video/" in url_path:
                ext = ".mp4"
            else:
                ext = ".png"  # Default for images

        # Generate unique filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        local_path = temp_dir / f"asset_{url_hash}{ext}"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                self.log(f"Downloading: {url[:80]}...")

                # Use retry with backoff
                content = await download_with_retry(
                    client, url,
                    max_retries=MAX_RETRIES,
                    log_func=self.log
                )

                if content:
                    local_path.write_bytes(content)
                    self._asset_cache[url] = str(local_path)
                    self.log(f"Downloaded: {local_path.name} ({len(content)} bytes)")
                    return str(local_path)
                else:
                    self.log(f"Download failed after retries for {url[:60]}")
                    return None

        except Exception as e:
            self.log(f"Download error for {url[:60]}: {e}")
            return None

    async def compose(
        self,
        timeline: Timeline,
        output_filename: str,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30
    ) -> CompositionResult:
        """
        Compose using concat demuxer - simpler but less flexible.
        Each event becomes a segment with exact duration.

        Uses global semaphore to limit concurrent FFmpeg processes
        and prevent memory exhaustion.
        """
        # Extract job_id from filename for tracking
        job_id = output_filename.split("_")[0] if "_" in output_filename else "unknown"

        # Acquire FFmpeg slot (blocks if too many concurrent processes)
        async with ffmpeg_manager.acquire(job_id, "compose"):
            return await self._compose_internal(timeline, output_filename, resolution, fps)

    async def _compose_internal(
        self,
        timeline: Timeline,
        output_filename: str,
        resolution: Tuple[int, int],
        fps: int
    ) -> CompositionResult:
        """Internal compose logic, called within semaphore context."""
        temp_dir = Path(tempfile.mkdtemp(prefix="simple_compose_"))
        self._asset_cache = {}  # Clear cache for new composition

        total_events = len(timeline.visual_events)
        self.log(f"=== Starting composition: {total_events} events, duration={timeline.total_duration:.2f}s ===")
        self.log(f"Output: {output_filename}, Resolution: {resolution}, FPS: {fps}")
        self.log(f"Temp dir: {temp_dir}")

        try:
            # Step 1: Create video segment for each event
            segments = []
            failed_segments = []

            for i, event in enumerate(timeline.visual_events):
                if event.event_type in [VisualEventType.BULLET_REVEAL, VisualEventType.HIGHLIGHT]:
                    self.log(f"[{i+1}/{total_events}] Skipping overlay event: {event.event_type}")
                    continue  # Skip overlay-only events

                source = event.asset_path or event.asset_url
                self.log(f"[{i+1}/{total_events}] Processing: type={event.event_type}, duration={event.duration:.2f}s")
                self.log(f"  Source: {source[:100] if source else 'None'}...")

                segment_path = temp_dir / f"segment_{i:04d}.mp4"
                success = await self._create_segment(
                    event, segment_path, resolution, fps, temp_dir
                )

                if success:
                    segments.append(str(segment_path))
                    self.log(f"  -> Segment created: {segment_path.name}")
                else:
                    failed_segments.append(i)
                    self.log(f"  -> FAILED to create segment")

            self.log(f"=== Segment creation complete: {len(segments)} success, {len(failed_segments)} failed ===")
            if failed_segments:
                self.log(f"Failed segment indices: {failed_segments}")

                # CRITICAL: Never deliver incomplete videos
                # Option 1: Retry failed segments with increased timeout
                retry_success = await self._retry_failed_segments(
                    timeline, failed_segments, segments, temp_dir, resolution, fps
                )

                if not retry_success:
                    # Option 2: Fail hard - don't deliver corrupted video
                    error_msg = f"Video incomplete: {len(failed_segments)} segments failed after retry. Indices: {failed_segments}"
                    self.log(f"CRITICAL ERROR: {error_msg}")
                    return CompositionResult(success=False, error=error_msg)

            if not segments:
                self.log("ERROR: No segments created, aborting composition")
                return CompositionResult(success=False, error="No segments created")

            # Step 2: Create concat file
            self.log(f"=== Step 2: Creating concat file with {len(segments)} segments ===")
            concat_file = temp_dir / "concat.txt"
            with open(concat_file, "w") as f:
                for seg in segments:
                    f.write(f"file '{seg}'\n")

            # Step 3: Concat segments
            self.log("=== Step 3: Concatenating segments ===")
            concat_output = temp_dir / "video_only.mp4"
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(concat_output)
            ]

            process = await asyncio.create_subprocess_exec(
                *concat_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            except asyncio.TimeoutError:
                self.log("ERROR: Concat timeout after 300s, killing process")
                process.kill()
                await process.wait()
                return CompositionResult(success=False, error="Concat timeout after 300s")

            if process.returncode != 0:
                error_msg = stderr.decode()[:500] if stderr else "Unknown error"
                self.log(f"ERROR: Concat failed with code {process.returncode}: {error_msg}")
                return CompositionResult(success=False, error=f"Concat failed: {error_msg}")

            self.log(f"Concat successful: {concat_output.name}")

            # Step 4: Add audio
            self.log("=== Step 4: Adding audio track ===")
            output_path = self.output_dir / output_filename
            audio_source = timeline.audio_track_path or timeline.audio_track_url

            self.log(f"Audio source: {audio_source[:100] if audio_source else 'None'}")

            if audio_source:
                # Download audio if remote URL
                if audio_source.startswith("http"):
                    self.log(f"Audio is remote URL, downloading...")
                    local_audio = await self._download_asset(audio_source, temp_dir)
                    if local_audio:
                        self.log(f"Audio downloaded: {local_audio}")
                        audio_source = local_audio
                    else:
                        self.log(f"WARNING: Failed to download audio, proceeding without: {audio_source[:60]}")
                        audio_source = None

            if audio_source:
                self.log(f"Muxing audio with video (target duration: {timeline.total_duration:.3f}s)...")
                # SYNC FIX: Reset timestamps and ensure audio-video alignment
                # - Use -ss 0 to start both streams from the beginning
                # - Use explicit mapping to control which streams are used
                # - Use -t for exact duration control
                # - Do NOT use -async as it can cause audio drift
                mux_cmd = [
                    "ffmpeg", "-y",
                    "-ss", "0",  # Start from beginning
                    "-i", str(concat_output),
                    "-ss", "0",  # Start audio from beginning too
                    "-i", audio_source,
                    "-map", "0:v:0",  # Take video from first input
                    "-map", "1:a:0",  # Take audio from second input
                    "-t", str(round(timeline.total_duration, 3)),  # Explicit duration
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-vsync", "cfr",  # Constant frame rate for better sync
                    str(output_path)
                ]
            else:
                self.log("No audio, copying video directly")
                mux_cmd = ["cp", str(concat_output), str(output_path)]

            process = await asyncio.create_subprocess_exec(
                *mux_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            except asyncio.TimeoutError:
                self.log("ERROR: Audio mux timeout after 300s, killing process")
                process.kill()
                await process.wait()
                return CompositionResult(success=False, error="Audio mux timeout after 300s")

            if process.returncode != 0:
                error_msg = stderr.decode()[:500] if stderr else "Unknown error"
                self.log(f"ERROR: Audio mux failed with code {process.returncode}: {error_msg}")
                return CompositionResult(success=False, error=f"Audio mux failed: {error_msg}")

            if output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                self.log(f"=== COMPOSITION COMPLETE ===")
                self.log(f"Output: {output_path}")
                self.log(f"Size: {file_size:.2f} MB, Duration: {timeline.total_duration:.2f}s")
                return CompositionResult(
                    success=True,
                    output_path=str(output_path),
                    duration=timeline.total_duration
                )

            self.log("ERROR: Final output file not created")
            return CompositionResult(success=False, error="Final output not created")

        except Exception as e:
            self.log(f"ERROR: Unexpected exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return CompositionResult(success=False, error=str(e))

        finally:
            self.log(f"Cleaning up temp dir: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _create_segment(
        self,
        event: VisualEvent,
        output_path: Path,
        resolution: Tuple[int, int],
        fps: int,
        temp_dir: Path
    ) -> bool:
        """Create a video segment from an event with exact duration"""
        source = event.asset_path or event.asset_url
        if not source:
            self.log(f"  No source for event, skipping")
            return False

        original_source = source

        # Download remote URLs to local files first
        if source.startswith("http"):
            self.log(f"  Downloading remote asset...")
            local_source = await self._download_asset(source, temp_dir)
            if not local_source:
                self.log(f"  FAILED to download: {source[:80]}")
                return False
            source = local_source
            self.log(f"  Using local file: {source}")

        width, height = resolution
        duration = event.duration

        # Determine if source is image or video
        is_image = source.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        self.log(f"  FFmpeg encoding: {'image' if is_image else 'video'} -> {duration:.2f}s segment")

        # Use ultrafast preset for speed (less CPU/memory usage)
        # Trade-off: larger file size but much faster encoding
        preset = os.getenv("FFMPEG_PRESET", "ultrafast")

        # Check for diagram focus animations (SSVS-D hybrid sync)
        # Can be disabled via ENABLE_DIAGRAM_FOCUS=false
        diagram_focus = event.metadata.get("diagram_focus", {}) if event.metadata else {}
        diagram_focus_filter = ""
        if diagram_focus.get("enabled") and diagram_focus.get("ffmpeg_filter"):
            diagram_focus_filter = "," + diagram_focus["ffmpeg_filter"]
            self.log(f"  Applying diagram focus filter ({diagram_focus.get('focus_points', 0)} focus points)")

        # Build video filter chain
        # Base filters: scale, pad, format
        # Optional: diagram focus filters (drawbox for highlighting elements)
        base_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        video_filter = f"{base_filter}{diagram_focus_filter},format=yuv420p"

        if is_image:
            # Image to video
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", source,
                "-t", str(duration),
                "-vf", video_filter,
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", preset,
                "-threads", "2",  # 2 threads per process (optimal for 4 vCPU with 2 concurrent)
                "-an",
                str(output_path)
            ]
        else:
            # Video - adjust duration
            cmd = [
                "ffmpeg", "-y",
                "-i", source,
                "-t", str(duration),
                "-vf", video_filter,
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", preset,
                "-threads", "2",  # 2 threads per process (optimal for 4 vCPU with 2 concurrent)
                "-an",
                str(output_path)
            ]

        # Dynamic timeout: at least 120s, or 3x segment duration (generous for slow CPUs)
        timeout_seconds = max(120, int(duration * 3))

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)

            if process.returncode != 0:
                error_msg = stderr.decode()[:500] if stderr else "Unknown error"
                self.log(f"  FFmpeg FAILED (code {process.returncode}): {error_msg}")
                return False

            # Verify output file was created
            if output_path.exists():
                size_kb = output_path.stat().st_size / 1024
                self.log(f"  Segment OK: {output_path.name} ({size_kb:.1f} KB)")
                return True
            else:
                self.log(f"  FFmpeg returned 0 but output not created: {output_path}")
                return False
        except asyncio.TimeoutError:
            self.log(f"  FFmpeg TIMEOUT after {timeout_seconds}s for: {source}")
            try:
                process.kill()
                await process.wait()
            except:
                pass
            # Cleanup on timeout
            gc.collect()
            return False
        except Exception as e:
            self.log(f"Segment creation failed: {type(e).__name__}: {e}")
            gc.collect()
            return False
        finally:
            # Always cleanup after FFmpeg to prevent memory accumulation
            gc.collect()

    async def _retry_failed_segments(
        self,
        timeline: Timeline,
        failed_indices: List[int],
        segments: List[str],
        temp_dir: Path,
        resolution: Tuple[int, int],
        fps: int,
        max_retries: int = 2
    ) -> bool:
        """
        Retry failed segments with increased timeout.

        Returns True if all segments eventually succeed, False otherwise.
        """
        self.log(f"=== RETRYING {len(failed_indices)} failed segments ===")

        all_recovered = True
        events = [e for e in timeline.visual_events
                  if e.event_type not in [VisualEventType.BULLET_REVEAL, VisualEventType.HIGHLIGHT]]

        for idx in failed_indices:
            if idx >= len(events):
                self.log(f"  Invalid segment index {idx}, skipping")
                all_recovered = False
                continue

            event = events[idx]
            segment_path = temp_dir / f"segment_{idx:04d}.mp4"

            for attempt in range(1, max_retries + 1):
                self.log(f"  Retry {attempt}/{max_retries} for segment {idx}...")

                # Increase timeout for retry (4x original duration, minimum 240s)
                original_timeout = max(120, int(event.duration * 3))
                extended_timeout = max(240, original_timeout * 2)

                # Temporarily modify the timeout for this attempt
                success = await self._create_segment_with_timeout(
                    event, segment_path, resolution, fps, temp_dir, extended_timeout
                )

                if success:
                    # Insert at correct position in segments list
                    # Find where to insert based on index
                    insert_pos = 0
                    for i, seg in enumerate(segments):
                        seg_idx = int(Path(seg).stem.split('_')[1])
                        if seg_idx > idx:
                            insert_pos = i
                            break
                        insert_pos = i + 1
                    segments.insert(insert_pos, str(segment_path))
                    self.log(f"  Segment {idx} recovered on retry {attempt}")
                    break
            else:
                # All retries failed for this segment
                self.log(f"  Segment {idx} FAILED after {max_retries} retries")
                all_recovered = False

        return all_recovered

    async def _create_segment_with_timeout(
        self,
        event: VisualEvent,
        output_path: Path,
        resolution: Tuple[int, int],
        fps: int,
        temp_dir: Path,
        timeout_seconds: int
    ) -> bool:
        """Create segment with specified timeout (for retries with extended timeout)."""
        source = event.asset_path or event.asset_url
        if not source:
            return False

        # Download remote URLs
        if source.startswith("http"):
            local_source = await self._download_asset(source, temp_dir)
            if not local_source:
                return False
            source = local_source

        width, height = resolution
        duration = event.duration
        is_image = source.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        preset = os.getenv("FFMPEG_PRESET", "ultrafast")

        base_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        video_filter = f"{base_filter},format=yuv420p"

        if is_image:
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", source,
                "-t", str(duration),
                "-vf", video_filter,
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", preset,
                "-threads", "2",  # 2 threads per process (optimal for 4 vCPU)
                "-an",
                str(output_path)
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", source,
                "-t", str(duration),
                "-vf", video_filter,
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", preset,
                "-threads", "2",  # 2 threads per process (optimal for 4 vCPU)
                "-an",
                str(output_path)
            ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)

            if process.returncode == 0 and output_path.exists():
                return True
            return False
        except asyncio.TimeoutError:
            self.log(f"  Extended timeout ({timeout_seconds}s) also failed")
            try:
                process.kill()
                await process.wait()
            except:
                pass
            return False
        except Exception as e:
            self.log(f"  Retry segment creation failed: {e}")
            return False
