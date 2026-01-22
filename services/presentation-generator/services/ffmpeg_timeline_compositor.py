"""
FFmpeg Timeline Compositor

Composes video from a Timeline using FFmpeg filtercomplex.
This provides millisecond-precision synchronization between audio and visuals.

Key Features:
- Uses overlay with enable='between(t,start,end)' for precise timing
- Handles freeze frames for padding
- Supports multiple layers (background, content, overlays)
- Crossfade transitions between slides
"""

import os
import asyncio
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .timeline_builder import Timeline, VisualEvent, VisualEventType


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
        """
        import httpx

        asset_map = {}
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i, event in enumerate(timeline.visual_events):
                source = event.asset_url or event.asset_path

                if not source:
                    continue

                # Determine file extension
                ext = Path(source.split("?")[0]).suffix or ".mp4"
                local_path = self.temp_dir / f"asset_{i}{ext}"

                try:
                    if source.startswith("http"):
                        # Download from URL
                        response = await client.get(source)
                        if response.status_code == 200:
                            local_path.write_bytes(response.content)
                            asset_map[source] = str(local_path)
                            self.log(f"Downloaded: {source} -> {local_path.name}")
                        else:
                            self.log(f"Failed to download {source}: {response.status_code}")
                    elif os.path.exists(source):
                        # Copy local file
                        shutil.copy(source, local_path)
                        asset_map[source] = str(local_path)
                        self.log(f"Copied: {source} -> {local_path.name}")
                    else:
                        self.log(f"Asset not found: {source}")

                except Exception as e:
                    self.log(f"Error preparing asset {source}: {e}")

        self.log(f"Prepared {len(asset_map)} assets")
        return asset_map

    async def _prepare_audio(self, timeline: Timeline) -> Optional[str]:
        """Download/copy audio track to temp directory"""
        import httpx

        source = timeline.audio_track_url or timeline.audio_track_path

        if not source:
            return None

        ext = Path(source.split("?")[0]).suffix or ".mp3"
        local_path = self.temp_dir / f"audio{ext}"

        try:
            if source.startswith("http"):
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.get(source)
                    if response.status_code == 200:
                        local_path.write_bytes(response.content)
                        self.log(f"Downloaded audio: {local_path.name}")
                        return str(local_path)
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

        # Quality presets
        quality_presets = {
            "low": {"crf": 28, "preset": "veryfast"},
            "medium": {"crf": 23, "preset": "medium"},
            "high": {"crf": 18, "preset": "slow"}
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
        cmd.extend([
            "-c:v", "libx264",
            "-crf", str(preset["crf"]),
            "-preset", preset["preset"],
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",  # End when shortest input ends
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

    def log(self, message: str):
        print(f"[SIMPLE_COMPOSITOR] {message}", flush=True)

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
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="simple_compose_"))

        try:
            # Step 1: Create video segment for each event
            segments = []

            for i, event in enumerate(timeline.visual_events):
                if event.event_type in [VisualEventType.BULLET_REVEAL, VisualEventType.HIGHLIGHT]:
                    continue  # Skip overlay-only events

                segment_path = temp_dir / f"segment_{i:04d}.mp4"
                success = await self._create_segment(
                    event, segment_path, resolution, fps
                )

                if success:
                    segments.append(str(segment_path))

            if not segments:
                return CompositionResult(success=False, error="No segments created")

            # Step 2: Create concat file
            concat_file = temp_dir / "concat.txt"
            with open(concat_file, "w") as f:
                for seg in segments:
                    f.write(f"file '{seg}'\n")

            # Step 3: Concat segments
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
            await process.communicate()

            if process.returncode != 0:
                return CompositionResult(success=False, error="Concat failed")

            # Step 4: Add audio
            output_path = self.output_dir / output_filename
            audio_source = timeline.audio_track_path or timeline.audio_track_url

            if audio_source:
                mux_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(concat_output),
                    "-i", audio_source,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    str(output_path)
                ]
            else:
                mux_cmd = ["cp", str(concat_output), str(output_path)]

            process = await asyncio.create_subprocess_exec(
                *mux_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if output_path.exists():
                return CompositionResult(
                    success=True,
                    output_path=str(output_path),
                    duration=timeline.total_duration
                )

            return CompositionResult(success=False, error="Final output not created")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _create_segment(
        self,
        event: VisualEvent,
        output_path: Path,
        resolution: Tuple[int, int],
        fps: int
    ) -> bool:
        """Create a video segment from an event with exact duration"""
        source = event.asset_path or event.asset_url
        if not source:
            return False

        width, height = resolution
        duration = event.duration

        # Determine if source is image or video
        is_image = source.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))

        if is_image:
            # Image to video
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", source,
                "-t", str(duration),
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", "fast",
                "-an",
                str(output_path)
            ]
        else:
            # Video - adjust duration
            cmd = [
                "ffmpeg", "-y",
                "-i", source,
                "-t", str(duration),
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", "fast",
                "-an",
                str(output_path)
            ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(process.communicate(), timeout=60)
            return process.returncode == 0
        except Exception as e:
            self.log(f"Segment creation failed: {e}")
            return False
