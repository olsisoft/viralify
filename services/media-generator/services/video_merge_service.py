"""
Video Merge Service

Renders the final video by merging all timeline segments with transitions,
overlays, and audio mixing.
Phase 3: Video Editor feature.
"""
import asyncio
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from models.video_editor_models import (
    VideoProject,
    VideoSegment,
    SegmentType,
    ProjectStatus,
    TransitionType,
    TextOverlay,
    ImageOverlay,
)


class VideoMergeService:
    """
    Service for merging video segments and rendering final output.
    Uses FFmpeg for video processing.
    """

    # Quality presets
    QUALITY_PRESETS = {
        'low': {'crf': 28, 'preset': 'faster', 'audio_bitrate': '128k'},
        'medium': {'crf': 23, 'preset': 'medium', 'audio_bitrate': '192k'},
        'high': {'crf': 18, 'preset': 'slow', 'audio_bitrate': '256k'},
    }

    def __init__(self, output_path: str = "/tmp/viralify/editor/output"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"[MERGE] Video merge service initialized at {output_path}", flush=True)

    async def render_project(
        self,
        project: VideoProject,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Render the complete video project.

        Args:
            project: VideoProject to render
            progress_callback: Optional callback for progress updates (progress%, message)

        Returns:
            Path to rendered video file
        """
        print(f"[MERGE] Starting render for project {project.id}", flush=True)
        print(f"[MERGE] {len(project.segments)} segments, {project.total_duration}s duration", flush=True)

        if not project.segments:
            raise ValueError("No segments to render")

        # Get quality settings
        quality = self.QUALITY_PRESETS.get(project.output_quality, self.QUALITY_PRESETS['medium'])

        # Parse resolution
        try:
            width, height = map(int, project.output_resolution.split('x'))
        except:
            width, height = 1920, 1080

        # Output file path
        output_filename = f"{project.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{project.output_format}"
        output_file = self.output_path / output_filename

        # Progress tracking
        total_steps = len(project.segments) + 2  # segments + concat + finalize
        current_step = 0

        def update_progress(message: str):
            nonlocal current_step
            current_step += 1
            progress = (current_step / total_steps) * 100
            if progress_callback:
                progress_callback(progress, message)
            print(f"[MERGE] Progress {progress:.1f}%: {message}", flush=True)

        try:
            # Step 1: Prepare segments (normalize, trim, apply effects)
            prepared_segments = []

            for idx, segment in enumerate(project.segments):
                update_progress(f"Processing segment {idx + 1}/{len(project.segments)}")

                prepared_path = await self._prepare_segment(
                    segment,
                    width,
                    height,
                    project.output_fps,
                )
                prepared_segments.append(prepared_path)

            # Step 2: Concatenate segments with transitions
            update_progress("Merging segments...")

            concat_file = await self._concatenate_segments(
                prepared_segments,
                project.segments,
                width,
                height,
                project.output_fps,
            )

            # Step 3: Apply overlays and background music, finalize
            update_progress("Applying overlays and finalizing...")

            await self._finalize_video(
                concat_file,
                str(output_file),
                project,
                quality,
                width,
                height,
            )

            # Cleanup temp files
            for temp_file in prepared_segments:
                if Path(temp_file).exists() and 'temp' in temp_file:
                    Path(temp_file).unlink()

            if Path(concat_file).exists():
                Path(concat_file).unlink()

            print(f"[MERGE] Render complete: {output_file}", flush=True)

            return str(output_file)

        except Exception as e:
            print(f"[MERGE] Render error: {e}", flush=True)
            raise

    async def _prepare_segment(
        self,
        segment: VideoSegment,
        width: int,
        height: int,
        fps: int,
    ) -> str:
        """
        Prepare a single segment for merging.
        Normalizes resolution, applies trim, adjusts audio.
        """
        if not segment.source_url:
            raise ValueError(f"Segment {segment.id} has no source URL")

        source_path = segment.source_url

        # For slides/images, create a video
        if segment.segment_type == SegmentType.SLIDE:
            return await self._image_to_video(
                source_path,
                segment.duration,
                width,
                height,
                fps,
            )

        # Prepare ffmpeg command for video
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.mp4',
            delete=False,
            dir=str(self.output_path / 'temp'),
        )
        temp_path = temp_file.name
        temp_file.close()

        # Build filter chain
        filters = []

        # Scale and pad to target resolution
        filters.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease")
        filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black")

        # FPS
        filters.append(f"fps={fps}")

        # Opacity
        if segment.opacity < 1.0:
            filters.append(f"colorchannelmixer=aa={segment.opacity}")

        filter_str = ','.join(filters)

        # Build command
        cmd = ['ffmpeg', '-y']

        # Input with trim
        if segment.trim_start > 0:
            cmd.extend(['-ss', str(segment.trim_start)])

        cmd.extend(['-i', source_path])

        if segment.trim_end:
            duration = segment.trim_end - segment.trim_start
            cmd.extend(['-t', str(duration)])
        elif segment.duration:
            cmd.extend(['-t', str(segment.duration)])

        # Video filter
        cmd.extend(['-vf', filter_str])

        # Audio
        if segment.is_audio_muted:
            cmd.extend(['-an'])  # No audio
        else:
            cmd.extend([
                '-af', f'volume={segment.original_audio_volume}',
                '-c:a', 'aac',
                '-b:a', '192k',
            ])

        # Output
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '20',
            temp_path
        ])

        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        _, stderr = await result.communicate()

        if result.returncode != 0:
            print(f"[MERGE] Segment prep error: {stderr.decode()[:500]}", flush=True)
            # Fallback: just copy the source
            return source_path

        return temp_path

    async def _image_to_video(
        self,
        image_path: str,
        duration: float,
        width: int,
        height: int,
        fps: int,
    ) -> str:
        """Convert an image to a video of specified duration"""
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.mp4',
            delete=False,
            dir=str(self.output_path / 'temp'),
        )
        temp_path = temp_file.name
        temp_file.close()

        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', image_path,
            '-t', str(duration),
            '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={fps}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '20',
            '-pix_fmt', 'yuv420p',
            # Add silent audio track for consistency
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
            '-shortest',
            temp_path
        ]

        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        await result.communicate()

        return temp_path

    async def _concatenate_segments(
        self,
        segment_paths: List[str],
        segments: List[VideoSegment],
        width: int,
        height: int,
        fps: int,
    ) -> str:
        """Concatenate all segments with transitions"""
        if len(segment_paths) == 1:
            return segment_paths[0]

        # Create concat file
        concat_file = tempfile.NamedTemporaryFile(
            suffix='.txt',
            delete=False,
            mode='w',
            dir=str(self.output_path / 'temp'),
        )

        for path in segment_paths:
            concat_file.write(f"file '{path}'\n")

        concat_file.close()

        # Output file
        output_file = tempfile.NamedTemporaryFile(
            suffix='.mp4',
            delete=False,
            dir=str(self.output_path / 'temp'),
        )
        output_path = output_file.name
        output_file.close()

        # Build ffmpeg concat command
        # For transitions, we'd need complex filter graphs
        # For now, use simple concat demuxer
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file.name,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '20',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]

        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        _, stderr = await result.communicate()

        # Cleanup concat file
        Path(concat_file.name).unlink()

        if result.returncode != 0:
            print(f"[MERGE] Concat error: {stderr.decode()[:500]}", flush=True)
            raise RuntimeError("Failed to concatenate segments")

        return output_path

    async def _finalize_video(
        self,
        input_path: str,
        output_path: str,
        project: VideoProject,
        quality: dict,
        width: int,
        height: int,
    ) -> None:
        """
        Apply final touches: overlays, background music, encoding.
        """
        filters = []
        inputs = ['-i', input_path]
        filter_complex = []
        current_stream = '[0:v]'

        # Add image overlays (watermarks, logos)
        for idx, overlay in enumerate(project.image_overlays):
            inputs.extend(['-i', overlay.image_url])
            input_idx = idx + 1

            # Calculate position
            x = int(overlay.position_x * width - (overlay.scale * width / 2))
            y = int(overlay.position_y * height - (overlay.scale * height / 2))

            # Build overlay filter
            scale_w = int(overlay.scale * width)
            overlay_filter = f"[{input_idx}:v]scale={scale_w}:-1,format=rgba,colorchannelmixer=aa={overlay.opacity}[ovl{idx}]"
            filter_complex.append(overlay_filter)

            # Apply overlay
            new_stream = f'[vid{idx}]'
            enable_str = ""
            if overlay.start_time is not None and overlay.duration is not None:
                enable_str = f":enable='between(t,{overlay.start_time},{overlay.start_time + overlay.duration})'"

            filter_complex.append(f"{current_stream}[ovl{idx}]overlay={x}:{y}{enable_str}{new_stream}")
            current_stream = new_stream

        # Add text overlays
        for idx, text_overlay in enumerate(project.text_overlays):
            # Escape text for ffmpeg
            escaped_text = text_overlay.text.replace("'", "\\'").replace(":", "\\:")

            x_pos = int(text_overlay.position_x * width)
            y_pos = int(text_overlay.position_y * height)

            text_filter = (
                f"drawtext=text='{escaped_text}'"
                f":fontsize={text_overlay.font_size}"
                f":fontcolor={text_overlay.font_color}"
                f":x={x_pos}:y={y_pos}"
                f":enable='between(t,{text_overlay.start_time},{text_overlay.start_time + text_overlay.duration})'"
            )

            if text_overlay.background_color:
                text_filter += f":box=1:boxcolor={text_overlay.background_color}"

            new_stream = f'[txt{idx}]'
            filter_complex.append(f"{current_stream}{text_filter}{new_stream}")
            current_stream = new_stream

        # Build final command
        cmd = ['ffmpeg', '-y']
        cmd.extend(inputs)

        # Background music
        if project.background_music_url:
            cmd.extend(['-i', project.background_music_url])

        # Filters
        if filter_complex:
            cmd.extend(['-filter_complex', ';'.join(filter_complex)])
            cmd.extend(['-map', current_stream])
        else:
            cmd.extend(['-map', '0:v'])

        # Audio mixing
        if project.background_music_url:
            # Mix original audio with background music
            audio_filter = f"[0:a]volume=1[a0];[{len(inputs)//2}:a]volume={project.background_music_volume}[a1];[a0][a1]amix=inputs=2:duration=first[aout]"
            cmd.extend(['-filter_complex', audio_filter, '-map', '[aout]'])
        else:
            cmd.extend(['-map', '0:a?'])

        # Output encoding
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', quality['preset'],
            '-crf', str(quality['crf']),
            '-c:a', 'aac',
            '-b:a', quality['audio_bitrate'],
            '-movflags', '+faststart',  # Web optimization
            output_path
        ])

        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        _, stderr = await result.communicate()

        if result.returncode != 0:
            # Fallback: simple copy without filters
            print(f"[MERGE] Finalize with filters failed, falling back to simple copy", flush=True)
            cmd_simple = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', quality['preset'],
                '-crf', str(quality['crf']),
                '-c:a', 'aac',
                '-b:a', quality['audio_bitrate'],
                output_path
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd_simple,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await result.communicate()

            if result.returncode != 0:
                raise RuntimeError("Failed to finalize video")

    async def create_preview(
        self,
        project: VideoProject,
        start_time: float,
        duration: float = 10.0,
    ) -> str:
        """
        Create a short preview of the project at a specific timestamp.

        Args:
            project: VideoProject
            start_time: Start time in seconds
            duration: Preview duration (default 10 seconds)

        Returns:
            Path to preview video
        """
        # Find which segment(s) this covers
        segments_to_render = []
        current_time = 0.0

        for segment in project.segments:
            seg_end = segment.start_time + segment.duration

            if segment.start_time <= start_time < seg_end:
                segments_to_render.append(segment)
            elif start_time < segment.start_time < start_time + duration:
                segments_to_render.append(segment)

        if not segments_to_render:
            raise ValueError("No segments found for preview time range")

        # Create temporary project with just these segments
        preview_project = VideoProject(
            id=f"preview_{uuid.uuid4().hex[:8]}",
            user_id=project.user_id,
            title=f"Preview of {project.title}",
            segments=segments_to_render,
            output_resolution=project.output_resolution,
            output_fps=project.output_fps,
            output_quality='low',  # Fast preview
        )

        # Render preview
        output_path = await self.render_project(preview_project)

        return output_path
