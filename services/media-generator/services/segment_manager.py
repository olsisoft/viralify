"""
Segment Manager Service

Handles user video uploads, video processing, and thumbnail generation.
Phase 3: Video Editor feature.
"""
import asyncio
import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional, Tuple

from models.video_editor_models import (
    VideoSegment,
    SegmentType,
    SegmentStatus,
)


class SegmentManager:
    """
    Service for managing video segments.
    Handles uploads, video info extraction, and thumbnail generation.
    """

    # Supported video formats
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.aac', '.m4a', '.ogg'}
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

    # Max file sizes
    MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB
    MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 50 MB
    MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB

    def __init__(self, storage_path: str = "/tmp/viralify/editor"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.videos_path = self.storage_path / "videos"
        self.thumbnails_path = self.storage_path / "thumbnails"
        self.temp_path = self.storage_path / "temp"

        for path in [self.videos_path, self.thumbnails_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)

        print(f"[SEGMENT] Segment manager initialized at {storage_path}", flush=True)

    async def process_upload(
        self,
        file_content: bytes,
        filename: str,
        project_id: str,
        user_id: str,
    ) -> Tuple[str, float, Optional[str], SegmentType]:
        """
        Process an uploaded file (video, audio, or image).

        Args:
            file_content: Raw file bytes
            filename: Original filename
            project_id: Project ID
            user_id: User ID

        Returns:
            Tuple of (file_url, duration, thumbnail_url, segment_type)
        """
        print(f"[SEGMENT] Processing upload: {filename} ({len(file_content)} bytes)", flush=True)

        # Determine file type
        ext = Path(filename).suffix.lower()

        if ext in self.SUPPORTED_VIDEO_FORMATS:
            if len(file_content) > self.MAX_VIDEO_SIZE:
                raise ValueError(f"Video file too large (max {self.MAX_VIDEO_SIZE // 1024 // 1024} MB)")
            return await self._process_video(file_content, filename, project_id, user_id)

        elif ext in self.SUPPORTED_AUDIO_FORMATS:
            if len(file_content) > self.MAX_AUDIO_SIZE:
                raise ValueError(f"Audio file too large (max {self.MAX_AUDIO_SIZE // 1024 // 1024} MB)")
            return await self._process_audio(file_content, filename, project_id, user_id)

        elif ext in self.SUPPORTED_IMAGE_FORMATS:
            if len(file_content) > self.MAX_IMAGE_SIZE:
                raise ValueError(f"Image file too large (max {self.MAX_IMAGE_SIZE // 1024 // 1024} MB)")
            return await self._process_image(file_content, filename, project_id, user_id)

        else:
            raise ValueError(f"Unsupported file format: {ext}")

    async def _process_video(
        self,
        content: bytes,
        filename: str,
        project_id: str,
        user_id: str,
    ) -> Tuple[str, float, Optional[str], SegmentType]:
        """Process video upload"""
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        ext = Path(filename).suffix.lower()
        safe_name = f"{project_id}_{file_id}{ext}"

        # Save video file
        video_path = self.videos_path / safe_name
        with open(video_path, 'wb') as f:
            f.write(content)

        print(f"[SEGMENT] Video saved: {video_path}", flush=True)

        # Get video duration
        duration = await self._get_video_duration(video_path)

        # Generate thumbnail
        thumbnail_url = await self._generate_thumbnail(video_path, project_id, file_id)

        return str(video_path), duration, thumbnail_url, SegmentType.USER_VIDEO

    async def _process_audio(
        self,
        content: bytes,
        filename: str,
        project_id: str,
        user_id: str,
    ) -> Tuple[str, float, Optional[str], SegmentType]:
        """Process audio upload"""
        file_id = str(uuid.uuid4())[:8]
        ext = Path(filename).suffix.lower()
        safe_name = f"{project_id}_{file_id}{ext}"

        # Save audio file
        audio_path = self.videos_path / safe_name
        with open(audio_path, 'wb') as f:
            f.write(content)

        print(f"[SEGMENT] Audio saved: {audio_path}", flush=True)

        # Get audio duration
        duration = await self._get_audio_duration(audio_path)

        return str(audio_path), duration, None, SegmentType.USER_AUDIO

    async def _process_image(
        self,
        content: bytes,
        filename: str,
        project_id: str,
        user_id: str,
    ) -> Tuple[str, float, Optional[str], SegmentType]:
        """Process image upload (for slides)"""
        file_id = str(uuid.uuid4())[:8]
        ext = Path(filename).suffix.lower()
        safe_name = f"{project_id}_{file_id}{ext}"

        # Save image
        image_path = self.videos_path / safe_name
        with open(image_path, 'wb') as f:
            f.write(content)

        print(f"[SEGMENT] Image saved: {image_path}", flush=True)

        # Default duration for slides
        duration = 5.0

        return str(image_path), duration, str(image_path), SegmentType.SLIDE

    async def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                duration = float(stdout.decode().strip())
                print(f"[SEGMENT] Video duration: {duration}s", flush=True)
                return duration
            else:
                print(f"[SEGMENT] ffprobe error: {stderr.decode()}", flush=True)
                return 60.0  # Default fallback

        except Exception as e:
            print(f"[SEGMENT] Error getting duration: {e}", flush=True)
            return 60.0

    async def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration using ffprobe"""
        return await self._get_video_duration(audio_path)

    async def _generate_thumbnail(
        self,
        video_path: Path,
        project_id: str,
        file_id: str,
    ) -> Optional[str]:
        """Generate thumbnail from video"""
        try:
            thumbnail_name = f"{project_id}_{file_id}_thumb.jpg"
            thumbnail_path = self.thumbnails_path / thumbnail_name

            cmd = [
                'ffmpeg',
                '-y',  # Overwrite
                '-i', str(video_path),
                '-ss', '00:00:01',  # 1 second in
                '-vframes', '1',
                '-vf', 'scale=320:-1',  # 320px width
                str(thumbnail_path)
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await result.communicate()

            if result.returncode == 0 and thumbnail_path.exists():
                print(f"[SEGMENT] Thumbnail generated: {thumbnail_path}", flush=True)
                return str(thumbnail_path)

        except Exception as e:
            print(f"[SEGMENT] Error generating thumbnail: {e}", flush=True)

        return None

    async def get_video_info(self, video_path: str) -> dict:
        """
        Get detailed video information.

        Returns:
            Dict with duration, width, height, fps, codec, etc.
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,codec_name,duration',
                '-show_entries', 'format=duration,size',
                '-of', 'json',
                video_path
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await result.communicate()

            if result.returncode == 0:
                import json
                data = json.loads(stdout.decode())

                # Parse frame rate
                fps = 30
                if 'streams' in data and data['streams']:
                    stream = data['streams'][0]
                    if 'r_frame_rate' in stream:
                        fps_parts = stream['r_frame_rate'].split('/')
                        if len(fps_parts) == 2:
                            fps = int(fps_parts[0]) / int(fps_parts[1])

                return {
                    'width': data.get('streams', [{}])[0].get('width', 1920),
                    'height': data.get('streams', [{}])[0].get('height', 1080),
                    'fps': fps,
                    'codec': data.get('streams', [{}])[0].get('codec_name', 'unknown'),
                    'duration': float(data.get('format', {}).get('duration', 0)),
                    'size': int(data.get('format', {}).get('size', 0)),
                }

        except Exception as e:
            print(f"[SEGMENT] Error getting video info: {e}", flush=True)

        return {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'codec': 'unknown',
            'duration': 0,
            'size': 0,
        }

    async def trim_video(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: str,
    ) -> bool:
        """
        Trim a video segment.

        Args:
            video_path: Source video path
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            duration = end_time - start_time

            cmd = [
                'ffmpeg',
                '-y',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c', 'copy',  # Stream copy (fast)
                output_path
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            _, stderr = await result.communicate()

            if result.returncode == 0:
                print(f"[SEGMENT] Video trimmed: {output_path}", flush=True)
                return True
            else:
                print(f"[SEGMENT] Trim error: {stderr.decode()}", flush=True)

        except Exception as e:
            print(f"[SEGMENT] Error trimming video: {e}", flush=True)

        return False

    async def extract_audio(
        self,
        video_path: str,
        output_path: str,
    ) -> bool:
        """Extract audio from video"""
        try:
            cmd = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'aac',
                '-ab', '192k',
                output_path
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await result.communicate()

            return result.returncode == 0

        except Exception as e:
            print(f"[SEGMENT] Error extracting audio: {e}", flush=True)
            return False

    async def delete_segment_files(
        self,
        source_url: str,
        thumbnail_url: Optional[str] = None,
    ) -> None:
        """Delete segment files from storage"""
        try:
            if source_url:
                source_path = Path(source_url)
                if source_path.exists():
                    source_path.unlink()
                    print(f"[SEGMENT] Deleted: {source_url}", flush=True)

            if thumbnail_url:
                thumb_path = Path(thumbnail_url)
                if thumb_path.exists():
                    thumb_path.unlink()

        except Exception as e:
            print(f"[SEGMENT] Error deleting files: {e}", flush=True)

    def get_supported_formats(self) -> dict:
        """Get supported file formats"""
        return {
            'video': list(self.SUPPORTED_VIDEO_FORMATS),
            'audio': list(self.SUPPORTED_AUDIO_FORMATS),
            'image': list(self.SUPPORTED_IMAGE_FORMATS),
            'max_sizes': {
                'video_mb': self.MAX_VIDEO_SIZE // 1024 // 1024,
                'audio_mb': self.MAX_AUDIO_SIZE // 1024 // 1024,
                'image_mb': self.MAX_IMAGE_SIZE // 1024 // 1024,
            }
        }
