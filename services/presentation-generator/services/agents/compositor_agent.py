"""
Compositor Agent

Assembles the final video from pre-synced scene packages.
Each scene is already validated for sync, so this just concatenates.
"""

import os
import json
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Awaitable
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult, ScenePackage


# Callback type for scene progress reporting
SceneProgressCallback = Callable[[int, str, str, float], Awaitable[None]]
# Args: scene_index, video_url, status, duration


@dataclass
class CompositionConfig:
    """Configuration for video composition"""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 23
    preset: str = "medium"
    transition_duration: float = 0.3
    # Diagram-to-code transition anticipation (in seconds)
    # Shorten diagram slides when followed by code to prevent voiceover overlap
    diagram_to_code_anticipation: float = 1.5


class CompositorAgent(BaseAgent):
    """Assembles final video from pre-synced scene packages"""

    def __init__(self):
        super().__init__("COMPOSITOR")
        self.output_dir = os.getenv("VIDEO_OUTPUT_DIR", "/tmp/viralify/videos")
        self.ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
        self.config = CompositionConfig()

    async def execute(
        self,
        state: Dict[str, Any],
        on_scene_ready: Optional[SceneProgressCallback] = None
    ) -> AgentResult:
        """
        Compose final video from scene packages.

        Args:
            state: Execution state with scene_packages, job_id, title
            on_scene_ready: Optional callback called when each scene video is ready.
                           Signature: (scene_index, video_url, status, duration) -> None
                           This enables progressive download of individual lessons.
        """
        scene_packages = state.get("scene_packages", [])
        job_id = state.get("job_id", "unknown")
        title = state.get("title", "presentation")

        self.log(f"Composing {len(scene_packages)} scenes for job {job_id}")

        if not scene_packages:
            return AgentResult(
                success=False,
                errors=["No scene packages to compose"]
            )

        try:
            os.makedirs(self.output_dir, exist_ok=True)

            # Step 1: Render each scene to individual video files
            # Pass callback for progressive download support
            scene_videos = await self._render_scenes(
                scene_packages, job_id, on_scene_ready
            )

            if not scene_videos:
                return AgentResult(
                    success=False,
                    errors=["No scenes could be rendered"]
                )

            # Step 2: Create concat file
            concat_file = await self._create_concat_file(scene_videos, job_id)

            # Step 3: Concatenate all scenes
            output_path = os.path.join(self.output_dir, f"{job_id}_final.mp4")
            await self._concatenate_scenes(concat_file, output_path)

            # Step 4: Get final video info
            duration = await self._get_video_duration(output_path)

            self.log(f"Final video: {output_path} ({duration:.1f}s)")

            return AgentResult(
                success=True,
                data={
                    "output_url": output_path,
                    "output_path": output_path,
                    "duration": duration,
                    "scene_count": len(scene_packages),
                    "resolution": f"{self.config.width}x{self.config.height}",
                    "fps": self.config.fps
                }
            )

        except Exception as e:
            self.log(f"Composition failed: {e}")
            return AgentResult(
                success=False,
                errors=[str(e)]
            )

    async def _render_scenes(
        self,
        scene_packages: List[Dict[str, Any]],
        job_id: str,
        on_scene_ready: Optional[SceneProgressCallback] = None
    ) -> List[str]:
        """
        Render each scene package to video.

        Args:
            scene_packages: List of scene data to render
            job_id: Job identifier for file naming
            on_scene_ready: Callback for progressive download support
        """
        scene_videos = []

        # Pre-calculate adjusted durations for diagram→code transitions
        adjusted_durations = self._calculate_adjusted_durations(scene_packages)

        for i, scene in enumerate(scene_packages):
            scene_path = os.path.join(
                self.output_dir,
                f"{job_id}_scene_{i:03d}.mp4"
            )

            try:
                # Apply adjusted duration if calculated
                scene_duration = scene.get("total_duration") or scene.get("audio_duration") or 10
                if i in adjusted_durations:
                    original_duration = scene_duration
                    scene_duration = adjusted_durations[i]
                    scene = scene.copy()
                    scene["total_duration"] = scene_duration
                    self.log(f"[DIAGRAM→CODE] Scene {i}: {original_duration:.1f}s → {scene_duration:.1f}s (anticipation)")

                result = await self._render_single_scene(scene, scene_path)
                if result:
                    scene_videos.append(scene_path)
                    self.log(f"Scene {i} rendered: {scene_path}")

                    # Notify callback for progressive download
                    if on_scene_ready:
                        scene_url = self._build_scene_url(scene_path)
                        try:
                            await on_scene_ready(i, scene_url, "ready", scene_duration)
                        except Exception as cb_err:
                            self.log(f"Scene {i} callback error (non-fatal): {cb_err}")
                else:
                    self.log(f"Scene {i} render failed, skipping")
                    if on_scene_ready:
                        try:
                            await on_scene_ready(i, "", "failed", 0)
                        except Exception:
                            pass
            except Exception as e:
                self.log(f"Scene {i} error: {e}")
                if on_scene_ready:
                    try:
                        await on_scene_ready(i, "", "error", 0)
                    except Exception:
                        pass

        return scene_videos

    def _build_scene_url(self, scene_path: str) -> str:
        """Build a publicly accessible URL for a scene video."""
        filename = os.path.basename(scene_path)

        # Use PUBLIC_MEDIA_URL for browser-accessible URLs
        public_media_url = os.getenv("PUBLIC_MEDIA_URL", "")
        if public_media_url:
            return f"{public_media_url}/files/videos/{filename}"

        # Fallback to internal URL
        media_url = os.getenv("MEDIA_GENERATOR_URL", "http://media-generator:8004")
        return f"{media_url}/files/videos/{filename}"

    def _calculate_adjusted_durations(
        self,
        scene_packages: List[Dict[str, Any]]
    ) -> Dict[int, float]:
        """
        Calculate adjusted durations for diagram→code transitions.

        When a diagram slide is followed by a code slide, we shorten the diagram
        duration to prevent the voiceover from overlapping onto the code visual.
        """
        adjusted = {}
        anticipation = self.config.diagram_to_code_anticipation

        for i in range(len(scene_packages) - 1):
            current_scene = scene_packages[i]
            next_scene = scene_packages[i + 1]

            current_type = current_scene.get("content_type", "").lower()
            next_type = next_scene.get("content_type", "").lower()

            # Detect diagram→code transition
            is_diagram = current_type in ("diagram", "architecture", "flowchart", "visual")
            is_code_next = next_type in ("code", "code_demo", "terminal", "terminal_output")

            if is_diagram and is_code_next:
                current_duration = current_scene.get("total_duration") or current_scene.get("audio_duration") or 10
                # Shorten diagram duration by anticipation, but keep at least 2 seconds
                new_duration = max(2.0, current_duration - anticipation)
                adjusted[i] = new_duration

        return adjusted

    async def _render_single_scene(
        self,
        scene: Dict[str, Any],
        output_path: str
    ) -> bool:
        """Render a single scene to video"""
        # Get scene components (handle None values)
        audio_url = scene.get("audio_url") or ""
        visual_url = scene.get("primary_visual_url") or ""
        duration = scene.get("total_duration") or scene.get("audio_duration") or 10
        animations = scene.get("animations") or []

        # Handle different input scenarios
        if visual_url and audio_url:
            # Both audio and visual available
            return await self._compose_av(
                visual_url, audio_url, duration, output_path, animations
            )
        elif audio_url:
            # Audio only - create simple background with text
            return await self._compose_audio_only(
                audio_url, scene, duration, output_path
            )
        elif visual_url:
            # Visual only - create silent video
            return await self._compose_visual_only(
                visual_url, duration, output_path
            )
        else:
            # No input - create placeholder
            return await self._create_placeholder(duration, output_path)

    async def _compose_av(
        self,
        visual_url: str,
        audio_url: str,
        duration: float,
        output_path: str,
        animations: List[Dict[str, Any]]
    ) -> bool:
        """Compose audio and visual together"""
        # Handle file:// URLs
        visual_path = visual_url.replace("file://", "") if visual_url.startswith("file://") else visual_url
        audio_path = audio_url.replace("file://", "") if audio_url.startswith("file://") else audio_url

        # Determine if visual is image or video
        is_image = visual_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))

        if is_image:
            # Convert image to video with audio
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-loop", "1",
                "-i", visual_path,
                "-i", audio_path,
                "-c:v", self.config.video_codec,
                "-tune", "stillimage",
                "-c:a", self.config.audio_codec,
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest",
                "-t", str(duration),
                output_path
            ]
        else:
            # Combine video with audio
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", visual_path,
                "-i", audio_path,
                "-c:v", self.config.video_codec,
                "-c:a", self.config.audio_codec,
                "-b:a", "192k",
                "-shortest",
                "-t", str(duration),
                output_path
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    async def _compose_audio_only(
        self,
        audio_url: str,
        scene: Dict[str, Any],
        duration: float,
        output_path: str
    ) -> bool:
        """Create video with background and text from audio"""
        audio_path = audio_url.replace("file://", "") if audio_url.startswith("file://") else audio_url

        # Get title for display
        title = scene.get("title", "")

        # Create video with dark background and title text
        filter_complex = (
            f"color=c=#1a1a2e:s={self.config.width}x{self.config.height}:d={duration},"
            f"drawtext=text='{self._escape_ffmpeg_text(title)}':fontsize=48:"
            f"fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:font=Arial"
        )

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=#1a1a2e:s={self.config.width}x{self.config.height}:d={duration}",
            "-i", audio_path,
            "-vf", f"drawtext=text='{self._escape_ffmpeg_text(title)}':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v", self.config.video_codec,
            "-c:a", self.config.audio_codec,
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-t", str(duration),
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    async def _compose_visual_only(
        self,
        visual_url: str,
        duration: float,
        output_path: str
    ) -> bool:
        """Create silent video from visual"""
        visual_path = visual_url.replace("file://", "") if visual_url.startswith("file://") else visual_url
        is_image = visual_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))

        if is_image:
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-loop", "1",
                "-i", visual_path,
                "-c:v", self.config.video_codec,
                "-tune", "stillimage",
                "-pix_fmt", "yuv420p",
                "-t", str(duration),
                output_path
            ]
        else:
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", visual_path,
                "-c:v", self.config.video_codec,
                "-an",  # No audio
                "-t", str(duration),
                output_path
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    async def _create_placeholder(
        self,
        duration: float,
        output_path: str
    ) -> bool:
        """Create a placeholder video"""
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=#1a1a2e:s={self.config.width}x{self.config.height}:d={duration}",
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=stereo",
            "-c:v", self.config.video_codec,
            "-c:a", self.config.audio_codec,
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-t", str(duration),
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    async def _create_concat_file(
        self,
        scene_videos: List[str],
        job_id: str
    ) -> str:
        """Create FFmpeg concat demuxer file"""
        concat_path = os.path.join(self.output_dir, f"{job_id}_concat.txt")

        with open(concat_path, "w") as f:
            for video_path in scene_videos:
                # Escape single quotes in path
                escaped_path = video_path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        return concat_path

    async def _concatenate_scenes(
        self,
        concat_file: str,
        output_path: str
    ) -> bool:
        """Concatenate all scene videos into final output"""
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:v", self.config.video_codec,
            "-c:a", self.config.audio_codec,
            "-crf", str(self.config.crf),
            "-preset", self.config.preset,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.log(f"Concatenation failed: {result.stderr}")
            return False

        return True

    async def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                return 0

        return 0

    def _escape_ffmpeg_text(self, text: str) -> str:
        """Escape text for FFmpeg drawtext filter"""
        # Escape special characters
        text = text.replace("\\", "\\\\")
        text = text.replace(":", "\\:")
        text = text.replace("'", "\\'")
        text = text.replace("[", "\\[")
        text = text.replace("]", "\\]")
        return text

    async def create_preview(
        self,
        scene_packages: List[Dict[str, Any]],
        job_id: str,
        max_duration: float = 30
    ) -> AgentResult:
        """Create a quick preview of the first N seconds"""
        self.log(f"Creating preview (max {max_duration}s) for job {job_id}")

        # Take only scenes that fit in preview
        preview_scenes = []
        total_duration = 0

        for scene in scene_packages:
            scene_duration = scene.get("total_duration", scene.get("audio_duration", 10))
            if total_duration + scene_duration <= max_duration:
                preview_scenes.append(scene)
                total_duration += scene_duration
            else:
                # Take partial scene
                remaining = max_duration - total_duration
                if remaining > 2:  # At least 2 seconds
                    partial_scene = scene.copy()
                    partial_scene["total_duration"] = remaining
                    preview_scenes.append(partial_scene)
                break

        # Render preview
        preview_state = {
            "scene_packages": preview_scenes,
            "job_id": f"{job_id}_preview",
            "title": "Preview"
        }

        return await self.execute(preview_state)
