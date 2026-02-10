"""
Video Compositor Service
Uses FFmpeg to compose final videos from:
- Video clips
- Images (with Ken Burns effect)
- Voiceover audio
- Background music
- Text overlays
"""

import asyncio
import os
import tempfile
import shutil
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path
import httpx
from pydantic import BaseModel


class WordTimestamp(BaseModel):
    """Word-level timestamp for synchronized captions"""
    word: str
    start: float  # seconds from start
    end: float    # seconds from start


class CompositionScene(BaseModel):
    id: str
    order: int
    media_url: str
    media_type: str  # "video" or "image"
    duration: float  # seconds
    start_time: float  # seconds from start
    audio_url: Optional[str] = None  # Per-scene audio for sync
    text_overlay: Optional[str] = None
    transition: str = "fade"  # fade, cut, dissolve


class CompositionRequest(BaseModel):
    project_id: str
    scenes: List[CompositionScene]
    voiceover_url: Optional[str] = None
    voiceover_text: Optional[str] = None  # For generating captions
    word_timestamps: Optional[List[WordTimestamp]] = None  # For synchronized animated captions
    music_url: Optional[str] = None
    music_volume: float = 0.3  # 0-1
    format: str = "9:16"  # 9:16, 16:9, 1:1
    quality: str = "1080p"  # 720p, 1080p, 4k
    fps: int = 30
    caption_style: Optional[str] = None  # classic, bold, neon, minimal, karaoke, boxed, gradient
    caption_config: Optional[Dict[str, Any]] = None
    ken_burns_effect: bool = False  # Enable zoom/pan on images, False for static display
    # PIP Avatar overlay settings
    pip_avatar_url: Optional[str] = None  # URL/path to avatar video for Picture-in-Picture
    pip_position: str = "bottom-right"  # bottom-right, bottom-left, top-right, top-left
    pip_size: float = 0.35  # Size as fraction of screen width (0.2-0.5)
    pip_margin: int = 20  # Margin from edges in pixels
    pip_border_radius: int = 20  # Border radius for rounded corners
    pip_shadow: bool = True  # Add drop shadow for depth
    pip_remove_background: bool = True  # Remove avatar background for seamless blending
    pip_bg_color: Optional[str] = None  # Background color to remove (auto-detect if None)
    pip_bg_similarity: float = 0.3  # Color similarity threshold (0.0-1.0)
    pip_circular: bool = True  # Use circular mask for medallion style


class CompositionResult(BaseModel):
    success: bool
    output_url: Optional[str] = None
    duration: float = 0
    file_size_bytes: int = 0
    error_message: Optional[str] = None


class VideoCompositorService:
    """Composes videos using FFmpeg"""

    def __init__(self, output_dir: str = "/tmp/viralify/videos", service_base_url: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="viralify_"))
        # Use environment variable or default to Docker hostname for container communication
        # In development, set SERVICE_BASE_URL env var to override
        self.service_base_url = service_base_url or os.getenv("SERVICE_BASE_URL", "http://media-generator:8004")
        # Public base URL for user-facing URLs (via nginx proxy)
        self.public_base_url = os.getenv("PUBLIC_BASE_URL", "")

    async def compose_video(
        self,
        request: CompositionRequest,
        progress_callback: Optional[callable] = None
    ) -> CompositionResult:
        """
        Compose a complete video from scenes, audio, and music
        """
        try:
            # Log the format being used
            print(f"=== VIDEO COMPOSITION ===")
            print(f"Format: {request.format}")
            print(f"Quality: {request.quality}")
            print(f"Number of scenes: {len(request.scenes)}")

            # Create working directory
            work_dir = self.temp_dir / request.project_id
            work_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Download all assets
            if progress_callback:
                progress_callback(10, "Downloading assets...")

            scene_files = await self._download_scene_assets(request.scenes, work_dir)

            # Step 2: Download audio files
            if progress_callback:
                progress_callback(30, "Processing audio...")

            voiceover_file = None
            music_file = None

            # Check if scenes have per-scene audio (for sync)
            scenes_with_audio = [s for s in request.scenes if s.audio_url]

            if scenes_with_audio:
                # Per-scene audio mode: download and concatenate for perfect sync
                print(f"[AUDIO] Using per-scene audio ({len(scenes_with_audio)} scenes with audio)")
                voiceover_file = await self._concatenate_scene_audio(
                    request.scenes,
                    work_dir
                )
                if voiceover_file:
                    print(f"[AUDIO] Concatenated per-scene audio: {voiceover_file}")
            elif request.voiceover_url:
                # Single voiceover file mode
                if request.voiceover_url.startswith('/') or request.voiceover_url.startswith('C:'):
                    voiceover_file = Path(request.voiceover_url)
                else:
                    voiceover_file = await self._download_file(
                        request.voiceover_url,
                        work_dir / "voiceover.mp3"
                    )

            if request.music_url:
                try:
                    music_file = await self._download_file(
                        request.music_url,
                        work_dir / "music.mp3"
                    )
                except Exception as e:
                    print(f"Warning: Could not download music ({e}), continuing without background music")
                    music_file = None

            # Step 3: Process each scene
            if progress_callback:
                progress_callback(40, "Processing scenes...")

            processed_scenes = await self._process_scenes(
                scene_files,
                request.scenes,
                request.format,
                request.quality,
                request.fps,
                work_dir,
                request.ken_burns_effect
            )

            # Step 4: Concatenate all scenes
            if progress_callback:
                progress_callback(60, "Concatenating scenes...")

            concat_file = await self._concatenate_scenes(
                processed_scenes,
                work_dir
            )

            # Step 4.5: Add PIP Avatar overlay if provided
            if request.pip_avatar_url:
                if progress_callback:
                    progress_callback(70, "Adding avatar overlay...")

                print(f"[PIP] Adding avatar overlay from: {request.pip_avatar_url}")
                concat_file = await self._add_pip_overlay(
                    concat_file,
                    request.pip_avatar_url,
                    request.pip_position,
                    request.pip_size,
                    request.pip_margin,
                    request.pip_border_radius,
                    request.pip_shadow,
                    request.format,
                    work_dir,
                    request.pip_remove_background,
                    request.pip_bg_color,
                    request.pip_bg_similarity,
                    request.pip_circular
                )
                print(f"[PIP] Avatar overlay added successfully")

            # Step 5: Add audio layers
            if progress_callback:
                progress_callback(80, "Adding audio...")

            output_file = await self._add_audio(
                concat_file,
                voiceover_file,
                music_file,
                request.music_volume,
                work_dir
            )

            # Step 6: Add captions if requested
            if request.caption_style:
                voiceover_text = request.voiceover_text or ""
                # If no voiceover text, try to extract from scene text overlays
                if not voiceover_text:
                    voiceover_text = " ".join([s.text_overlay or "" for s in request.scenes if s.text_overlay])

                if voiceover_text.strip():
                    print(f"Adding captions with style '{request.caption_style}', text length: {len(voiceover_text)}")
                    has_timestamps = request.word_timestamps and len(request.word_timestamps) > 0
                    print(f"Word timestamps available: {has_timestamps} ({len(request.word_timestamps or [])} words)")

                    if progress_callback:
                        progress_callback(90, "Adding captions...")

                    output_file = await self._add_captions(
                        output_file,
                        voiceover_text,
                        request.caption_style,
                        request.caption_config or {},
                        request.format,
                        sum(s.duration for s in request.scenes),
                        work_dir,
                        request.word_timestamps  # Pass word timestamps for sync
                    )
                else:
                    print("No voiceover text available for captions")

            # Step 7: Move to output directory
            if progress_callback:
                progress_callback(95, "Finalizing...")

            final_output = self.output_dir / f"{request.project_id}.mp4"
            shutil.move(str(output_file), str(final_output))

            # Get file info
            file_size = final_output.stat().st_size
            duration = sum(s.duration for s in request.scenes)

            if progress_callback:
                progress_callback(100, "Complete!")

            # Generate HTTP URL - use public URL for user-facing responses
            if self.public_base_url:
                # PUBLIC_BASE_URL may already include path (e.g., https://olsitec.com/media)
                # Don't add /media/ prefix as it's configured in the URL itself
                output_http_url = f"{self.public_base_url}/files/videos/{request.project_id}.mp4"
            else:
                output_http_url = f"{self.service_base_url}/files/videos/{request.project_id}.mp4"

            return CompositionResult(
                success=True,
                output_url=output_http_url,
                duration=duration,
                file_size_bytes=file_size
            )

        except Exception as e:
            return CompositionResult(
                success=False,
                error_message=str(e)
            )

        finally:
            # Cleanup temp files
            try:
                shutil.rmtree(work_dir)
            except (OSError, IOError) as cleanup_err:
                print(f"[COMPOSITOR] Cleanup warning: {cleanup_err}", flush=True)

    async def _download_file(self, url: str, output_path: Path) -> Path:
        """Download a file from URL or copy if it's a local file"""
        import shutil

        # Handle local file paths (from lip-sync or local generation)
        if url.startswith('/') or url.startswith('/tmp/'):
            local_path = Path(url)
            if local_path.exists():
                shutil.copy2(str(local_path), str(output_path))
                return output_path
            else:
                raise FileNotFoundError(f"Local file not found: {url}")

        # Handle remote URLs
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=120.0, follow_redirects=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

        return output_path

    async def _detect_green_screen(self, video_path: Path, work_dir: Path) -> Optional[str]:
        """
        Detect if video has a green screen background (from D-ID Clips).
        Returns '0x00ff00' if green screen detected, None otherwise.
        """
        try:
            from PIL import Image

            # Extract first frame
            frame_path = work_dir / "green_detect_frame.png"
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vframes", "1", "-f", "image2",
                str(frame_path)
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                await asyncio.wait_for(process.communicate(), timeout=30)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return None

            if not frame_path.exists():
                return None

            # Open image and sample corners
            img = Image.open(frame_path).convert("RGB")
            w, h = img.size

            # Sample corners
            sample_size = 20
            corners = [
                (0, 0), (w - sample_size, 0),
                (0, h - sample_size), (w - sample_size, h - sample_size)
            ]

            green_pixels = 0
            total_pixels = 0

            for cx, cy in corners:
                region = img.crop((cx, cy, cx + sample_size, cy + sample_size))
                pixels = list(region.getdata())
                for r, g, b in pixels:
                    total_pixels += 1
                    # Check if pixel is green-ish (high green, low red/blue)
                    if g > 200 and r < 100 and b < 100:
                        green_pixels += 1

            # Clean up
            frame_path.unlink(missing_ok=True)

            # If more than 60% of corner pixels are green, it's a green screen
            if total_pixels > 0 and (green_pixels / total_pixels) > 0.6:
                print(f"[PIP] Green screen detected: {green_pixels}/{total_pixels} pixels are green")
                return "0x00ff00"

            return None

        except Exception as e:
            print(f"[PIP] Green screen detection failed: {e}")
            return None

    async def _detect_background_color(self, video_path: Path, work_dir: Path) -> Optional[str]:
        """
        Detect the dominant background color from the corners of the first frame.
        Returns hex color string like '0xRRGGBB' for FFmpeg colorkey.
        """
        try:
            from PIL import Image
            import io

            # Extract first frame
            frame_path = work_dir / "bg_detect_frame.png"
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vframes", "1", "-f", "image2",
                str(frame_path)
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                await asyncio.wait_for(process.communicate(), timeout=30)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return None

            if not frame_path.exists():
                return None

            # Open image and sample corners
            img = Image.open(frame_path).convert("RGB")
            w, h = img.size

            # Sample 10x10 pixel regions from each corner
            sample_size = 10
            corners = [
                (0, 0),  # top-left
                (w - sample_size, 0),  # top-right
                (0, h - sample_size),  # bottom-left
                (w - sample_size, h - sample_size)  # bottom-right
            ]

            colors = []
            for cx, cy in corners:
                region = img.crop((cx, cy, cx + sample_size, cy + sample_size))
                # Get average color of region
                pixels = list(region.getdata())
                avg_r = sum(p[0] for p in pixels) // len(pixels)
                avg_g = sum(p[1] for p in pixels) // len(pixels)
                avg_b = sum(p[2] for p in pixels) // len(pixels)
                colors.append((avg_r, avg_g, avg_b))

            # Find the most common color (corners should have similar background)
            # Use the average of all corner colors
            avg_r = sum(c[0] for c in colors) // len(colors)
            avg_g = sum(c[1] for c in colors) // len(colors)
            avg_b = sum(c[2] for c in colors) // len(colors)

            bg_color = f"0x{avg_r:02x}{avg_g:02x}{avg_b:02x}"
            print(f"[PIP] Detected background color: {bg_color} (RGB: {avg_r}, {avg_g}, {avg_b})")

            # Clean up
            frame_path.unlink(missing_ok=True)

            return bg_color

        except Exception as e:
            print(f"[PIP] Background detection failed: {e}")
            return None

    async def _add_pip_overlay(
        self,
        background_video: Path,
        pip_video_url: str,
        position: str,
        size: float,
        margin: int,
        border_radius: int,
        shadow: bool,
        format: str,
        work_dir: Path,
        remove_background: bool = True,
        bg_color: Optional[str] = None,
        bg_similarity: float = 0.3,
        circular: bool = True
    ) -> Path:
        """
        Add Picture-in-Picture avatar overlay on top of the background video.

        The avatar video is overlaid in a corner with optional:
        - Background removal (colorkey)
        - Rounded corners
        - Shadow effect
        The avatar video loops if it's shorter than the background video.
        """
        # Download/copy PIP video
        pip_file = work_dir / "pip_avatar.mp4"
        await self._download_file(pip_video_url, pip_file)

        if not pip_file.exists():
            print(f"[PIP] Warning: Avatar video not found at {pip_video_url}")
            return background_video

        # Get dimensions based on format
        dimensions = {
            "9:16": (1080, 1920),
            "16:9": (1920, 1080),
            "1:1": (1080, 1080)
        }
        width, height = dimensions.get(format, (1080, 1920))

        # Calculate PIP dimensions (maintain aspect ratio)
        pip_width = int(width * size)
        # D-ID videos are typically 16:9 or similar, so calculate height accordingly
        pip_height = int(pip_width * 9 / 16)  # Assuming 16:9 avatar video

        # Calculate position
        if position == "bottom-right":
            x_pos = width - pip_width - margin
            y_pos = height - pip_height - margin - 100  # Extra margin for captions
        elif position == "bottom-left":
            x_pos = margin
            y_pos = height - pip_height - margin - 100
        elif position == "top-right":
            x_pos = width - pip_width - margin
            y_pos = margin
        elif position == "top-left":
            x_pos = margin
            y_pos = margin
        else:
            x_pos = width - pip_width - margin
            y_pos = height - pip_height - margin - 100

        output_file = work_dir / "with_pip.mp4"

        # Background removal strategy:
        # 1. For D-ID Clips (presenters), use green screen chromakey (bg_color=0x00ff00)
        # 2. For D-ID Talks, background is removed at source (pre-processing with rembg)
        detected_bg_color = None
        if remove_background:
            if bg_color:
                # Specific color provided (likely green screen from D-ID Clips)
                detected_bg_color = bg_color
                print(f"[PIP] Using chromakey with color: {detected_bg_color}")
            else:
                # Try to detect if this is a green screen video (D-ID Clips use #00FF00)
                detected_bg_color = await self._detect_green_screen(pip_file, work_dir)
                if detected_bg_color:
                    print(f"[PIP] Detected green screen, applying chromakey")
                else:
                    print(f"[PIP] No green screen detected, background handled at source")

        # Get video duration to decide on filter complexity
        try:
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(background_video)]
            process = await asyncio.create_subprocess_exec(
                *probe_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            video_duration = float(stdout.decode().strip()) if stdout else 0
        except (OSError, ValueError):
            video_duration = 0

        use_simple_filter = video_duration >= 30  # Use simple filter for videos >= 30 seconds
        if use_simple_filter:
            print(f"[PIP] Using optimized filter for video ({video_duration:.1f}s)", flush=True)

        # Build FFmpeg filter for PIP overlay
        # Base scaling and format conversion
        base_filter = f"[1:v]scale={pip_width}:{pip_height},format=rgba"

        # Add background removal with colorkey if enabled and color detected
        if detected_bg_color and remove_background:
            # colorkey removes a specific color, making it transparent
            # similarity: how close colors need to be (0.0-1.0)
            # blend: edge blending for smoother transitions
            base_filter += f",colorkey=color={detected_bg_color}:similarity={bg_similarity}:blend=0.1"
            print(f"[PIP] Applying colorkey filter: color={detected_bg_color}, similarity={bg_similarity}")

        # Determine mask type: circular (medallion) or rounded rectangle
        if circular:
            # Circular medallion mask - creates a perfect circle
            # Use the smaller dimension as the diameter for a perfect circle
            circle_radius = min(pip_width, pip_height) // 2
            cx = pip_width // 2
            cy = pip_height // 2
            mask_filter = f"geq=lum='lum(X,Y)':a='if(lte(hypot(X-{cx},Y-{cy}),{circle_radius}),255,0)'"
            print(f"[PIP] Using circular medallion mask (radius={circle_radius}px)")
        else:
            # Rounded rectangle mask
            mask_filter = (
                f"geq=lum='lum(X,Y)':a='if(gt(abs(X-{pip_width}/2),{pip_width}/2-{border_radius})*"
                f"gt(abs(Y-{pip_height}/2),{pip_height}/2-{border_radius}),"
                f"if(lte(hypot(abs(X-{pip_width}/2)-({pip_width}/2-{border_radius}),"
                f"abs(Y-{pip_height}/2)-({pip_height}/2-{border_radius})),{border_radius}),255,0),255)'"
            )

        if use_simple_filter:
            # Fast filter - just scale, colorkey (if enabled), and overlay
            # For simple filter, still apply circular mask if requested
            if circular:
                filter_complex = f"{base_filter},{mask_filter}[pip];[0:v][pip]overlay={x_pos}:{y_pos}:shortest=1"
            else:
                filter_complex = f"{base_filter}[pip];[0:v][pip]overlay={x_pos}:{y_pos}:shortest=1"
        elif shadow and not use_simple_filter:
            # Complex filter with shadow effect
            filter_complex = (
                f"{base_filter},{mask_filter}[pip];"
                f"[pip]split[pip1][pip_shadow];"
                f"[pip_shadow]colorchannelmixer=aa=0.4,boxblur=8:8[shadow];"
                f"[0:v][shadow]overlay={x_pos+8}:{y_pos+8}[bg_shadow];"
                f"[bg_shadow][pip1]overlay={x_pos}:{y_pos}:shortest=1"
            )
        else:
            # Medium filter with mask (circular or rounded)
            filter_complex = (
                f"{base_filter},{mask_filter}[pip];"
                f"[0:v][pip]overlay={x_pos}:{y_pos}:shortest=1"
            )

        # FFmpeg command with stream_loop to loop the PIP video
        cmd = [
            "ffmpeg", "-y",
            "-i", str(background_video),
            "-stream_loop", "-1",  # Loop PIP video infinitely
            "-i", str(pip_file),
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",
            "-shortest",  # End when the shortest input ends (background)
            str(output_file)
        ]

        print(f"[PIP] Running FFmpeg for PIP overlay...")
        print(f"[PIP] Position: {position}, Size: {pip_width}x{pip_height}, Location: ({x_pos}, {y_pos})")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        except asyncio.TimeoutError:
            print(f"[PIP] FFmpeg timeout after 300s, killing process", flush=True)
            process.kill()
            await process.wait()
            return background_video

        if process.returncode != 0:
            # If complex filter fails, try simpler overlay without rounded corners
            print(f"[PIP] Complex filter failed, trying simple overlay...")
            simple_filter = f"[1:v]scale={pip_width}:{pip_height}[pip];[0:v][pip]overlay={x_pos}:{y_pos}:shortest=1"

            cmd_simple = [
                "ffmpeg", "-y",
                "-i", str(background_video),
                "-stream_loop", "-1",
                "-i", str(pip_file),
                "-filter_complex", simple_filter,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "copy",
                "-shortest",
                str(output_file)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd_simple,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            except asyncio.TimeoutError:
                print(f"[PIP] Simple overlay timeout after 300s, killing process", flush=True)
                process.kill()
                await process.wait()
                return background_video

            if process.returncode != 0:
                print(f"[PIP] FFmpeg error: {stderr.decode()[:500]}")
                return background_video

        print(f"[PIP] Overlay complete: {output_file}")
        return output_file

    async def _download_scene_assets(
        self,
        scenes: List[CompositionScene],
        work_dir: Path
    ) -> Dict[str, Path]:
        """Download all scene media files in parallel"""

        async def download_scene(scene: CompositionScene) -> tuple:
            ext = ".mp4" if scene.media_type == "video" else ".jpg"
            output_path = work_dir / f"scene_{scene.order}{ext}"
            await self._download_file(scene.media_url, output_path)
            return (scene.id, output_path)

        tasks = [download_scene(scene) for scene in scenes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scene_files = {}
        for result in results:
            if isinstance(result, tuple):
                scene_id, path = result
                scene_files[scene_id] = path
            else:
                print(f"Download error: {result}")

        return scene_files

    async def _concatenate_scene_audio(
        self,
        scenes: List[CompositionScene],
        work_dir: Path
    ) -> Optional[Path]:
        """
        Download and concatenate per-scene audio files for perfect sync.
        For scenes without audio, inserts silence of the scene's duration.
        """
        audio_files = []
        sorted_scenes = sorted(scenes, key=lambda s: s.order)

        for i, scene in enumerate(sorted_scenes):
            audio_path = work_dir / f"audio_{scene.order:03d}.mp3"

            if scene.audio_url:
                try:
                    # Download the audio file
                    if scene.audio_url.startswith('/') or scene.audio_url.startswith('C:'):
                        # Local file - copy it
                        import shutil
                        shutil.copy(scene.audio_url, audio_path)
                    else:
                        # Remote file - download
                        await self._download_file(scene.audio_url, audio_path)

                    audio_files.append(str(audio_path))
                    print(f"[AUDIO] Scene {scene.order}: downloaded audio ({scene.duration:.1f}s)")
                except Exception as e:
                    print(f"[AUDIO] Scene {scene.order}: failed to download audio: {e}")
                    # Generate silence for this scene
                    silence_path = work_dir / f"silence_{scene.order:03d}.mp3"
                    await self._generate_silence(silence_path, scene.duration)
                    audio_files.append(str(silence_path))
            else:
                # No audio for this scene - generate silence
                silence_path = work_dir / f"silence_{scene.order:03d}.mp3"
                await self._generate_silence(silence_path, scene.duration)
                audio_files.append(str(silence_path))
                print(f"[AUDIO] Scene {scene.order}: using silence ({scene.duration:.1f}s)")

        if not audio_files:
            return None

        # Create concat file list
        concat_list = work_dir / "audio_concat.txt"
        with open(concat_list, 'w') as f:
            for audio_file in audio_files:
                f.write(f"file '{audio_file}'\n")

        # Concatenate all audio files
        output_file = work_dir / "voiceover_synced.mp3"
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            str(output_file)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"[AUDIO] Concat failed: {stderr.decode()}")
            return None

        print(f"[AUDIO] Concatenated {len(audio_files)} audio segments -> {output_file}")
        return output_file

    async def _generate_silence(self, output_path: Path, duration: float) -> Path:
        """Generate a silent audio file of specified duration"""
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=mono",
            "-t", str(duration),
            "-c:a", "libmp3lame",
            "-b:a", "128k",
            str(output_path)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError:
            print(f"[AUDIO] Silence generation timeout, killing process", flush=True)
            process.kill()
            await process.wait()
        return output_path

    async def _process_scenes(
        self,
        scene_files: Dict[str, Path],
        scenes: List[CompositionScene],
        format: str,
        quality: str,
        fps: int,
        work_dir: Path,
        ken_burns_effect: bool = False
    ) -> List[Path]:
        """Process each scene to target format"""

        # Get dimensions based on format
        dimensions = {
            "9:16": (1080, 1920),
            "16:9": (1920, 1080),
            "1:1": (1080, 1080)
        }
        width, height = dimensions.get(format, (1080, 1920))
        print(f"Processing scenes with format '{format}' -> dimensions: {width}x{height}")
        print(f"Ken Burns effect: {'enabled' if ken_burns_effect else 'disabled (static)'}")

        processed = []

        for scene in sorted(scenes, key=lambda s: s.order):
            input_file = scene_files.get(scene.id)
            if not input_file or not input_file.exists():
                continue

            output_file = work_dir / f"processed_{scene.order}.mp4"

            if scene.media_type == "video":
                # Process video: resize, loop if needed, trim to duration
                # Use stream_loop to repeat video if it's shorter than required duration
                cmd = [
                    "ffmpeg", "-y",
                    "-stream_loop", "-1",  # Loop video infinitely
                    "-i", str(input_file),
                    "-t", str(scene.duration),  # Then cut to exact duration
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
                    "-c:v", "libx264",
                    "-profile:v", "high",
                    "-level", "4.0",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",  # Required for browser compatibility
                    "-r", str(fps),
                    "-an",  # Remove audio (we'll add voiceover separately)
                    str(output_file)
                ]
                print(f"Processing scene {scene.order}: {scene.duration}s video")
            elif ken_burns_effect:
                # Process image with Ken Burns effect (zoom/pan)
                # Creates a video from still image with subtle motion
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", str(input_file),
                    "-t", str(scene.duration),
                    "-vf", f"scale=8000:-1,zoompan=z='min(zoom+0.0015,1.5)':d={int(scene.duration * fps)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}",
                    "-c:v", "libx264",
                    "-profile:v", "high",
                    "-level", "4.0",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-r", str(fps),
                    str(output_file)
                ]
                print(f"Processing scene {scene.order}: {scene.duration}s image (with Ken Burns)")
            else:
                # Process image: STATIC display - scale and pad to fit dimensions
                # No zoom/pan, perfect for presentations with text/code
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", str(input_file),
                    "-t", str(scene.duration),
                    "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
                    "-c:v", "libx264",
                    "-profile:v", "high",
                    "-level", "4.0",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-r", str(fps),
                    str(output_file)
                ]
                print(f"Processing scene {scene.order}: {scene.duration}s image (static)")

            # Run FFmpeg with timeout (60s base + 2x duration for safety)
            timeout_seconds = 60 + int(scene.duration * 2)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                print(f"FFmpeg timeout for scene {scene.order} after {timeout_seconds}s, killing process")
                process.kill()
                await process.wait()
                continue

            if process.returncode != 0:
                print(f"FFmpeg error for scene {scene.order}: {stderr.decode()}")
                continue

            # Add text overlay if present
            if scene.text_overlay:
                overlay_file = work_dir / f"overlay_{scene.order}.mp4"
                await self._add_text_overlay(
                    output_file,
                    overlay_file,
                    scene.text_overlay,
                    width,
                    height
                )
                output_file = overlay_file

            processed.append(output_file)

        return processed

    async def _add_text_overlay(
        self,
        input_file: Path,
        output_file: Path,
        text: str,
        width: int,
        height: int
    ):
        """Add text overlay to video"""

        # Escape special characters in text
        safe_text = text.replace("'", "\\'").replace(":", "\\:")

        # Calculate font size based on video height (larger for better readability)
        font_size = int(height / 14)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-vf", f"drawtext=text='{safe_text}':fontcolor=white:fontsize={font_size}:box=1:boxcolor=black@0.5:boxborderw=10:x=(w-text_w)/2:y=h-th-50",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-level", "4.0",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(output_file)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
            if process.returncode != 0:
                print(f"[TEXT_OVERLAY] FFmpeg error: {stderr.decode()[:300]}", flush=True)
        except asyncio.TimeoutError:
            print(f"[TEXT_OVERLAY] FFmpeg timeout after 120s, killing process", flush=True)
            process.kill()
            await process.wait()

    async def _concatenate_scenes(
        self,
        scene_files: List[Path],
        work_dir: Path
    ) -> Path:
        """Concatenate all scene videos into one"""

        if not scene_files:
            raise Exception("No scenes to concatenate")

        if len(scene_files) == 1:
            return scene_files[0]

        # Create concat file list
        concat_list = work_dir / "concat_list.txt"
        with open(concat_list, 'w') as f:
            for scene_file in scene_files:
                f.write(f"file '{scene_file}'\n")

        output_file = work_dir / "concatenated.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_file)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300  # 5 minutes max for concatenation
            )
        except asyncio.TimeoutError:
            print(f"FFmpeg concatenation timeout after 300s, killing process")
            process.kill()
            await process.wait()
            raise Exception("Concatenation timeout")

        if process.returncode != 0:
            raise Exception(f"Concatenation failed: {stderr.decode()}")

        return output_file

    async def _add_audio(
        self,
        video_file: Path,
        voiceover_file: Optional[Path],
        music_file: Optional[Path],
        music_volume: float,
        work_dir: Path
    ) -> Path:
        """Add voiceover and/or background music to video with proper ducking"""

        if not voiceover_file and not music_file:
            return video_file

        output_file = work_dir / "final.mp4"

        # Build FFmpeg command based on available audio
        if voiceover_file and music_file:
            # Both voiceover and music
            # Music: low volume (0.15) with fade in/out for smooth transitions
            # Voice: boosted (1.3) for clarity over music
            # amix combines them with voice priority
            # NOTE: Removed -shortest flag to prevent audio cutoff, added apad for safety
            filter_complex = (
                f"[2:a]volume=0.15,afade=t=in:st=0:d=0.5,afade=t=out:st=13:d=2[music];"
                "[1:a]volume=1.3,apad=pad_dur=2[voice];"
                "[voice][music]amix=inputs=2:duration=first:weights=3 1[aout]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_file),
                "-i", str(voiceover_file),
                "-i", str(music_file),
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(output_file)
            ]
        elif voiceover_file:
            # Only voiceover - boost and compress for clarity
            # NOTE: Removed -shortest flag to prevent audio cutoff
            # Instead, we pad audio with silence if needed or let video extend
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_file),
                "-i", str(voiceover_file),
                "-filter_complex", "[1:a]volume=1.2,acompressor=threshold=0.1:ratio=3,apad=pad_dur=2[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(output_file)
            ]
        else:
            # Only music - fade in/out
            # NOTE: Removed -shortest flag to prevent audio cutoff
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_file),
                "-i", str(music_file),
                "-filter_complex", f"[1:a]volume={music_volume},afade=t=in:st=0:d=1,afade=t=out:st=14:d=1,apad=pad_dur=2[music]",
                "-map", "0:v",
                "-map", "[music]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                str(output_file)
            ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300  # 5 minutes max for audio mixing
            )
        except asyncio.TimeoutError:
            print(f"FFmpeg audio mixing timeout after 300s, killing process")
            process.kill()
            await process.wait()
            raise Exception("Audio mixing timeout")

        if process.returncode != 0:
            raise Exception(f"Audio mixing failed: {stderr.decode()}")

        return output_file

    async def _add_captions(
        self,
        video_file: Path,
        voiceover_text: str,
        caption_style: str,
        caption_config: Dict[str, Any],
        video_format: str,
        duration: float,
        work_dir: Path,
        word_timestamps: Optional[List[WordTimestamp]] = None
    ) -> Path:
        """Add styled captions to video with optional word-by-word animation"""

        output_file = work_dir / "with_captions.mp4"

        # Get video dimensions based on format
        if video_format == "9:16":
            width, height = 1080, 1920
        elif video_format == "16:9":
            width, height = 1920, 1080
        else:  # 1:1
            width, height = 1080, 1080

        # Style configurations - font sizes optimized for mobile (9:16)
        # Increased for better readability
        base_size = 56 if video_format == "9:16" else 64

        # Define color schemes for different styles
        style_colors = {
            "classic": {"primary": "white", "highlight": "#FFD700", "bg": "black@0.7"},
            "bold": {"primary": "white", "highlight": "#FF6B6B", "bg": None},
            "neon": {"primary": "#00ff88", "highlight": "#ffffff", "bg": None},
            "minimal": {"primary": "white", "highlight": "#4ECDC4", "bg": None},
            "karaoke": {"primary": "white", "highlight": "#FFD700", "bg": "black@0.8"},
            "boxed": {"primary": "black", "highlight": "#FF6B6B", "bg": "white@1"},
            "gradient": {"primary": "white", "highlight": "#FF6B9D", "bg": "#667eea@0.9"},
        }

        colors = style_colors.get(caption_style, style_colors["classic"])

        # Override with custom config if provided
        if caption_config:
            if caption_config.get("fontSize") == "small":
                base_size = int(base_size * 0.75)
            elif caption_config.get("fontSize") == "large":
                base_size = int(base_size * 1.4)

        # If we have word timestamps, create animated word-by-word captions
        if word_timestamps and len(word_timestamps) > 0:
            print(f"Creating animated captions with {len(word_timestamps)} words")
            return await self._create_animated_captions(
                video_file, word_timestamps, colors, base_size,
                width, height, video_format, caption_config, work_dir
            )

        # Fallback: sentence-based captions without animation
        print("No word timestamps, using sentence-based captions")
        return await self._create_sentence_captions(
            video_file, voiceover_text, colors, base_size,
            width, height, duration, work_dir
        )

    async def _create_animated_captions(
        self,
        video_file: Path,
        word_timestamps: List[WordTimestamp],
        colors: Dict[str, str],
        font_size: int,
        width: int,
        height: int,
        video_format: str,
        caption_config: Dict[str, Any],
        work_dir: Path
    ) -> Path:
        """Create animated word-by-word captions using ASS subtitles"""

        output_file = work_dir / "with_captions.mp4"
        ass_file = work_dir / "captions.ass"

        # Group words into phrases (3-5 words per phrase for readability)
        words_per_phrase = 4
        phrases = []
        current_phrase = []

        for i, wt in enumerate(word_timestamps):
            current_phrase.append(wt)
            # End phrase at punctuation or every N words
            is_end_punct = wt.word.rstrip() and wt.word.rstrip()[-1] in '.!?,;:'
            if len(current_phrase) >= words_per_phrase or is_end_punct or i == len(word_timestamps) - 1:
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []

        print(f"Created {len(phrases)} phrases from {len(word_timestamps)} words")

        # Convert colors to ASS format (BGR with alpha)
        primary_color = self._color_to_ass_full(colors["primary"])
        highlight_color = self._color_to_ass_full(colors["highlight"])
        bg_color = self._color_to_ass_full(colors.get("bg", "black@0.7"))

        # Determine Y position based on config
        y_pos = height - 350 if video_format == "9:16" else height - 200
        if caption_config and caption_config.get("position") == "top":
            y_pos = 150
        elif caption_config and caption_config.get("position") == "center":
            y_pos = height // 2

        margin_v = height - y_pos - 50

        # Build ASS file with karaoke effects
        ass_content = f"""[Script Info]
Title: Animated Captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{font_size},{primary_color},{highlight_color},&H00000000,{bg_color},1,0,0,0,100,100,0,0,3,2,0,2,10,10,{margin_v},1
Style: Highlight,DejaVu Sans,{font_size},{highlight_color},{primary_color},&H00000000,{bg_color},1,0,0,0,100,100,0,0,3,2,0,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        # Generate dialogue lines with karaoke effects
        for phrase in phrases:
            if not phrase:
                continue

            phrase_start = phrase[0].start
            phrase_end = phrase[-1].end

            # Build the karaoke text with \k tags
            # Each word gets a \k tag with duration in centiseconds
            karaoke_text = ""
            for j, wt in enumerate(phrase):
                # Calculate duration in centiseconds
                word_duration = int((wt.end - wt.start) * 100)
                # Clean the word
                clean_word = wt.word.replace("{", "").replace("}", "").replace("\\", "")

                # Use \kf for smooth fill effect
                karaoke_text += f"{{\\kf{word_duration}}}{clean_word} "

            karaoke_text = karaoke_text.strip()

            # Format times as ASS format (H:MM:SS.cc)
            start_ass = self._seconds_to_ass_time(phrase_start)
            end_ass = self._seconds_to_ass_time(phrase_end + 0.3)  # Small buffer

            # Add dialogue line
            ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{karaoke_text}\n"

        # Write ASS file
        with open(ass_file, 'w', encoding='utf-8') as f:
            f.write(ass_content)

        print(f"Created ASS file with {len(phrases)} animated phrases")

        # Apply subtitles using FFmpeg
        ass_path_escaped = str(ass_file).replace('\\', '/').replace(':', r'\:')

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_file),
            "-vf", f"ass={ass_path_escaped}",
            "-c:a", "copy",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-level", "4.0",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_file)
        ]

        print(f"Applying animated captions...")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode()
            print(f"Animated caption rendering failed: {error_msg[:500]}")
            # Fallback to simpler approach
            return await self._create_simple_animated_captions(
                video_file, word_timestamps, colors, font_size, width, height, work_dir
            )

        print(f"Animated captions added successfully")
        return output_file

    async def _create_simple_animated_captions(
        self,
        video_file: Path,
        word_timestamps: List[WordTimestamp],
        colors: Dict[str, str],
        font_size: int,
        width: int,
        height: int,
        work_dir: Path
    ) -> Path:
        """Fallback: Create simple animated captions using multiple drawtext filters"""

        output_file = work_dir / "with_simple_animated_captions.mp4"

        # Group into phrases
        words_per_phrase = 4
        phrases = []
        current_phrase = []

        for i, wt in enumerate(word_timestamps):
            current_phrase.append(wt)
            if len(current_phrase) >= words_per_phrase or i == len(word_timestamps) - 1:
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []

        # Build drawtext filter chain - show each phrase during its time
        filters = []
        y_pos = height - 300

        for phrase in phrases:
            if not phrase:
                continue

            phrase_start = phrase[0].start
            phrase_end = phrase[-1].end

            # Create text for the phrase
            phrase_text = " ".join([wt.word for wt in phrase])
            # Escape special characters
            phrase_text = phrase_text.replace("'", "").replace('"', '').replace(":", " ").replace("\\", "")

            # Add drawtext filter with enable condition
            filter_str = (
                f"drawtext=text='{phrase_text}'"
                f":fontcolor=white:fontsize={font_size}"
                f":box=1:boxcolor=black@0.7:boxborderw=8"
                f":x=(w-text_w)/2:y={y_pos}"
                f":enable='between(t,{phrase_start:.2f},{phrase_end:.2f})'"
            )
            filters.append(filter_str)

        if not filters:
            return video_file

        # Limit to 20 filters to avoid command line length issues
        filters = filters[:20]
        filter_chain = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_file),
            "-vf", filter_chain,
            "-c:a", "copy",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-level", "4.0",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_file)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"Simple animated captions failed: {stderr.decode()[:300]}")
            return video_file

        return output_file

    async def _create_sentence_captions(
        self,
        video_file: Path,
        voiceover_text: str,
        colors: Dict[str, str],
        font_size: int,
        width: int,
        height: int,
        duration: float,
        work_dir: Path
    ) -> Path:
        """Create sentence-based captions (fallback when no word timestamps)"""

        output_file = work_dir / "with_captions.mp4"

        import re
        sentences = re.split(r'[.!?]', voiceover_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return video_file

        time_per_sentence = duration / len(sentences)

        # Create SRT file
        srt_file = work_dir / "captions.srt"
        srt_content = []

        for i, sentence in enumerate(sentences):
            start_time = i * time_per_sentence
            end_time = (i + 1) * time_per_sentence

            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)

            clean_text = sentence.strip()
            if len(clean_text) > 50:
                mid = len(clean_text) // 2
                space_pos = clean_text.rfind(' ', max(0, mid - 15), min(len(clean_text), mid + 15))
                if space_pos > 0:
                    clean_text = clean_text[:space_pos] + "\n" + clean_text[space_pos+1:]

            srt_content.append(f"{i+1}")
            srt_content.append(f"{start_srt} --> {end_srt}")
            srt_content.append(clean_text)
            srt_content.append("")

        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_content))

        # Apply with force_style
        force_style = f"FontName=DejaVu Sans,FontSize={font_size},PrimaryColour=&H00FFFFFF"
        force_style += ",BorderStyle=3,BackColour=&H80000000,Alignment=2"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_file),
            "-vf", f"subtitles={str(srt_file).replace(chr(92), '/')}:force_style='{force_style}'",
            "-c:a", "copy",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-level", "4.0",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_file)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"Sentence captions failed: {stderr.decode()[:300]}")
            return video_file

        return output_file

    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format (H:MM:SS.cc)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"

    def _color_to_ass_full(self, color: str) -> str:
        """Convert color to full ASS format (&HAABBGGRR) with alpha support"""
        if not color:
            return "&H80000000"

        alpha = "00"
        color_part = color

        # Handle alpha in color@alpha format
        if "@" in color:
            parts = color.split("@")
            color_part = parts[0]
            try:
                alpha_val = float(parts[1])
                # ASS uses inverse alpha (00=opaque, FF=transparent)
                alpha = f"{int((1 - alpha_val) * 255):02X}"
            except (ValueError, IndexError):
                alpha = "00"

        # Color name mapping
        color_map = {
            "white": "FFFFFF",
            "black": "000000",
            "yellow": "00FFFF",
            "red": "0000FF",
            "green": "00FF00",
            "blue": "FF0000",
            "orange": "00A5FF",
            "pink": "CBC0FF",
            "cyan": "FFFF00",
            "magenta": "FF00FF",
        }

        color_lower = color_part.lower()
        if color_lower in color_map:
            bgr = color_map[color_lower]
        elif color_lower.startswith("#"):
            # Convert hex #RRGGBB to BGR
            hex_color = color_lower[1:]
            if len(hex_color) == 6:
                r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
                bgr = f"{b}{g}{r}".upper()
            else:
                bgr = "FFFFFF"
        else:
            bgr = "FFFFFF"

        return f"&H{alpha}{bgr}"

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _color_to_ass(self, color: str) -> str:
        """Convert color to ASS format (&HAABBGGRR)"""
        # Handle common color names
        color_map = {
            "white": "&H00FFFFFF",
            "black": "&H00000000",
            "yellow": "&H0000FFFF",
            "red": "&H000000FF",
            "green": "&H0000FF00",
            "blue": "&H00FF0000",
        }

        color_lower = color.lower().split('@')[0]  # Remove alpha if present

        if color_lower in color_map:
            return color_map[color_lower]

        # Handle hex colors like #00ff88
        if color_lower.startswith('#'):
            hex_color = color_lower[1:]
            if len(hex_color) == 6:
                r = hex_color[0:2]
                g = hex_color[2:4]
                b = hex_color[4:6]
                return f"&H00{b}{g}{r}".upper()

        # Handle rgba format
        if 'black@' in color.lower():
            alpha_match = color.split('@')
            if len(alpha_match) > 1:
                try:
                    alpha = int(float(alpha_match[1]) * 255)
                    return f"&H{alpha:02X}000000"
                except ValueError:
                    pass
            return "&H80000000"

        return "&H00FFFFFF"  # Default white

    def _create_ass_style(self, style_cfg: dict, width: int, height: int) -> str:
        """Create ASS style string"""
        return ""  # Not needed with force_style

    def cleanup(self):
        """Clean up temporary directory"""
        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            pass
