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


class CompositionResult(BaseModel):
    success: bool
    output_url: Optional[str] = None
    duration: float = 0
    file_size_bytes: int = 0
    error_message: Optional[str] = None


class VideoCompositorService:
    """Composes videos using FFmpeg"""

    def __init__(self, output_dir: str = "/tmp/viralify/videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="viralify_"))

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

            if request.voiceover_url:
                # Voiceover is usually a local file path from TTS generation
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
                work_dir
            )

            # Step 4: Concatenate all scenes
            if progress_callback:
                progress_callback(60, "Concatenating scenes...")

            concat_file = await self._concatenate_scenes(
                processed_scenes,
                work_dir
            )

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

            return CompositionResult(
                success=True,
                output_url=str(final_output),
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
            except:
                pass

    async def _download_file(self, url: str, output_path: Path) -> Path:
        """Download a file from URL"""
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=120.0, follow_redirects=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

        return output_path

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

    async def _process_scenes(
        self,
        scene_files: Dict[str, Path],
        scenes: List[CompositionScene],
        format: str,
        quality: str,
        fps: int,
        work_dir: Path
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
            else:
                # Process image: add Ken Burns effect (zoom/pan)
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

            # Run FFmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

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
        await process.communicate()

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
        stdout, stderr = await process.communicate()

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
            filter_complex = (
                f"[2:a]volume=0.15,afade=t=in:st=0:d=0.5,afade=t=out:st=13:d=2[music];"
                "[1:a]volume=1.3[voice];"
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
                "-shortest",
                "-movflags", "+faststart",
                str(output_file)
            ]
        elif voiceover_file:
            # Only voiceover - boost and compress for clarity
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_file),
                "-i", str(voiceover_file),
                "-filter_complex", "[1:a]volume=1.2,acompressor=threshold=0.1:ratio=3[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                str(output_file)
            ]
        else:
            # Only music - fade in/out
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_file),
                "-i", str(music_file),
                "-filter_complex", f"[1:a]volume={music_volume},afade=t=in:st=0:d=1,afade=t=out:st=14:d=1[music]",
                "-map", "0:v",
                "-map", "[music]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
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
            except:
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
                except:
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
        except:
            pass
