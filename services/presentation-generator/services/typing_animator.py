"""
Typing Animator Service

Creates typing animation videos showing code being written character by character.
Designed to feel natural and human-like, as if a developer is typing while explaining.

OPTIMIZED: Streams frames directly to disk instead of accumulating in memory.
This reduces memory usage from ~12GB to ~50MB for typical animations.
"""
import asyncio
import gc
import io
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Generator

from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.styles import get_style_by_name
from pygments.token import Token


class TypingAnimatorService:
    """Service for creating typing animation videos with human-like feel"""

    # Video dimensions
    WIDTH = 1920
    HEIGHT = 1080

    # Margins
    MARGIN_X = 100
    MARGIN_Y = 100

    # Animation settings - configurable via environment
    # TYPING_ANIMATION_FPS: 15 = faster generation, 30 = smoother animation
    DEFAULT_FPS = int(os.getenv("TYPING_ANIMATION_FPS", "15"))  # Reduced from 30 for speed

    # Speed presets (chars per second) - designed for teaching/explaining context
    # TYPING_ANIMATION_SPEED: slow, natural, moderate, fast, turbo
    SPEED_PRESETS = {
        "slow": 2.0,       # Very deliberate, teaching beginners
        "natural": 4.0,    # Human explaining while typing
        "moderate": 6.0,   # Confident developer
        "fast": 10.0,      # Quick demo
        "turbo": 20.0      # Production speed - minimal animation time
    }

    # Skip frame generation for very fast mode (just show final result)
    SKIP_ANIMATION_THRESHOLD = 25.0  # chars/sec above this = skip animation

    # ANIMATION MODE SETTINGS:
    # TYPING_ANIMATION_MODE: animated, static, optimized (default: animated)
    # - animated: frame-by-frame typing (shows actual character-by-character typing)
    # - static: single image, no animation (fastest, just shows final code)
    # - optimized: single image + FFmpeg reveal animation (fast, line-by-line reveal)
    #
    # When SSVS-C sync is active (sync_mode=True), a "reveal" animation is used
    # that shows code appearing in sync with voiceover (line-by-line).
    # To force character-by-character typing animation even when SSVS-C is available,
    # set FORCE_TYPING_ANIMATION=true in the compositor.
    ANIMATION_MODE = os.getenv("TYPING_ANIMATION_MODE", "animated").lower()

    # Threshold for auto-switching to optimized mode (chars)
    # Codes longer than this will use optimized mode automatically
    OPTIMIZED_THRESHOLD_CHARS = int(os.getenv("TYPING_OPTIMIZED_THRESHOLD", "300"))

    # Keywords that deserve a pause (as if thinking/explaining)
    PAUSE_KEYWORDS = {
        'def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while',
        'import', 'from', 'try', 'except', 'finally', 'with', 'async',
        'await', 'yield', 'lambda', 'raise', 'assert', 'function', 'const',
        'let', 'var', 'public', 'private', 'static', 'void', 'int', 'string'
    }

    def __init__(self):
        self.fonts_dir = Path(__file__).parent.parent / "fonts"

    def _load_font(self, style: str, size: int) -> ImageFont.FreeTypeFont:
        """Load a font with fallbacks"""
        font_files = {
            "mono": "DejaVuSansMono.ttf",
            "bold": "DejaVuSans-Bold.ttf",
            "regular": "DejaVuSans.ttf"
        }

        font_path = self.fonts_dir / font_files.get(style, "DejaVuSansMono.ttf")
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except:
                pass

        # Fallback
        return ImageFont.load_default()

    async def create_typing_animation(
        self,
        code: str,
        language: str,
        output_path: str,
        title: Optional[str] = None,
        typing_speed: str = "natural",
        fps: int = None,
        target_duration: float = None,
        execution_output: Optional[str] = None,
        background_color: str = "#1e1e2e",
        text_color: str = "#cdd6f4",
        accent_color: str = "#89b4fa",
        pygments_style: str = "monokai",
        # SSVS-C: Synced mode parameters
        reveal_points: Optional[List[dict]] = None,
        sync_mode: bool = False,
        # Code display mode parameters (from user choice)
        force_static: bool = False,
        force_typing: bool = False
    ) -> Tuple[str, float]:
        """
        Create a typing animation video with human-like feel.

        Args:
            code: The code to animate
            language: Programming language for syntax highlighting
            output_path: Path for the output video file
            title: Optional title to show above the code
            typing_speed: Speed preset - 'slow', 'natural', 'moderate', 'fast'
            fps: Frames per second (default: 30)
            target_duration: Target duration in seconds (adjusts typing speed)
            execution_output: Optional code execution output to show at the end
            background_color: Background color hex
            text_color: Text color hex
            accent_color: Accent color hex
            pygments_style: Pygments style for syntax highlighting

        Returns:
            Tuple of (path to the generated video file, actual duration in seconds)
        """
        fps = fps or self.DEFAULT_FPS

        # Get base speed from preset
        base_speed = self.SPEED_PRESETS.get(typing_speed, self.SPEED_PRESETS["natural"])

        # Calculate typing speed based on target duration if provided
        if target_duration and target_duration > 0:
            # Reserve time for intro, output display, and outro hold
            output_display_time = 4.0 if execution_output else 0  # Longer output display
            outro_time = 2.0
            intro_time = 0.5

            # Calculate ACTUAL pause overhead based on code content
            # This is critical - pauses for newlines, colons, etc add significant time
            pause_overhead = self._calculate_pause_overhead(code)

            typing_time = (target_duration - intro_time - outro_time - output_display_time) / pause_overhead
            if typing_time > 0 and len(code) > 0:
                chars_per_second = len(code) / typing_time
                # Clamp to reasonable range - allow turbo speed for production
                max_speed = self.SPEED_PRESETS["turbo"]  # Allow up to turbo speed
                min_speed = self.SPEED_PRESETS["slow"]
                chars_per_second = max(min_speed, min(chars_per_second, max_speed))
            else:
                chars_per_second = base_speed
        else:
            chars_per_second = base_speed

        # ========================================================================
        # CODE DISPLAY MODE ROUTING
        # Priority: User explicit choice > SSVS-C sync > auto-detection
        # ========================================================================

        # 1. STATIC mode (user choice: instant code display)
        if force_static:
            print(f"[TYPING] STATIC MODE (user choice): {len(code)} chars (instant display)", flush=True)
            return await self._create_static_video(
                code=code,
                language=language,
                output_path=output_path,
                title=title,
                target_duration=target_duration or 5.0,
                fps=fps,
                execution_output=execution_output,
                background_color=background_color,
                text_color=text_color,
                accent_color=accent_color,
                pygments_style=pygments_style
            )

        # 2. TYPING mode (user choice: character-by-character animation)
        if force_typing:
            print(f"[TYPING] TYPING MODE (user choice): {len(code)} chars at {chars_per_second:.1f} chars/sec", flush=True)
            # Skip to the frame-by-frame animation section below
            pass
        else:
            # 3. REVEAL mode: SSVS-C synced line-by-line reveal
            if sync_mode and reveal_points:
                print(f"[TYPING] REVEAL MODE (synced): {len(code)} chars, {len(reveal_points)} reveal points", flush=True)
                return await self._create_synced_reveal_video(
                    code=code,
                    language=language,
                    output_path=output_path,
                    title=title,
                    target_duration=target_duration or 10.0,
                    fps=fps,
                    reveal_points=reveal_points,
                    execution_output=execution_output,
                    background_color=background_color,
                    text_color=text_color,
                    accent_color=accent_color,
                    pygments_style=pygments_style
                )

            # 4. Auto-detection fallback (legacy behavior)
            # Priority: explicit static > optimized (default) > animated (legacy)
            use_static = (
                chars_per_second >= self.SKIP_ANIMATION_THRESHOLD or
                self.ANIMATION_MODE == "static"
            )

            use_optimized = (
                not use_static and
                self.ANIMATION_MODE == "optimized"
            )

            if use_static:
                print(f"[TYPING] STATIC MODE (auto): {len(code)} chars (no animation)", flush=True)
                return await self._create_static_video(
                    code=code,
                    language=language,
                    output_path=output_path,
                    title=title,
                    target_duration=target_duration or 5.0,
                    fps=fps,
                    execution_output=execution_output,
                    background_color=background_color,
                    text_color=text_color,
                    accent_color=accent_color,
                    pygments_style=pygments_style
                )

            if use_optimized:
                print(f"[TYPING] OPTIMIZED MODE: {len(code)} chars (FFmpeg reveal animation)", flush=True)
                return await self._create_optimized_reveal_video(
                    code=code,
                    language=language,
                    output_path=output_path,
                    title=title,
                    target_duration=target_duration or (len(code) / chars_per_second + 3.0),
                    fps=fps,
                    execution_output=execution_output,
                    background_color=background_color,
                    text_color=text_color,
                    accent_color=accent_color,
                    pygments_style=pygments_style
                )

        print(f"[TYPING] Creating animation: {len(code)} chars at {chars_per_second:.1f} chars/sec (speed: {typing_speed}, target: {target_duration or 'auto'}s)", flush=True)
        if execution_output:
            print(f"[TYPING] Will show execution output: {execution_output[:50]}...", flush=True)

        # OPTIMIZED: Stream frames directly to disk instead of memory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate frames directly to disk
            frame_count = await self._generate_frames_to_disk(
                code=code,
                language=language,
                title=title,
                chars_per_second=chars_per_second,
                fps=fps,
                execution_output=execution_output,
                background_color=background_color,
                text_color=text_color,
                accent_color=accent_color,
                pygments_style=pygments_style,
                output_dir=temp_path
            )

            if frame_count == 0:
                raise ValueError("No frames generated")

            # Calculate actual duration
            actual_duration = frame_count / fps
            print(f"[TYPING] Generated {frame_count} frames ({actual_duration:.1f}s)", flush=True)

            # Pad with extra frames if shorter than target duration
            if target_duration and actual_duration < target_duration:
                frames_needed = int((target_duration - actual_duration) * fps)
                if frames_needed > 0:
                    # Copy the last frame for padding
                    last_frame_path = temp_path / f"frame_{frame_count - 1:06d}.png"
                    for i in range(frames_needed):
                        new_frame_path = temp_path / f"frame_{frame_count + i:06d}.png"
                        shutil.copy(last_frame_path, new_frame_path)
                    frame_count += frames_needed
                    actual_duration = frame_count / fps
                    print(f"[TYPING] Padded to {frame_count} frames ({actual_duration:.1f}s) to match target", flush=True)

            # Convert frames to video
            video_path = await self._frames_dir_to_video(temp_path, output_path, fps)

            print(f"[TYPING] Video created: {video_path}", flush=True)

            return video_path, actual_duration

    async def _create_static_video(
        self,
        code: str,
        language: str,
        output_path: str,
        title: Optional[str],
        target_duration: float,
        fps: int,
        execution_output: Optional[str],
        background_color: str,
        text_color: str,
        accent_color: str,
        pygments_style: str
    ) -> Tuple[str, float]:
        """
        Create a static video showing the final code (no animation).
        Much faster than frame-by-frame animation.

        Uses FFmpeg to create a video from a single image.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate a single frame with the complete code
            if execution_output:
                frame = await self._render_frame_with_output(
                    code=code,
                    output=execution_output,
                    language=language,
                    title=title,
                    background_color=background_color,
                    text_color=text_color,
                    accent_color=accent_color,
                    pygments_style=pygments_style
                )
            else:
                frame = await self._render_frame(
                    text=code,
                    language=language,
                    title=title,
                    show_cursor=False,  # No cursor for static mode
                    background_color=background_color,
                    text_color=text_color,
                    accent_color=accent_color,
                    pygments_style=pygments_style
                )

            # Save the frame
            frame_path = temp_path / "static_frame.png"
            frame.save(str(frame_path), format="PNG")
            frame.close()

            # Use FFmpeg to create video from single image
            # This is MUCH faster than generating many frames
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(frame_path),
                "-c:v", "libx264",
                "-t", str(target_duration),
                "-pix_fmt", "yuv420p",
                "-r", str(fps),
                "-preset", "ultrafast",
                "-tune", "stillimage",
                str(output_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg failed to create static video")

            print(f"[TYPING] Static video created: {output_path} ({target_duration:.1f}s)", flush=True)
            return output_path, target_duration

    async def _create_optimized_reveal_video(
        self,
        code: str,
        language: str,
        output_path: str,
        title: Optional[str],
        target_duration: float,
        fps: int,
        execution_output: Optional[str],
        background_color: str,
        text_color: str,
        accent_color: str,
        pygments_style: str
    ) -> Tuple[str, float]:
        """
        OPTIMIZED: Create reveal animation using FFmpeg drawbox.

        Instead of generating 14000+ frames with Pillow:
        1. Render ONE image with complete syntax-highlighted code
        2. Use FFmpeg drawbox filter to create reveal animation
        3. Result: ~2-3 seconds instead of 7+ minutes

        The drawbox filter draws a black rectangle that shrinks from right to left,
        progressively revealing the code underneath.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Calculate timing
            # Reserve time for: intro hold (0.5s) + typing reveal + final hold (2s) + output (3s if present)
            intro_hold = 0.5
            final_hold = 2.0
            output_hold = 3.0 if execution_output else 0

            # Typing reveal gets the remaining time
            reveal_duration = max(2.0, target_duration - intro_hold - final_hold - output_hold)

            # 1. Render the complete code image with syntax highlighting
            code_frame = await self._render_frame(
                text=code,
                language=language,
                title=title,
                show_cursor=False,
                background_color=background_color,
                text_color=text_color,
                accent_color=accent_color,
                pygments_style=pygments_style
            )

            code_image_path = temp_path / "code_complete.png"
            code_frame.save(str(code_image_path), format="PNG")
            code_frame.close()

            # 2. Create the reveal animation with FFmpeg
            # Strategy: Use drawbox to cover the image with black, then animate its position

            # Calculate the code area boundaries (where the actual code is)
            # Based on _render_frame: code starts at MARGIN_X + 60 (after line numbers)
            code_start_x = self.MARGIN_X + 60
            code_end_x = self.WIDTH - self.MARGIN_X

            # We'll reveal from left to right within the code area
            reveal_width = code_end_x - code_start_x

            reveal_video_path = temp_path / "reveal.mp4"

            # FFmpeg filter: drawbox that shrinks from right to left
            # x position starts at code_start_x and moves right as time progresses
            # The box covers everything to the right of the revealed portion
            filter_complex = (
                f"drawbox=x='min({code_start_x}+({reveal_width}*t/{reveal_duration}),{code_end_x})':"
                f"y=0:w={self.WIDTH}:h={self.HEIGHT}:c={background_color}@1:t=fill"
            )

            cmd_reveal = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(code_image_path),
                "-vf", filter_complex,
                "-t", str(reveal_duration),
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                str(reveal_video_path)
            ]

            print(f"[TYPING] Creating reveal animation ({reveal_duration:.1f}s)...", flush=True)

            process = await asyncio.create_subprocess_exec(
                *cmd_reveal,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                print(f"[TYPING] FFmpeg reveal error: {stderr.decode()}", flush=True)
                # Fallback to static mode
                return await self._create_static_video(
                    code=code,
                    language=language,
                    output_path=output_path,
                    title=title,
                    target_duration=target_duration,
                    fps=fps,
                    execution_output=execution_output,
                    background_color=background_color,
                    text_color=text_color,
                    accent_color=accent_color,
                    pygments_style=pygments_style
                )

            # 3. Build the final video with intro, reveal, hold, and optional output
            segments = []

            # Intro: blank/cursor frame held for 0.5s
            intro_frame = await self._render_frame(
                text="",
                language=language,
                title=title,
                show_cursor=True,
                background_color=background_color,
                text_color=text_color,
                accent_color=accent_color,
                pygments_style=pygments_style
            )
            intro_image_path = temp_path / "intro.png"
            intro_frame.save(str(intro_image_path), format="PNG")
            intro_frame.close()

            intro_video_path = temp_path / "intro.mp4"
            cmd_intro = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(intro_image_path),
                "-t", str(intro_hold),
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                str(intro_video_path)
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd_intro,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            segments.append(str(intro_video_path))

            # Reveal animation
            segments.append(str(reveal_video_path))

            # Final hold: complete code held for comprehension
            final_video_path = temp_path / "final_hold.mp4"
            cmd_final = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(code_image_path),
                "-t", str(final_hold),
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                str(final_video_path)
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd_final,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            segments.append(str(final_video_path))

            # Output display if provided
            if execution_output:
                output_frame = await self._render_frame_with_output(
                    code=code,
                    output=execution_output,
                    language=language,
                    title=title,
                    background_color=background_color,
                    text_color=text_color,
                    accent_color=accent_color,
                    pygments_style=pygments_style
                )
                output_image_path = temp_path / "output.png"
                output_frame.save(str(output_image_path), format="PNG")
                output_frame.close()

                output_video_path = temp_path / "output.mp4"
                cmd_output = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", str(output_image_path),
                    "-t", str(output_hold),
                    "-r", str(fps),
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p",
                    str(output_video_path)
                ]
                process = await asyncio.create_subprocess_exec(
                    *cmd_output,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                segments.append(str(output_video_path))

            # 4. Concatenate all segments
            concat_list_path = temp_path / "concat.txt"
            with open(concat_list_path, "w") as f:
                for seg in segments:
                    f.write(f"file '{seg}'\n")

            cmd_concat = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd_concat,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                print(f"[TYPING] FFmpeg concat error: {stderr.decode()}", flush=True)
                raise RuntimeError(f"FFmpeg concat failed: {stderr.decode()}")

            actual_duration = intro_hold + reveal_duration + final_hold + output_hold
            print(f"[TYPING] Optimized video created: {output_path} ({actual_duration:.1f}s)", flush=True)

            return output_path, actual_duration

    async def _create_synced_reveal_video(
        self,
        code: str,
        language: str,
        output_path: str,
        title: Optional[str],
        target_duration: float,
        fps: int,
        reveal_points: List[dict],
        execution_output: Optional[str],
        background_color: str,
        text_color: str,
        accent_color: str,
        pygments_style: str
    ) -> Tuple[str, float]:
        """
        SSVS-C: Create video with line-by-line reveal synced to voiceover.

        Strategy:
        1. Render complete code image with syntax highlighting
        2. Use FFmpeg drawbox filters to mask unrevealed lines
        3. Masks disappear at reveal_points times (synced to voiceover)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 1. Render complete code image
            code_frame = await self._render_frame(
                text=code,
                language=language,
                title=title,
                show_cursor=False,
                background_color=background_color,
                text_color=text_color,
                accent_color=accent_color,
                pygments_style=pygments_style
            )

            code_image_path = temp_path / "code_complete.png"
            code_frame.save(str(code_image_path), format="PNG")
            code_frame.close()

            # 2. Calculate line metrics - MUST match _render_frame exactly
            lines = code.split('\n')
            total_lines = len(lines)

            # Calculate line_height dynamically to match _render_frame
            # _render_frame uses: bbox = draw.textbbox((0, 0), "Ay", font=code_font)
            # code_font is self._load_font("mono", 24) -> CODE_FONT_SIZE = 24
            # line_height = bbox[3] - bbox[1] + 8
            # For 24pt monospace, this is approximately 32 pixels
            line_height = 32

            # Calculate code_start_y to match _render_frame:
            # - y_offset starts at MARGIN_Y (100)
            # - If title: y_offset += 60 (title) + 30 (separator) = 190
            # - Code text starts at y_offset + padding (padding = 20)
            # - So: with title = 190 + 20 = 210, without title = 100 + 20 = 120
            padding = 20
            code_start_y = (190 + padding) if title else (self.MARGIN_Y + padding)  # 210 or 120
            code_start_x = self.MARGIN_X + 60  # After line numbers
            code_width = self.WIDTH - code_start_x - self.MARGIN_X

            print(f"[TYPING] Synced reveal metrics: title={title is not None}, code_start_y={code_start_y}, line_height={line_height}, total_lines={total_lines}", flush=True)

            # 3. Build reveal timeline from reveal_points
            revealed_at: dict = {}  # line_num -> reveal_time

            for rp in reveal_points:
                start_line = rp.get('start_line', 1)
                end_line = rp.get('end_line', start_line)
                reveal_time = rp.get('reveal_at', rp.get('reveal_time', 0))

                for line in range(start_line, min(end_line + 1, total_lines + 1)):
                    if line not in revealed_at:
                        revealed_at[line] = reveal_time

            # Unrevealed lines get revealed near the end
            for line in range(1, total_lines + 1):
                if line not in revealed_at:
                    revealed_at[line] = max(0, target_duration - 1.5)

            # 4. Generate FFmpeg filter complex for line masking
            filters = []
            bg_color_hex = background_color.replace('#', '0x')

            for line_num, reveal_time in sorted(revealed_at.items()):
                y = code_start_y + (line_num - 1) * line_height

                # Mask this line UNTIL reveal_time
                filter_str = (
                    f"drawbox=x={code_start_x}:y={y}:"
                    f"w={code_width}:h={line_height}:"
                    f"c={bg_color_hex}@1:t=fill:"
                    f"enable='lt(t,{reveal_time:.2f})'"
                )
                filters.append(filter_str)

            # Combine filters
            filter_complex = ",".join(filters) if filters else "null"

            # 5. Create reveal video with FFmpeg
            reveal_video_path = temp_path / "synced_reveal.mp4"

            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(code_image_path),
                "-vf", filter_complex,
                "-t", str(target_duration),
                "-r", str(fps),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(reveal_video_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                print(f"[TYPING] FFmpeg synced reveal error: {stderr.decode()}", flush=True)
                # Fallback to optimized mode
                return await self._create_optimized_reveal_video(
                    code=code,
                    language=language,
                    output_path=output_path,
                    title=title,
                    target_duration=target_duration,
                    fps=fps,
                    execution_output=execution_output,
                    background_color=background_color,
                    text_color=text_color,
                    accent_color=accent_color,
                    pygments_style=pygments_style
                )

            # 6. Handle execution output if provided
            if execution_output:
                # Append output display segment
                output_hold = 3.0
                output_frame = await self._render_frame_with_output(
                    code=code,
                    output=execution_output,
                    language=language,
                    title=title,
                    background_color=background_color,
                    text_color=text_color,
                    accent_color=accent_color,
                    pygments_style=pygments_style
                )
                output_image_path = temp_path / "output.png"
                output_frame.save(str(output_image_path), format="PNG")
                output_frame.close()

                output_video_path = temp_path / "output.mp4"
                cmd_output = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", str(output_image_path),
                    "-t", str(output_hold),
                    "-r", str(fps),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    str(output_video_path)
                ]
                process = await asyncio.create_subprocess_exec(
                    *cmd_output,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()

                # Concatenate reveal + output
                concat_list_path = temp_path / "concat.txt"
                with open(concat_list_path, "w") as f:
                    f.write(f"file '{reveal_video_path}'\n")
                    f.write(f"file '{output_video_path}'\n")

                cmd_concat = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_list_path),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    output_path
                ]
                process = await asyncio.create_subprocess_exec(
                    *cmd_concat,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()

                actual_duration = target_duration + output_hold
            else:
                # Just copy the reveal video
                shutil.copy(str(reveal_video_path), output_path)
                actual_duration = target_duration

            print(f"[TYPING] Synced reveal video created: {output_path} ({actual_duration:.1f}s, {len(reveal_points)} reveals)", flush=True)

            return output_path, actual_duration

    def _get_word_at_position(self, text: str, pos: int) -> str:
        """Extract the word that just completed at this position"""
        # Find the start of the word
        start = pos
        while start > 0 and text[start - 1].isalnum() or (start > 0 and text[start - 1] == '_'):
            start -= 1
        return text[start:pos]

    def _calculate_pause_overhead(self, code: str) -> float:
        """
        Calculate the actual pause overhead multiplier based on code content.

        The human-like pauses for newlines, colons, brackets, keywords etc.
        add significant time. This calculates the real overhead.

        Returns:
            Multiplier to apply to typing time (e.g., 2.5 means 2.5x base time)
        """
        if not code:
            return 1.5

        # Count pause-inducing characters
        newlines = code.count('\n')
        colons = code.count(':')
        equals = code.count('=')
        open_brackets = code.count('(') + code.count('{') + code.count('[')
        close_brackets = code.count(')') + code.count('}') + code.count(']')
        commas = code.count(',')
        spaces = code.count(' ')

        # Count keywords (after spaces)
        keyword_count = 0
        words = code.split()
        for word in words:
            word_clean = word.strip('():{}[].,')
            if word_clean.lower() in self.PAUSE_KEYWORDS:
                keyword_count += 1

        total_chars = len(code)
        if total_chars == 0:
            return 1.5

        # Calculate weighted pause contribution
        # Base: 1 frame per char at target speed
        # Newlines add 3 extra frames (4x - 1 = 3 extra)
        # Colons add 2 extra (3x - 1 = 2 extra)
        # etc.
        pause_frames = (
            newlines * 3.0 +      # 4x multiplier = 3 extra
            colons * 2.0 +        # 3x multiplier = 2 extra
            equals * 1.0 +        # 2x multiplier = 1 extra
            open_brackets * 1.0 + # 2x multiplier = 1 extra
            close_brackets * 0.5 + # 1.5x = 0.5 extra
            commas * 0.5 +        # 1.5x = 0.5 extra
            keyword_count * 2.0 + # 3x after keyword space
            spaces * 0.2          # 1.2x = 0.2 extra
        )

        # Calculate overhead ratio
        # overhead = (base_frames + pause_frames) / base_frames
        base_frames = total_chars
        overhead = (base_frames + pause_frames) / base_frames

        # Add hold frames: intro (0.5s) + final code hold (3s) + output (3s)
        # At 30fps: 0.5*30 + 3*30 + 3*30 = 195 frames
        # For typical code of 500 chars at 30fps/6cps = ~2500 base frames
        # hold_overhead = 195/2500 = ~0.08
        estimated_base_frames = total_chars * 5  # ~6 chars/sec at 30fps
        hold_frames = 15 + 90 + 90  # intro + code hold + output hold
        hold_overhead = 1 + (hold_frames / max(estimated_base_frames, 100))

        final_overhead = overhead * hold_overhead

        # Clamp to reasonable range
        final_overhead = max(1.5, min(final_overhead, 5.0))

        print(f"[TYPING] Pause overhead calculated: {final_overhead:.2f}x (newlines:{newlines}, colons:{colons}, keywords:{keyword_count})", flush=True)

        return final_overhead

    async def _generate_frames(
        self,
        code: str,
        language: str,
        title: Optional[str],
        chars_per_second: float,
        fps: int,
        execution_output: Optional[str],
        background_color: str,
        text_color: str,
        accent_color: str,
        pygments_style: str
    ) -> List[Image.Image]:
        """Generate all frames for the typing animation with human-like timing"""

        frames = []
        current_text = ""

        # Calculate base frames per character
        base_frames_per_char = fps / chars_per_second

        # Get lexer for syntax highlighting
        try:
            lexer = get_lexer_by_name(language, stripall=True)
        except:
            lexer = TextLexer()

        try:
            style = get_style_by_name(pygments_style)
        except:
            style = get_style_by_name("monokai")

        frame_count = 0
        char_index = 0

        # Initial empty frame (hold for 0.5 seconds - developer getting ready)
        initial_frame = await self._render_frame(
            text="",
            language=language,
            title=title,
            show_cursor=True,
            background_color=background_color,
            text_color=text_color,
            accent_color=accent_color,
            pygments_style=pygments_style
        )
        for _ in range(int(fps * 0.5)):
            frames.append(initial_frame)

        # Track current word for keyword detection
        current_word = ""

        # Generate frames for each character with human-like timing
        for i, char in enumerate(code):
            current_text += char
            char_index += 1

            # Build current word
            if char.isalnum() or char == '_':
                current_word += char
            else:
                # Word just ended - check if it was a keyword
                word_just_ended = current_word
                current_word = ""

            # Determine how many frames to show this state (human-like pauses)
            if char == '\n':
                # New line: pause as if thinking about next line
                num_frames = int(base_frames_per_char * 4)
            elif char == ':':
                # After colon (def, class, if, etc.): thinking pause
                num_frames = int(base_frames_per_char * 3)
            elif char == '=' and i > 0 and code[i-1] != '=' and (i + 1 >= len(code) or code[i+1] != '='):
                # Assignment: slight pause
                num_frames = int(base_frames_per_char * 2)
            elif char in '({[':
                # Opening brackets: slight pause before typing contents
                num_frames = int(base_frames_per_char * 2)
            elif char in ')}]':
                # Closing brackets: slight pause
                num_frames = int(base_frames_per_char * 1.5)
            elif char == ',':
                # Comma: brief pause
                num_frames = int(base_frames_per_char * 1.5)
            elif char == ' ' and word_just_ended and word_just_ended.lower() in self.PAUSE_KEYWORDS:
                # Space after a keyword: pause as if explaining
                num_frames = int(base_frames_per_char * 3)
            elif char == ' ':
                # Regular space: slight natural pause between words
                num_frames = int(base_frames_per_char * 1.2)
            else:
                # Regular character: add slight randomness for human feel
                variation = random.uniform(0.8, 1.2)
                num_frames = max(1, int(base_frames_per_char * variation))

            # Generate frame with current text
            frame = await self._render_frame(
                text=current_text,
                language=language,
                title=title,
                show_cursor=True,
                background_color=background_color,
                text_color=text_color,
                accent_color=accent_color,
                pygments_style=pygments_style
            )

            # Add frames (with cursor blinking on longer pauses)
            for i in range(num_frames):
                # Blink cursor every 15 frames
                if num_frames > 10 and i % 15 > 7:
                    frame_no_cursor = await self._render_frame(
                        text=current_text,
                        language=language,
                        title=title,
                        show_cursor=False,
                        background_color=background_color,
                        text_color=text_color,
                        accent_color=accent_color,
                        pygments_style=pygments_style
                    )
                    frames.append(frame_no_cursor)
                else:
                    frames.append(frame)

            frame_count += num_frames

            # Progress update every 50 characters
            if char_index % 50 == 0:
                print(f"[TYPING] Progress: {char_index}/{len(code)} characters", flush=True)

        # Final code frame (hold for 3 seconds for learner comprehension)
        # Increased from 0.5s to give time for code assimilation
        comprehension_hold_seconds = 3.0
        for i in range(int(fps * comprehension_hold_seconds)):
            show_cursor = (i % 15) < 8  # Blink cursor
            final_frame = await self._render_frame(
                text=current_text,
                language=language,
                title=title,
                show_cursor=show_cursor,
                background_color=background_color,
                text_color=text_color,
                accent_color=accent_color,
                pygments_style=pygments_style
            )
            frames.append(final_frame)

        # Add execution output display if provided
        if execution_output:
            # Hold output for 3 seconds
            output_frames = int(fps * 3)
            output_frame = await self._render_frame_with_output(
                code=current_text,
                output=execution_output,
                language=language,
                title=title,
                background_color=background_color,
                text_color=text_color,
                accent_color=accent_color,
                pygments_style=pygments_style
            )
            for _ in range(output_frames):
                frames.append(output_frame)

        return frames

    async def _generate_frames_to_disk(
        self,
        code: str,
        language: str,
        title: Optional[str],
        chars_per_second: float,
        fps: int,
        execution_output: Optional[str],
        background_color: str,
        text_color: str,
        accent_color: str,
        pygments_style: str,
        output_dir: Path
    ) -> int:
        """
        OPTIMIZED: Generate frames directly to disk instead of memory.

        Returns the total number of frames generated.
        """
        current_text = ""
        frame_index = 0

        # Calculate base frames per character
        base_frames_per_char = fps / chars_per_second

        # Get lexer for syntax highlighting
        try:
            lexer = get_lexer_by_name(language, stripall=True)
        except:
            lexer = TextLexer()

        try:
            style = get_style_by_name(pygments_style)
        except:
            style = get_style_by_name("monokai")

        char_index = 0

        # Initial empty frame (hold for 0.5 seconds)
        initial_frame = await self._render_frame(
            text="",
            language=language,
            title=title,
            show_cursor=True,
            background_color=background_color,
            text_color=text_color,
            accent_color=accent_color,
            pygments_style=pygments_style
        )
        # Save initial frames
        initial_count = int(fps * 0.5)
        for i in range(initial_count):
            frame_path = output_dir / f"frame_{frame_index:06d}.png"
            if i == 0:
                initial_frame.save(frame_path, "PNG")
                first_frame_path = frame_path
            else:
                shutil.copy(first_frame_path, frame_path)
            frame_index += 1
        # Clear from memory
        del initial_frame
        gc.collect()

        # Track current word for keyword detection
        current_word = ""
        last_frame_path = None

        # Generate frames for each character with human-like timing
        for i, char in enumerate(code):
            current_text += char
            char_index += 1

            # Build current word
            word_just_ended = ""
            if char.isalnum() or char == '_':
                current_word += char
            else:
                word_just_ended = current_word
                current_word = ""

            # Determine how many frames for this state
            if char == '\n':
                num_frames = int(base_frames_per_char * 4)
            elif char == ':':
                num_frames = int(base_frames_per_char * 3)
            elif char == '=' and i > 0 and code[i-1] != '=' and (i + 1 >= len(code) or code[i+1] != '='):
                num_frames = int(base_frames_per_char * 2)
            elif char in '({[':
                num_frames = int(base_frames_per_char * 2)
            elif char in ')}]':
                num_frames = int(base_frames_per_char * 1.5)
            elif char == ',':
                num_frames = int(base_frames_per_char * 1.5)
            elif char == ' ' and word_just_ended and word_just_ended.lower() in self.PAUSE_KEYWORDS:
                num_frames = int(base_frames_per_char * 3)
            elif char == ' ':
                num_frames = int(base_frames_per_char * 1.2)
            else:
                variation = random.uniform(0.8, 1.2)
                num_frames = max(1, int(base_frames_per_char * variation))

            # Render frame with cursor
            frame = await self._render_frame(
                text=current_text,
                language=language,
                title=title,
                show_cursor=True,
                background_color=background_color,
                text_color=text_color,
                accent_color=accent_color,
                pygments_style=pygments_style
            )

            # Save frame to disk
            frame_path = output_dir / f"frame_{frame_index:06d}.png"
            frame.save(frame_path, "PNG")
            last_frame_path = frame_path
            frame_index += 1

            # For additional frames (pauses), copy the file instead of re-rendering
            if num_frames > 1:
                # Check if we need cursor blinking
                if num_frames > 10:
                    # Render frame without cursor for blinking
                    frame_no_cursor = await self._render_frame(
                        text=current_text,
                        language=language,
                        title=title,
                        show_cursor=False,
                        background_color=background_color,
                        text_color=text_color,
                        accent_color=accent_color,
                        pygments_style=pygments_style
                    )
                    no_cursor_path = output_dir / f"frame_{frame_index:06d}_nc.png"
                    frame_no_cursor.save(no_cursor_path, "PNG")
                    del frame_no_cursor

                    for j in range(1, num_frames):
                        dest_path = output_dir / f"frame_{frame_index:06d}.png"
                        if j % 15 > 7:
                            shutil.copy(no_cursor_path, dest_path)
                        else:
                            shutil.copy(last_frame_path, dest_path)
                        frame_index += 1

                    # Clean up temp no-cursor frame
                    no_cursor_path.unlink()
                else:
                    # Just copy the frame for shorter pauses
                    for j in range(1, num_frames):
                        dest_path = output_dir / f"frame_{frame_index:06d}.png"
                        shutil.copy(last_frame_path, dest_path)
                        frame_index += 1

            # Clear frame from memory
            del frame

            # Periodic garbage collection
            if char_index % 20 == 0:
                gc.collect()

            # Progress update
            if char_index % 50 == 0:
                print(f"[TYPING] Progress: {char_index}/{len(code)} characters", flush=True)

        # Final code frame (hold for 3 seconds)
        comprehension_hold_seconds = 3.0
        hold_frames = int(fps * comprehension_hold_seconds)

        # Render with and without cursor for blinking
        final_with_cursor = await self._render_frame(
            text=current_text,
            language=language,
            title=title,
            show_cursor=True,
            background_color=background_color,
            text_color=text_color,
            accent_color=accent_color,
            pygments_style=pygments_style
        )
        final_cursor_path = output_dir / f"frame_{frame_index:06d}.png"
        final_with_cursor.save(final_cursor_path, "PNG")
        del final_with_cursor
        frame_index += 1

        final_no_cursor = await self._render_frame(
            text=current_text,
            language=language,
            title=title,
            show_cursor=False,
            background_color=background_color,
            text_color=text_color,
            accent_color=accent_color,
            pygments_style=pygments_style
        )
        final_no_cursor_path = output_dir / f"final_nc.png"
        final_no_cursor.save(final_no_cursor_path, "PNG")
        del final_no_cursor

        for j in range(1, hold_frames):
            dest_path = output_dir / f"frame_{frame_index:06d}.png"
            if j % 15 < 8:
                shutil.copy(final_cursor_path, dest_path)
            else:
                shutil.copy(final_no_cursor_path, dest_path)
            frame_index += 1

        final_no_cursor_path.unlink()

        # Add execution output if provided
        if execution_output:
            output_frames = int(fps * 3)
            output_frame = await self._render_frame_with_output(
                code=current_text,
                output=execution_output,
                language=language,
                title=title,
                background_color=background_color,
                text_color=text_color,
                accent_color=accent_color,
                pygments_style=pygments_style
            )
            output_frame_path = output_dir / f"frame_{frame_index:06d}.png"
            output_frame.save(output_frame_path, "PNG")
            del output_frame
            frame_index += 1

            for j in range(1, output_frames):
                dest_path = output_dir / f"frame_{frame_index:06d}.png"
                shutil.copy(output_frame_path, dest_path)
                frame_index += 1

        gc.collect()
        return frame_index

    async def _frames_dir_to_video(
        self,
        frames_dir: Path,
        output_path: str,
        fps: int
    ) -> str:
        """Convert frames directory to video using FFmpeg"""
        frames_pattern = str(frames_dir / "frame_%06d.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frames_pattern,
            "-c:v", "libx264",
            "-profile:v", "high",
            "-level", "4.0",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]

        print(f"[TYPING] Creating video with FFmpeg...", flush=True)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"[TYPING] FFmpeg error: {stderr.decode()}", flush=True)
            raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")

        return output_path

    def _get_token_color(self, token_type, style) -> str:
        """Get color for a pygments token type"""
        # Get color from style
        style_dict = style.style_for_token(token_type)
        color = style_dict.get('color')
        if color:
            return f"#{color}"
        # Default to white
        return "#ffffff"

    async def _render_frame(
        self,
        text: str,
        language: str,
        title: Optional[str],
        show_cursor: bool,
        background_color: str,
        text_color: str,
        accent_color: str,
        pygments_style: str
    ) -> Image.Image:
        """Render a single frame with manual syntax highlighting"""

        # Create base image
        img = Image.new("RGB", (self.WIDTH, self.HEIGHT), background_color)
        draw = ImageDraw.Draw(img)

        y_offset = self.MARGIN_Y

        # Draw title if provided
        if title:
            title_font = self._load_font("bold", 42)
            draw.text(
                (self.MARGIN_X, y_offset),
                title,
                font=title_font,
                fill=accent_color
            )
            y_offset += 60

            # Separator line
            draw.line(
                [(self.MARGIN_X, y_offset), (self.WIDTH - self.MARGIN_X, y_offset)],
                fill=accent_color,
                width=2
            )
            y_offset += 30

        # Add cursor to text if needed
        display_text = text + ("█" if show_cursor else " ")

        # Draw code background
        code_bg = "#181825"
        padding = 20
        code_area_width = self.WIDTH - 2 * self.MARGIN_X
        code_area_height = self.HEIGHT - y_offset - self.MARGIN_Y

        draw.rectangle(
            [
                (self.MARGIN_X - padding, y_offset),
                (self.WIDTH - self.MARGIN_X + padding, y_offset + code_area_height)
            ],
            fill=code_bg,
            outline=accent_color,
            width=2
        )

        # Render syntax highlighted code manually
        if display_text.strip():
            try:
                lexer = get_lexer_by_name(language, stripall=True)
            except:
                lexer = TextLexer()

            try:
                style_obj = get_style_by_name(pygments_style)
            except:
                style_obj = get_style_by_name("monokai")

            # Load monospace font for code
            code_font = self._load_font("mono", 24)

            # Calculate line height using textbbox (new Pillow API)
            bbox = draw.textbbox((0, 0), "Ay", font=code_font)
            line_height = bbox[3] - bbox[1] + 8  # Add some padding

            # Calculate character width for cursor positioning
            char_bbox = draw.textbbox((0, 0), "M", font=code_font)
            char_width = char_bbox[2] - char_bbox[0]

            # Starting position for code
            code_x = self.MARGIN_X + 60  # Leave space for line numbers
            code_y = y_offset + padding

            # Draw line numbers background
            line_num_bg = "#11111b"
            draw.rectangle(
                [
                    (self.MARGIN_X - padding, y_offset),
                    (self.MARGIN_X + 50, y_offset + code_area_height)
                ],
                fill=line_num_bg
            )

            # Tokenize and render
            current_x = code_x
            current_y = code_y
            line_num = 1
            line_num_font = self._load_font("mono", 20)
            line_num_color = "#6c7086"

            # Draw first line number
            draw.text(
                (self.MARGIN_X, current_y + 2),
                str(line_num).rjust(3),
                font=line_num_font,
                fill=line_num_color
            )

            # Get all tokens
            tokens = list(lex(display_text, lexer))

            for token_type, token_value in tokens:
                color = self._get_token_color(token_type, style_obj)

                for char in token_value:
                    if char == '\n':
                        # Move to next line
                        current_x = code_x
                        current_y += line_height
                        line_num += 1

                        # Check if we're still in visible area
                        if current_y < y_offset + code_area_height - line_height:
                            # Draw line number
                            draw.text(
                                (self.MARGIN_X, current_y + 2),
                                str(line_num).rjust(3),
                                font=line_num_font,
                                fill=line_num_color
                            )
                    else:
                        # Check if we're still in visible area
                        if current_y < y_offset + code_area_height - line_height:
                            if current_x < self.WIDTH - self.MARGIN_X - 20:
                                # Draw character with syntax highlighting color
                                draw.text(
                                    (current_x, current_y),
                                    char,
                                    font=code_font,
                                    fill=color
                                )
                        # Move cursor right
                        current_x += char_width

        return img

    async def _render_frame_with_output(
        self,
        code: str,
        output: str,
        language: str,
        title: Optional[str],
        background_color: str,
        text_color: str,
        accent_color: str,
        pygments_style: str
    ) -> Image.Image:
        """Render a frame showing code and its execution output"""

        # Create base image
        img = Image.new("RGB", (self.WIDTH, self.HEIGHT), background_color)
        draw = ImageDraw.Draw(img)

        y_offset = self.MARGIN_Y

        # Draw title if provided
        if title:
            title_font = self._load_font("bold", 36)
            draw.text(
                (self.MARGIN_X, y_offset),
                title,
                font=title_font,
                fill=accent_color
            )
            y_offset += 50

        # Split the vertical space: code on left (60%), output on right (40%)
        code_width = int((self.WIDTH - 3 * self.MARGIN_X) * 0.6)
        output_width = int((self.WIDTH - 3 * self.MARGIN_X) * 0.4)
        code_height = self.HEIGHT - y_offset - self.MARGIN_Y

        # Draw code section
        code_bg = "#181825"
        padding = 15
        draw.rectangle(
            [
                (self.MARGIN_X - padding, y_offset),
                (self.MARGIN_X + code_width + padding, y_offset + code_height)
            ],
            fill=code_bg,
            outline=accent_color,
            width=2
        )

        # Draw "Code" label
        label_font = self._load_font("bold", 20)
        draw.text(
            (self.MARGIN_X, y_offset + 10),
            "Code",
            font=label_font,
            fill=accent_color
        )

        # Draw code text (simplified, no syntax highlighting for output frame)
        code_font = self._load_font("mono", 18)
        bbox = draw.textbbox((0, 0), "M", font=code_font)
        line_height = bbox[3] - bbox[1] + 4

        code_lines = code.split('\n')
        code_y = y_offset + 45
        for i, line in enumerate(code_lines[:20]):  # Limit lines
            if code_y + line_height > y_offset + code_height - 20:
                break
            draw.text(
                (self.MARGIN_X + 10, code_y),
                line[:60],  # Truncate long lines
                font=code_font,
                fill=text_color
            )
            code_y += line_height

        # Draw output section
        output_x = self.MARGIN_X + code_width + self.MARGIN_X
        output_bg = "#1a1a2e"
        draw.rectangle(
            [
                (output_x - padding, y_offset),
                (output_x + output_width + padding, y_offset + code_height)
            ],
            fill=output_bg,
            outline="#50fa7b",  # Green for output
            width=2
        )

        # Draw "Output" label
        draw.text(
            (output_x, y_offset + 10),
            "Output",
            font=label_font,
            fill="#50fa7b"
        )

        # Draw chevron/arrow between code and output
        arrow_x = self.MARGIN_X + code_width + self.MARGIN_X // 2
        arrow_y = y_offset + code_height // 2
        arrow_font = self._load_font("bold", 40)
        draw.text(
            (arrow_x - 15, arrow_y - 20),
            "→",
            font=arrow_font,
            fill="#50fa7b"
        )

        # Draw output text
        output_font = self._load_font("mono", 20)
        output_lines = output.split('\n')
        output_y = y_offset + 45
        for i, line in enumerate(output_lines[:15]):  # Limit lines
            if output_y + line_height > y_offset + code_height - 20:
                break
            draw.text(
                (output_x + 10, output_y),
                line[:30],  # Truncate long lines
                font=output_font,
                fill="#50fa7b"  # Green text for output
            )
            output_y += line_height + 2

        return img

    async def _frames_to_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int
    ) -> str:
        """Convert frames to video using FFmpeg"""

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save frames as images
            print(f"[TYPING] Saving {len(frames)} frames...", flush=True)
            for i, frame in enumerate(frames):
                frame_path = temp_path / f"frame_{i:06d}.png"
                frame.save(frame_path, "PNG")

            # Use FFmpeg to create video
            frames_pattern = str(temp_path / "frame_%06d.png")

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frames_pattern,
                "-c:v", "libx264",
                "-profile:v", "high",
                "-level", "4.0",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ]

            print(f"[TYPING] Creating video with FFmpeg...", flush=True)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                print(f"[TYPING] FFmpeg error: {stderr.decode()}", flush=True)
                raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")

            return output_path

    async def create_typing_with_output(
        self,
        code: str,
        output_result: str,
        language: str,
        output_path: str,
        title: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create typing animation followed by showing execution output.

        Args:
            code: The code to animate typing
            output_result: The output to show after typing completes
            language: Programming language
            output_path: Path for output video
            title: Optional title
            **kwargs: Additional animation parameters

        Returns:
            Path to the generated video
        """
        # First create the typing animation
        typing_video = await self.create_typing_animation(
            code=code,
            language=language,
            output_path=output_path.replace('.mp4', '_typing.mp4'),
            title=title,
            **kwargs
        )

        # TODO: Add output display animation after typing
        # For now, just return the typing video
        return typing_video
