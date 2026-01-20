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

    # Animation settings - default is "natural" pace for explaining code
    DEFAULT_FPS = 30

    # Speed presets (chars per second) - designed for teaching/explaining context
    SPEED_PRESETS = {
        "slow": 2.0,       # Very deliberate, teaching beginners
        "natural": 4.0,    # Human explaining while typing (default)
        "moderate": 6.0,   # Confident developer
        "fast": 10.0       # Quick demo
    }

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
        pygments_style: str = "monokai"
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
            # Account for human-like pauses (roughly 50% overhead for natural feel)
            pause_overhead = 1.5
            typing_time = (target_duration - intro_time - outro_time - output_display_time) / pause_overhead
            if typing_time > 0 and len(code) > 0:
                chars_per_second = len(code) / typing_time
                # Clamp based on preset - never go faster than moderate even with time pressure
                max_speed = min(base_speed * 1.5, self.SPEED_PRESETS["moderate"])
                min_speed = base_speed * 0.5
                chars_per_second = max(min_speed, min(chars_per_second, max_speed))
            else:
                chars_per_second = base_speed
        else:
            chars_per_second = base_speed

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

    def _get_word_at_position(self, text: str, pos: int) -> str:
        """Extract the word that just completed at this position"""
        # Find the start of the word
        start = pos
        while start > 0 and text[start - 1].isalnum() or (start > 0 and text[start - 1] == '_'):
            start -= 1
        return text[start:pos]

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
