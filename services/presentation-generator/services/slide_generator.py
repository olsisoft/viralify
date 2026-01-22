"""
Slide Generator Service

Generates slide images using Pygments for syntax highlighting and PIL for rendering.
"""
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import ImageFormatter
from pygments.styles import get_style_by_name
import httpx

from models.presentation_models import (
    Slide,
    SlideType,
    CodeBlock,
    PresentationStyle,
)
from services.diagram_generator import DiagramGeneratorService, DiagramType


class SlideGeneratorService:
    """Service for generating slide images"""

    # Slide dimensions (1080p)
    WIDTH = 1920
    HEIGHT = 1080

    # Font sizes
    TITLE_FONT_SIZE = 72
    SUBTITLE_FONT_SIZE = 48
    CONTENT_FONT_SIZE = 36
    BULLET_FONT_SIZE = 32
    CODE_FONT_SIZE = 24

    # Margins
    MARGIN_X = 100
    MARGIN_Y = 80

    def __init__(self):
        self.config_path = Path(__file__).parent.parent / "config" / "languages.json"
        self.styles_config = self._load_config()

        # Load fonts (fallback to default if custom fonts not available)
        self.title_font = self._load_font("bold", self.TITLE_FONT_SIZE)
        self.subtitle_font = self._load_font("regular", self.SUBTITLE_FONT_SIZE)
        self.content_font = self._load_font("regular", self.CONTENT_FONT_SIZE)
        self.code_font = self._load_font("mono", self.CODE_FONT_SIZE)

        # Cloudinary config
        self.cloudinary_url = os.getenv("CLOUDINARY_URL")

        # Diagram generator
        self.diagram_generator = DiagramGeneratorService()

    def _load_config(self) -> dict:
        """Load language and style configuration"""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {"languages": {}, "styles": {}}

    def _load_font(self, style: str, size: int) -> ImageFont.FreeTypeFont:
        """Load a font, with fallback to default"""
        fonts_dir = Path(__file__).parent.parent / "fonts"

        font_files = {
            "bold": "DejaVuSans-Bold.ttf",
            "regular": "DejaVuSans.ttf",
            "mono": "DejaVuSansMono.ttf"
        }

        # Try to load custom font
        font_path = fonts_dir / font_files.get(style, "DejaVuSans.ttf")
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception:
                pass

        # Try system fonts
        system_fonts = {
            "bold": ["Arial Bold", "Helvetica Bold", "DejaVuSans-Bold"],
            "regular": ["Arial", "Helvetica", "DejaVuSans"],
            "mono": ["Consolas", "Monaco", "DejaVuSansMono", "Courier New"]
        }

        for font_name in system_fonts.get(style, ["Arial"]):
            try:
                return ImageFont.truetype(font_name, size)
            except Exception:
                continue

        # Final fallback
        return ImageFont.load_default()

    def get_style_colors(self, style: PresentationStyle) -> Dict[str, str]:
        """Get color scheme for a style"""
        style_config = self.styles_config.get("styles", {}).get(style.value, {})

        return {
            "background": style_config.get("background_color", "#1e1e2e"),
            "text": style_config.get("text_color", "#cdd6f4"),
            "accent": style_config.get("accent_color", "#89b4fa"),
            "code_bg": style_config.get("code_background", "#181825"),
            "line_number": style_config.get("line_number_color", "#6c7086"),
            "pygments_style": style_config.get("pygments_style", "monokai")
        }

    async def generate_slide_image(
        self,
        slide: Slide,
        style: PresentationStyle
    ) -> bytes:
        """
        Generate an image for a single slide.

        Args:
            slide: The slide to render
            style: Visual style/theme

        Returns:
            PNG image as bytes
        """
        colors = self.get_style_colors(style)

        # Create base image
        if "linear-gradient" in colors["background"]:
            # Handle gradient backgrounds
            img = self._create_gradient_background(colors["background"])
        else:
            bg_color = colors["background"]
            img = Image.new("RGB", (self.WIDTH, self.HEIGHT), bg_color)

        draw = ImageDraw.Draw(img)

        # Render based on slide type
        if slide.type == SlideType.TITLE:
            img = self._render_title_slide(img, draw, slide, colors)
        elif slide.type == SlideType.CODE or slide.type == SlideType.CODE_DEMO:
            img = self._render_code_slide(img, draw, slide, colors)
        elif slide.type == SlideType.DIAGRAM:
            img = await self._render_diagram_slide(img, draw, slide, colors, style)
        elif slide.type == SlideType.CONCLUSION:
            img = self._render_conclusion_slide(img, draw, slide, colors)
        else:
            img = self._render_content_slide(img, draw, slide, colors)

        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", quality=95)
        buffer.seek(0)

        return buffer.getvalue()

    def _create_gradient_background(self, gradient_str: str) -> Image.Image:
        """Create a gradient background image"""
        # Parse gradient (simplified - assumes linear-gradient format)
        # Default purple gradient
        color1 = "#667eea"
        color2 = "#764ba2"

        img = Image.new("RGB", (self.WIDTH, self.HEIGHT))

        # Convert hex to RGB
        r1, g1, b1 = self._hex_to_rgb(color1)
        r2, g2, b2 = self._hex_to_rgb(color2)

        # Create gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            for x in range(self.WIDTH):
                img.putpixel((x, y), (r, g, b))

        return img

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _render_title_slide(
        self,
        img: Image.Image,
        draw: ImageDraw.Draw,
        slide: Slide,
        colors: Dict[str, str]
    ) -> Image.Image:
        """Render a title slide"""
        text_color = colors["text"]
        accent_color = colors["accent"]

        max_width = self.WIDTH - 2 * self.MARGIN_X
        line_height_title = 80
        line_height_subtitle = 55

        # Wrap and center the title
        title_lines = []
        if slide.title:
            title_lines = self._wrap_text(slide.title, self.title_font, max_width)
            # Limit to 3 lines max
            title_lines = title_lines[:3]

        # Wrap and center the subtitle
        subtitle_lines = []
        if slide.subtitle:
            subtitle_lines = self._wrap_text(slide.subtitle, self.subtitle_font, max_width)
            # Limit to 2 lines max
            subtitle_lines = subtitle_lines[:2]

        # Calculate total height of all text
        total_height = (len(title_lines) * line_height_title +
                       len(subtitle_lines) * line_height_subtitle +
                       (40 if title_lines and subtitle_lines else 0))  # Gap between title and subtitle

        # Start y position to center everything vertically
        y_offset = (self.HEIGHT - total_height) // 2

        # Draw title lines (centered)
        for line in title_lines:
            line_bbox = draw.textbbox((0, 0), line, font=self.title_font)
            line_width = line_bbox[2] - line_bbox[0]
            line_x = (self.WIDTH - line_width) // 2

            draw.text(
                (line_x, y_offset),
                line,
                font=self.title_font,
                fill=text_color
            )
            y_offset += line_height_title

        # Add gap between title and subtitle
        if title_lines and subtitle_lines:
            y_offset += 40

        # Draw subtitle lines (centered)
        for line in subtitle_lines:
            line_bbox = draw.textbbox((0, 0), line, font=self.subtitle_font)
            line_width = line_bbox[2] - line_bbox[0]
            line_x = (self.WIDTH - line_width) // 2

            draw.text(
                (line_x, y_offset),
                line,
                font=self.subtitle_font,
                fill=accent_color
            )
            y_offset += line_height_subtitle

        return img

    def _render_content_slide(
        self,
        img: Image.Image,
        draw: ImageDraw.Draw,
        slide: Slide,
        colors: Dict[str, str]
    ) -> Image.Image:
        """Render a content slide with bullet points"""
        text_color = colors["text"]
        accent_color = colors["accent"]

        y_offset = self.MARGIN_Y
        max_width = self.WIDTH - 2 * self.MARGIN_X
        bottom_margin = 60
        max_y = self.HEIGHT - bottom_margin

        # Title
        if slide.title:
            title_wrapped = self._wrap_text(slide.title, self.subtitle_font, max_width)
            for line in title_wrapped:
                if y_offset >= max_y:
                    break
                draw.text(
                    (self.MARGIN_X, y_offset),
                    line,
                    font=self.subtitle_font,
                    fill=accent_color
                )
                y_offset += 60
            y_offset += 20

        # Draw accent line under title
        if y_offset < max_y:
            draw.line(
                [(self.MARGIN_X, y_offset), (self.WIDTH - self.MARGIN_X, y_offset)],
                fill=accent_color,
                width=3
            )
            y_offset += 40

        # Content text
        if slide.content and y_offset < max_y:
            wrapped = self._wrap_text(slide.content, self.content_font, max_width)
            for line in wrapped:
                if y_offset >= max_y:
                    break
                draw.text(
                    (self.MARGIN_X, y_offset),
                    line,
                    font=self.content_font,
                    fill=text_color
                )
                y_offset += 45
            y_offset += 15

        # Bullet points
        for point in slide.bullet_points:
            if y_offset >= max_y:
                break
            bullet = "  •  "
            point_text = bullet + point
            # Wrap long bullet points
            wrapped_point = self._wrap_text(point_text, self.content_font, max_width - 40)
            for i, line in enumerate(wrapped_point):
                if y_offset >= max_y:
                    break
                # Indent continuation lines
                x_pos = self.MARGIN_X if i == 0 else self.MARGIN_X + 60
                draw.text(
                    (x_pos, y_offset),
                    line if i == 0 else line.lstrip(),
                    font=self.content_font,
                    fill=text_color
                )
                y_offset += 45
            y_offset += 15  # Extra space between bullet points

        return img

    def _render_code_slide(
        self,
        img: Image.Image,
        draw: ImageDraw.Draw,
        slide: Slide,
        colors: Dict[str, str]
    ) -> Image.Image:
        """Render a code slide with syntax highlighting"""
        text_color = colors["text"]
        accent_color = colors["accent"]
        code_bg = colors["code_bg"]
        pygments_style = colors["pygments_style"]

        y_offset = self.MARGIN_Y
        max_width = self.WIDTH - 2 * self.MARGIN_X

        # Title at the top - use smaller font for code slides to save space
        title_font = self._load_font("bold", 42)
        if slide.title:
            # Wrap title if too long
            title_wrapped = self._wrap_text(slide.title, title_font, max_width)
            # Limit to 2 lines max for code slides
            title_wrapped = title_wrapped[:2]
            for line in title_wrapped:
                draw.text(
                    (self.MARGIN_X, y_offset),
                    line,
                    font=title_font,
                    fill=accent_color
                )
                y_offset += 50
            y_offset += 15  # Spacing after title

        # Draw separator line
        draw.line(
            [(self.MARGIN_X, y_offset), (self.WIDTH - self.MARGIN_X, y_offset)],
            fill=accent_color,
            width=2
        )
        y_offset += 25

        # Calculate available space for code
        bottom_margin = 50
        max_code_height = self.HEIGHT - y_offset - bottom_margin

        # Render each code block
        for code_block in slide.code_blocks:
            # Filename header - single line, truncate if needed
            if code_block.filename:
                filename_font = self._load_font("mono", 20)
                filename_text = f"  {code_block.filename}"
                # Truncate filename if too long
                while len(filename_text) > 50:
                    filename_text = filename_text[:47] + "..."
                draw.text(
                    (self.MARGIN_X, y_offset),
                    filename_text,
                    font=filename_font,
                    fill=accent_color
                )
                y_offset += 45  # Increased spacing after filename

            # Generate syntax highlighted code image
            code_img = self._highlight_code(
                code_block.code,
                code_block.language,
                pygments_style,
                code_block.highlight_lines
            )

            # Calculate position and sizing
            code_x = self.MARGIN_X
            code_width = max_width

            # Convert to RGBA for transparency
            if code_img.mode != "RGBA":
                code_img = code_img.convert("RGBA")

            # Calculate available height for this code block
            available_height = max_code_height - 30

            # Scale code image to fit available space
            scale_factor = 1.0

            # Scale down if width exceeds available width
            if code_img.width > code_width:
                scale_factor = min(scale_factor, code_width / code_img.width)

            # Scale down if height exceeds available height
            if code_img.height > available_height and available_height > 0:
                scale_factor = min(scale_factor, available_height / code_img.height)

            # Apply scaling if needed (minimum scale 0.5 to keep readable)
            if scale_factor < 1.0:
                scale_factor = max(scale_factor, 0.5)
                new_width = int(code_img.width * scale_factor)
                new_height = int(code_img.height * scale_factor)
                code_img = code_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Draw code background - starts AT y_offset, no negative offset to avoid text overlap
            padding_x = 15
            padding_y = 10

            # Background starts at current position (no overlap with text above)
            bg_top = y_offset
            bg_bottom = y_offset + code_img.height + padding_y * 2

            draw.rectangle(
                [
                    (code_x - padding_x, bg_top),
                    (code_x + code_img.width + padding_x, bg_bottom)
                ],
                fill=code_bg,
                outline=accent_color,
                width=2
            )

            # Paste code image with small offset from background top
            code_y = y_offset + padding_y
            img.paste(code_img, (code_x, code_y), code_img)

            # Move y_offset past the entire code block (background + spacing)
            y_offset = bg_bottom + 25

            # Update remaining available height for next code block
            max_code_height = self.HEIGHT - y_offset - bottom_margin

        return img

    def _render_conclusion_slide(
        self,
        img: Image.Image,
        draw: ImageDraw.Draw,
        slide: Slide,
        colors: Dict[str, str]
    ) -> Image.Image:
        """Render a conclusion slide"""
        text_color = colors["text"]
        accent_color = colors["accent"]

        y_offset = self.MARGIN_Y + 50
        max_width = self.WIDTH - 2 * self.MARGIN_X
        max_y = self.HEIGHT - 60  # Bottom margin

        # Title - wrap if too long
        if slide.title:
            title_wrapped = self._wrap_text(slide.title, self.subtitle_font, max_width)
            for line in title_wrapped[:2]:  # Max 2 lines for title
                if y_offset >= max_y:
                    break
                line_bbox = draw.textbbox((0, 0), line, font=self.subtitle_font)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = (self.WIDTH - line_width) // 2

                draw.text(
                    (line_x, y_offset),
                    line,
                    font=self.subtitle_font,
                    fill=accent_color
                )
                y_offset += 65
            y_offset += 35  # Gap after title

        # Summary points from bullet_points - wrap each point
        if slide.bullet_points:
            for point in slide.bullet_points:
                if y_offset >= max_y:
                    break
                bullet = "  ✓  "
                point_text = bullet + point
                # Wrap the bullet point if too long
                wrapped_lines = self._wrap_text(point_text, self.content_font, max_width - 40)
                for i, line in enumerate(wrapped_lines):
                    if y_offset >= max_y:
                        break
                    line_bbox = draw.textbbox((0, 0), line, font=self.content_font)
                    line_width = line_bbox[2] - line_bbox[0]
                    line_x = (self.WIDTH - line_width) // 2

                    draw.text(
                        (line_x, y_offset),
                        line,
                        font=self.content_font,
                        fill=text_color
                    )
                    y_offset += 50
                y_offset += 20  # Extra space between bullet points
        # Fallback: use content field if no bullet_points
        elif slide.content:
            # Split content by newlines or periods to create points
            content_lines = slide.content.replace(". ", ".\n").split("\n")
            for line in content_lines:
                line = line.strip()
                if not line:
                    continue
                bullet = "  ✓  "
                point_text = bullet + line
                # Wrap long lines
                wrapped = self._wrap_text(point_text, self.content_font, self.WIDTH - 2 * self.MARGIN_X)
                for wrapped_line in wrapped:
                    point_bbox = draw.textbbox((0, 0), wrapped_line, font=self.content_font)
                    point_width = point_bbox[2] - point_bbox[0]
                    point_x = (self.WIDTH - point_width) // 2

                    draw.text(
                        (point_x, y_offset),
                        wrapped_line,
                        font=self.content_font,
                        fill=text_color
                    )
                    y_offset += 50
                y_offset += 20
        # Last fallback: show a default message
        else:
            default_text = "Merci pour votre attention!"
            text_bbox = draw.textbbox((0, 0), default_text, font=self.content_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (self.WIDTH - text_width) // 2

            draw.text(
                (text_x, y_offset),
                default_text,
                font=self.content_font,
                fill=text_color
            )

        return img

    async def _render_diagram_slide(
        self,
        img: Image.Image,
        draw: ImageDraw.Draw,
        slide: Slide,
        colors: Dict[str, str],
        style: PresentationStyle
    ) -> Image.Image:
        """Render a diagram slide using the DiagramGeneratorService"""
        try:
            # Determine diagram type from slide metadata or content
            diagram_type = DiagramType.FLOWCHART  # default

            # Check if diagram_type is specified in slide metadata
            if hasattr(slide, 'diagram_type') and slide.diagram_type:
                try:
                    diagram_type = DiagramType(slide.diagram_type)
                except ValueError:
                    pass
            elif slide.content:
                # Try to infer diagram type from content
                content_lower = slide.content.lower()
                if 'architecture' in content_lower or 'system' in content_lower:
                    diagram_type = DiagramType.ARCHITECTURE
                elif 'process' in content_lower or 'step' in content_lower:
                    diagram_type = DiagramType.PROCESS
                elif 'compare' in content_lower or 'vs' in content_lower:
                    diagram_type = DiagramType.COMPARISON
                elif 'hierarchy' in content_lower or 'tree' in content_lower:
                    diagram_type = DiagramType.HIERARCHY

            # Map presentation style to diagram theme
            theme_map = {
                PresentationStyle.DARK: "tech",
                PresentationStyle.LIGHT: "light",
                PresentationStyle.GRADIENT: "gradient",
                PresentationStyle.OCEAN: "tech"
            }
            theme = theme_map.get(style, "tech")

            # Generate the diagram
            diagram_path = await self.diagram_generator.generate_diagram(
                diagram_type=diagram_type,
                description=slide.content or slide.title or "Diagram",
                title=slide.title or "",
                job_id=getattr(slide, 'job_id', 'unknown'),
                slide_index=getattr(slide, 'index', 0),
                theme=theme,
                width=self.WIDTH,
                height=self.HEIGHT
            )

            if diagram_path and os.path.exists(diagram_path):
                # Load the generated diagram and return it
                diagram_img = Image.open(diagram_path)
                return diagram_img.convert("RGB")
            else:
                # Fallback: render as content slide with a note
                print(f"[SLIDE] Diagram generation failed, falling back to content slide", flush=True)
                return self._render_content_slide(img, draw, slide, colors)

        except Exception as e:
            print(f"[SLIDE] Error generating diagram: {e}", flush=True)
            # Fallback to content slide
            return self._render_content_slide(img, draw, slide, colors)

    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess code to convert literal escape sequences to actual characters.
        This handles cases where code contains '\\n' as two characters instead of newline.
        """
        if not code:
            return code

        # Map of literal escape sequences to their actual characters
        escape_map = {
            '\\n': '\n',      # Literal backslash-n to newline
            '\\t': '\t',      # Literal backslash-t to tab
            '\\r': '\r',      # Literal backslash-r to carriage return
            '\\\\': '\\',     # Double backslash to single backslash
        }

        processed = code

        # First, handle double-escaped sequences (\\\\n -> \n for display)
        processed = processed.replace('\\\\n', '\n')
        processed = processed.replace('\\\\t', '\t')
        processed = processed.replace('\\\\r', '\r')

        # Then handle single-escaped sequences
        for literal, actual in escape_map.items():
            if literal != '\\\\':  # Skip double backslash, already handled
                processed = processed.replace(literal, actual)

        # Clean up any remaining artifacts
        # Remove trailing whitespace on each line but preserve newlines
        lines = processed.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        processed = '\n'.join(cleaned_lines)

        return processed

    def _normalize_language(self, language: str) -> str:
        """
        Normalize language name to a valid Pygments lexer name.
        Handles common aliases and variations.
        """
        if not language:
            return "text"

        # Normalize to lowercase and strip whitespace
        lang = language.lower().strip()

        # Common language aliases mapping to Pygments lexer names
        language_map = {
            # JavaScript variants
            "js": "javascript",
            "jsx": "jsx",
            "ts": "typescript",
            "tsx": "tsx",
            "node": "javascript",
            "nodejs": "javascript",
            "es6": "javascript",
            "ecmascript": "javascript",

            # Python variants
            "py": "python",
            "py3": "python3",
            "python2": "python",
            "python3": "python3",
            "ipython": "python",

            # Shell variants
            "sh": "bash",
            "shell": "bash",
            "zsh": "bash",
            "terminal": "bash",
            "console": "console",
            "cmd": "batch",
            "bat": "batch",
            "powershell": "powershell",
            "ps1": "powershell",

            # Web technologies
            "htm": "html",
            "xhtml": "html",
            "vue": "vue",
            "svelte": "html",
            "scss": "scss",
            "sass": "sass",
            "less": "less",
            "styl": "stylus",

            # Data formats
            "yml": "yaml",
            "jsonc": "json",
            "json5": "json",

            # C-family
            "c++": "cpp",
            "cxx": "cpp",
            "cc": "cpp",
            "h": "c",
            "hpp": "cpp",
            "objective-c": "objectivec",
            "objc": "objectivec",
            "c#": "csharp",
            "cs": "csharp",

            # JVM languages
            "kt": "kotlin",
            "kts": "kotlin",
            "groovy": "groovy",
            "scala": "scala",

            # Database
            "psql": "postgresql",
            "pgsql": "postgresql",
            "mysql": "mysql",
            "plsql": "plpgsql",
            "nosql": "javascript",  # Usually JSON-like

            # Markup and config
            "md": "markdown",
            "rst": "rst",
            "tex": "latex",
            "dockerfile": "docker",
            "nginx": "nginx",
            "apache": "apacheconf",
            "ini": "ini",
            "cfg": "ini",
            "conf": "ini",
            "toml": "toml",
            "env": "bash",
            ".env": "bash",

            # Other languages
            "rb": "ruby",
            "rs": "rust",
            "go": "go",
            "golang": "go",
            "swift": "swift",
            "r": "r",
            "rlang": "r",
            "matlab": "matlab",
            "m": "matlab",
            "pl": "perl",
            "ex": "elixir",
            "exs": "elixir",
            "erl": "erlang",
            "hs": "haskell",
            "clj": "clojure",
            "lisp": "common-lisp",
            "scm": "scheme",
            "rkt": "racket",
            "f#": "fsharp",
            "fs": "fsharp",
            "vb": "vbnet",
            "vba": "vbnet",
            "asm": "nasm",
            "assembly": "nasm",

            # DevOps
            "tf": "terraform",
            "hcl": "terraform",
            "ansible": "yaml",
            "k8s": "yaml",
            "kubernetes": "yaml",
            "helm": "yaml",

            # Misc
            "graphql": "graphql",
            "gql": "graphql",
            "proto": "protobuf",
            "protobuf": "protobuf",
            "sol": "solidity",
            "solidity": "solidity",
            "txt": "text",
            "plain": "text",
            "plaintext": "text",
            "none": "text",
        }

        # Return mapped language or original if it's already valid
        normalized = language_map.get(lang, lang)

        # Verify the lexer exists, fallback to text if not
        try:
            get_lexer_by_name(normalized, stripall=True)
            return normalized
        except Exception:
            print(f"[SLIDE] Unknown language '{language}', falling back to 'text'", flush=True)
            return "text"

    def _highlight_code(
        self,
        code: str,
        language: str,
        style: str,
        highlight_lines: List[int] = None
    ) -> Image.Image:
        """Generate syntax highlighted code image using Pygments"""
        # Preprocess code to convert literal escape sequences to actual characters
        processed_code = self._preprocess_code(code)

        # Normalize language name to valid Pygments lexer
        normalized_language = self._normalize_language(language)

        try:
            lexer = get_lexer_by_name(normalized_language, stripall=True)
        except Exception:
            lexer = TextLexer()

        try:
            style_obj = get_style_by_name(style)
        except Exception:
            style_obj = get_style_by_name("monokai")

        # Configure formatter - use Consolas on Windows, DejaVu Sans Mono on Linux
        import platform
        mono_font = "Consolas" if platform.system() == "Windows" else "DejaVu Sans Mono"
        formatter = ImageFormatter(
            style=style_obj,
            font_name=mono_font,
            font_size=self.CODE_FONT_SIZE,
            line_numbers=True,
            line_number_bg="#181825",
            line_number_fg="#6c7086",
            hl_lines=highlight_lines or []
        )

        # Generate image
        result = highlight(processed_code, lexer, formatter)

        # Convert bytes to PIL Image
        return Image.open(io.BytesIO(result))

    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int
    ) -> List[str]:
        """Wrap text to fit within max_width with character-level breaking for long words"""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            # First, check if the word itself is too long for the line
            word_bbox = font.getbbox(word)
            word_width = word_bbox[2] - word_bbox[0]

            if word_width > max_width:
                # Word is too long - break it into chunks
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = []

                # Break the long word into character chunks that fit
                broken_word = self._break_long_word(word, font, max_width)
                for chunk in broken_word[:-1]:
                    lines.append(chunk)
                # Last chunk becomes the start of a new line
                if broken_word:
                    current_line = [broken_word[-1]]
            else:
                test_line = " ".join(current_line + [word])
                bbox = font.getbbox(test_line)
                if bbox[2] - bbox[0] <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def _break_long_word(
        self,
        word: str,
        font: ImageFont.FreeTypeFont,
        max_width: int
    ) -> List[str]:
        """Break a long word into chunks that fit within max_width"""
        chunks = []
        current_chunk = ""

        for char in word:
            test_chunk = current_chunk + char
            bbox = font.getbbox(test_chunk)
            chunk_width = bbox[2] - bbox[0]

            if chunk_width <= max_width - 20:  # Leave room for hyphen
                current_chunk = test_chunk
            else:
                # Add hyphen and start new chunk
                if current_chunk:
                    chunks.append(current_chunk + "-")
                current_chunk = char

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [word]

    async def upload_to_cloudinary(self, image_bytes: bytes, filename: str) -> str:
        """Upload image to Cloudinary and return URL"""
        # Get service URL for local fallback (Docker hostname for internal communication)
        service_url = os.getenv("SERVICE_URL", "http://presentation-generator:8006")

        if not self.cloudinary_url or self.cloudinary_url.strip() == "":
            # Save locally and return HTTP URL
            temp_dir = Path(tempfile.gettempdir()) / "presentations"
            temp_dir.mkdir(exist_ok=True)
            file_path = temp_dir / filename
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            # Return HTTP URL for inter-service communication
            return f"{service_url}/files/presentations/{filename}"

        # Parse Cloudinary URL
        try:
            import cloudinary
            import cloudinary.uploader

            cloudinary.config(cloudinary_url=self.cloudinary_url)

            result = cloudinary.uploader.upload(
                image_bytes,
                folder="presentations/slides",
                public_id=filename.replace(".png", ""),
                resource_type="image"
            )

            return result["secure_url"]
        except Exception as e:
            print(f"[CLOUDINARY] Error uploading: {e}", flush=True)
            # Fallback to local storage with HTTP URL
            temp_dir = Path(tempfile.gettempdir()) / "presentations"
            temp_dir.mkdir(exist_ok=True)
            file_path = temp_dir / filename
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            return f"{service_url}/files/presentations/{filename}"
