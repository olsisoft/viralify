"""
Slide Generator Service

Generates slide images using:
- PPTX Service (PptxGenJS) - Primary, when USE_PPTX_SERVICE=true
- PIL/Pygments - Fallback

The PPTX service provides better rendering with transitions, themes, and syntax highlighting.
"""
import io
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from services.viralify_diagram_service import (
    ViralifyDiagramService,
    ViralifyLayoutType,
    ViralifyExportFormat,
    get_viralify_diagram_service,
)

# PPTX Service integration (optional)
try:
    from services.pptx_client import (
        PptxClient,
        Slide as PptxSlide,
        SlideType as PptxSlideType,
        CodeBlock as PptxCodeBlock,
        BulletPoint as PptxBulletPoint,
        PresentationTheme,
        ThemeStyle,
        get_pptx_client,
    )
    PPTX_CLIENT_AVAILABLE = True
except ImportError:
    PPTX_CLIENT_AVAILABLE = False


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

        # Diagram generators
        # USE_VIRALIFY_DIAGRAMS=true uses viralify-diagrams with themed SVG + Graphviz layout
        # This provides professional diagrams that match the slide theme colors
        self.use_viralify_diagrams = os.getenv("USE_VIRALIFY_DIAGRAMS", "true").lower() == "true"
        self.diagram_generator = DiagramGeneratorService()
        self.viralify_diagram_service = get_viralify_diagram_service() if self.use_viralify_diagrams else None

        # PPTX Service configuration
        # USE_PPTX_SERVICE=true uses PptxGenJS microservice for better slide rendering
        # Features: transitions, professional themes, better syntax highlighting
        # Fallback: PIL/Pygments if service unavailable
        self.use_pptx_service = os.getenv("USE_PPTX_SERVICE", "false").lower() == "true"
        self.pptx_client: Optional[PptxClient] = None
        self._pptx_service_available: Optional[bool] = None  # Cache availability check

        if self.use_pptx_service and PPTX_CLIENT_AVAILABLE:
            try:
                self.pptx_client = get_pptx_client()
                print("[SLIDE_GEN] PPTX Service enabled - will use PptxGenJS for slide rendering", flush=True)
            except Exception as e:
                print(f"[SLIDE_GEN] Failed to initialize PPTX client: {e}", flush=True)
                self.pptx_client = None

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
        style: PresentationStyle,
        target_audience: str = "intermediate developers",
        target_career: Optional[str] = None,
        rag_context: Optional[str] = None,
        course_context: Optional[Dict[str, Any]] = None,
        rag_images: Optional[List[Dict[str, Any]]] = None,
        job_id: Optional[str] = None
    ) -> bytes:
        """
        Generate an image for a single slide.

        Uses PPTX Service (PptxGenJS) if enabled and available, otherwise falls back to PIL.

        Args:
            slide: The slide to render
            style: Visual style/theme
            target_audience: Target audience for diagram complexity adjustment
            target_career: Target career for diagram focus (e.g., "data_engineer", "cloud_architect")
            rag_context: RAG context from source documents (for diagram accuracy)
            course_context: Course context dict with topic, section, description, etc.
            rag_images: List of RAG image references extracted from documents
            job_id: Job ID for file organization

        Returns:
            PNG image as bytes
        """
        # Try PPTX Service first (if enabled and available)
        if self.use_pptx_service and self.pptx_client:
            try:
                pptx_result = await self._generate_with_pptx_service(
                    slide, style, job_id or "slide"
                )
                if pptx_result:
                    return pptx_result
            except Exception as e:
                print(f"[SLIDE_GEN] PPTX Service failed, falling back to PIL: {e}", flush=True)
                traceback.print_exc()

        # Fallback to PIL rendering
        return await self._generate_with_pil(
            slide, style, target_audience, target_career,
            rag_context, course_context, rag_images, job_id
        )

    async def _generate_with_pil(
        self,
        slide: Slide,
        style: PresentationStyle,
        target_audience: str = "intermediate developers",
        target_career: Optional[str] = None,
        rag_context: Optional[str] = None,
        course_context: Optional[Dict[str, Any]] = None,
        rag_images: Optional[List[Dict[str, Any]]] = None,
        job_id: Optional[str] = None
    ) -> bytes:
        """Generate slide using PIL/Pygments (fallback method)."""
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
            img = await self._render_diagram_slide(
                img, draw, slide, colors, style, target_audience, target_career,
                rag_context=rag_context, course_context=course_context,
                rag_images=rag_images, job_id=job_id
            )
        elif slide.type == SlideType.CONCLUSION:
            img = self._render_conclusion_slide(img, draw, slide, colors)
        else:
            img = self._render_content_slide(img, draw, slide, colors)

        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", quality=95)
        buffer.seek(0)

        return buffer.getvalue()

    async def _generate_with_pptx_service(
        self,
        slide: Slide,
        style: PresentationStyle,
        job_id: str
    ) -> Optional[bytes]:
        """
        Generate slide using PPTX Service (PptxGenJS).

        Returns PNG bytes if successful, None if failed.
        """
        if not self.pptx_client or not PPTX_CLIENT_AVAILABLE:
            return None

        # Check service availability (cache result)
        if self._pptx_service_available is None:
            self._pptx_service_available = await self.pptx_client.is_available()
            if not self._pptx_service_available:
                print("[SLIDE_GEN] PPTX Service not available, will use PIL", flush=True)
                return None

        if not self._pptx_service_available:
            return None

        # Convert internal Slide model to PPTX service format
        pptx_slide = self._convert_to_pptx_slide(slide)
        if not pptx_slide:
            return None

        # Map PresentationStyle to ThemeStyle
        theme_style = self._map_style_to_theme(style)
        theme = PresentationTheme(style=theme_style)

        # Generate preview (single slide as PNG)
        try:
            png_bytes = await self.pptx_client.generate_preview(
                slide=pptx_slide,
                theme=theme,
                width=self.WIDTH,
                height=self.HEIGHT
            )

            if png_bytes:
                print(f"[SLIDE_GEN] Generated slide via PPTX Service ({len(png_bytes)} bytes)", flush=True)
                return png_bytes

        except Exception as e:
            print(f"[SLIDE_GEN] PPTX Service preview failed: {e}", flush=True)

        return None

    def _convert_to_pptx_slide(self, slide: Slide) -> Optional["PptxSlide"]:
        """Convert internal Slide model to PPTX service Slide format."""
        if not PPTX_CLIENT_AVAILABLE:
            return None

        # Map SlideType to PptxSlideType
        type_mapping = {
            SlideType.TITLE: PptxSlideType.TITLE,
            SlideType.CONTENT: PptxSlideType.CONTENT,
            SlideType.CODE: PptxSlideType.CODE,
            SlideType.CODE_DEMO: PptxSlideType.CODE_DEMO,
            SlideType.DIAGRAM: PptxSlideType.DIAGRAM,
            SlideType.COMPARISON: PptxSlideType.COMPARISON,
            SlideType.CONCLUSION: PptxSlideType.CONCLUSION,
            SlideType.QUOTE: PptxSlideType.QUOTE,
            SlideType.IMAGE: PptxSlideType.IMAGE,
        }

        pptx_type = type_mapping.get(slide.type, PptxSlideType.CONTENT)

        # Convert code blocks
        pptx_code_blocks = None
        if slide.code_blocks:
            pptx_code_blocks = []
            for cb in slide.code_blocks:
                pptx_code_blocks.append(PptxCodeBlock(
                    code=cb.code,
                    language=cb.language,
                    title=cb.title,
                    show_line_numbers=cb.show_line_numbers if hasattr(cb, 'show_line_numbers') else True,
                ))

        # Convert bullet points
        pptx_bullets = None
        if slide.bullet_points:
            pptx_bullets = []
            for i, bp in enumerate(slide.bullet_points):
                if isinstance(bp, str):
                    pptx_bullets.append(PptxBulletPoint(text=bp, level=0))
                elif isinstance(bp, dict):
                    pptx_bullets.append(PptxBulletPoint(
                        text=bp.get("text", str(bp)),
                        level=bp.get("level", 0)
                    ))
                else:
                    pptx_bullets.append(PptxBulletPoint(text=str(bp), level=0))

        # Build PPTX slide
        pptx_slide = PptxSlide(
            type=pptx_type,
            title=slide.title,
            subtitle=slide.subtitle,
            content=slide.content,
            bullet_points=pptx_bullets,
            code_blocks=pptx_code_blocks,
            voiceover=slide.voiceover_text if hasattr(slide, 'voiceover_text') else None,
        )

        return pptx_slide

    def _map_style_to_theme(self, style: PresentationStyle) -> "ThemeStyle":
        """Map PresentationStyle to PPTX ThemeStyle."""
        if not PPTX_CLIENT_AVAILABLE:
            return None

        style_mapping = {
            PresentationStyle.DARK: ThemeStyle.DARK,
            PresentationStyle.LIGHT: ThemeStyle.LIGHT,
            PresentationStyle.CORPORATE: ThemeStyle.CORPORATE,
            PresentationStyle.GRADIENT: ThemeStyle.GRADIENT,
            PresentationStyle.OCEAN: ThemeStyle.OCEAN,
            PresentationStyle.NEON: ThemeStyle.NEON,
            PresentationStyle.MINIMAL: ThemeStyle.MINIMAL,
        }

        # Handle string styles
        style_value = style.value if hasattr(style, 'value') else str(style)

        for ps, ts in style_mapping.items():
            if ps.value == style_value:
                return ts

        return ThemeStyle.DARK  # Default

    async def generate_slides_batch(
        self,
        slides: List[Slide],
        style: PresentationStyle,
        job_id: str,
        target_audience: str = "intermediate developers",
        target_career: Optional[str] = None,
    ) -> List[bytes]:
        """
        Generate multiple slides as PNG images.

        Uses PPTX Service for batch generation if available (more efficient),
        otherwise generates each slide individually with PIL.

        Args:
            slides: List of slides to render
            style: Visual style/theme
            job_id: Job ID for file organization
            target_audience: Target audience for complexity
            target_career: Target career for focus

        Returns:
            List of PNG images as bytes
        """
        # Try batch generation with PPTX Service
        if self.use_pptx_service and self.pptx_client and PPTX_CLIENT_AVAILABLE:
            try:
                batch_result = await self._generate_batch_with_pptx_service(
                    slides, style, job_id
                )
                if batch_result and len(batch_result) == len(slides):
                    print(f"[SLIDE_GEN] Generated {len(batch_result)} slides via PPTX Service batch", flush=True)
                    return batch_result
            except Exception as e:
                print(f"[SLIDE_GEN] PPTX batch failed, falling back to individual generation: {e}", flush=True)

        # Fallback: generate each slide individually
        results = []
        for i, slide in enumerate(slides):
            try:
                img_bytes = await self.generate_slide_image(
                    slide, style, target_audience, target_career,
                    job_id=f"{job_id}_{i}"
                )
                results.append(img_bytes)
            except Exception as e:
                print(f"[SLIDE_GEN] Failed to generate slide {i}: {e}", flush=True)
                # Generate placeholder
                results.append(self._generate_error_slide(str(e)))

        return results

    async def _generate_batch_with_pptx_service(
        self,
        slides: List[Slide],
        style: PresentationStyle,
        job_id: str
    ) -> Optional[List[bytes]]:
        """Generate multiple slides using PPTX Service batch endpoint."""
        if not self.pptx_client or not PPTX_CLIENT_AVAILABLE:
            return None

        # Check availability
        if self._pptx_service_available is None:
            self._pptx_service_available = await self.pptx_client.is_available()

        if not self._pptx_service_available:
            return None

        # Convert all slides
        pptx_slides = []
        for slide in slides:
            pptx_slide = self._convert_to_pptx_slide(slide)
            if pptx_slide:
                pptx_slides.append(pptx_slide)

        if not pptx_slides:
            return None

        # Map style to theme
        theme = PresentationTheme(style=self._map_style_to_theme(style))

        # Generate PPTX and convert to PNGs
        result = await self.pptx_client.generate(
            job_id=job_id,
            slides=pptx_slides,
            theme=theme,
            output_format="png",
            png_width=self.WIDTH,
            png_height=self.HEIGHT,
        )

        if not result.success or not result.png_urls:
            print(f"[SLIDE_GEN] PPTX batch generation failed: {result.error}", flush=True)
            return None

        # Download all PNG files
        png_list = []
        for url in result.png_urls:
            png_bytes = await self.pptx_client.download_file(url)
            if png_bytes:
                png_list.append(png_bytes)
            else:
                # If download fails, return None to trigger fallback
                return None

        return png_list

    def _generate_error_slide(self, error_message: str) -> bytes:
        """Generate a simple error placeholder slide."""
        img = Image.new("RGB", (self.WIDTH, self.HEIGHT), "#1e1e2e")
        draw = ImageDraw.Draw(img)

        # Error message
        draw.text(
            (self.WIDTH // 2, self.HEIGHT // 2),
            f"Error: {error_message[:100]}",
            fill="#ff6b6b",
            font=self.content_font,
            anchor="mm"
        )

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
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

        # Bullet points (defensive: handle None case)
        for point in (slide.bullet_points or []):
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
        style: PresentationStyle,
        target_audience: str = "intermediate developers",
        target_career: Optional[str] = None,
        rag_context: Optional[str] = None,
        course_context: Optional[Dict[str, Any]] = None,
        rag_images: Optional[List[Dict[str, Any]]] = None,
        job_id: Optional[str] = None
    ) -> Image.Image:
        """Render a diagram slide using themed diagrams with Graphviz layout.

        Priority order:
        1. RAG images from documents (if available and relevant, score >= 0.7)
        2. ViralifyDiagramService (themed SVG with Graphviz layout)
        3. DiagramGeneratorService (Python Diagrams / Mermaid)
        4. Content slide fallback

        Uses ViralifyDiagramService (when enabled) for:
        - Professional themed SVG rendering matching slide colors
        - Graphviz-based layout for complex diagrams (50+ components)
        - Edge crossing minimization and proper clustering

        Falls back to DiagramGeneratorService if ViralifyDiagramService fails.
        """
        try:
            # 1. First, try to use a RAG image from documents
            if rag_images and self._should_use_rag_images():
                rag_image = self._find_matching_rag_image(
                    slide_topic=slide.title or slide.content or "",
                    rag_images=rag_images,
                    min_score=float(os.getenv("RAG_IMAGE_MIN_SCORE", "0.7"))
                )
                if rag_image:
                    result = await self._use_rag_image(rag_image, job_id)
                    if result:
                        print(f"[SLIDE] Using RAG image for diagram: {rag_image.get('file_name', 'unknown')} "
                              f"(score: {rag_image.get('relevance_score', 0):.2f})", flush=True)
                        return result

            # 2. Build enriched description with RAG context and course context
            base_description = slide.content or slide.title or "Diagram"
            enriched_description = self._build_enriched_diagram_description(
                base_description=base_description,
                slide_title=slide.title,
                voiceover_text=getattr(slide, 'voiceover_text', None),
                rag_context=rag_context,
                course_context=course_context
            )

            # Map presentation style to viralify-diagrams theme
            # These themes match the slide background colors
            viralify_theme_map = {
                PresentationStyle.DARK: "dark",
                PresentationStyle.LIGHT: "light",
                PresentationStyle.GRADIENT: "gradient",
                PresentationStyle.OCEAN: "ocean"
            }
            viralify_theme = viralify_theme_map.get(style, "dark")

            # Try ViralifyDiagramService first (themed SVG with Graphviz layout)
            if self.use_viralify_diagrams and self.viralify_diagram_service:
                diagram_result = await self._render_with_viralify(
                    description=enriched_description,
                    title=slide.title or "Diagram",
                    theme=viralify_theme,
                    target_audience=target_audience
                )
                if diagram_result:
                    print(f"[SLIDE] Diagram rendered with ViralifyDiagramService (theme: {viralify_theme})", flush=True)
                    return diagram_result

            # Fallback to DiagramGeneratorService (Python Diagrams / Mermaid)
            print(f"[SLIDE] Falling back to DiagramGeneratorService", flush=True)

            # Determine diagram type from slide metadata or content
            # Default to ARCHITECTURE to use Python Diagrams (professional icons) instead of Mermaid
            diagram_type = DiagramType.ARCHITECTURE

            # Check if diagram_type is specified in slide metadata
            if hasattr(slide, 'diagram_type') and slide.diagram_type:
                try:
                    diagram_type = DiagramType(slide.diagram_type)
                except ValueError:
                    pass
            elif slide.content:
                # Try to infer diagram type from content
                # Prioritize types that use Python Diagrams (architecture, process, hierarchy)
                content_lower = slide.content.lower()

                # Mermaid-specific types (only use if explicitly mentioned)
                if 'sequence diagram' in content_lower or 'sequence flow' in content_lower:
                    diagram_type = DiagramType.SEQUENCE
                elif 'mindmap' in content_lower:
                    diagram_type = DiagramType.MINDMAP
                elif 'timeline' in content_lower:
                    diagram_type = DiagramType.TIMELINE
                # Python Diagrams types (preferred for quality)
                elif 'hierarchy' in content_lower or 'tree' in content_lower or 'org chart' in content_lower:
                    diagram_type = DiagramType.HIERARCHY
                elif 'process' in content_lower or 'step' in content_lower or 'workflow' in content_lower or 'pipeline' in content_lower:
                    diagram_type = DiagramType.PROCESS
                elif 'compare' in content_lower or 'vs' in content_lower or 'versus' in content_lower:
                    diagram_type = DiagramType.COMPARISON
                else:
                    # Default: ARCHITECTURE (uses Python Diagrams with professional icons)
                    diagram_type = DiagramType.ARCHITECTURE

            # Map presentation style to diagram theme
            theme_map = {
                PresentationStyle.DARK: "tech",
                PresentationStyle.LIGHT: "light",
                PresentationStyle.GRADIENT: "gradient",
                PresentationStyle.OCEAN: "tech"
            }
            theme = theme_map.get(style, "tech")

            # Generate the diagram with audience-based complexity and career-based focus
            # Ensure job_id is never None (can happen if explicitly set to None on slide)
            safe_job_id = getattr(slide, 'job_id', None) or 'unknown'
            safe_slide_index = getattr(slide, 'index', None) or 0

            diagram_path = await self.diagram_generator.generate_diagram(
                diagram_type=diagram_type,
                description=enriched_description,
                title=slide.title or "",
                job_id=safe_job_id,
                slide_index=safe_slide_index,
                theme=theme,
                width=self.WIDTH,
                height=self.HEIGHT,
                target_audience=target_audience,
                target_career=target_career
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

    async def _render_with_viralify(
        self,
        description: str,
        title: str,
        theme: str,
        target_audience: str = "senior"
    ) -> Optional[Image.Image]:
        """
        Render diagram using ViralifyDiagramService with themed SVG and Graphviz layout.

        This provides:
        - Professional diagrams with colors matching the slide theme
        - Graphviz-based layout for optimal positioning (edge crossing minimization)
        - Support for 50+ component complex diagrams

        Args:
            description: Enriched diagram description
            title: Diagram title
            theme: Viralify theme name (dark, light, gradient, ocean)
            target_audience: Audience level for complexity

        Returns:
            PIL Image if successful, None otherwise
        """
        try:
            # Select layout based on diagram complexity
            # Use AUTO to let viralify-diagrams choose between simple/Graphviz layout
            layout = ViralifyLayoutType.AUTO

            # Map audience to diagram type
            diagram_type = "architecture"  # Default
            if 'pipeline' in description.lower() or 'flow' in description.lower():
                diagram_type = "flowchart"
            elif 'process' in description.lower() or 'step' in description.lower():
                diagram_type = "process"

            print(f"[SLIDE] Rendering with ViralifyDiagramService: theme={theme}, layout={layout.value}", flush=True)

            result = await self.viralify_diagram_service.generate_from_ai_description(
                description=description,
                title=title,
                diagram_type=diagram_type,
                layout=layout,
                theme=theme,
                export_format=ViralifyExportFormat.PNG_SINGLE,
                generate_narration=False,
                target_audience=target_audience,
                width=self.WIDTH,
                height=self.HEIGHT
            )

            if result.success and result.file_path and os.path.exists(result.file_path):
                # Load the generated diagram
                diagram_img = Image.open(result.file_path)
                return diagram_img.convert("RGB")
            else:
                error_msg = result.error or "Unknown error"
                print(f"[SLIDE] ViralifyDiagramService failed: {error_msg}", flush=True)
                return None

        except Exception as e:
            print(f"[SLIDE] ViralifyDiagramService error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    def _should_use_rag_images(self) -> bool:
        """Check if RAG images should be used for diagram slides."""
        return os.getenv("USE_RAG_IMAGES", "true").lower() == "true"

    def _find_matching_rag_image(
        self,
        slide_topic: str,
        rag_images: List[Dict[str, Any]],
        min_score: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best matching RAG image for a slide topic.

        Args:
            slide_topic: The topic/title of the slide
            rag_images: List of RAG image references
            min_score: Minimum relevance score to consider

        Returns:
            Best matching image dict or None
        """
        if not rag_images:
            return None

        # Image types suitable for diagram slides
        diagram_types = ["diagram", "chart", "architecture", "flowchart", "schema"]
        slide_topic_lower = slide_topic.lower()

        candidates = []
        for img in rag_images:
            # Check relevance score
            score = img.get("relevance_score", 0)
            if score < min_score:
                continue

            # Check image type
            img_type = img.get("detected_type", "unknown")
            if img_type not in diagram_types:
                continue

            # Additional matching: check if context/caption relates to slide topic
            context = (img.get("context_text") or "").lower()
            caption = (img.get("caption") or "").lower()

            # Boost score if slide topic words appear in image context
            topic_words = [w for w in slide_topic_lower.split() if len(w) > 3]
            topic_match_bonus = 0
            for word in topic_words:
                if word in context or word in caption:
                    topic_match_bonus += 0.05

            adjusted_score = min(1.0, score + topic_match_bonus)
            candidates.append({**img, "adjusted_score": adjusted_score})

        if not candidates:
            return None

        # Sort by adjusted score and return best match
        candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return candidates[0]

    async def _use_rag_image(
        self,
        rag_image: Dict[str, Any],
        job_id: Optional[str] = None
    ) -> Optional[Image.Image]:
        """
        Load and return a RAG image as a PIL Image.

        Args:
            rag_image: RAG image reference dict
            job_id: Current job ID for file organization

        Returns:
            PIL Image if successful, None otherwise
        """
        try:
            file_path = rag_image.get("file_path")
            if not file_path:
                print(f"[SLIDE] RAG image has no file_path", flush=True)
                return None

            # Check if file exists
            if not os.path.exists(file_path):
                print(f"[SLIDE] RAG image file not found: {file_path}", flush=True)
                return None

            # Load the image
            img = Image.open(file_path)

            # Resize to fit slide dimensions if needed
            img_width, img_height = img.size
            if img_width != self.WIDTH or img_height != self.HEIGHT:
                # Calculate aspect ratio preserving resize
                ratio = min(self.WIDTH / img_width, self.HEIGHT / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Center on slide-sized canvas
                canvas = Image.new("RGB", (self.WIDTH, self.HEIGHT), (30, 30, 30))
                x_offset = (self.WIDTH - new_width) // 2
                y_offset = (self.HEIGHT - new_height) // 2
                canvas.paste(img, (x_offset, y_offset))
                img = canvas

            return img.convert("RGB")

        except Exception as e:
            print(f"[SLIDE] Error loading RAG image: {e}", flush=True)
            return None

    def _build_enriched_diagram_description(
        self,
        base_description: str,
        slide_title: Optional[str] = None,
        voiceover_text: Optional[str] = None,
        rag_context: Optional[str] = None,
        course_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build an enriched diagram description with full context.

        This prevents LLM hallucinations by providing specific context from:
        - The slide's voiceover (explains what the diagram should show)
        - RAG context (source documents with real architectures/processes)
        - Course context (topic, section, learning objectives)

        Args:
            base_description: Original diagram description from slide.content
            slide_title: Title of the slide
            voiceover_text: Voiceover narration that explains the diagram
            rag_context: Content from uploaded documents (PDFs, etc.)
            course_context: Dict with topic, section_title, description, objectives

        Returns:
            Enriched description string for accurate diagram generation
        """
        parts = []

        # 1. Start with the base description
        parts.append(f"DIAGRAM TO CREATE: {base_description}")

        if slide_title:
            parts.append(f"SLIDE TITLE: {slide_title}")

        # 2. Add voiceover context (explains what the diagram should show)
        if voiceover_text:
            # Extract key technical terms from voiceover
            parts.append(f"""
VOICEOVER EXPLANATION (use this to understand what to diagram):
{voiceover_text[:1500]}
""")

        # 3. Add course context
        if course_context:
            topic = course_context.get('topic', '')
            section = course_context.get('section_title', '')
            description = course_context.get('description', '')
            objectives = course_context.get('objectives', [])

            context_parts = []
            if topic:
                context_parts.append(f"Course Topic: {topic}")
            if section:
                context_parts.append(f"Current Section: {section}")
            if description:
                context_parts.append(f"Lecture Description: {description[:500]}")
            if objectives:
                obj_text = ", ".join(objectives[:5]) if isinstance(objectives, list) else str(objectives)
                context_parts.append(f"Learning Objectives: {obj_text}")

            if context_parts:
                parts.append(f"""
COURSE CONTEXT (diagram must be relevant to this):
{chr(10).join(context_parts)}
""")

        # 4. Add RAG context (source documents - most important for accuracy!)
        if rag_context:
            # Extract diagram-relevant content from RAG (limit to avoid token overflow)
            # Focus on architecture, components, flow descriptions
            rag_excerpt = self._extract_diagram_relevant_content(rag_context, max_chars=3000)
            if rag_excerpt:
                parts.append(f"""
=== SOURCE DOCUMENT CONTENT (CRITICAL - BASE DIAGRAM ON THIS) ===
The following is from the user's uploaded documents. The diagram MUST accurately
represent the architecture/process/flow described here. Do NOT invent components
that are not mentioned in this source material:

{rag_excerpt}

=== END SOURCE CONTENT ===
""")

        # 5. Add strict instructions to prevent hallucination
        parts.append("""
STRICT REQUIREMENTS:
- ONLY include components/services that are explicitly mentioned above
- Do NOT add AWS/Azure/GCP services unless specifically mentioned
- Use generic icons if the exact technology is not specified
- Keep the diagram focused on what the voiceover and source documents describe
- If unsure about a component, use a generic representation
""")

        enriched = "\n".join(parts)
        print(f"[SLIDE] Built enriched diagram description: {len(enriched)} chars (RAG: {'yes' if rag_context else 'no'})", flush=True)

        return enriched

    def _extract_diagram_relevant_content(self, rag_context: str, max_chars: int = 3000) -> str:
        """
        Extract diagram-relevant content from RAG context.
        Focuses on architecture, components, flows, and technical descriptions.
        """
        if not rag_context:
            return ""

        # Keywords that indicate diagram-relevant content
        diagram_keywords = [
            'architecture', 'component', 'service', 'layer', 'module',
            'flow', 'pipeline', 'process', 'step', 'stage',
            'database', 'storage', 'cache', 'queue', 'message',
            'api', 'endpoint', 'gateway', 'load balancer',
            'kubernetes', 'docker', 'container', 'pod',
            'aws', 'azure', 'gcp', 'cloud',
            'server', 'client', 'frontend', 'backend',
            'input', 'output', 'transform', 'etl',
            'diagram', 'schema', 'structure', 'topology'
        ]

        # Split into paragraphs and score them
        paragraphs = rag_context.split('\n\n')
        scored_paragraphs = []

        for para in paragraphs:
            if len(para.strip()) < 20:
                continue
            para_lower = para.lower()
            score = sum(1 for kw in diagram_keywords if kw in para_lower)
            if score > 0:
                scored_paragraphs.append((score, para))

        # Sort by relevance score and take top paragraphs
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

        result = []
        current_length = 0
        for score, para in scored_paragraphs:
            if current_length + len(para) > max_chars:
                break
            result.append(para)
            current_length += len(para)

        # If we didn't find relevant paragraphs, just take the beginning
        if not result and rag_context:
            return rag_context[:max_chars]

        return "\n\n".join(result)

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

            # Technical concepts (not real languages - LLM sometimes confuses these)
            "esb": "xml",       # Enterprise Service Bus (configs usually XML)
            "api": "json",      # API usually JSON
            "rest": "json",     # REST API
            "soap": "xml",      # SOAP is XML-based
            "etl": "sql",       # ETL often uses SQL
            "data": "json",     # Generic data format
            "config": "yaml",   # Configuration files
            "diagram": "text",  # Diagram descriptions
            "architecture": "text",  # Architecture descriptions
            "pseudocode": "text",    # Pseudocode
            "pseudo": "text",

            # Natural languages (LLM sometimes passes these instead of programming languages)
            # These should fall back to text since they're not code
            "fr": "text",        # French
            "french": "text",
            "français": "text",
            "en": "text",        # English
            "english": "text",
            "es": "text",        # Spanish
            "spanish": "text",
            "español": "text",
            "de": "text",        # German
            "german": "text",
            "deutsch": "text",
            "pt": "text",        # Portuguese
            "portuguese": "text",
            "português": "text",
            "it": "text",        # Italian
            "italian": "text",
            "italiano": "text",
            "zh": "text",        # Chinese
            "chinese": "text",
            "中文": "text",
            "ja": "text",        # Japanese
            "japanese": "text",
            "日本語": "text",
            "ko": "text",        # Korean
            "korean": "text",
            "한국어": "text",
            "ru": "text",        # Russian
            "russian": "text",
            "русский": "text",
            "ar": "text",        # Arabic
            "arabic": "text",
            "العربية": "text",
            "nl": "text",        # Dutch
            "dutch": "text",
            "pl": "text",        # Polish
            "polish": "text",
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
