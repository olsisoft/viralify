"""
PPTX Service Client

Client for communicating with the PPTX generation microservice.
Provides async methods for generating PPTX files and PNG slides.
"""

import os
import httpx
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


# ===========================================
# CONFIGURATION
# ===========================================

PPTX_SERVICE_URL = os.getenv("PPTX_SERVICE_URL", "http://pptx-service:8013")
PPTX_SERVICE_TIMEOUT = int(os.getenv("PPTX_SERVICE_TIMEOUT", "120"))


# ===========================================
# ENUMS
# ===========================================

class SlideType(str, Enum):
    TITLE = "title"
    CONTENT = "content"
    CODE = "code"
    CODE_DEMO = "code_demo"
    DIAGRAM = "diagram"
    COMPARISON = "comparison"
    QUOTE = "quote"
    IMAGE = "image"
    VIDEO = "video"
    QUIZ = "quiz"
    CONCLUSION = "conclusion"
    SECTION_HEADER = "section_header"
    TWO_COLUMN = "two_column"
    BULLET_POINTS = "bullet_points"


class ThemeStyle(str, Enum):
    DARK = "dark"
    LIGHT = "light"
    CORPORATE = "corporate"
    GRADIENT = "gradient"
    OCEAN = "ocean"
    NEON = "neon"
    MINIMAL = "minimal"


class TransitionType(str, Enum):
    NONE = "none"
    FADE = "fade"
    PUSH = "push"
    WIPE = "wipe"
    ZOOM = "zoom"
    SPLIT = "split"
    REVEAL = "reveal"
    COVER = "cover"


# ===========================================
# DATA CLASSES
# ===========================================

@dataclass
class BulletPoint:
    text: str
    level: int = 0
    bullet: bool = True
    color: Optional[str] = None
    font_size: Optional[int] = None


@dataclass
class CodeBlock:
    code: str
    language: str
    title: Optional[str] = None
    highlight_lines: Optional[List[int]] = None
    show_line_numbers: bool = True
    font_size: Optional[int] = None
    theme: str = "dark"


@dataclass
class ImageElement:
    path: Optional[str] = None
    data: Optional[str] = None  # Base64
    url: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    w: Optional[str] = None
    h: Optional[str] = None
    alt_text: Optional[str] = None


@dataclass
class SlideTransition:
    type: TransitionType = TransitionType.FADE
    duration: float = 0.5


@dataclass
class Slide:
    type: SlideType
    title: Optional[str] = None
    subtitle: Optional[str] = None
    content: Optional[str] = None
    bullet_points: Optional[List[BulletPoint]] = None
    code_blocks: Optional[List[CodeBlock]] = None
    images: Optional[List[ImageElement]] = None
    transition: Optional[SlideTransition] = None
    speaker_notes: Optional[str] = None
    voiceover: Optional[str] = None
    duration: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        result = {
            "type": self.type.value if isinstance(self.type, SlideType) else self.type,
        }

        if self.title:
            result["title"] = self.title
        if self.subtitle:
            result["subtitle"] = self.subtitle
        if self.content:
            result["content"] = self.content

        if self.bullet_points:
            result["bulletPoints"] = [
                {
                    "text": bp.text,
                    "level": bp.level,
                    "bullet": bp.bullet,
                    **({"color": bp.color} if bp.color else {}),
                    **({"fontSize": bp.font_size} if bp.font_size else {}),
                }
                for bp in self.bullet_points
            ]

        if self.code_blocks:
            result["codeBlocks"] = [
                {
                    "code": cb.code,
                    "language": cb.language,
                    **({"title": cb.title} if cb.title else {}),
                    **({"highlightLines": cb.highlight_lines} if cb.highlight_lines else {}),
                    "showLineNumbers": cb.show_line_numbers,
                    **({"fontSize": cb.font_size} if cb.font_size else {}),
                    "theme": cb.theme,
                }
                for cb in self.code_blocks
            ]

        if self.images:
            result["images"] = [
                {
                    **({"path": img.path} if img.path else {}),
                    **({"data": img.data} if img.data else {}),
                    **({"url": img.url} if img.url else {}),
                    **({"x": img.x} if img.x else {}),
                    **({"y": img.y} if img.y else {}),
                    **({"w": img.w} if img.w else {}),
                    **({"h": img.h} if img.h else {}),
                    **({"altText": img.alt_text} if img.alt_text else {}),
                }
                for img in self.images
            ]

        if self.transition:
            result["transition"] = {
                "type": self.transition.type.value if isinstance(self.transition.type, TransitionType) else self.transition.type,
                "duration": self.transition.duration,
            }

        if self.speaker_notes:
            result["speakerNotes"] = self.speaker_notes
        if self.voiceover:
            result["voiceover"] = self.voiceover
        if self.duration:
            result["duration"] = self.duration

        return result


@dataclass
class PresentationTheme:
    style: ThemeStyle = ThemeStyle.DARK
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    accent_color: Optional[str] = None
    background_color: Optional[str] = None
    text_color: Optional[str] = None
    font_family: Optional[str] = None
    heading_font_family: Optional[str] = None
    code_font_family: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "style": self.style.value if isinstance(self.style, ThemeStyle) else self.style,
        }
        if self.primary_color:
            result["primaryColor"] = self.primary_color
        if self.secondary_color:
            result["secondaryColor"] = self.secondary_color
        if self.accent_color:
            result["accentColor"] = self.accent_color
        if self.background_color:
            result["backgroundColor"] = self.background_color
        if self.text_color:
            result["textColor"] = self.text_color
        if self.font_family:
            result["fontFamily"] = self.font_family
        if self.heading_font_family:
            result["headingFontFamily"] = self.heading_font_family
        if self.code_font_family:
            result["codeFontFamily"] = self.code_font_family
        return result


@dataclass
class PresentationMetadata:
    title: str
    author: Optional[str] = None
    company: Optional[str] = None
    subject: Optional[str] = None
    category: Optional[str] = None
    keywords: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"title": self.title}
        if self.author:
            result["author"] = self.author
        if self.company:
            result["company"] = self.company
        if self.subject:
            result["subject"] = self.subject
        if self.category:
            result["category"] = self.category
        if self.keywords:
            result["keywords"] = self.keywords
        return result


@dataclass
class GenerationResult:
    success: bool
    job_id: str
    pptx_url: Optional[str] = None
    png_urls: Optional[List[str]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None


@dataclass
class JobStatus:
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    pptx_url: Optional[str] = None
    png_urls: Optional[List[str]] = None
    slide_count: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None


# ===========================================
# PPTX CLIENT
# ===========================================

class PptxClient:
    """
    Async client for the PPTX generation microservice.

    Usage:
        client = PptxClient()

        # Generate PPTX
        result = await client.generate(
            job_id="course_123",
            slides=[
                Slide(type=SlideType.TITLE, title="Hello World"),
                Slide(type=SlideType.CODE, code_blocks=[...]),
            ],
            theme=PresentationTheme(style=ThemeStyle.DARK),
            output_format="both",  # pptx + png
        )

        if result.success:
            print(f"PPTX: {result.pptx_url}")
            print(f"PNGs: {result.png_urls}")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or PPTX_SERVICE_URL).rstrip("/")
        self.timeout = timeout or PPTX_SERVICE_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        client = await self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()

    async def is_available(self) -> bool:
        """Check if the service is available."""
        try:
            health = await self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False

    async def get_themes(self) -> List[Dict[str, Any]]:
        """Get available themes."""
        client = await self._get_client()
        response = await client.get("/api/v1/pptx/themes")
        response.raise_for_status()
        return response.json().get("themes", [])

    async def get_slide_types(self) -> Dict[str, Any]:
        """Get available slide types."""
        client = await self._get_client()
        response = await client.get("/api/v1/pptx/slide-types")
        response.raise_for_status()
        return response.json()

    async def generate(
        self,
        job_id: str,
        slides: List[Slide],
        theme: Optional[PresentationTheme] = None,
        metadata: Optional[PresentationMetadata] = None,
        output_format: str = "pptx",  # pptx, png, both
        png_width: int = 1920,
        png_height: int = 1080,
        default_transition: Optional[SlideTransition] = None,
    ) -> GenerationResult:
        """
        Generate PPTX file synchronously.

        Args:
            job_id: Unique job identifier
            slides: List of Slide objects
            theme: Presentation theme
            metadata: Presentation metadata
            output_format: Output format (pptx, png, both)
            png_width: PNG width in pixels
            png_height: PNG height in pixels
            default_transition: Default transition for all slides

        Returns:
            GenerationResult with URLs to generated files
        """
        request_data = {
            "job_id": job_id,
            "slides": [s.to_dict() for s in slides],
            "outputFormat": output_format,
            "pngWidth": png_width,
            "pngHeight": png_height,
        }

        if theme:
            request_data["theme"] = theme.to_dict()
        if metadata:
            request_data["metadata"] = metadata.to_dict()
        if default_transition:
            request_data["defaultTransition"] = {
                "type": default_transition.type.value,
                "duration": default_transition.duration,
            }

        client = await self._get_client()
        response = await client.post("/api/v1/pptx/generate", json=request_data)

        if response.status_code != 200:
            error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"error": response.text}
            return GenerationResult(
                success=False,
                job_id=job_id,
                error=error_data.get("error", "Generation failed"),
            )

        data = response.json()
        return GenerationResult(
            success=data.get("success", False),
            job_id=data.get("job_id", job_id),
            pptx_url=self._make_full_url(data.get("pptx_url")),
            png_urls=[self._make_full_url(u) for u in data.get("png_urls", [])] if data.get("png_urls") else None,
            error=data.get("error"),
            processing_time_ms=data.get("processing_time_ms"),
        )

    async def generate_async(
        self,
        job_id: str,
        slides: List[Slide],
        theme: Optional[PresentationTheme] = None,
        metadata: Optional[PresentationMetadata] = None,
        output_format: str = "pptx",
        png_width: int = 1920,
        png_height: int = 1080,
    ) -> str:
        """
        Start asynchronous PPTX generation.

        Returns:
            Job ID for status polling
        """
        request_data = {
            "job_id": job_id,
            "slides": [s.to_dict() for s in slides],
            "outputFormat": output_format,
            "pngWidth": png_width,
            "pngHeight": png_height,
        }

        if theme:
            request_data["theme"] = theme.to_dict()
        if metadata:
            request_data["metadata"] = metadata.to_dict()

        client = await self._get_client()
        response = await client.post("/api/v1/pptx/generate-async", json=request_data)
        response.raise_for_status()

        return response.json().get("job_id", job_id)

    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of an async generation job."""
        client = await self._get_client()
        response = await client.get(f"/api/v1/pptx/jobs/{job_id}")

        if response.status_code == 404:
            return JobStatus(job_id=job_id, status="not_found")

        response.raise_for_status()
        data = response.json()

        return JobStatus(
            job_id=data.get("job_id", job_id),
            status=data.get("status", "unknown"),
            created_at=data.get("created_at"),
            completed_at=data.get("completed_at"),
            pptx_url=self._make_full_url(data.get("pptx_url")),
            png_urls=[self._make_full_url(u) for u in data.get("png_urls", [])] if data.get("png_urls") else None,
            slide_count=data.get("slide_count"),
            processing_time_ms=data.get("processing_time_ms"),
            error=data.get("error"),
        )

    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 1.0,
        max_wait: float = 300.0,
    ) -> JobStatus:
        """
        Wait for an async job to complete.

        Args:
            job_id: Job ID to wait for
            poll_interval: Polling interval in seconds
            max_wait: Maximum wait time in seconds

        Returns:
            Final JobStatus
        """
        elapsed = 0.0

        while elapsed < max_wait:
            status = await self.get_job_status(job_id)

            if status.status in ("completed", "failed", "not_found"):
                return status

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return JobStatus(job_id=job_id, status="timeout", error="Maximum wait time exceeded")

    async def generate_preview(
        self,
        slide: Slide,
        theme: Optional[PresentationTheme] = None,
        width: int = 1920,
        height: int = 1080,
    ) -> Optional[bytes]:
        """
        Generate a PNG preview of a single slide.

        Returns:
            PNG image data as bytes, or None if failed
        """
        request_data = {
            "slide": slide.to_dict(),
            "width": width,
            "height": height,
        }

        if theme:
            request_data["theme"] = theme.to_dict()

        client = await self._get_client()
        response = await client.post("/api/v1/pptx/preview", json=request_data)

        if response.status_code != 200:
            return None

        return response.content

    async def download_file(self, url: str) -> Optional[bytes]:
        """Download a generated file."""
        client = await self._get_client()

        # Handle relative URLs
        if url.startswith("/"):
            response = await client.get(url)
        else:
            # For full URLs, create a new request
            async with httpx.AsyncClient() as temp_client:
                response = await temp_client.get(url)

        if response.status_code != 200:
            return None

        return response.content

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job and its files."""
        client = await self._get_client()
        response = await client.delete(f"/api/v1/pptx/jobs/{job_id}")
        return response.status_code == 200

    async def cleanup(self, max_age_ms: int = 3600000) -> Dict[str, int]:
        """Trigger cleanup of old files."""
        client = await self._get_client()
        response = await client.post(f"/api/v1/pptx/cleanup?max_age_ms={max_age_ms}")
        response.raise_for_status()
        return response.json().get("cleaned", {})

    def _make_full_url(self, path: Optional[str]) -> Optional[str]:
        """Convert relative path to full URL."""
        if not path:
            return None
        if path.startswith("http"):
            return path
        return f"{self.base_url}{path}"


# ===========================================
# SINGLETON & CONVENIENCE FUNCTIONS
# ===========================================

_client: Optional[PptxClient] = None


def get_pptx_client() -> PptxClient:
    """Get the singleton PPTX client instance."""
    global _client
    if _client is None:
        _client = PptxClient()
    return _client


async def generate_pptx(
    job_id: str,
    slides: List[Slide],
    theme: Optional[PresentationTheme] = None,
    output_format: str = "pptx",
) -> GenerationResult:
    """Convenience function for generating PPTX."""
    client = get_pptx_client()
    return await client.generate(
        job_id=job_id,
        slides=slides,
        theme=theme,
        output_format=output_format,
    )


async def generate_slide_pngs(
    job_id: str,
    slides: List[Slide],
    theme: Optional[PresentationTheme] = None,
) -> List[str]:
    """
    Generate PNG images for slides.

    Returns:
        List of PNG URLs
    """
    client = get_pptx_client()
    result = await client.generate(
        job_id=job_id,
        slides=slides,
        theme=theme,
        output_format="png",
    )

    if result.success and result.png_urls:
        return result.png_urls
    return []
