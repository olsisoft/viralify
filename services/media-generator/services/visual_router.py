"""
Visual Router - Intelligent routing of visual generation requests.

Analyzes scene content and routes to the most appropriate generator:
- Mermaid.js for technical diagrams
- DALL-E for abstract concepts
- D-ID for avatar presenters
- Stock footage for real-world scenes
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from openai import AsyncOpenAI

from models.visual_types import (
    VisualType,
    DiagramType,
    VisualAnalysis,
    VisualGenerationRequest,
    DiagramResult,
)
from models.avatar_models import (
    AvatarVideoRequest,
    AvatarVideoResult,
)
from services.visual_context_analyzer import VisualContextAnalyzer
from services.diagram_generator import DiagramGenerator
from services.avatar_service import AvatarService

logger = logging.getLogger(__name__)


@dataclass
class RoutedVisualResult:
    """Result from visual routing."""
    visual_type: VisualType
    asset_url: str
    asset_type: str  # "image", "video"
    generator_used: str  # "mermaid", "dalle", "did", "stock", "pexels"
    metadata: Dict[str, Any]
    analysis: Optional[VisualAnalysis] = None
    mermaid_code: Optional[str] = None
    duration: Optional[float] = None


class VisualRouter:
    """
    Intelligent visual routing service that analyzes scene content
    and delegates to the appropriate visual generator.
    """

    def __init__(
        self,
        openai_api_key: str,
        did_api_key: Optional[str] = None,
        heygen_api_key: Optional[str] = None,
        pexels_api_key: Optional[str] = None,
        output_dir: str = "/tmp/visuals"
    ):
        """
        Initialize visual router with all required services.

        Args:
            openai_api_key: OpenAI API key for GPT-4 and DALL-E
            did_api_key: D-ID API key for avatars
            heygen_api_key: HeyGen API key (fallback)
            pexels_api_key: Pexels API key for stock footage
            output_dir: Base output directory
        """
        self.openai_api_key = openai_api_key
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize services
        self.analyzer = VisualContextAnalyzer(openai_api_key)
        self.diagram_gen = DiagramGenerator(
            openai_api_key=openai_api_key,
            output_dir=os.path.join(output_dir, "diagrams")
        )

        # Avatar service (optional - requires D-ID key)
        self.avatar_service = None
        if did_api_key:
            self.avatar_service = AvatarService(
                did_api_key=did_api_key,
                heygen_api_key=heygen_api_key,
                output_dir=os.path.join(output_dir, "avatars")
            )

        # OpenAI client for DALL-E
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)

        # Pexels for stock (will be initialized on demand)
        self.pexels_api_key = pexels_api_key

    async def route_scene(
        self,
        request: VisualGenerationRequest,
        skip_analysis: bool = False
    ) -> RoutedVisualResult:
        """
        Analyze and route a scene to the appropriate visual generator.

        Args:
            request: Visual generation request with description
            skip_analysis: Skip GPT-4 analysis and use preferred_type directly

        Returns:
            RoutedVisualResult with generated visual
        """
        logger.info(f"Routing scene: {request.description[:100]}...")

        # Perform analysis unless skipped
        analysis = None
        if not skip_analysis and not request.preferred_type:
            analysis = await self.analyzer.analyze_scene(
                description=request.description,
                script_context=request.script_context
            )
            visual_type = analysis.visual_type
            logger.info(f"Analysis result: {visual_type.value} (confidence: {analysis.confidence})")
        else:
            visual_type = request.preferred_type or VisualType.STOCK

        # Route to appropriate generator
        if visual_type == VisualType.DIAGRAM:
            return await self._route_to_diagram(request, analysis)

        elif visual_type == VisualType.AVATAR:
            return await self._route_to_avatar(request, analysis)

        elif visual_type == VisualType.CONCEPT:
            return await self._route_to_dalle_concept(request, analysis)

        elif visual_type == VisualType.AI_IMAGE:
            return await self._route_to_dalle_image(request, analysis)

        elif visual_type == VisualType.CHART:
            return await self._route_to_chart(request, analysis)

        else:  # STOCK
            return await self._route_to_stock(request, analysis)

    async def _route_to_diagram(
        self,
        request: VisualGenerationRequest,
        analysis: Optional[VisualAnalysis]
    ) -> RoutedVisualResult:
        """Route to diagram generator (Mermaid or DALL-E)."""
        diagram_type = None
        if analysis and analysis.diagram_type:
            diagram_type = analysis.diagram_type

        # Get dimensions from format
        width, height = self._get_dimensions(request.output_format)

        result = await self.diagram_gen.generate(
            description=request.description,
            diagram_type=diagram_type,
            analysis=analysis,
            width=width,
            height=height
        )

        return RoutedVisualResult(
            visual_type=VisualType.DIAGRAM,
            asset_url=result.image_url,
            asset_type="image",
            generator_used=result.generator,
            metadata={
                "diagram_type": diagram_type.value if diagram_type else "flowchart",
                "fallback_used": result.fallback_used
            },
            analysis=analysis,
            mermaid_code=result.mermaid_code
        )

    async def _route_to_avatar(
        self,
        request: VisualGenerationRequest,
        analysis: Optional[VisualAnalysis]
    ) -> RoutedVisualResult:
        """Route to avatar service (D-ID)."""
        if not self.avatar_service:
            logger.warning("Avatar service not configured, falling back to DALL-E")
            return await self._route_to_dalle_concept(request, analysis)

        if not request.avatar_id:
            # Use default avatar
            avatar = self.avatar_service.get_default_avatar()
            avatar_id = avatar.id
        else:
            avatar_id = request.avatar_id

        if not request.voiceover_url:
            logger.warning("No voiceover URL for avatar, falling back to DALL-E")
            return await self._route_to_dalle_concept(request, analysis)

        avatar_request = AvatarVideoRequest(
            avatar_id=avatar_id,
            voiceover_url=request.voiceover_url,
            output_format=request.output_format
        )

        result = await self.avatar_service.generate_avatar_video(avatar_request)

        return RoutedVisualResult(
            visual_type=VisualType.AVATAR,
            asset_url=result.video_url,
            asset_type="video",
            generator_used=result.provider.value,
            metadata={
                "avatar_id": avatar_id,
                "job_id": result.job_id
            },
            analysis=analysis,
            duration=result.duration
        )

    async def _route_to_dalle_concept(
        self,
        request: VisualGenerationRequest,
        analysis: Optional[VisualAnalysis]
    ) -> RoutedVisualResult:
        """Route to DALL-E for abstract/concept visuals."""
        # Use suggested prompt from analysis if available
        if analysis and analysis.suggested_prompt:
            prompt = analysis.suggested_prompt
        else:
            prompt = self._build_concept_prompt(request.description, request.style)

        # Get DALL-E size
        width, height = self._get_dimensions(request.output_format)
        dalle_size = self._get_dalle_size(width, height)

        try:
            response = await self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size=dalle_size,
                quality="hd",
                style="vivid"
            )

            image_url = response.data[0].url

            # Download locally
            local_path = await self._download_image(image_url, "concept")

            return RoutedVisualResult(
                visual_type=VisualType.CONCEPT,
                asset_url=local_path,
                asset_type="image",
                generator_used="dalle",
                metadata={
                    "prompt": prompt[:500],
                    "model": "dall-e-3"
                },
                analysis=analysis
            )

        except Exception as e:
            logger.error(f"DALL-E concept generation failed: {e}")
            # Fallback to stock
            return await self._route_to_stock(request, analysis)

    async def _route_to_dalle_image(
        self,
        request: VisualGenerationRequest,
        analysis: Optional[VisualAnalysis]
    ) -> RoutedVisualResult:
        """Route to DALL-E for general AI images."""
        # Similar to concept but with different prompt style
        prompt = self._build_image_prompt(request.description, request.style)

        width, height = self._get_dimensions(request.output_format)
        dalle_size = self._get_dalle_size(width, height)

        try:
            response = await self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size=dalle_size,
                quality="hd",
                style="natural"
            )

            image_url = response.data[0].url
            local_path = await self._download_image(image_url, "ai_image")

            return RoutedVisualResult(
                visual_type=VisualType.AI_IMAGE,
                asset_url=local_path,
                asset_type="image",
                generator_used="dalle",
                metadata={"prompt": prompt[:500]},
                analysis=analysis
            )

        except Exception as e:
            logger.error(f"DALL-E image generation failed: {e}")
            return await self._route_to_stock(request, analysis)

    async def _route_to_chart(
        self,
        request: VisualGenerationRequest,
        analysis: Optional[VisualAnalysis]
    ) -> RoutedVisualResult:
        """Route to chart generation (using DALL-E for now)."""
        # For now, use DALL-E for charts
        # Future: implement matplotlib/plotly chart generation
        prompt = f"""Create a professional data visualization chart:

{request.description}

Style:
- Dark theme with deep blue background (#1E1B4B)
- Neon accent colors (blue #3B82F6, purple #8B5CF6)
- Clean, minimalist design with clear labels
- Modern data viz aesthetic
- Suitable for video content
- {request.output_format} format"""

        width, height = self._get_dimensions(request.output_format)
        dalle_size = self._get_dalle_size(width, height)

        try:
            response = await self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size=dalle_size,
                quality="hd",
                style="vivid"
            )

            image_url = response.data[0].url
            local_path = await self._download_image(image_url, "chart")

            return RoutedVisualResult(
                visual_type=VisualType.CHART,
                asset_url=local_path,
                asset_type="image",
                generator_used="dalle",
                metadata={
                    "chart_type": analysis.chart_type.value if analysis and analysis.chart_type else "bar"
                },
                analysis=analysis
            )

        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return await self._route_to_stock(request, analysis)

    async def _route_to_stock(
        self,
        request: VisualGenerationRequest,
        analysis: Optional[VisualAnalysis]
    ) -> RoutedVisualResult:
        """Route to stock footage (Pexels)."""
        # Import here to avoid circular dependency
        # This integrates with existing asset_fetcher if available

        if self.pexels_api_key:
            try:
                video_url = await self._fetch_pexels_video(request.description)
                if video_url:
                    return RoutedVisualResult(
                        visual_type=VisualType.STOCK,
                        asset_url=video_url,
                        asset_type="video",
                        generator_used="pexels",
                        metadata={"query": request.description[:100]},
                        analysis=analysis
                    )
            except Exception as e:
                logger.warning(f"Pexels fetch failed: {e}")

        # Fallback to DALL-E image
        logger.info("No stock footage found, generating with DALL-E")
        return await self._route_to_dalle_image(request, analysis)

    async def _fetch_pexels_video(self, query: str) -> Optional[str]:
        """Fetch video from Pexels API."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": self.pexels_api_key},
                params={
                    "query": query,
                    "per_page": 1,
                    "orientation": "portrait"
                }
            )

            if response.status_code == 200:
                data = response.json()
                videos = data.get("videos", [])
                if videos:
                    video_files = videos[0].get("video_files", [])
                    # Get highest quality
                    for vf in sorted(video_files, key=lambda x: x.get("height", 0), reverse=True):
                        if vf.get("height", 0) >= 720:
                            return vf.get("link")
                    if video_files:
                        return video_files[0].get("link")

        return None

    def _build_concept_prompt(self, description: str, style: Optional[str]) -> str:
        """Build DALL-E prompt for concept visualization."""
        style_desc = style or "modern, professional"

        return f"""Create an abstract, conceptual visualization representing:

{description}

Visual style:
- {style_desc} aesthetic
- Dark theme with deep blue/purple gradients (#1E1B4B to #312E81)
- Neon accent colors (electric blue #3B82F6, purple #8B5CF6, teal #14B8A6)
- Abstract geometric shapes and flowing lines
- High contrast, cinematic lighting
- Suitable for professional video content
- Clean composition with focal point
- 9:16 vertical format optimized for social media"""

    def _build_image_prompt(self, description: str, style: Optional[str]) -> str:
        """Build DALL-E prompt for general image."""
        style_desc = style or "photorealistic"

        return f"""Create a high-quality image of:

{description}

Style: {style_desc}
- Professional quality
- Suitable for video content
- Clear subject focus
- Good lighting and composition"""

    def _get_dimensions(self, output_format: str) -> tuple:
        """Get dimensions from output format string."""
        formats = {
            "9:16": (1080, 1920),
            "16:9": (1920, 1080),
            "1:1": (1080, 1080),
            "4:5": (1080, 1350),
        }
        return formats.get(output_format, (1080, 1920))

    def _get_dalle_size(self, width: int, height: int) -> str:
        """Map dimensions to DALL-E supported sizes."""
        aspect = width / height
        if aspect < 0.7:
            return "1024x1792"
        elif aspect > 1.3:
            return "1792x1024"
        else:
            return "1024x1024"

    async def _download_image(self, url: str, prefix: str) -> str:
        """Download image to local storage."""
        import httpx
        import uuid

        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                return filepath
            else:
                raise RuntimeError(f"Failed to download: {response.status_code}")

    async def analyze_only(self, description: str, context: Optional[str] = None) -> VisualAnalysis:
        """Analyze a scene without generating visuals."""
        return await self.analyzer.analyze_scene(
            description=description,
            script_context=context
        )
