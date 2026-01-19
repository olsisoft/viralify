"""
Visual Generator Service
Main orchestrator that detects diagram needs and routes to appropriate renderer.
"""

import os
import uuid
import time
from typing import Optional, Dict, Any
from datetime import datetime

from ..models.visual_models import (
    DiagramType,
    DiagramStyle,
    AnimationComplexity,
    RenderFormat,
    DetectionResult,
    VisualGenerationRequest,
    VisualGenerationResult,
)
from .diagram_detector import DiagramDetector
from ..renderers.mermaid_renderer import MermaidRenderer
from ..renderers.matplotlib_renderer import MatplotlibRenderer
from ..renderers.manim_renderer import ManimRenderer


class VisualGeneratorService:
    """
    Main orchestrator for visual content generation.

    Workflow:
    1. Analyze content to detect if visualization is needed
    2. Determine the best diagram type
    3. Route to appropriate renderer (Mermaid, Matplotlib, or Manim)
    4. Return the generated visual
    """

    # Mapping of diagram types to renderers
    MERMAID_TYPES = {
        DiagramType.FLOWCHART,
        DiagramType.SEQUENCE,
        DiagramType.CLASS_DIAGRAM,
        DiagramType.STATE_DIAGRAM,
        DiagramType.ER_DIAGRAM,
        DiagramType.GANTT,
        DiagramType.PIE_CHART,
        DiagramType.MINDMAP,
        DiagramType.TIMELINE,
        DiagramType.ARCHITECTURE,
    }

    MATPLOTLIB_TYPES = {
        DiagramType.LINE_CHART,
        DiagramType.BAR_CHART,
        DiagramType.SCATTER_PLOT,
        DiagramType.HISTOGRAM,
        DiagramType.HEATMAP,
        DiagramType.BOX_PLOT,
    }

    MANIM_TYPES = {
        DiagramType.ANIMATION,
        DiagramType.MATH_FORMULA,
        DiagramType.GRAPH_THEORY,
        DiagramType.TRANSFORMATION,
        DiagramType.CODE_VISUALIZATION,
        DiagramType.DATA_STRUCTURE,
        DiagramType.ALGORITHM,
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        output_dir: str = "/tmp/viralify/visuals"
    ):
        """Initialize the Visual Generator Service."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.output_dir = output_dir

        # Initialize sub-services
        self.detector = DiagramDetector(openai_api_key=self.api_key)
        self.mermaid_renderer = MermaidRenderer(
            openai_api_key=self.api_key,
            output_dir=f"{output_dir}/mermaid"
        )
        self.matplotlib_renderer = MatplotlibRenderer(
            openai_api_key=self.api_key,
            output_dir=f"{output_dir}/charts"
        )
        self.manim_renderer = ManimRenderer(
            openai_api_key=self.api_key,
            output_dir=f"{output_dir}/animations"
        )

    async def generate(
        self,
        request: VisualGenerationRequest
    ) -> VisualGenerationResult:
        """
        Main entry point for visual generation.

        1. Detect if diagram is needed
        2. Route to appropriate renderer
        3. Return result
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            # Step 1: Detect diagram need
            detection = await self.detector.detect(
                content=request.content,
                slide_type=request.slide_type,
                lesson_context=request.lesson_context
            )

            # If no diagram needed, return early
            if not detection.needs_diagram:
                return VisualGenerationResult(
                    request_id=request_id,
                    success=True,
                    detection=detection,
                    generation_time_ms=int((time.time() - start_time) * 1000)
                )

            # Step 2: Determine diagram type
            diagram_type = request.preferred_type or detection.suggested_type
            if not diagram_type:
                diagram_type = DiagramType.FLOWCHART  # Default fallback

            # Step 3: Route to renderer
            description = detection.suggested_description or request.content

            if diagram_type in self.MERMAID_TYPES:
                result = await self._render_mermaid(
                    description=description,
                    diagram_type=diagram_type,
                    request=request
                )
                renderer_used = "mermaid"
            elif diagram_type in self.MATPLOTLIB_TYPES:
                result = await self._render_matplotlib(
                    description=description,
                    diagram_type=diagram_type,
                    request=request
                )
                renderer_used = "matplotlib"
            elif diagram_type in self.MANIM_TYPES:
                result = await self._render_manim(
                    description=description,
                    diagram_type=diagram_type,
                    request=request
                )
                renderer_used = "manim"
            else:
                # Fallback to Mermaid for unknown types
                result = await self._render_mermaid(
                    description=description,
                    diagram_type=DiagramType.FLOWCHART,
                    request=request
                )
                renderer_used = "mermaid"

            generation_time = int((time.time() - start_time) * 1000)

            return VisualGenerationResult(
                request_id=request_id,
                success=result.success,
                detection=detection,
                visual_type=diagram_type,
                file_path=result.file_path,
                file_url=result.file_url,
                format=result.format,
                duration_seconds=result.metadata.get("duration_estimate") if result.metadata else None,
                generation_time_ms=generation_time,
                renderer_used=renderer_used,
                error=result.error,
                raw_specification=result.metadata
            )

        except Exception as e:
            generation_time = int((time.time() - start_time) * 1000)
            return VisualGenerationResult(
                request_id=request_id,
                success=False,
                detection=DetectionResult(
                    needs_diagram=True,
                    confidence=0.5,
                    reasoning="Detection failed"
                ),
                generation_time_ms=generation_time,
                error=str(e)
            )

    async def _render_mermaid(
        self,
        description: str,
        diagram_type: DiagramType,
        request: VisualGenerationRequest
    ):
        """Render using Mermaid."""
        return await self.mermaid_renderer.generate_and_render(
            description=description,
            diagram_type=diagram_type,
            style=request.style,
            format=request.format if request.format != RenderFormat.MP4 else RenderFormat.PNG,
            width=request.width,
            height=request.height,
            context=request.lesson_context,
            language=request.language
        )

    async def _render_matplotlib(
        self,
        description: str,
        diagram_type: DiagramType,
        request: VisualGenerationRequest
    ):
        """Render using Matplotlib."""
        return await self.matplotlib_renderer.generate_and_render(
            description=description,
            chart_type=diagram_type,
            style=request.style,
            format=request.format if request.format != RenderFormat.MP4 else RenderFormat.PNG,
            width=request.width,
            height=request.height,
            context=request.lesson_context,
            language=request.language
        )

    async def _render_manim(
        self,
        description: str,
        diagram_type: DiagramType,
        request: VisualGenerationRequest
    ):
        """Render using Manim."""
        # Determine complexity based on max duration
        if request.max_duration_seconds <= 10:
            complexity = AnimationComplexity.SIMPLE
        elif request.max_duration_seconds <= 20:
            complexity = AnimationComplexity.MODERATE
        elif request.max_duration_seconds <= 45:
            complexity = AnimationComplexity.COMPLEX
        else:
            complexity = AnimationComplexity.CINEMATIC

        return await self.manim_renderer.generate_and_render(
            description=description,
            animation_type=diagram_type,
            complexity=complexity,
            style=request.style,
            resolution="1080p",
            fps=30,
            context=request.lesson_context,
            language=request.language
        )

    async def generate_from_slide(
        self,
        slide_content: Dict[str, Any],
        lesson_context: Optional[str] = None,
        style: DiagramStyle = DiagramStyle.DARK,
    ) -> VisualGenerationResult:
        """
        Generate visual from a slide specification.

        Convenience method for integration with Presentation Generator.

        Args:
            slide_content: Dictionary with slide data (title, content, type, voiceover, etc.)
            lesson_context: Optional context about the lesson for better visual generation
            style: Visual style (dark, light, colorful)

        Returns:
            VisualGenerationResult with success status and file path
        """
        # Extract content from slide
        content = slide_content.get("content", "")
        title = slide_content.get("title", "")
        slide_type = slide_content.get("type", "")
        voiceover = slide_content.get("voiceover", "") or slide_content.get("voiceover_text", "")
        code = slide_content.get("code", "")

        # Combine for better context
        full_content = f"{title}\n{content}\n{voiceover}".strip()
        if code:
            full_content += f"\nCode context: {code[:200]}"  # Include code snippet for context

        request = VisualGenerationRequest(
            content=full_content,
            slide_type=slide_type,
            lesson_context=lesson_context,
            preferred_type=self._infer_type_from_slide(slide_content),
            style=style,
            format=RenderFormat.PNG,
            language=slide_content.get("language", "en")
        )

        return await self.generate(request)

    def _infer_type_from_slide(
        self,
        slide_content: Dict[str, Any]
    ) -> Optional[DiagramType]:
        """Infer diagram type from slide metadata."""
        slide_type = slide_content.get("type", "").lower()

        type_mappings = {
            "architecture": DiagramType.ARCHITECTURE,
            "flow": DiagramType.FLOWCHART,
            "sequence": DiagramType.SEQUENCE,
            "class": DiagramType.CLASS_DIAGRAM,
            "state": DiagramType.STATE_DIAGRAM,
            "database": DiagramType.ER_DIAGRAM,
            "chart": DiagramType.BAR_CHART,
            "graph": DiagramType.LINE_CHART,
            "algorithm": DiagramType.ALGORITHM,
            "data_structure": DiagramType.DATA_STRUCTURE,
            "math": DiagramType.MATH_FORMULA,
            "animation": DiagramType.ANIMATION,
        }

        return type_mappings.get(slide_type)

    async def batch_generate(
        self,
        slides: list[Dict[str, Any]],
        lesson_context: Optional[str] = None
    ) -> list[VisualGenerationResult]:
        """
        Generate visuals for multiple slides.

        Useful for processing entire presentations.
        """
        import asyncio

        tasks = [
            self.generate_from_slide(slide, lesson_context)
            for slide in slides
        ]

        return await asyncio.gather(*tasks)

    async def close(self):
        """Cleanup resources."""
        await self.mermaid_renderer.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Factory function for easy instantiation
def create_visual_generator(
    openai_api_key: Optional[str] = None,
    output_dir: str = "/tmp/viralify/visuals"
) -> VisualGeneratorService:
    """Create a configured VisualGeneratorService instance."""
    return VisualGeneratorService(
        openai_api_key=openai_api_key,
        output_dir=output_dir
    )
