"""
Visual Generator Module
Generates diagrams, charts, and animations for educational content.

Usage:
    from visual_generator import VisualGeneratorService, VisualGenerationRequest

    async with VisualGeneratorService() as generator:
        result = await generator.generate(
            VisualGenerationRequest(
                content="Kafka architecture with producers, brokers, and consumers",
                slide_type="concept"
            )
        )
        print(f"Generated: {result.file_path}")
"""

from .models import (
    DiagramType,
    DiagramStyle,
    AnimationComplexity,
    RenderFormat,
    DiagramRequest,
    DiagramResult,
    MermaidDiagram,
    MatplotlibChart,
    ManimAnimation,
    DetectionResult,
    VisualGenerationRequest,
    VisualGenerationResult,
)

from .services import (
    DiagramDetector,
    VisualGeneratorService,
)

from .renderers import (
    MermaidRenderer,
    MatplotlibRenderer,
    ManimRenderer,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "DiagramType",
    "DiagramStyle",
    "AnimationComplexity",
    "RenderFormat",
    "DiagramRequest",
    "DiagramResult",
    "MermaidDiagram",
    "MatplotlibChart",
    "ManimAnimation",
    "DetectionResult",
    "VisualGenerationRequest",
    "VisualGenerationResult",
    # Services
    "DiagramDetector",
    "VisualGeneratorService",
    # Renderers
    "MermaidRenderer",
    "MatplotlibRenderer",
    "ManimRenderer",
]
