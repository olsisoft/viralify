"""
Integration Example: Visual Generator with Presentation Generator

This example shows how to integrate the Visual Generator module
with the existing Presentation Generator service.
"""

import asyncio
from typing import Dict, Any, List

# Import from Visual Generator module
import sys
sys.path.insert(0, '..')

from models.visual_models import (
    DiagramType,
    DiagramStyle,
    RenderFormat,
    VisualGenerationRequest,
)
from services.visual_generator_service import VisualGeneratorService


async def process_slide_for_visuals(
    slide: Dict[str, Any],
    lesson_context: str,
    visual_generator: VisualGeneratorService
) -> Dict[str, Any]:
    """
    Process a single slide and generate visuals if needed.

    This function can be called from the Presentation Generator's
    slide processing pipeline.
    """
    # Create request from slide
    request = VisualGenerationRequest(
        content=slide.get("content", "") + " " + slide.get("voiceover", ""),
        slide_type=slide.get("type", ""),
        lesson_context=lesson_context,
        style=DiagramStyle.DARK,
        format=RenderFormat.PNG,
        width=1920,
        height=1080,
        language=slide.get("language", "en")
    )

    # Generate visual
    result = await visual_generator.generate(request)

    # If visual was generated, update slide
    if result.success and result.file_path:
        slide["visual"] = {
            "type": result.visual_type.value if result.visual_type else "diagram",
            "file_path": result.file_path,
            "file_url": result.file_url,
            "renderer": result.renderer_used,
            "duration": result.duration_seconds
        }
        print(f"[VisualGenerator] Generated {result.visual_type} for slide: {slide.get('title', 'Untitled')}")
    elif result.detection.needs_diagram:
        print(f"[VisualGenerator] Visual needed but generation failed: {result.error}")

    return slide


async def process_presentation_slides(
    slides: List[Dict[str, Any]],
    lesson_title: str
) -> List[Dict[str, Any]]:
    """
    Process all slides in a presentation and add visuals.

    Example integration point for Presentation Generator V3.
    """
    async with VisualGeneratorService() as visual_generator:
        processed_slides = []

        for slide in slides:
            processed = await process_slide_for_visuals(
                slide=slide,
                lesson_context=lesson_title,
                visual_generator=visual_generator
            )
            processed_slides.append(processed)

        return processed_slides


# Example usage
async def main():
    # Sample slides from a course
    sample_slides = [
        {
            "type": "concept",
            "title": "Understanding Kafka Architecture",
            "content": "Kafka is a distributed streaming platform with producers, brokers, and consumers",
            "voiceover": "Let me show you how Kafka works. Think of it like a postal system...",
            "duration": 60
        },
        {
            "type": "code_demo",
            "title": "Creating a Kafka Producer",
            "content": "from kafka import KafkaProducer\n\nproducer = KafkaProducer(bootstrap_servers='localhost:9092')",
            "voiceover": "Let's write a simple producer in Python...",
            "duration": 90
        },
        {
            "type": "visualization",
            "title": "Message Flow",
            "content": "This diagram shows how messages flow from producers through brokers to consumers",
            "voiceover": "Here's a visual representation of the message flow...",
            "duration": 45
        },
        {
            "type": "metrics",
            "title": "Kafka Performance",
            "content": "Kafka can handle millions of messages per second with latency under 10ms",
            "voiceover": "Let's look at some performance metrics...",
            "duration": 30
        }
    ]

    print("Processing slides for visual generation...")
    processed = await process_presentation_slides(
        slides=sample_slides,
        lesson_title="Introduction to Apache Kafka"
    )

    print("\n=== Results ===")
    for slide in processed:
        visual_info = slide.get("visual", {})
        if visual_info:
            print(f"  - {slide['title']}: Generated {visual_info.get('type')} using {visual_info.get('renderer')}")
        else:
            print(f"  - {slide['title']}: No visual needed")


if __name__ == "__main__":
    asyncio.run(main())
