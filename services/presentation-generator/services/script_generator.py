"""
Script Generator Service

Wrapper for generating presentation scripts using GPT-4.
Used by the multi-agent system for V3 generation.
"""

from typing import Optional
from models.presentation_models import (
    GeneratePresentationRequest,
    PresentationScript,
    PresentationStyle,
)
from services.presentation_planner import PresentationPlannerService


class ScriptGenerator:
    """Generates presentation scripts using GPT-4"""

    def __init__(self):
        self.planner = PresentationPlannerService()

    async def generate_script(
        self,
        topic: str,
        language: str = "python",
        style: str = "dark",
        duration: int = 300,
        execute_code: bool = True
    ) -> PresentationScript:
        """
        Generate a presentation script from a topic.

        Args:
            topic: The topic to create a presentation about
            language: Programming language for code examples
            style: Visual style (dark, light, etc.)
            duration: Target duration in seconds
            execute_code: Whether to include executable code demos

        Returns:
            PresentationScript with slides and metadata
        """
        # Create a request object for the planner
        request = GeneratePresentationRequest(
            topic=topic,
            language=language,
            style=PresentationStyle(style) if style in ["dark", "light", "catppuccin", "nord"] else PresentationStyle.DARK,
            duration=duration,
            execute_code=execute_code
        )

        # Use the existing planner to generate the script
        script = await self.planner.generate_script(request)

        return script
