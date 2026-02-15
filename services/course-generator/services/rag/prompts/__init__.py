"""
Prompt Engineering Module

Provides well-structured prompts following the Viralify prompt engineering pattern:
- Role definition with expertise
- Context (Viralify platform)
- Input signals
- Responsibilities
- Decision rules (HARD constraints)
- Self-validation checklist
- Examples (correct and incorrect)
- Output contract
"""

from .base_prompt import (
    BasePromptBuilder,
    PromptSection,
    PromptExample,
)
from .summary_prompts import DocumentSummaryPromptBuilder
from .structure_prompts import StructureExtractionPromptBuilder

__all__ = [
    "BasePromptBuilder",
    "PromptSection",
    "PromptExample",
    "DocumentSummaryPromptBuilder",
    "StructureExtractionPromptBuilder",
]
