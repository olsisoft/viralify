"""
Planner module for presentation generation.

This module contains the refactored presentation planner with
extracted prompts and utilities for better maintainability.
"""

from .prompts import (
    PRACTICAL_FOCUS_CONFIG,
    parse_practical_focus,
    get_practical_focus_instructions,
    get_practical_focus_slide_ratio,
    PLANNING_SYSTEM_PROMPT,
    VALIDATED_PLANNING_PROMPT,
    build_rag_section,
)

__all__ = [
    "PRACTICAL_FOCUS_CONFIG",
    "parse_practical_focus",
    "get_practical_focus_instructions",
    "get_practical_focus_slide_ratio",
    "PLANNING_SYSTEM_PROMPT",
    "VALIDATED_PLANNING_PROMPT",
    "build_rag_section",
]
