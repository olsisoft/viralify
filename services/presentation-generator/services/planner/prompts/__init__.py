"""
Prompts module for presentation planner.

Contains all prompt templates and configurations used by the planner.
"""

from .practical_focus import (
    PRACTICAL_FOCUS_CONFIG,
    parse_practical_focus,
    get_practical_focus_instructions,
    get_practical_focus_slide_ratio,
)

from .system_prompts import (
    PLANNING_SYSTEM_PROMPT,
    VALIDATED_PLANNING_PROMPT,
    RAG_STRICT_HEADER,
    RAG_STRICT_FOOTER,
)

from .rag_prompts import (
    build_rag_section,
    RAG_STRICT_MODE_TEMPLATE,
    RAG_TOPIC_LOCK_TEMPLATE,
)

__all__ = [
    # Practical focus
    "PRACTICAL_FOCUS_CONFIG",
    "parse_practical_focus",
    "get_practical_focus_instructions",
    "get_practical_focus_slide_ratio",
    # System prompts
    "PLANNING_SYSTEM_PROMPT",
    "VALIDATED_PLANNING_PROMPT",
    # RAG Strict Mode (Sandwich Structure)
    "RAG_STRICT_HEADER",
    "RAG_STRICT_FOOTER",
    # RAG prompts (legacy)
    "build_rag_section",
    "RAG_STRICT_MODE_TEMPLATE",
    "RAG_TOPIC_LOCK_TEMPLATE",
]
