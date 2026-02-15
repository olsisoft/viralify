"""
Course Generator Prompts Module

Centralized, well-structured prompts for all LLM interactions.
Each prompt follows the agentic structure:
- Role definition with expertise
- Context (Viralify platform)
- Responsibilities
- Decision rules (hard constraints)
- Examples (correct and incorrect)
- Output contract
"""

from .script_prompts import (
    SCRIPT_WRITER_SYSTEM_PROMPT,
    SCRIPT_SIMPLIFIER_SYSTEM_PROMPT,
    CODE_SIMPLIFIER_SYSTEM_PROMPT,
)

from .knowledge_prompts import (
    CONCEPT_EXTRACTOR_SYSTEM_PROMPT,
    RELATIONSHIP_ANALYZER_SYSTEM_PROMPT,
    DEFINITION_SYNTHESIZER_SYSTEM_PROMPT,
)

from .analysis_prompts import (
    COHERENCE_ANALYZER_SYSTEM_PROMPT,
    CROSS_REFERENCE_ANALYZER_SYSTEM_PROMPT,
    DIFFICULTY_CALIBRATOR_SYSTEM_PROMPT,
)

from .content_prompts import (
    EXERCISE_GENERATOR_SYSTEM_PROMPT,
    SUMMARY_GENERATOR_SYSTEM_PROMPT,
)

__all__ = [
    "SCRIPT_WRITER_SYSTEM_PROMPT",
    "SCRIPT_SIMPLIFIER_SYSTEM_PROMPT",
    "CODE_SIMPLIFIER_SYSTEM_PROMPT",
    "CONCEPT_EXTRACTOR_SYSTEM_PROMPT",
    "RELATIONSHIP_ANALYZER_SYSTEM_PROMPT",
    "DEFINITION_SYNTHESIZER_SYSTEM_PROMPT",
    "COHERENCE_ANALYZER_SYSTEM_PROMPT",
    "CROSS_REFERENCE_ANALYZER_SYSTEM_PROMPT",
    "DIFFICULTY_CALIBRATOR_SYSTEM_PROMPT",
    "EXERCISE_GENERATOR_SYSTEM_PROMPT",
    "SUMMARY_GENERATOR_SYSTEM_PROMPT",
]
