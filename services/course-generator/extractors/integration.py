"""
Course Planner Integration

Shows how to integrate the Document Structure Extractor into course_planner.py.

This provides:
1. Pre-LLM structure analysis
2. Adaptive constraints based on detected structure
3. Post-LLM validation against structure

Author: Viralify Team
Version: 1.0
"""

from typing import Optional, List
from dataclasses import dataclass

from extractors.structure_extractor import (
    DocumentStructure,
    DocumentStructureExtractor,
    StructurePromptFormatter,
    extract_document_structure,
    format_structure_for_prompt,
)


# =============================================================================
#                         INTEGRATION HELPERS
# =============================================================================

@dataclass
class StructureAwareConstraints:
    """Constraints adapted to document structure."""
    section_count: int
    lectures_per_section: List[int]
    is_from_documents: bool
    confidence: float
    structure_prompt: str
    warnings: List[str]


def get_adaptive_constraints(
    rag_context: Optional[str],
    target_sections: int,
    target_lectures: int
) -> StructureAwareConstraints:
    """
    Analyze documents and return adaptive constraints.

    If documents have clear structure: use document structure
    If no clear structure: use target values

    Args:
        rag_context: Document text (or None)
        target_sections: User's requested section count
        target_lectures: User's requested lectures per section

    Returns:
        StructureAwareConstraints with either document or target values
    """
    # No RAG context: use targets
    if not rag_context or len(rag_context.strip()) < 100:
        return StructureAwareConstraints(
            section_count=target_sections,
            lectures_per_section=[target_lectures] * target_sections,
            is_from_documents=False,
            confidence=0.0,
            structure_prompt="",
            warnings=["No documents provided, using target values"]
        )

    # Extract structure
    structure = extract_document_structure(rag_context)

    # Check if structure is usable
    if structure.section_count > 0 and structure.detection_confidence >= 0.3:
        # Use document structure
        return StructureAwareConstraints(
            section_count=structure.section_count,
            lectures_per_section=structure.lectures_per_section,
            is_from_documents=True,
            confidence=structure.detection_confidence,
            structure_prompt=format_structure_for_prompt(structure),
            warnings=structure.warnings
        )
    else:
        # Structure not usable, use targets as guide
        return StructureAwareConstraints(
            section_count=target_sections,
            lectures_per_section=[target_lectures] * target_sections,
            is_from_documents=False,
            confidence=structure.detection_confidence,
            structure_prompt=format_structure_for_prompt(structure),
            warnings=structure.warnings + ["Document structure unclear, using targets as guide"]
        )


def validate_output_against_constraints(
    curriculum: dict,
    constraints: StructureAwareConstraints
) -> dict:
    """
    Validate generated curriculum against constraints.

    Args:
        curriculum: Generated curriculum dict with sections
        constraints: Expected constraints from document structure

    Returns:
        Dict with 'valid' bool and 'issues' list
    """
    issues = []

    sections = curriculum.get("sections", [])

    # Check section count
    if len(sections) != constraints.section_count:
        issues.append(
            f"Section count mismatch: expected {constraints.section_count}, "
            f"got {len(sections)}"
        )

    # Check lecture counts per section
    for i, (section, expected_count) in enumerate(
        zip(sections, constraints.lectures_per_section)
    ):
        actual_count = len(section.get("lectures", []))
        if actual_count != expected_count:
            issues.append(
                f"Section {i+1} lecture count: expected {expected_count}, "
                f"got {actual_count}"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues
    }
