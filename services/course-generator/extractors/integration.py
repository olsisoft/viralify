"""
Course Planner Integration

Shows how to integrate the Document Structure Extractor into course_planner.py.

This provides:
1. Pre-LLM structure analysis
2. Adaptive constraints based on detected structure
3. Post-LLM validation against structure
4. Source reference validation for RAG traceability

Author: Viralify Team
Version: 1.0
"""

import re
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
    # Conditions:
    # 1. Has sections
    # 2. Confidence >= 30%
    # 3. NOT all sections have 0 lectures (useless structure)
    # 4. Reasonable section count (not more than 20)
    has_useful_structure = (
        structure.section_count > 0 and
        structure.detection_confidence >= 0.3 and
        sum(structure.lectures_per_section) > 0 and  # At least some lectures detected
        structure.section_count <= 20  # Not too many sections (probably false positives)
    )

    if has_useful_structure:
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
        # Log why structure was rejected
        rejection_reasons = []
        if structure.section_count == 0:
            rejection_reasons.append("no sections detected")
        elif structure.section_count > 20:
            rejection_reasons.append(f"too many sections ({structure.section_count})")
        if structure.detection_confidence < 0.3:
            rejection_reasons.append(f"low confidence ({structure.detection_confidence:.0%})")
        if sum(structure.lectures_per_section) == 0:
            rejection_reasons.append("all sections have 0 lectures")

        rejection_msg = f"Structure rejected: {', '.join(rejection_reasons)}" if rejection_reasons else "Structure unclear"

        # Structure not usable, use targets as guide
        return StructureAwareConstraints(
            section_count=target_sections,
            lectures_per_section=[target_lectures] * target_sections,
            is_from_documents=False,
            confidence=structure.detection_confidence,
            structure_prompt="",  # Don't pass unusable structure to prompt
            warnings=structure.warnings + [rejection_msg, "Using user targets as guide"]
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


# =============================================================================
#                    SOURCE REFERENCE VALIDATION
# =============================================================================

@dataclass
class SourceReferenceValidation:
    """Result of source reference validation."""
    valid: bool
    total: int
    valid_count: int
    invalid_count: int
    valid_refs: List[str]
    invalid_refs: List[str]
    coverage: float


def validate_source_references(curriculum: dict, document: str) -> SourceReferenceValidation:
    """
    Validate that all source_reference values exist in the document.

    Args:
        curriculum: The generated curriculum JSON
        document: The source document text

    Returns:
        SourceReferenceValidation with validation results
    """
    document_lower = document.lower()

    invalid_refs = []
    valid_refs = []

    # Check each section
    for section in curriculum.get("sections", []):
        section_title = section.get("title", "Unknown")
        section_ref = section.get("source_reference", "")

        # Validate section reference
        if not section_ref or section_ref.lower() in ["n/a", "none", ""]:
            invalid_refs.append(f"Section '{section_title}': missing/empty source_reference")
        elif section_ref.lower() not in document_lower and not _fuzzy_match_reference(section_ref, document):
            invalid_refs.append(f"Section '{section_title}': reference not found: '{section_ref}'")
        else:
            valid_refs.append(f"Section '{section_title}': ✓")

        # Check each lecture
        for lecture in section.get("lectures", []):
            lecture_title = lecture.get("title", "Unknown")
            lecture_ref = lecture.get("source_reference", "")

            if not lecture_ref or lecture_ref.lower() in ["n/a", "none", ""]:
                invalid_refs.append(f"Lecture '{lecture_title}': missing/empty source_reference")
            elif lecture_ref.lower() not in document_lower and not _fuzzy_match_reference(lecture_ref, document):
                invalid_refs.append(f"Lecture '{lecture_title}': reference not found: '{lecture_ref}'")
            else:
                valid_refs.append(f"Lecture '{lecture_title}': ✓")

    total = len(valid_refs) + len(invalid_refs)
    coverage = len(valid_refs) / total if total > 0 else 0

    return SourceReferenceValidation(
        valid=len(invalid_refs) == 0,
        total=total,
        valid_count=len(valid_refs),
        invalid_count=len(invalid_refs),
        valid_refs=valid_refs,
        invalid_refs=invalid_refs,
        coverage=coverage
    )


def _fuzzy_match_reference(reference: str, document: str) -> bool:
    """Check if reference partially matches document content."""
    # Extract key words (4+ chars)
    words = re.findall(r'\b[a-zà-ÿ]{4,}\b', reference.lower())
    doc_lower = document.lower()

    if not words:
        return False

    # At least 60% of words should be in document
    matches = sum(1 for w in words if w in doc_lower)
    return matches / len(words) >= 0.6


# Source reference prompt section for RAG mode
SOURCE_REFERENCE_PROMPT = '''
╔══════════════════════════════════════════════════════════════════════════════╗
║              🔗 SOURCE REFERENCE - MANDATORY FOR EVERY ITEM                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

REQUIREMENT: Every section AND every lecture MUST include a source_reference field.

WHAT IS source_reference?
The EXACT heading text from the source documents that this item comes from.

FORMAT:
┌─────────────────────────────────────────────────────────────────────────────┐
│  "source_reference": "[Heading marker] Exact Text (line N)"                 │
└─────────────────────────────────────────────────────────────────────────────┘

EXAMPLES OF VALID source_reference:
✅ "## Chapitre 1: Introduction aux EIP"
✅ "### 1.1 Qu'est-ce que l'intégration d'entreprise?"
✅ "2.3 Datatype Channel (line 45)"
✅ "Section 'Message Router' - heading at line 78"

EXAMPLES OF INVALID source_reference (REJECTED):
❌ ""                          (empty)
❌ "N/A"                       (placeholder)
❌ "From document"             (too vague)
❌ "Introduction"              (no heading marker)
❌ "See section 1"             (not exact text)

RULES:
1. Copy the heading EXACTLY as it appears in the document
2. Include the heading marker (##, ###, 1.1, etc.) if present
3. Add line number in parentheses if known
4. If you CANNOT find a source_reference → DO NOT include that section/lecture

⚠️ WARNING: All source_reference values will be VALIDATED against the documents.
   Fake references will be detected and flagged as hallucinations.
'''
