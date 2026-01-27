"""
Document Extractors

Pre-LLM extraction modules for analyzing document structure before sending to LLM.
"""

from extractors.structure_extractor import (
    HeadingLevel,
    ExtractedHeading,
    DocumentSection,
    DocumentStructure,
    HeadingPatterns,
    DocumentStructureExtractor,
    StructurePromptFormatter,
    extract_document_structure,
    format_structure_for_prompt,
    analyze_and_format,
)

from extractors.integration import (
    StructureAwareConstraints,
    get_adaptive_constraints,
    validate_output_against_constraints,
)

__all__ = [
    # Structure Extractor
    "HeadingLevel",
    "ExtractedHeading",
    "DocumentSection",
    "DocumentStructure",
    "HeadingPatterns",
    "DocumentStructureExtractor",
    "StructurePromptFormatter",
    "extract_document_structure",
    "format_structure_for_prompt",
    "analyze_and_format",
    # Integration
    "StructureAwareConstraints",
    "get_adaptive_constraints",
    "validate_output_against_constraints",
]
