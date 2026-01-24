"""
Traceability Models for Course Generation

Provides models for tracking which sources were used to generate
each piece of content in a course, enabling:
1. Source attribution per slide/concept
2. Optional vocal citations in voiceover
3. User-visible traceability panel
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Citation Configuration
# =============================================================================

class CitationStyle(str, Enum):
    """How sources are cited in the voiceover"""
    NATURAL = "natural"       # "As explained in the Enterprise Integration Patterns book..."
    ACADEMIC = "academic"     # "According to Hohpe and Woolf (2003)..."
    MINIMAL = "minimal"       # "According to the documentation..."
    NONE = "none"             # No vocal citations


class SourceCitationConfig(BaseModel):
    """
    User configuration for source citations in generated courses.
    Stored as part of the course generation request.
    """
    # Vocal citations in voiceover
    enable_vocal_citations: bool = Field(
        default=False,
        description="Include source citations in the voiceover narration"
    )
    citation_style: CitationStyle = Field(
        default=CitationStyle.NATURAL,
        description="Style of vocal citations"
    )

    # Traceability panel (always available, this controls default visibility)
    show_traceability_panel: bool = Field(
        default=True,
        description="Show source traceability panel by default in the UI"
    )

    # What to include in traceability
    include_page_numbers: bool = Field(
        default=True,
        description="Include page numbers for PDF sources"
    )
    include_timestamps: bool = Field(
        default=True,
        description="Include timestamps for video sources"
    )
    include_quote_excerpts: bool = Field(
        default=True,
        description="Include short excerpts from the original source"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enable_vocal_citations": True,
                "citation_style": "natural",
                "show_traceability_panel": True,
                "include_page_numbers": True,
                "include_timestamps": True,
                "include_quote_excerpts": True,
            }
        }


# =============================================================================
# Content References (Traceability)
# =============================================================================

class ContentReference(BaseModel):
    """
    A reference linking generated content to its source.
    Used for traceability - showing users where information came from.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Source identification
    source_id: str = Field(..., description="ID of the source in the library")
    source_name: str = Field(..., description="Human-readable source name")
    source_type: str = Field(..., description="Type: file, url, youtube, note")
    pedagogical_role: str = Field(default="auto", description="Role: theory, example, etc.")

    # Location in source
    location: str = Field(
        default="",
        description="Location in source: 'page 45', 'timestamp 4:12', 'section 3.2'"
    )
    page_number: Optional[int] = Field(None, description="Page number for PDFs")
    timestamp_seconds: Optional[int] = Field(None, description="Timestamp for videos")
    section_title: Optional[str] = Field(None, description="Section/chapter title")

    # Content excerpt
    quote_excerpt: str = Field(
        default="",
        description="Short excerpt from the source that was used"
    )

    # Relevance
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="How confident we are this source was used (0-1)"
    )
    relevance_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="How relevant this source is to the content"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "src-123",
                "source_name": "Enterprise Integration Patterns",
                "source_type": "file",
                "pedagogical_role": "theory",
                "location": "Chapter 3, page 67",
                "page_number": 67,
                "quote_excerpt": "The Message Channel pattern provides...",
                "confidence": 0.92,
                "relevance_score": 0.88,
            }
        }


class SlideTraceability(BaseModel):
    """
    Traceability information for a single slide.
    Links all content in the slide to its sources.
    """
    slide_index: int = Field(..., description="Index of the slide in the presentation")
    slide_type: str = Field(..., description="Type of slide: title, content, code, etc.")
    slide_title: Optional[str] = Field(None, description="Title of the slide")

    # References for different parts of the slide
    title_references: List[ContentReference] = Field(
        default_factory=list,
        description="Sources for the slide title"
    )
    content_references: List[ContentReference] = Field(
        default_factory=list,
        description="Sources for the main content/bullet points"
    )
    voiceover_references: List[ContentReference] = Field(
        default_factory=list,
        description="Sources for the voiceover narration"
    )
    code_references: List[ContentReference] = Field(
        default_factory=list,
        description="Sources for code examples"
    )
    diagram_references: List[ContentReference] = Field(
        default_factory=list,
        description="Sources for diagrams/visuals"
    )

    # Aggregated metrics
    primary_source_id: Optional[str] = Field(
        None,
        description="ID of the main source used for this slide"
    )
    source_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of slide content backed by sources"
    )

    # Timestamp in the final video (for synchronization)
    video_start_seconds: Optional[float] = Field(None)
    video_end_seconds: Optional[float] = Field(None)


class ConceptTraceability(BaseModel):
    """
    Traceability for a single concept/topic covered in the course.
    Links concepts to their source definitions and examples.
    """
    concept_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    concept_name: str = Field(..., description="Name of the concept")

    # Where this concept appears
    lecture_ids: List[str] = Field(
        default_factory=list,
        description="IDs of lectures where this concept appears"
    )
    slide_indices: List[int] = Field(
        default_factory=list,
        description="Slide indices where this concept is mentioned"
    )

    # Source references by type
    definition_sources: List[ContentReference] = Field(
        default_factory=list,
        description="Sources providing definitions (theory role)"
    )
    example_sources: List[ContentReference] = Field(
        default_factory=list,
        description="Sources providing examples (example role)"
    )
    reference_sources: List[ContentReference] = Field(
        default_factory=list,
        description="Sources providing official references"
    )

    # Consolidated information
    consolidated_definition: Optional[str] = Field(
        None,
        description="Synthesized definition from multiple sources"
    )
    complexity_level: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Complexity level 1-5"
    )


class LectureTraceability(BaseModel):
    """
    Complete traceability for a lecture.
    Aggregates slide and concept traceability.
    """
    lecture_id: str = Field(..., description="ID of the lecture")
    lecture_title: str = Field(..., description="Title of the lecture")
    section_id: Optional[str] = Field(None)
    section_title: Optional[str] = Field(None)

    # Slide-level traceability
    slides: List[SlideTraceability] = Field(
        default_factory=list,
        description="Traceability for each slide"
    )

    # Concept-level traceability
    concepts_covered: List[str] = Field(
        default_factory=list,
        description="IDs of concepts covered in this lecture"
    )

    # Source summary
    sources_used: List[str] = Field(
        default_factory=list,
        description="IDs of all sources used in this lecture"
    )
    primary_sources: List[str] = Field(
        default_factory=list,
        description="IDs of primary sources (most heavily used)"
    )

    # Metrics
    overall_source_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of lecture content backed by sources"
    )

    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CourseTraceability(BaseModel):
    """
    Complete traceability for an entire course.
    This is what gets stored and displayed to the user.
    """
    course_id: str = Field(..., description="ID of the course/job")
    course_title: str = Field(..., description="Title of the course")

    # Configuration used
    citation_config: SourceCitationConfig = Field(
        default_factory=SourceCitationConfig,
        description="Citation configuration used for this course"
    )

    # Lecture traceability
    lectures: List[LectureTraceability] = Field(
        default_factory=list,
        description="Traceability for each lecture"
    )

    # Concept index (for quick lookup)
    concepts: List[ConceptTraceability] = Field(
        default_factory=list,
        description="All concepts covered in the course"
    )

    # Source summary
    all_sources_used: List[str] = Field(
        default_factory=list,
        description="IDs of all sources used"
    )
    source_usage_stats: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Usage statistics per source: {source_id: {usage_count, lectures, slides}}"
    )

    # Overall metrics
    overall_source_coverage: float = Field(
        default=0.0,
        description="Percentage of course content backed by sources"
    )
    total_references: int = Field(default=0)

    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "course_id": "job-123",
                "course_title": "Enterprise Integration Patterns",
                "overall_source_coverage": 0.85,
                "total_references": 47,
                "all_sources_used": ["src-1", "src-2", "src-3"],
            }
        }


# =============================================================================
# API Response Models
# =============================================================================

class TraceabilityResponse(BaseModel):
    """API response with course traceability"""
    course_id: str
    course_title: str
    traceability: CourseTraceability

    # Quick access
    sources_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary of each source with usage stats"
    )


class SlideTraceabilityResponse(BaseModel):
    """API response for single slide traceability"""
    lecture_id: str
    slide_index: int
    slide: SlideTraceability

    # Detailed source info
    source_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full details of sources referenced"
    )
