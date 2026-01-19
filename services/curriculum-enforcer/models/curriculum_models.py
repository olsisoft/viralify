"""
Curriculum Enforcer Models
Models for lesson structure templates and validation.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ContextType(str, Enum):
    """Context types for different learning environments."""
    EDUCATION = "education"         # Traditional learning (Hook → Concept → Code → Recap)
    ENTERPRISE = "enterprise"       # Corporate training (Problem → Solution → ROI → Action)
    BOOTCAMP = "bootcamp"          # Intensive learning (Concept → Practice → Practice → Test)
    TUTORIAL = "tutorial"          # Quick how-to (Goal → Steps → Result)
    WORKSHOP = "workshop"          # Hands-on (Intro → Exercise → Exercise → Debrief)
    CERTIFICATION = "certification" # Exam prep (Theory → Examples → Practice → Quiz)
    CUSTOM = "custom"              # User-defined


class LessonPhase(str, Enum):
    """Standard phases that can appear in a lesson."""
    # Engagement
    HOOK = "hook"                   # Emotional/problem-based opener
    TEASER = "teaser"              # Preview of what's coming

    # Introduction
    CONTEXT = "context"            # Why this matters
    OBJECTIVES = "objectives"       # What you'll learn
    PREREQUISITES = "prerequisites" # What you need to know

    # Core Content
    CONCEPT = "concept"            # Main idea explained simply
    THEORY = "theory"              # Formal/technical explanation
    ANALOGY = "analogy"            # Relatable comparison
    VISUALIZATION = "visualization" # Diagram/animation

    # Practice
    CODE_DEMO = "code_demo"        # Live coding
    EXAMPLE = "example"            # Worked example
    EXERCISE = "exercise"          # Hands-on practice
    CHALLENGE = "challenge"        # Advanced exercise

    # Validation
    QUIZ = "quiz"                  # Knowledge check
    REVIEW = "review"              # Self-assessment

    # Business Context
    USE_CASE = "use_case"          # Real-world application
    ROI = "roi"                    # Business value
    CASE_STUDY = "case_study"      # Success story
    METRICS = "metrics"            # Measurable outcomes

    # Closure
    RECAP = "recap"                # Summary of key points
    NEXT_STEPS = "next_steps"      # What to do next
    RESOURCES = "resources"        # Additional materials
    ACTION_ITEMS = "action_items"  # Takeaways to implement

    # Meta
    TRANSITION = "transition"       # Bridge between sections
    BREAK = "break"                # Pause point


class PhaseConfig(BaseModel):
    """Configuration for a lesson phase."""
    phase: LessonPhase
    required: bool = True
    order: int = Field(..., ge=0, description="Position in the lesson flow")
    min_duration_seconds: int = Field(default=30, ge=10)
    max_duration_seconds: int = Field(default=300, le=600)
    slide_count: int = Field(default=1, ge=1, le=5)

    # Content guidance
    prompt_template: Optional[str] = None
    required_elements: List[str] = Field(default_factory=list)  # e.g., ["code_block", "diagram"]
    forbidden_elements: List[str] = Field(default_factory=list)

    # Style guidance
    tone: Optional[str] = None  # e.g., "conversational", "formal", "energetic"
    target_audience: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "phase": "hook",
                "required": True,
                "order": 0,
                "min_duration_seconds": 15,
                "max_duration_seconds": 45,
                "slide_count": 1,
                "prompt_template": "Start with an engaging question or problem statement",
                "tone": "energetic"
            }
        }


class LessonTemplate(BaseModel):
    """Template defining the structure of a single lesson."""
    name: str
    description: str
    phases: List[PhaseConfig]
    total_duration_target_seconds: int = Field(default=600, ge=120, le=3600)

    # Flexibility settings
    allow_phase_reordering: bool = False
    allow_optional_phases: bool = True
    strict_duration_enforcement: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Standard Education Lesson",
                "description": "Hook → Concept → Theory → Code → Recap",
                "total_duration_target_seconds": 600,
                "phases": [
                    {"phase": "hook", "required": True, "order": 0},
                    {"phase": "concept", "required": True, "order": 1},
                    {"phase": "theory", "required": True, "order": 2},
                    {"phase": "code_demo", "required": True, "order": 3},
                    {"phase": "recap", "required": True, "order": 4}
                ]
            }
        }


class CurriculumTemplate(BaseModel):
    """
    Complete curriculum template for a specific context.
    Defines how lessons should be structured.
    """
    id: str
    name: str
    context_type: ContextType
    description: str

    # Lesson templates for different lesson types
    default_lesson_template: LessonTemplate
    lesson_templates: Dict[str, LessonTemplate] = Field(default_factory=dict)
    # e.g., {"intro": intro_template, "deep_dive": deep_dive_template}

    # Course-level structure
    include_course_intro: bool = True
    include_course_conclusion: bool = True
    quiz_frequency: str = Field(default="per_section", pattern="^(per_lesson|per_section|end_only|custom)$")

    # Branding/style
    brand_guidelines: Optional[Dict[str, Any]] = None
    language: str = Field(default="en")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class LessonContent(BaseModel):
    """Actual content of a lesson to be validated."""
    lesson_id: str
    title: str
    slides: List[Dict[str, Any]]  # Each slide with type, content, duration

    # Metadata
    lesson_type: Optional[str] = None  # Maps to lesson_templates key
    section_position: int = Field(default=0, ge=0)
    total_lessons_in_section: int = Field(default=1, ge=1)


class PhaseViolation(BaseModel):
    """A specific violation of the lesson structure."""
    phase: LessonPhase
    violation_type: str  # "missing", "wrong_order", "too_short", "too_long", "wrong_content"
    message: str
    severity: str = Field(default="warning", pattern="^(error|warning|info)$")
    slide_index: Optional[int] = None


class ValidationResult(BaseModel):
    """Result of validating lesson content against a template."""
    is_valid: bool
    score: float = Field(ge=0.0, le=1.0, description="Compliance score 0-1")
    violations: List[PhaseViolation] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    # Phase detection
    detected_phases: List[Dict[str, Any]] = Field(default_factory=list)
    missing_required_phases: List[LessonPhase] = Field(default_factory=list)
    extra_phases: List[LessonPhase] = Field(default_factory=list)


class EnforcementRequest(BaseModel):
    """Request to enforce curriculum structure on content."""
    content: LessonContent
    template_id: Optional[str] = None
    context_type: ContextType = ContextType.EDUCATION

    # Enforcement options
    auto_fix: bool = True  # Automatically restructure content
    strict_mode: bool = False  # Fail on any violation
    preserve_content: bool = True  # Keep original content, just reorganize


class EnforcementResult(BaseModel):
    """Result of enforcing curriculum structure."""
    request_id: str
    success: bool

    # Validation
    original_validation: ValidationResult
    final_validation: Optional[ValidationResult] = None

    # Restructured content
    restructured_content: Optional[LessonContent] = None
    changes_made: List[str] = Field(default_factory=list)

    # Metadata
    template_used: str
    processing_time_ms: int
    error: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
