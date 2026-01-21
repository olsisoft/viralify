"""
Pedagogical Agent State Definition

Defines the TypedDict state that flows through the LangGraph agent.
"""
from typing import Any, Dict, List, Optional, TypedDict

from models.course_models import (
    CourseOutline,
    ProfileCategory,
    DifficultyLevel,
)


class ContentPreferences(TypedDict):
    """Weights for different content types based on profile"""
    code_weight: float  # 0-1, how much code to include
    diagram_weight: float  # 0-1, how many diagrams
    demo_weight: float  # 0-1, live demonstrations
    theory_weight: float  # 0-1, theoretical content
    case_study_weight: float  # 0-1, real-world examples


class RAGImage(TypedDict):
    """Reference to an image/diagram from RAG documents"""
    document_id: str
    image_path: str
    description: str
    suggested_use: str  # e.g., "architecture_diagram", "process_flow"


class QuizPlacement(TypedDict):
    """Quiz placement and configuration"""
    lecture_id: str
    quiz_type: str  # "section_review", "lecture_check", "final_assessment"
    difficulty: str
    question_count: int
    topics_covered: List[str]


class ValidationResult(TypedDict):
    """Results from structure and language validation"""
    is_valid: bool
    warnings: List[str]
    suggestions: List[str]
    pedagogical_score: float  # 0-100


class PedagogicalAgentState(TypedDict, total=False):
    """
    Complete state for the Pedagogical Agent.

    This state flows through all nodes in the LangGraph workflow.
    """
    # Input fields (set at start)
    topic: str
    description: Optional[str]
    profile_category: ProfileCategory
    difficulty_start: DifficultyLevel
    difficulty_end: DifficultyLevel
    target_language: str  # Content language (en, fr, es, etc.)
    target_audience: str
    structure_sections: int
    structure_lectures_per_section: int
    total_duration_minutes: int
    rag_context: Optional[str]
    document_ids: List[str]
    quiz_enabled: bool
    quiz_frequency: str

    # Analysis results (set by analyze_context node)
    detected_persona: str  # e.g., "developer", "architect", "manager", "student"
    topic_complexity: str  # "basic", "intermediate", "advanced", "expert"
    requires_code: bool
    requires_diagrams: bool
    requires_hands_on: bool
    domain_keywords: List[str]

    # Content adaptation (set by adapt_for_profile node)
    content_preferences: ContentPreferences
    recommended_elements: List[str]  # Suggested lesson element IDs

    # RAG integration (set by fetch_rag_images node)
    rag_images: List[RAGImage]
    rag_diagrams_available: bool

    # Element mapping (set by suggest_elements node)
    element_mapping: Dict[str, List[str]]  # lecture_id -> list of element IDs

    # Generated outline (set by generate_outline node)
    outline: Optional[CourseOutline]
    outline_json: Optional[Dict[str, Any]]  # Raw JSON for modifications

    # Quiz planning (set by plan_quizzes node)
    quiz_placement: List[QuizPlacement]
    quiz_total_count: int

    # Validation (set by validate_* nodes)
    language_validated: bool
    structure_validated: bool
    validation_result: ValidationResult

    # Final output (set by finalize_plan node)
    final_outline: Optional[CourseOutline]
    generation_metadata: Dict[str, Any]

    # Error handling
    errors: List[str]
    current_node: str
