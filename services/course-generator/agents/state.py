"""
Hierarchical State Definitions for Course Generation

This module defines separate states for each subgraph in the hierarchical
architecture, following the "Async-Ready" pattern.

Architecture:
- OrchestratorState: Lightweight global state for the main graph
- PlanningState: State for curriculum planning subgraph
- ProductionState: State for media production subgraph (per-lecture)

Benefits:
- Isolation of concerns: Planning bugs don't affect production logic
- Granular error handling: Each subgraph has its own recovery loops
- Testability: Subgraphs can be tested in isolation
- Future-proof: Ready for async/event-driven scaling
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union


# =============================================================================
# ENUMS
# =============================================================================

class RecoveryStrategy(str, Enum):
    """Strategy for recovering from media generation failures"""
    RETRY = "retry"  # Simple retry
    SIMPLIFY_SCRIPT = "simplify_script"  # Reduce complexity
    REDUCE_ANIMATIONS = "reduce_animations"  # Remove animations
    SKIP = "skip"  # Skip this lecture


class ProductionStatus(str, Enum):
    """Status of a lecture in production"""
    PENDING = "pending"
    WRITING_SCRIPT = "writing_script"
    GENERATING_CODE = "generating_code"
    REVIEWING_CODE = "reviewing_code"
    REFINING_CODE = "refining_code"
    GENERATING_MEDIA = "generating_media"
    SIMPLIFYING = "simplifying"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanningStatus(str, Enum):
    """Status of curriculum planning"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    GENERATING_OUTLINE = "generating_outline"
    ENFORCING_STRUCTURE = "enforcing_structure"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# SHARED TYPES
# =============================================================================

class CodeBlockInfo(TypedDict):
    """Information about a code block to generate"""
    concept: str
    language: str
    description: str
    persona_level: str
    complexity_target: int  # 1-5


class GeneratedCodeBlock(TypedDict):
    """A code block that has been generated and reviewed"""
    concept: str
    language: str
    code: str
    explanation: str
    expected_output: Optional[str]
    review_status: str  # "approved", "rejected", "needs_refinement"
    quality_score: int  # 1-10
    retry_count: int


class LecturePlan(TypedDict):
    """Plan for a single lecture (from outline)"""
    lecture_id: str
    section_id: str
    title: str
    description: str
    objectives: List[str]
    difficulty: str
    duration_seconds: int
    position: int  # Position in course (1-based)
    total_lectures: int
    # Section context
    section_title: str
    section_description: str
    # Course context
    course_title: str
    target_audience: str
    # Content elements
    requires_code: bool
    requires_diagrams: bool
    code_blocks: List[CodeBlockInfo]
    diagram_descriptions: List[str]
    voiceover_script: Optional[str]
    # Metadata
    lesson_elements: List[str]  # Element IDs to include


class MediaResult(TypedDict):
    """Result of media generation for a lecture"""
    lecture_id: str
    video_url: Optional[str]
    thumbnail_url: Optional[str]
    duration_seconds: float
    error: Optional[str]
    job_id: Optional[str]


# =============================================================================
# PLANNING STATE (Subgraph 1)
# =============================================================================

class PlanningState(TypedDict, total=False):
    """
    State for the Planning Subgraph.

    Handles:
    - Pedagogical analysis
    - Curriculum structure generation
    - Element suggestion
    - Outline validation

    Isolated from production concerns.
    """
    # === INPUT ===
    topic: str
    description: Optional[str]
    profile_category: str  # "tech", "business", etc.
    difficulty_start: str
    difficulty_end: str
    content_language: str
    programming_language: Optional[str]
    target_audience: str

    # Structure config
    total_duration_minutes: int
    number_of_sections: int
    lectures_per_section: int

    # Content config
    lesson_elements_enabled: Dict[str, bool]
    quiz_enabled: bool
    quiz_frequency: str

    # RAG context
    rag_context: Optional[str]
    document_ids: List[str]

    # === PEDAGOGICAL ANALYSIS OUTPUT ===
    detected_persona: str
    topic_complexity: str  # "basic", "intermediate", "advanced", "expert"
    domain_keywords: List[str]
    requires_code: bool
    requires_diagrams: bool
    requires_hands_on: bool
    content_preferences: Dict[str, float]  # Element type -> weight
    recommended_elements: List[str]
    rag_images: List[Dict[str, Any]]

    # === OUTLINE OUTPUT ===
    outline: Optional[Dict[str, Any]]  # Serialized CourseOutline
    sections: List[Dict[str, Any]]  # Serialized sections with lectures
    lecture_plans: List[LecturePlan]  # Flattened lecture plans
    total_lectures: int

    # === WORKFLOW ===
    status: PlanningStatus
    errors: List[str]
    warnings: List[str]


# =============================================================================
# PRODUCTION STATE (Subgraph 2 - Per Lecture)
# =============================================================================

class ProductionState(TypedDict, total=False):
    """
    State for the Production Subgraph.

    Handles a SINGLE lecture's production:
    - Script writing
    - Code generation & review loop
    - Media generation
    - Error recovery

    This subgraph is invoked once per lecture.
    """
    # === INPUT (from Planning) ===
    lecture_plan: LecturePlan
    content_preferences: Dict[str, float]  # From pedagogical analysis
    profile_category: str
    content_language: str
    programming_language: Optional[str]

    # Course context (for building proper prompts)
    course_title: str
    target_audience: str
    section_title: str
    section_description: str

    # Lesson elements configuration
    lesson_elements: Dict[str, bool]

    # Voice/Visual config (passed from orchestrator)
    voice_id: str
    style: str
    typing_speed: str
    include_avatar: bool
    avatar_id: Optional[str]

    # RAG context from source documents (passed to presentation-generator)
    rag_context: Optional[str]
    document_ids: List[str]

    # === SCRIPT STATE ===
    voiceover_script: Optional[str]
    script_version: int  # Incremented on simplification
    script_complexity_score: int  # 1-10, reduced on simplification

    # === CODE GENERATION STATE ===
    pending_code_blocks: List[CodeBlockInfo]
    current_code_block: Optional[CodeBlockInfo]
    generated_code_blocks: List[GeneratedCodeBlock]
    code_review_iterations: int  # Track for this lecture
    max_code_iterations: int  # Default: 3

    # === MEDIA GENERATION STATE ===
    media_job_id: Optional[str]
    media_generation_attempts: int
    max_media_attempts: int  # Default: 3
    last_media_error: Optional[str]

    # === RECOVERY STATE ===
    recovery_strategy: Optional[RecoveryStrategy]
    recovery_attempts: int
    max_recovery_attempts: int  # Default: 2
    simplification_applied: bool
    animations_disabled: bool

    # === OUTPUT ===
    media_result: Optional[MediaResult]

    # === WORKFLOW ===
    status: ProductionStatus
    errors: List[str]
    warnings: List[str]
    started_at: str
    completed_at: Optional[str]


# =============================================================================
# ORCHESTRATOR STATE (Main Graph)
# =============================================================================

class OrchestratorState(TypedDict, total=False):
    """
    Lightweight state for the Main Orchestrator Graph.

    This is the "chef d'orchestre" that:
    - Validates input
    - Invokes Planning subgraph
    - Iterates over lectures, invoking Production subgraph for each
    - Packages final output

    Keeps minimal state - delegates details to subgraphs.
    """
    # === JOB TRACKING ===
    job_id: str
    started_at: str
    completed_at: Optional[str]

    # === INPUT (from API request) ===
    # These are passed to subgraphs as needed
    topic: str
    description: Optional[str]
    profile_category: str
    difficulty_start: str
    difficulty_end: str
    content_language: str
    programming_language: Optional[str]
    target_audience: str

    # Structure
    total_duration_minutes: int
    number_of_sections: int
    lectures_per_section: int

    # Content
    lesson_elements_enabled: Dict[str, bool]
    quiz_enabled: bool
    quiz_frequency: str

    # RAG
    rag_context: Optional[str]
    document_ids: List[str]

    # Voice/Visual
    voice_id: str
    style: str
    typing_speed: str
    include_avatar: bool
    avatar_id: Optional[str]

    # === VALIDATION ===
    input_validated: bool
    validation_errors: List[str]

    # === PLANNING OUTPUT (from Planning subgraph) ===
    planning_completed: bool
    outline: Optional[Dict[str, Any]]
    lecture_plans: List[LecturePlan]
    total_lectures: int
    content_preferences: Dict[str, float]

    # === COHERENCE CHECK (Phase 2) ===
    coherence_checked: bool
    coherence_score: float  # 0-100 score
    coherence_issues: List[Dict[str, Any]]  # List of issues found

    # === PRODUCTION TRACKING ===
    current_lecture_index: int
    lectures_completed: List[str]  # lecture_ids
    lectures_failed: List[Dict[str, Any]]  # {lecture_id, error, attempts}
    lectures_skipped: List[str]  # lecture_ids

    # Media results (aggregated from production)
    video_urls: Dict[str, str]  # lecture_id -> video_url

    # === OUTPUT ===
    output_zip_url: Optional[str]
    final_status: str  # "success", "partial", "failed"

    # === WORKFLOW ===
    current_stage: str  # "validating", "planning", "producing", "packaging", "done"
    errors: List[str]
    warnings: List[str]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_planning_state_from_orchestrator(
    orchestrator_state: OrchestratorState
) -> PlanningState:
    """
    Create a PlanningState from OrchestratorState.

    Called when entering the Planning subgraph.
    """
    return PlanningState(
        topic=orchestrator_state.get("topic", ""),
        description=orchestrator_state.get("description"),
        profile_category=orchestrator_state.get("profile_category", "education"),
        difficulty_start=orchestrator_state.get("difficulty_start", "beginner"),
        difficulty_end=orchestrator_state.get("difficulty_end", "intermediate"),
        content_language=orchestrator_state.get("content_language", "en"),
        programming_language=orchestrator_state.get("programming_language"),
        target_audience=orchestrator_state.get("target_audience", "general learners"),
        # Extract structure config - try nested dict first, then flat keys
        total_duration_minutes=orchestrator_state.get("structure", {}).get("total_duration_minutes", orchestrator_state.get("total_duration_minutes", 60)),
        number_of_sections=orchestrator_state.get("structure", {}).get("number_of_sections", orchestrator_state.get("number_of_sections", 4)),
        lectures_per_section=orchestrator_state.get("structure", {}).get("lectures_per_section", orchestrator_state.get("lectures_per_section", 3)),
        lesson_elements_enabled=orchestrator_state.get("lesson_elements", orchestrator_state.get("lesson_elements_enabled", {})),
        quiz_enabled=orchestrator_state.get("quiz_enabled", True),
        quiz_frequency=orchestrator_state.get("quiz_frequency", "per_section"),
        rag_context=orchestrator_state.get("rag_context"),
        document_ids=orchestrator_state.get("document_ids", []),
        # Initialize workflow
        status=PlanningStatus.PENDING,
        errors=[],
        warnings=[],
        lecture_plans=[],
        sections=[],
    )


def create_production_state_for_lecture(
    orchestrator_state: OrchestratorState,
    lecture_plan: LecturePlan,
) -> ProductionState:
    """
    Create a ProductionState for a single lecture.

    Called for each lecture during the production phase.
    """
    # Get outline for course title
    outline = orchestrator_state.get("outline", {})
    course_title = outline.get("title", orchestrator_state.get("topic", ""))

    # Get section info from lecture_plan (should be populated by planning graph)
    section_title = lecture_plan.get("section_title", "")
    section_description = lecture_plan.get("section_description", "")

    return ProductionState(
        lecture_plan=lecture_plan,
        content_preferences=orchestrator_state.get("content_preferences", {}),
        profile_category=orchestrator_state.get("profile_category", "education"),
        content_language=orchestrator_state.get("content_language", "en"),
        programming_language=orchestrator_state.get("programming_language"),
        # Course context for building prompts
        course_title=course_title,
        target_audience=orchestrator_state.get("target_audience", "general learners"),
        section_title=section_title,
        section_description=section_description,
        # Lesson elements configuration
        lesson_elements=orchestrator_state.get("lesson_elements_enabled", {
            "concept_intro": True,
            "diagram_schema": True,
            "code_typing": True,
            "code_execution": False,
            "voiceover_explanation": True,
            "curriculum_slide": True,
        }),
        # Voice/Visual config
        voice_id=orchestrator_state.get("voice_id", "default"),
        style=orchestrator_state.get("style", "modern"),
        typing_speed=orchestrator_state.get("typing_speed", "natural"),
        include_avatar=orchestrator_state.get("include_avatar", False),
        avatar_id=orchestrator_state.get("avatar_id"),
        # Initialize script state
        voiceover_script=lecture_plan.get("voiceover_script"),
        script_version=1,
        script_complexity_score=5,  # Default mid-complexity
        # Initialize code state
        pending_code_blocks=lecture_plan.get("code_blocks", []),
        current_code_block=None,
        generated_code_blocks=[],
        code_review_iterations=0,
        max_code_iterations=3,
        # Initialize media state
        media_job_id=None,
        media_generation_attempts=0,
        max_media_attempts=3,
        last_media_error=None,
        # Initialize recovery state
        recovery_strategy=None,
        recovery_attempts=0,
        max_recovery_attempts=2,
        simplification_applied=False,
        animations_disabled=False,
        # Initialize output
        media_result=None,
        # Initialize workflow
        status=ProductionStatus.PENDING,
        errors=[],
        warnings=[],
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
        # RAG context from source documents (passed to presentation-generator)
        rag_context=orchestrator_state.get("rag_context"),
        document_ids=orchestrator_state.get("document_ids", []),
    )


def merge_production_result_to_orchestrator(
    orchestrator_state: OrchestratorState,
    production_state: ProductionState,
) -> OrchestratorState:
    """
    Merge ProductionState result back into OrchestratorState.

    Called after each lecture production completes.
    """
    lecture_plan = production_state.get("lecture_plan", {})
    lecture_id = lecture_plan.get("lecture_id", "unknown")
    status = production_state.get("status", ProductionStatus.FAILED)
    media_result = production_state.get("media_result")

    # Copy current state
    new_state = dict(orchestrator_state)

    if status == ProductionStatus.COMPLETED and media_result:
        # Success
        new_state["lectures_completed"] = orchestrator_state.get("lectures_completed", []) + [lecture_id]
        if media_result.get("video_url"):
            video_urls = dict(orchestrator_state.get("video_urls", {}))
            video_urls[lecture_id] = media_result["video_url"]
            new_state["video_urls"] = video_urls

    elif status == ProductionStatus.SKIPPED:
        # Skipped
        new_state["lectures_skipped"] = orchestrator_state.get("lectures_skipped", []) + [lecture_id]

    else:
        # Failed
        failed_entry = {
            "lecture_id": lecture_id,
            "error": production_state.get("last_media_error") or "Unknown error",
            "attempts": production_state.get("media_generation_attempts", 0),
        }
        new_state["lectures_failed"] = orchestrator_state.get("lectures_failed", []) + [failed_entry]

    # Add any warnings/errors
    new_state["warnings"] = orchestrator_state.get("warnings", []) + production_state.get("warnings", [])

    # Move to next lecture
    new_state["current_lecture_index"] = orchestrator_state.get("current_lecture_index", 0) + 1

    return OrchestratorState(**new_state)


def merge_planning_result_to_orchestrator(
    orchestrator_state: OrchestratorState,
    planning_state: PlanningState,
) -> OrchestratorState:
    """
    Merge PlanningState result back into OrchestratorState.

    Called after planning subgraph completes.
    """
    new_state = dict(orchestrator_state)

    if planning_state.get("status") == PlanningStatus.COMPLETED:
        new_state["planning_completed"] = True
        new_state["outline"] = planning_state.get("outline")
        new_state["lecture_plans"] = planning_state.get("lecture_plans", [])
        new_state["total_lectures"] = planning_state.get("total_lectures", 0)
        new_state["content_preferences"] = planning_state.get("content_preferences", {})
    else:
        new_state["planning_completed"] = False
        new_state["errors"] = orchestrator_state.get("errors", []) + planning_state.get("errors", [])

    new_state["warnings"] = orchestrator_state.get("warnings", []) + planning_state.get("warnings", [])

    return OrchestratorState(**new_state)


def create_orchestrator_state(
    job_id: str,
    **kwargs
) -> OrchestratorState:
    """
    Create initial OrchestratorState from API request parameters.

    Args:
        job_id: Unique job identifier
        **kwargs: Request fields

    Returns:
        Initialized OrchestratorState
    """
    return OrchestratorState(
        job_id=job_id,
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
        topic=kwargs.get("topic", ""),
        description=kwargs.get("description"),
        profile_category=kwargs.get("profile_category", "education"),
        difficulty_start=kwargs.get("difficulty_start", "beginner"),
        difficulty_end=kwargs.get("difficulty_end", "intermediate"),
        content_language=kwargs.get("content_language", "en"),
        programming_language=kwargs.get("programming_language"),
        target_audience=kwargs.get("target_audience", "general learners"),
        # Extract structure config - try nested dict first, then flat keys
        total_duration_minutes=kwargs.get("structure", {}).get("total_duration_minutes", kwargs.get("total_duration_minutes", 60)),
        number_of_sections=kwargs.get("structure", {}).get("number_of_sections", kwargs.get("number_of_sections", 4)),
        lectures_per_section=kwargs.get("structure", {}).get("lectures_per_section", kwargs.get("lectures_per_section", 3)),
        lesson_elements_enabled=kwargs.get("lesson_elements", kwargs.get("lesson_elements_enabled", {})),
        quiz_enabled=kwargs.get("quiz_enabled", True),
        quiz_frequency=kwargs.get("quiz_frequency", "per_section"),
        rag_context=kwargs.get("rag_context"),
        document_ids=kwargs.get("document_ids", []),
        voice_id=kwargs.get("voice_id", "default"),
        style=kwargs.get("style", "modern"),
        typing_speed=kwargs.get("typing_speed", "natural"),
        include_avatar=kwargs.get("include_avatar", False),
        avatar_id=kwargs.get("avatar_id"),
        # Initialize validation
        input_validated=False,
        validation_errors=[],
        # Initialize planning
        planning_completed=False,
        outline=None,
        lecture_plans=[],
        total_lectures=0,
        content_preferences={},
        # Initialize production
        current_lecture_index=0,
        lectures_completed=[],
        lectures_failed=[],
        lectures_skipped=[],
        video_urls={},
        # Initialize output
        output_zip_url=None,
        final_status="pending",
        # Initialize workflow
        current_stage="validating",
        errors=[],
        warnings=[],
    )
