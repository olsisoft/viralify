"""
Base Agent Classes and Shared State

This module defines the foundational classes for the multi-agent architecture.
All agents inherit from BaseAgent and share a common CourseGenerationState.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union
from datetime import datetime
import os

# Use shared LLM provider for multi-provider support
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    _USE_SHARED_LLM = False


class AgentType(str, Enum):
    """Types of agents in the system"""
    INPUT_VALIDATOR = "input_validator"
    TECHNICAL_REVIEWER = "technical_reviewer"
    PEDAGOGICAL = "pedagogical"
    SCRIPT_WRITER = "script_writer"
    CODE_EXPERT = "code_expert"
    CODE_REVIEWER = "code_reviewer"
    CODE_EXECUTOR = "code_executor"
    DIAGRAM = "diagram"
    QUIZ = "quiz"
    VISUAL = "visual"
    COMPOSITOR = "compositor"


class AgentStatus(str, Enum):
    """Status of agent execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"


@dataclass
class AgentResult:
    """Result from an agent execution"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_needed: bool = False
    retry_prompt: Optional[str] = None


class LessonElementConfig(TypedDict):
    """Configuration for lesson elements from frontend"""
    concept_intro: bool
    diagram_schema: bool
    code_typing: bool
    code_execution: bool
    voiceover_explanation: bool
    curriculum_slide: bool


class QuizConfig(TypedDict):
    """Quiz configuration from frontend"""
    enabled: bool
    frequency: str  # "per_lecture", "per_section", "end_only", "custom"
    custom_interval: Optional[int]
    question_types: List[str]  # "mcq", "true_false", "short_answer"


class CourseStructureConfig(TypedDict):
    """Course structure configuration from frontend"""
    total_duration_minutes: int
    number_of_sections: int
    lectures_per_section: int
    random_structure: bool


class CodeBlockState(TypedDict):
    """State for a code block being processed"""
    raw_code: str
    refined_code: Optional[str]
    language: str
    concept: str
    persona_level: str
    complexity_score: int
    review_status: str  # "pending", "approved", "rejected"
    rejection_reasons: List[str]
    execution_result: Optional[Dict[str, Any]]
    retry_count: int
    max_retries: int


class SlideState(TypedDict):
    """State for a slide being processed"""
    slide_id: str
    slide_type: str
    title: str
    content: Optional[str]
    bullet_points: List[str]
    code_blocks: List[CodeBlockState]
    diagram_description: Optional[str]
    voiceover_text: Optional[str]
    duration: float
    visual_url: Optional[str]


class LectureState(TypedDict):
    """State for a lecture being processed"""
    lecture_id: str
    title: str
    description: str
    objectives: List[str]
    difficulty: str
    duration_seconds: int
    slides: List[SlideState]
    quiz_questions: List[Dict[str, Any]]
    video_url: Optional[str]
    status: str


class ValidationError(TypedDict):
    """A validation error"""
    field: str
    message: str
    severity: str  # "error", "warning", "info"


class CourseGenerationState(TypedDict, total=False):
    """
    Complete state for the Course Generation Multi-Agent System.

    This state flows through all agents in the LangGraph workflow.
    It extends the pedagogical state with additional fields for
    input validation, code generation, and review processes.
    """
    # =========================================================================
    # REQUEST INPUT (from frontend)
    # =========================================================================
    job_id: str
    topic: str
    description: Optional[str]
    profile_id: Optional[str]

    # Profile/Persona
    profile_category: str  # "tech", "business", "creative", etc.
    target_audience: str
    persona_level: str  # "beginner", "intermediate", "advanced", "expert"

    # Difficulty
    difficulty_start: str
    difficulty_end: str

    # Language
    content_language: str  # "en", "fr", "es", etc.
    programming_language: str  # "python", "javascript", etc.

    # Structure
    structure: CourseStructureConfig

    # Lesson Elements (frontend choices)
    lesson_elements: LessonElementConfig

    # Quiz Configuration
    quiz_config: QuizConfig

    # RAG/Documents
    document_ids: List[str]
    rag_context: Optional[str]

    # Visual/Audio
    voice_id: str
    style: str
    typing_speed: str
    include_avatar: bool
    avatar_id: Optional[str]

    # =========================================================================
    # INPUT VALIDATION (InputValidatorAgent)
    # =========================================================================
    input_validated: bool
    input_validation_errors: List[ValidationError]
    missing_required_fields: List[str]

    # =========================================================================
    # TECHNICAL REVIEW (TechnicalReviewerAgent)
    # =========================================================================
    config_reviewed: bool
    prompt_enrichments: Dict[str, str]  # Field -> enrichment to add to prompts
    config_warnings: List[str]
    config_suggestions: List[str]

    # =========================================================================
    # PEDAGOGICAL ANALYSIS (PedagogicalAgent)
    # =========================================================================
    detected_persona: str
    topic_complexity: str
    requires_code: bool
    requires_diagrams: bool
    requires_hands_on: bool
    domain_keywords: List[str]
    content_preferences: Dict[str, float]
    recommended_elements: List[str]
    rag_images: List[Dict[str, Any]]

    # =========================================================================
    # COURSE OUTLINE
    # =========================================================================
    outline: Optional[Dict[str, Any]]
    outline_validated: bool

    # =========================================================================
    # LECTURE PROCESSING
    # =========================================================================
    current_lecture_index: int
    lectures: List[LectureState]

    # =========================================================================
    # CODE PROCESSING (CodeExpertAgent + CodeReviewerAgent)
    # =========================================================================
    current_code_block: Optional[CodeBlockState]
    code_blocks_processed: int
    code_blocks_approved: int
    code_blocks_rejected: int
    code_expert_prompt: str  # The enriched prompt for code generation

    # =========================================================================
    # EXECUTION (CodeExecutorAgent)
    # =========================================================================
    sandbox_enabled: bool
    execution_timeout: int

    # =========================================================================
    # OUTPUT
    # =========================================================================
    output_videos: List[str]
    output_zip_url: Optional[str]

    # =========================================================================
    # WORKFLOW CONTROL
    # =========================================================================
    current_agent: str
    agent_history: List[Dict[str, Any]]  # Track agent executions
    errors: List[str]
    warnings: List[str]

    # =========================================================================
    # TIMESTAMPS
    # =========================================================================
    started_at: str
    completed_at: Optional[str]


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.

    All agents must implement the `process` method which takes the
    current state and returns an updated state or AgentResult.
    """

    # Default model tier for this agent type. Subclasses can override.
    MODEL_TIER: str = "fast"

    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.name = agent_type.value
        if _USE_SHARED_LLM:
            self.client = get_llm_client()
            self.model = get_model_name(self.MODEL_TIER)
        else:
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=120.0,
                max_retries=2
            )
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"

    def log(self, message: str):
        """Log a message with agent name prefix"""
        if self.debug:
            print(f"[{self.name.upper()}] {message}", flush=True)

    @abstractmethod
    async def process(self, state: CourseGenerationState) -> CourseGenerationState:
        """
        Process the current state and return updated state.

        Args:
            state: The current course generation state

        Returns:
            Updated state with this agent's contributions
        """
        pass

    def validate_required_fields(
        self,
        state: CourseGenerationState,
        required_fields: List[str]
    ) -> List[ValidationError]:
        """Check that required fields are present in state"""
        errors = []
        for field in required_fields:
            if field not in state or state.get(field) is None:
                errors.append({
                    "field": field,
                    "message": f"Required field '{field}' is missing",
                    "severity": "error"
                })
        return errors

    def add_to_history(
        self,
        state: CourseGenerationState,
        status: AgentStatus,
        result: Optional[AgentResult] = None
    ) -> CourseGenerationState:
        """Add this agent's execution to the history"""
        history = state.get("agent_history", [])
        history.append({
            "agent": self.name,
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result.data if result else None,
            "errors": result.errors if result else [],
        })
        state["agent_history"] = history
        state["current_agent"] = self.name
        return state


def create_initial_state(
    job_id: str,
    topic: str,
    **kwargs
) -> CourseGenerationState:
    """
    Create an initial CourseGenerationState from request parameters.

    Args:
        job_id: Unique job identifier
        topic: Course topic
        **kwargs: Additional fields to include

    Returns:
        Initialized CourseGenerationState
    """
    state: CourseGenerationState = {
        "job_id": job_id,
        "topic": topic,
        "started_at": datetime.utcnow().isoformat(),
        "agent_history": [],
        "errors": [],
        "warnings": [],
        "lectures": [],
        "output_videos": [],
        "input_validated": False,
        "config_reviewed": False,
        "outline_validated": False,
        "current_lecture_index": 0,
        "code_blocks_processed": 0,
        "code_blocks_approved": 0,
        "code_blocks_rejected": 0,
        "sandbox_enabled": True,
        "execution_timeout": 30,
    }

    # Add any additional fields from kwargs
    for key, value in kwargs.items():
        if key in CourseGenerationState.__annotations__:
            state[key] = value

    return state
