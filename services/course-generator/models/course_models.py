"""
Course Generator Data Models

Defines all Pydantic models for the course generation service.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field
import uuid

if TYPE_CHECKING:
    from .lesson_elements import Quiz


class DifficultyLevel(str, Enum):
    """Course difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    VERY_ADVANCED = "very_advanced"
    EXPERT = "expert"


class ProfileCategory(str, Enum):
    """Categories of creator profiles for contextual questions"""
    BUSINESS = "business"
    TECH = "tech"
    CREATIVE = "creative"
    HEALTH = "health"
    EDUCATION = "education"
    LIFESTYLE = "lifestyle"


class ContextQuestion(BaseModel):
    """A contextual question for course creation"""
    id: str = Field(..., description="Unique question identifier")
    question: str = Field(..., description="The question text")
    type: str = Field(default="select", description="Question type: select, text, multiselect")
    options: Optional[List[str]] = Field(None, description="Options for select/multiselect types")
    placeholder: Optional[str] = Field(None, description="Placeholder for text inputs")
    required: bool = Field(default=True, description="Whether the question is required")


class CourseContext(BaseModel):
    """Enriched context for course generation based on profile and user answers"""

    # Category detected from profile
    category: ProfileCategory = Field(..., description="Profile category")

    # Relevant profile data
    profile_niche: str = Field(..., description="Creator's niche")
    profile_tone: str = Field(default="educational", description="Communication tone")
    profile_audience_level: str = Field(default="intermediate", description="Audience skill level")
    profile_language_level: str = Field(default="moderate", description="Language complexity")
    profile_primary_goal: str = Field(default="educate", description="Primary content goal")
    profile_audience_description: str = Field(default="", description="Target audience description")

    # User answers to contextual questions
    context_answers: Dict[str, str] = Field(default_factory=dict, description="Answers to context questions")

    # Optional additional context
    specific_tools: Optional[str] = Field(None, description="Specific tools/technologies to cover")
    practical_focus: Optional[str] = Field(None, description="theoretical, hands-on, or mixed")
    expected_outcome: Optional[str] = Field(None, description="Expected learning outcome")


class ContextQuestionsRequest(BaseModel):
    """Request to get contextual questions"""
    category: ProfileCategory = Field(..., description="Profile category")
    topic: Optional[str] = Field(None, description="Course topic for AI-generated questions")
    generate_ai_questions: bool = Field(default=False, description="Whether to generate AI questions")


class ContextQuestionsResponse(BaseModel):
    """Response with contextual questions"""
    category: ProfileCategory
    base_questions: List[ContextQuestion] = Field(default_factory=list)
    ai_questions: List[ContextQuestion] = Field(default_factory=list)


class CourseStage(str, Enum):
    """Stages of course generation pipeline"""
    QUEUED = "queued"
    PLANNING = "planning"
    GENERATING_LECTURES = "generating_lectures"
    COMPILING = "compiling"
    COMPLETED = "completed"
    FAILED = "failed"


class LessonElementConfig(BaseModel):
    """
    Legacy configuration for lesson elements.
    Use AdaptiveLessonElementConfig from lesson_elements.py for new implementations.
    """
    concept_intro: bool = Field(default=True, description="Include concept introduction slide")
    diagram_schema: bool = Field(default=True, description="Include diagram/schema explanations")
    code_typing: bool = Field(default=True, description="Show typing animation for code")
    code_execution: bool = Field(default=False, description="Execute code and show output")
    voiceover_explanation: bool = Field(default=True, description="Include voiceover during code")
    curriculum_slide: bool = Field(default=True, description="Always include curriculum slide (readonly)")

    # New: Quiz configuration (required for all courses)
    quiz_enabled: bool = Field(default=True, description="Enable quizzes (required)")

    def to_dict_for_category(self, category: str) -> Dict[str, bool]:
        """Convert to dict with category-aware defaults"""
        base = self.model_dump()
        # Add category context for presentation generator
        base["_category"] = category
        return base


class CourseStructureConfig(BaseModel):
    """Configuration for course structure"""
    total_duration_minutes: int = Field(default=60, ge=10, le=600, description="Total duration in minutes")
    number_of_sections: int = Field(default=5, ge=1, le=20, description="Number of sections")
    lectures_per_section: int = Field(default=3, ge=1, le=10, description="Lectures per section")
    random_structure: bool = Field(default=False, description="Let AI decide structure")


class Lecture(BaseModel):
    """A single lecture in a section"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = Field(..., description="Lecture title")
    description: str = Field(default="", description="Brief description")
    objectives: List[str] = Field(default_factory=list, description="Learning objectives")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.INTERMEDIATE)
    duration_seconds: int = Field(default=300, description="Target duration in seconds")
    order: int = Field(default=0, description="Order within section")

    # Generation status
    status: str = Field(default="pending", description="pending, generating, completed, failed, retrying")
    presentation_job_id: Optional[str] = Field(None, description="ID of presentation-generator job")
    video_url: Optional[str] = Field(None, description="Generated video URL")
    error: Optional[str] = Field(None, description="Error if failed")

    # Progress tracking
    progress_percent: float = Field(default=0.0, description="Current generation progress 0-100")
    current_stage: Optional[str] = Field(None, description="Current stage: script, slides, voiceover, animations, composing")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Quiz for this lecture (generated if quiz_config enabled)
    quiz: Optional[Any] = Field(None, description="Quiz for this lecture")


class Section(BaseModel):
    """A section containing multiple lectures"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = Field(..., description="Section title")
    description: str = Field(default="", description="Section description")
    order: int = Field(default=0, description="Order within course")
    lectures: List[Lecture] = Field(default_factory=list, description="Lectures in this section")

    # Quiz for this section (generated if quiz_config frequency is PER_SECTION)
    quiz: Optional[Any] = Field(None, description="Quiz for this section")

    @property
    def lecture_count(self) -> int:
        return len(self.lectures)

    @property
    def completed_lectures(self) -> int:
        return len([l for l in self.lectures if l.status == "completed"])


class CourseOutline(BaseModel):
    """Complete course outline/curriculum"""
    title: str = Field(..., description="Course title")
    description: str = Field(..., description="Course description")
    target_audience: str = Field(default="", description="Target audience")
    category: Optional[ProfileCategory] = Field(None, description="Course category")
    context_summary: Optional[str] = Field(None, description="Summary of course context")
    language: str = Field(default="en", description="Course language code (e.g., 'en', 'fr')")
    difficulty_start: DifficultyLevel = Field(default=DifficultyLevel.BEGINNER)
    difficulty_end: DifficultyLevel = Field(default=DifficultyLevel.INTERMEDIATE)
    total_duration_minutes: int = Field(default=60, description="Total estimated duration")
    sections: List[Section] = Field(default_factory=list, description="Course sections")

    # Final course quiz (generated if quiz_config frequency is END_OF_COURSE)
    final_quiz: Optional[Any] = Field(None, description="Final course assessment quiz")

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def total_lectures(self) -> int:
        return sum(s.lecture_count for s in self.sections)

    @property
    def completed_lectures(self) -> int:
        return sum(s.completed_lectures for s in self.sections)


class PreviewOutlineResponse(BaseModel):
    """Response for preview outline endpoint - includes RAG context for optimization"""
    outline: CourseOutline = Field(..., description="Generated course outline")
    rag_context: Optional[str] = Field(None, description="Pre-fetched RAG context to pass to generate")

    class Config:
        # Allow arbitrary types for forward compatibility
        arbitrary_types_allowed = True


class QuizFrequencyConfig(str, Enum):
    """How often quizzes should appear"""
    PER_LECTURE = "per_lecture"
    PER_SECTION = "per_section"
    END_OF_COURSE = "end_of_course"
    CUSTOM = "custom"


class QuizConfigRequest(BaseModel):
    """Quiz configuration in request"""
    enabled: bool = Field(default=True, description="Quizzes are always enabled")
    frequency: QuizFrequencyConfig = Field(default=QuizFrequencyConfig.PER_SECTION)
    custom_frequency: Optional[int] = Field(None, description="Every N lectures if frequency=custom")
    questions_per_quiz: int = Field(default=5, ge=1, le=20)
    passing_score: int = Field(default=70, ge=0, le=100)
    show_explanations: bool = Field(default=True)


class AdaptiveElementsRequest(BaseModel):
    """Adaptive lesson elements configuration in request"""
    # Common elements (required ones can't be disabled)
    concept_intro: bool = Field(default=True)
    voiceover: bool = Field(default=True)
    curriculum_slide: bool = Field(default=True)
    conclusion: bool = Field(default=True)

    # Category-specific elements (keys are element IDs)
    category_elements: Dict[str, bool] = Field(default_factory=dict)

    # Let AI suggest elements
    use_ai_suggestions: bool = Field(default=True)


class GenerateCourseRequest(BaseModel):
    """Request to generate a complete course"""
    profile_id: str = Field(..., description="Creator profile ID")
    topic: str = Field(..., description="Course topic/subject", min_length=5)
    description: Optional[str] = Field(None, description="Additional description/context")

    # Difficulty range
    difficulty_start: DifficultyLevel = Field(default=DifficultyLevel.BEGINNER)
    difficulty_end: DifficultyLevel = Field(default=DifficultyLevel.INTERMEDIATE)

    # Structure configuration
    structure: CourseStructureConfig = Field(default_factory=CourseStructureConfig)

    # Legacy lesson elements configuration (for backwards compatibility)
    lesson_elements: LessonElementConfig = Field(default_factory=LessonElementConfig)

    # NEW: Adaptive lesson elements (category-aware)
    adaptive_elements: Optional[AdaptiveElementsRequest] = Field(
        None, description="Adaptive lesson elements based on category"
    )

    # NEW: Quiz configuration (required for all courses)
    quiz_config: QuizConfigRequest = Field(default_factory=QuizConfigRequest)

    # Course context (replaces simple 'language' field)
    context: Optional[CourseContext] = Field(None, description="Enriched course context from profile and questions")

    # Presentation options (passed to presentation-generator)
    voice_id: str = Field(default="alloy", description="Voice ID for narration")
    style: str = Field(default="dark", description="Visual style")
    typing_speed: str = Field(default="natural", description="Typing animation speed")
    include_avatar: bool = Field(default=False, description="Include avatar presenter")
    avatar_id: Optional[str] = Field(None, description="Avatar ID if include_avatar is True")

    # Optional pre-approved outline (from preview)
    approved_outline: Optional[CourseOutline] = Field(None, description="Pre-approved outline from preview")

    # RAG document references (Phase 2)
    document_ids: List[str] = Field(default_factory=list, description="IDs of uploaded documents to use as source")
    # OPTIMIZED: Pre-fetched RAG context from preview (avoids double fetch)
    rag_context: Optional[str] = Field(None, description="Pre-fetched RAG context from preview (set by server)")

    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": "tech-expert-001",
                "topic": "Python for Data Science",
                "description": "A comprehensive course on Python programming for data analysis",
                "difficulty_start": "beginner",
                "difficulty_end": "advanced",
                "structure": {
                    "total_duration_minutes": 120,
                    "number_of_sections": 5,
                    "lectures_per_section": 3,
                    "random_structure": False
                },
                "lesson_elements": {
                    "concept_intro": True,
                    "diagram_schema": True,
                    "code_typing": True,
                    "code_execution": True,
                    "voiceover_explanation": True
                },
                "context": {
                    "category": "tech",
                    "profile_niche": "Technology",
                    "profile_tone": "educational",
                    "profile_audience_level": "beginner",
                    "profile_language_level": "moderate",
                    "profile_primary_goal": "educate",
                    "profile_audience_description": "Developers learning data science",
                    "context_answers": {
                        "tech_domain": "Data/IA",
                        "specific_tools": "Python, Pandas, NumPy",
                        "practical_focus": "Très pratique (projets)"
                    }
                },
                "voice_id": "alloy",
                "style": "dark"
            }
        }


class PreviewOutlineRequest(BaseModel):
    """Request to preview course outline before generation"""
    profile_id: Optional[str] = Field(None, description="Creator profile ID for context")
    topic: str = Field(..., description="Course topic", min_length=5)
    description: Optional[str] = Field(None, description="Additional context")
    difficulty_start: DifficultyLevel = Field(default=DifficultyLevel.BEGINNER)
    difficulty_end: DifficultyLevel = Field(default=DifficultyLevel.INTERMEDIATE)
    structure: CourseStructureConfig = Field(default_factory=CourseStructureConfig)
    context: Optional[CourseContext] = Field(None, description="Enriched course context")
    # RAG document references (Phase 2)
    document_ids: List[str] = Field(default_factory=list, description="IDs of uploaded documents to use as source")
    rag_context: Optional[str] = Field(None, description="Pre-fetched RAG context (set by server)")


class ReorderRequest(BaseModel):
    """Request to reorder sections/lectures"""
    sections: List[Dict[str, Any]] = Field(..., description="Reordered sections with lecture IDs")


class CourseJob(BaseModel):
    """Tracks the status of a course generation job"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = Field(default="queued", description="Current status")
    current_stage: CourseStage = Field(default=CourseStage.QUEUED)
    progress: float = Field(default=0.0, description="Progress percentage 0-100")
    message: str = Field(default="", description="Current status message")

    # Request parameters
    request: Optional[GenerateCourseRequest] = None

    # Generated content
    outline: Optional[CourseOutline] = Field(None, description="Course outline/curriculum")
    lectures_total: int = Field(default=0, description="Total number of lectures")
    lectures_completed: int = Field(default=0, description="Completed lectures")
    current_lecture_title: Optional[str] = Field(None, description="Currently generating lecture")

    # Output
    output_urls: List[str] = Field(default_factory=list, description="Generated video URLs")
    zip_url: Optional[str] = Field(None, description="ZIP download URL")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Error handling
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Curriculum Enforcer context (Phase 6)
    curriculum_context: Optional[str] = Field(
        None,
        description="Curriculum context type: education, enterprise, bootcamp, tutorial, workshop, certification"
    )

    def update_progress(self, stage: CourseStage, progress: float, message: str = ""):
        """Update job progress"""
        self.current_stage = stage
        self.progress = progress
        self.message = message
        self.updated_at = datetime.utcnow()

        if stage == CourseStage.COMPLETED:
            self.status = "completed"
            self.completed_at = datetime.utcnow()
        elif stage == CourseStage.FAILED:
            self.status = "failed"
        else:
            self.status = "processing"

    def update_lecture_progress(self, completed: int, total: int, current_title: Optional[str] = None):
        """Update lecture generation progress"""
        self.lectures_completed = completed
        self.lectures_total = total
        self.current_lecture_title = current_title
        # Progress: 10-90% for lecture generation
        if total > 0:
            self.progress = 10 + (80 * completed / total)
        self.updated_at = datetime.utcnow()


class CourseJobResponse(BaseModel):
    """Response with course job status"""
    job_id: str
    status: str
    current_stage: CourseStage
    progress: float
    message: str
    outline: Optional[CourseOutline] = None
    lectures_total: int = 0
    lectures_completed: int = 0
    current_lecture_title: Optional[str] = None
    output_urls: List[str] = []
    zip_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
