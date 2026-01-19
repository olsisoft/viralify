"""
Practice Models for the Practice Agent

Defines models for practice sessions, exercises, and learner progress.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DifficultyLevel(str, Enum):
    """Exercise difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ExerciseType(str, Enum):
    """Types of exercises"""
    CODING = "coding"                    # Write code from scratch
    DEBUGGING = "debugging"              # Fix broken code
    CONFIGURATION = "configuration"      # Configure systems/tools
    ARCHITECTURE = "architecture"        # Design system architecture
    MULTIPLE_CHOICE = "multiple_choice"  # Quiz-style questions
    FILL_IN_BLANK = "fill_in_blank"     # Complete partial code
    CODE_REVIEW = "code_review"         # Review and improve code
    TROUBLESHOOTING = "troubleshooting" # Diagnose and fix issues


class ExerciseCategory(str, Enum):
    """Categories of exercises by domain"""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    CI_CD = "ci_cd"
    LINUX = "linux"
    GIT = "git"
    PYTHON = "python"
    DATABASES = "databases"
    NETWORKING = "networking"
    MONITORING = "monitoring"
    SECURITY = "security"


class SessionStatus(str, Enum):
    """Status of a practice session"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ValidationCheck(BaseModel):
    """A single validation check for an exercise"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Name of the check")
    description: Optional[str] = Field(None, description="What this check validates")
    check_type: str = Field(..., description="Type: code_contains, output, file, command, state")
    # Flexible validation options
    expected_value: Optional[Any] = Field(None, description="Expected value")
    patterns: Optional[List[str]] = Field(None, description="Patterns to match")
    expected_output: Optional[str] = Field(None, description="Expected output string")
    contains: Optional[List[str]] = Field(None, description="Strings that must be present")
    # Scoring
    points: int = Field(default=10, description="Points awarded for passing")
    required: bool = Field(default=False, description="Must pass to complete exercise")


class ExpectedOutput(BaseModel):
    """Expected output for validation"""
    type: str = Field(..., description="stdout, stderr, file, state")
    pattern: Optional[str] = Field(None, description="Regex pattern to match")
    exact_match: Optional[str] = Field(None, description="Exact string to match")
    contains: Optional[List[str]] = Field(None, description="Strings that must be present")
    not_contains: Optional[List[str]] = Field(None, description="Strings that must NOT be present")


class Exercise(BaseModel):
    """A practice exercise"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Basic info
    title: str = Field(..., description="Exercise title")
    description: str = Field(..., description="Brief description")
    instructions: str = Field(..., description="Detailed instructions in markdown")

    # Classification
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.BEGINNER)
    type: ExerciseType = Field(default=ExerciseType.CODING)
    category: ExerciseCategory = Field(..., description="Domain category")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")

    # Content
    starter_code: Optional[str] = Field(None, description="Initial code provided")
    starter_files: Dict[str, str] = Field(default_factory=dict, description="Multiple starter files")

    # Validation
    expected_outputs: List[ExpectedOutput] = Field(default_factory=list)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    validation_script: Optional[str] = Field(None, description="Custom validation script")

    # Help system
    hints: List[str] = Field(default_factory=list, description="Progressive hints")
    solution: Optional[str] = Field(None, description="Full solution code")
    solution_explanation: Optional[str] = Field(None, description="Why this solution works")

    # Relationships
    course_id: Optional[str] = Field(None, description="Related course")
    lecture_ids: List[str] = Field(default_factory=list, description="Related lectures")
    prerequisite_exercises: List[str] = Field(default_factory=list, description="Must complete first")

    # Sandbox configuration
    sandbox_type: str = Field(default="docker", description="Required sandbox type")
    sandbox_config: Dict[str, Any] = Field(default_factory=dict, description="Sandbox-specific config")
    timeout_seconds: int = Field(default=300, description="Max execution time")

    # Metadata
    estimated_minutes: int = Field(default=15, description="Estimated time to complete")
    points: int = Field(default=100, description="Points awarded for completion")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "docker-001",
                "title": "Create Your First Dockerfile",
                "difficulty": "beginner",
                "type": "coding",
                "category": "docker",
                "estimated_minutes": 15,
                "points": 100
            }
        }


class ExerciseAttempt(BaseModel):
    """A single attempt at an exercise"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exercise_id: str = Field(..., description="Exercise being attempted")

    # Code submitted
    submitted_code: str = Field(..., description="Code submitted by learner")
    submitted_files: Dict[str, str] = Field(default_factory=dict, description="Multiple files")

    # Results
    passed: bool = Field(default=False)
    score: int = Field(default=0, description="Points earned")
    checks_passed: List[str] = Field(default_factory=list, description="Validation checks passed")
    checks_failed: List[str] = Field(default_factory=list, description="Validation checks failed")

    # Execution details
    execution_output: Optional[str] = Field(None)
    execution_errors: Optional[str] = Field(None)
    execution_time_ms: int = Field(default=0)

    # Hints used
    hints_used: int = Field(default=0)
    solution_viewed: bool = Field(default=False)

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)


class LearnerProgress(BaseModel):
    """Tracks learner's overall progress"""
    user_id: str = Field(..., description="User ID")

    # Overall stats
    total_exercises_completed: int = Field(default=0)
    total_points: int = Field(default=0)
    current_streak: int = Field(default=0, description="Days in a row")
    longest_streak: int = Field(default=0)

    # By category
    category_progress: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Progress per category: {category: {completed, total, points}}"
    )

    # By difficulty
    difficulty_stats: Dict[str, int] = Field(
        default_factory=lambda: {
            "beginner": 0,
            "intermediate": 0,
            "advanced": 0,
            "expert": 0
        }
    )

    # Exercise history
    completed_exercises: List[str] = Field(default_factory=list, description="Exercise IDs completed")
    in_progress_exercises: List[str] = Field(default_factory=list)

    # Badges/Achievements
    badges: List[str] = Field(default_factory=list)

    # Learning patterns
    average_attempts_per_exercise: float = Field(default=0.0)
    average_hints_used: float = Field(default=0.0)
    preferred_time_of_day: Optional[str] = Field(None)

    # Timestamps
    first_exercise_at: Optional[datetime] = Field(None)
    last_exercise_at: Optional[datetime] = Field(None)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Message(BaseModel):
    """A message in the practice conversation"""
    role: str = Field(..., description="user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PracticeSession(BaseModel):
    """An active practice session"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="Learner's user ID")
    course_id: Optional[str] = Field(None, description="Associated course if any")

    # Current state
    status: SessionStatus = Field(default=SessionStatus.ACTIVE)
    current_exercise: Optional[Exercise] = Field(None)
    current_attempt: Optional[ExerciseAttempt] = Field(None)

    # Conversation
    conversation_history: List[Message] = Field(default_factory=list)

    # Session progress
    exercises_completed: List[str] = Field(default_factory=list)
    exercises_attempted: Dict[str, List[ExerciseAttempt]] = Field(default_factory=dict)
    points_earned: int = Field(default=0)
    hints_used_total: int = Field(default=0)

    # Sandbox state
    sandbox_id: Optional[str] = Field(None, description="Active sandbox ID")
    sandbox_type: Optional[str] = Field(None)

    # Configuration
    difficulty_preference: DifficultyLevel = Field(default=DifficultyLevel.BEGINNER)
    categories_focus: List[ExerciseCategory] = Field(default_factory=list)
    pair_programming_enabled: bool = Field(default=False)
    voice_enabled: bool = Field(default=False)

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "session-123",
                "user_id": "user-456",
                "course_id": "course-789",
                "status": "active",
                "points_earned": 250
            }
        }


# Request/Response models for API

class CreateSessionRequest(BaseModel):
    """Request to create a practice session"""
    user_id: str
    course_id: Optional[str] = None
    difficulty_preference: DifficultyLevel = DifficultyLevel.BEGINNER
    categories_focus: List[ExerciseCategory] = Field(default_factory=list)
    pair_programming_enabled: bool = False
    voice_enabled: bool = False


class CreateSessionResponse(BaseModel):
    """Response after creating a session"""
    session_id: str
    status: str
    first_exercise: Optional[Exercise]
    message: str


class SubmitCodeRequest(BaseModel):
    """Request to submit code for evaluation"""
    code: str
    files: Dict[str, str] = Field(default_factory=dict)


class SubmitCodeResponse(BaseModel):
    """Response after code submission"""
    passed: bool
    score: int
    feedback: str
    checks_passed: List[str]
    checks_failed: List[str]
    execution_output: Optional[str]
    next_exercise: Optional[Exercise]


class ChatRequest(BaseModel):
    """Request to chat with the practice agent"""
    message: str
    include_code_context: bool = True


class ChatResponse(BaseModel):
    """Response from practice agent chat"""
    message: str
    suggestions: List[str] = Field(default_factory=list)
    code_snippet: Optional[str] = None
    action_required: Optional[str] = None


class HintRequest(BaseModel):
    """Request for a hint"""
    hint_level: int = Field(default=1, ge=1, le=5)


class HintResponse(BaseModel):
    """Response with hint"""
    hint: str
    hint_number: int
    hints_remaining: int
    points_deduction: int
