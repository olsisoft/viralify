"""
Assessment Models for Practice Agent

Defines models for evaluating learner solutions and providing feedback.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HintLevel(int, Enum):
    """Levels of hints from subtle to explicit"""
    NUDGE = 1          # Very subtle hint, just a direction
    GUIDANCE = 2       # Points to relevant concept
    EXPLANATION = 3    # Explains what to look for
    PARTIAL = 4        # Shows part of the solution
    SOLUTION = 5       # Full solution revealed


class UnderstandingLevel(str, Enum):
    """Assessed level of understanding"""
    NONE = "none"               # No understanding demonstrated
    SURFACE = "surface"         # Can repeat but not apply
    DEVELOPING = "developing"   # Starting to understand
    FUNCTIONAL = "functional"   # Can apply with guidance
    PROFICIENT = "proficient"   # Can apply independently
    EXPERT = "expert"           # Can teach others


class FeedbackType(str, Enum):
    """Types of feedback"""
    SUCCESS = "success"             # Positive reinforcement
    ENCOURAGEMENT = "encouragement" # Keep trying
    CORRECTION = "correction"       # Fix this specific thing
    EXPLANATION = "explanation"     # Explains a concept
    SUGGESTION = "suggestion"       # Try this approach
    WARNING = "warning"             # Potential issue
    HINT = "hint"                   # Subtle guidance
    QUESTION = "question"           # Socratic questioning


class CodeQualityMetric(BaseModel):
    """A code quality measurement"""
    name: str = Field(..., description="Metric name")
    score: float = Field(..., ge=0, le=100, description="Score 0-100")
    feedback: str = Field(..., description="Specific feedback")
    suggestions: List[str] = Field(default_factory=list)


class CodeAnalysis(BaseModel):
    """Analysis of submitted code"""
    # Correctness
    compiles: bool = Field(default=True)
    runs_without_error: bool = Field(default=False)
    produces_expected_output: bool = Field(default=False)

    # Quality metrics
    metrics: List[CodeQualityMetric] = Field(default_factory=list)
    overall_quality_score: float = Field(default=0.0)

    # Issues found
    syntax_errors: List[Dict[str, Any]] = Field(default_factory=list)
    runtime_errors: List[Dict[str, Any]] = Field(default_factory=list)
    logic_errors: List[str] = Field(default_factory=list)

    # Best practices
    follows_best_practices: bool = Field(default=False)
    best_practice_violations: List[str] = Field(default_factory=list)

    # Security (for DevOps exercises)
    security_issues: List[Dict[str, Any]] = Field(default_factory=list)

    # Efficiency
    time_complexity: Optional[str] = Field(None)
    space_complexity: Optional[str] = Field(None)
    performance_notes: List[str] = Field(default_factory=list)


class PedagogicalFeedback(BaseModel):
    """Pedagogically-designed feedback for the learner"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Type and tone
    feedback_type: FeedbackType
    tone: str = Field(default="encouraging", description="encouraging, neutral, direct")

    # Content
    title: str = Field(..., description="Brief headline")
    message: str = Field(..., description="Main feedback message")
    details: Optional[str] = Field(None, description="Additional explanation")

    # Code-specific
    code_reference: Optional[str] = Field(None, description="Line or section of code")
    suggested_fix: Optional[str] = Field(None, description="Code suggestion")

    # Learning resources
    related_concept: Optional[str] = Field(None)
    documentation_link: Optional[str] = Field(None)
    video_reference: Optional[str] = Field(None, description="Timestamp in course video")

    # Actions
    actionable_steps: List[str] = Field(default_factory=list)

    # Socratic elements
    follow_up_question: Optional[str] = Field(None, description="Question to prompt thinking")

    # Priority
    priority: int = Field(default=1, ge=1, le=5, description="1=highest priority")


class AssessmentResult(BaseModel):
    """Complete assessment of a learner's submission"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exercise_id: str
    attempt_id: str

    # Overall result
    passed: bool = Field(default=False)
    score: int = Field(default=0)
    max_score: int = Field(default=100)
    percentage: float = Field(default=0.0)

    # Detailed analysis
    code_analysis: Optional[CodeAnalysis] = Field(None)

    # Validation results
    checks_passed: List[str] = Field(default_factory=list)
    checks_failed: List[str] = Field(default_factory=list)
    partial_credit: Dict[str, int] = Field(default_factory=dict, description="Check -> points earned")

    # Feedback
    feedback_items: List[PedagogicalFeedback] = Field(default_factory=list)
    summary_feedback: str = Field(default="")

    # Understanding assessment
    understanding_level: UnderstandingLevel = Field(default=UnderstandingLevel.DEVELOPING)
    misconceptions_detected: List[str] = Field(default_factory=list)
    strengths_observed: List[str] = Field(default_factory=list)

    # Recommendations
    recommended_review_topics: List[str] = Field(default_factory=list)
    recommended_exercises: List[str] = Field(default_factory=list)

    # Timing
    assessed_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "passed": True,
                "score": 85,
                "max_score": 100,
                "understanding_level": "functional",
                "summary_feedback": "Good job! Your Dockerfile is functional..."
            }
        }


class SocraticQuestion(BaseModel):
    """A Socratic question to guide learning"""
    question: str = Field(..., description="The question to ask")
    purpose: str = Field(..., description="Why we're asking this")
    expected_insight: str = Field(..., description="What we hope they'll realize")

    # If they don't get it
    follow_up_if_wrong: str = Field(default="")
    simpler_version: Optional[str] = Field(None, description="Easier version of same question")

    # Context
    relates_to_code: Optional[str] = Field(None, description="Specific code section")
    concept: str = Field(..., description="The concept being explored")


class LearningMoment(BaseModel):
    """A captured teachable moment"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # What triggered it
    trigger: str = Field(..., description="What the learner did/said")
    trigger_type: str = Field(..., description="error, question, misconception, breakthrough")

    # The learning opportunity
    concept: str = Field(..., description="Concept to teach")
    explanation: str = Field(..., description="Clear explanation")

    # Examples
    example_code: Optional[str] = Field(None)
    counter_example: Optional[str] = Field(None, description="What NOT to do")

    # Engagement
    socratic_questions: List[SocraticQuestion] = Field(default_factory=list)

    # Verification
    understanding_check: Optional[str] = Field(None, description="Question to verify understanding")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProgressAssessment(BaseModel):
    """Assessment of overall learning progress"""
    user_id: str
    assessment_date: datetime = Field(default_factory=datetime.utcnow)

    # Skill levels by category
    skill_levels: Dict[str, UnderstandingLevel] = Field(
        default_factory=dict,
        description="Category -> understanding level"
    )

    # Trends
    improving_areas: List[str] = Field(default_factory=list)
    struggling_areas: List[str] = Field(default_factory=list)
    mastered_areas: List[str] = Field(default_factory=list)

    # Learning patterns
    learns_best_from: List[str] = Field(
        default_factory=list,
        description="e.g., 'examples', 'theory', 'hands-on'"
    )
    common_mistakes: List[str] = Field(default_factory=list)

    # Recommendations
    next_topics: List[str] = Field(default_factory=list)
    recommended_exercises: List[str] = Field(default_factory=list)
    recommended_review: List[str] = Field(default_factory=list)

    # Engagement
    engagement_score: float = Field(default=0.0, ge=0, le=100)
    persistence_score: float = Field(default=0.0, ge=0, le=100, description="How they handle difficulty")

    # Predictions
    ready_for_next_level: bool = Field(default=False)
    estimated_time_to_proficiency: Optional[str] = Field(None, description="e.g., '2 weeks'")


class ExerciseGenerationRequest(BaseModel):
    """Request to generate exercises from course content"""
    course_id: str
    lecture_ids: Optional[List[str]] = Field(None, description="Specific lectures or all")

    # Difficulty distribution
    difficulty_distribution: Dict[str, int] = Field(
        default_factory=lambda: {
            "beginner": 40,
            "intermediate": 35,
            "advanced": 20,
            "expert": 5
        },
        description="Percentage per difficulty"
    )

    # Exercise types to generate
    exercise_types: List[str] = Field(
        default_factory=lambda: ["coding", "debugging", "configuration"],
        description="Types to include"
    )

    # Count
    exercises_per_lecture: int = Field(default=3)
    max_total_exercises: int = Field(default=20)

    # Focus areas
    focus_concepts: List[str] = Field(default_factory=list, description="Concepts to emphasize")


class ExerciseGenerationResult(BaseModel):
    """Result from exercise generation"""
    course_id: str
    exercises_generated: int
    exercises: List[Dict[str, Any]] = Field(default_factory=list, description="Generated exercises")

    # Coverage
    concepts_covered: List[str] = Field(default_factory=list)
    concepts_not_covered: List[str] = Field(default_factory=list)

    # Distribution achieved
    difficulty_distribution: Dict[str, int] = Field(default_factory=dict)
    type_distribution: Dict[str, int] = Field(default_factory=dict)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_time_seconds: float = Field(default=0.0)
