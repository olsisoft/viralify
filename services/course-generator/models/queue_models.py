"""
Queue Models for Distributed Course Generation

Defines dataclass models for the queue-based lecture generation system.
Supports 3-phase architecture:
- Phase 1: Orchestration (generates outline, creates lecture jobs)
- Phase 2: Lecture Generation (workers process individual lectures)
- Phase 3: Finalization (assembles course, generates quizzes, creates ZIP)
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class CourseJobStatus(str, Enum):
    """Status of a course generation job"""
    QUEUED = "queued"
    ORCHESTRATING = "orchestrating"
    GENERATING_LECTURES = "generating_lectures"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class LectureJobStatus(str, Enum):
    """Status of a lecture generation job"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class QueuedLectureJob:
    """
    Job for generating a SINGLE lecture.

    Published to lecture_queue after orchestration phase.
    Consumed by lecture workers for parallel processing.
    """
    job_id: str                   # UUID unique for this lecture job
    course_job_id: str            # Reference to parent course job
    section_index: int            # 0-based section index
    lecture_index: int            # 0-based lecture index within section
    lecture_id: str               # ID of the lecture in the outline

    # Data needed to generate the lecture
    lecture_title: str
    lecture_description: str
    section_title: str
    course_topic: str
    difficulty: str               # "beginner", "intermediate", "advanced", etc.
    language: str                 # "en", "fr", etc.

    # Target audience for content adaptation
    target_audience: Optional[str] = None

    # RAG context if applicable
    rag_context: Optional[str] = None

    # Config
    selected_elements: Optional[List[str]] = None
    element_weights: Optional[Dict[str, float]] = None

    # Quiz config
    quiz_config: Optional[Dict] = None

    # Timing
    duration_seconds: int = 300   # Target lecture duration

    # Pedagogical analysis results (from course orchestrator)
    detected_persona: Optional[str] = None
    topic_complexity: Optional[str] = None
    requires_code: Optional[bool] = None
    requires_diagrams: Optional[bool] = None
    content_preferences: Optional[Dict[str, float]] = None
    recommended_elements: Optional[List[str]] = None

    # Presentation options (passed through to presentation-generator)
    voice_id: str = "alloy"
    style: str = "dark"
    typing_speed: str = "natural"
    title_style: str = "engaging"
    code_display_mode: str = "reveal"
    include_avatar: bool = False
    avatar_id: Optional[str] = None

    # Metadata
    created_at: Optional[str] = None
    priority: int = 5             # 1-10, lower = higher priority
    retry_count: int = 0
    max_retries: int = 3

    def to_json(self) -> str:
        """Serialize to JSON for queue message"""
        data = asdict(self)
        if not data.get('created_at'):
            data['created_at'] = datetime.utcnow().isoformat()
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'QueuedLectureJob':
        """Deserialize from JSON queue message"""
        data = json.loads(json_str)
        # Filter to only known fields for backward compatibility
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        if not data.get('created_at'):
            data['created_at'] = datetime.utcnow().isoformat()
        return data


@dataclass
class LectureResult:
    """
    Result of a lecture generation.

    Stored in Redis under course:{course_job_id}:lectures:{lecture_id}
    """
    lecture_id: str
    status: LectureJobStatus = LectureJobStatus.QUEUED

    # Generated assets
    video_url: Optional[str] = None
    presentation_url: Optional[str] = None
    presentation_job_id: Optional[str] = None

    # Metadata
    duration_seconds: Optional[float] = None
    slides_count: Optional[int] = None

    # Error tracking
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    retry_count: int = 0

    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON for Redis storage"""
        data = asdict(self)
        # Convert enum to string
        if isinstance(data.get('status'), LectureJobStatus):
            data['status'] = data['status'].value
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'LectureResult':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        # Convert string back to enum
        if isinstance(data.get('status'), str):
            data['status'] = LectureJobStatus(data['status'])
        return cls(**data)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        if isinstance(data.get('status'), LectureJobStatus):
            data['status'] = data['status'].value
        return data


@dataclass
class CourseProgress:
    """
    Tracking progress of a course generation job.

    Stored in Redis under course:{course_job_id}
    """
    course_job_id: str
    status: CourseJobStatus = CourseJobStatus.QUEUED

    # Lecture tracking
    total_lectures: int = 0
    completed_lectures: int = 0
    failed_lectures: int = 0
    in_progress_lectures: int = 0

    # Failed lecture details
    failed_lecture_ids: List[str] = field(default_factory=list)
    failed_lecture_errors: Dict[str, str] = field(default_factory=dict)

    # Lecture results (lecture_id -> LectureResult as dict)
    lecture_results: Dict[str, Dict] = field(default_factory=dict)

    # Course data
    outline_json: Optional[str] = None

    # Output URLs
    zip_url: Optional[str] = None
    final_video_url: Optional[str] = None

    # Error tracking
    error: Optional[str] = None

    # Timing
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON for Redis storage"""
        data = asdict(self)
        if isinstance(data.get('status'), CourseJobStatus):
            data['status'] = data['status'].value
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'CourseProgress':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        if isinstance(data.get('status'), str):
            data['status'] = CourseJobStatus(data['status'])
        return cls(**data)

    def to_dict(self) -> Dict:
        """Convert to dictionary for Redis hash"""
        return {
            'course_job_id': self.course_job_id,
            'status': self.status.value if isinstance(self.status, CourseJobStatus) else self.status,
            'total_lectures': str(self.total_lectures),
            'completed_lectures': str(self.completed_lectures),
            'failed_lectures': str(self.failed_lectures),
            'in_progress_lectures': str(self.in_progress_lectures),
            'failed_lecture_ids': json.dumps(self.failed_lecture_ids),
            'failed_lecture_errors': json.dumps(self.failed_lecture_errors),
            'outline_json': self.outline_json or '',
            'zip_url': self.zip_url or '',
            'final_video_url': self.final_video_url or '',
            'error': self.error or '',
            'created_at': self.created_at or datetime.utcnow().isoformat(),
            'started_at': self.started_at or '',
            'completed_at': self.completed_at or '',
        }

    @classmethod
    def from_redis_hash(cls, data: Dict[str, str]) -> 'CourseProgress':
        """Create from Redis hash data"""
        return cls(
            course_job_id=data.get('course_job_id', ''),
            status=CourseJobStatus(data.get('status', 'queued')),
            total_lectures=int(data.get('total_lectures', 0)),
            completed_lectures=int(data.get('completed_lectures', 0)),
            failed_lectures=int(data.get('failed_lectures', 0)),
            in_progress_lectures=int(data.get('in_progress_lectures', 0)),
            failed_lecture_ids=json.loads(data.get('failed_lecture_ids', '[]')),
            failed_lecture_errors=json.loads(data.get('failed_lecture_errors', '{}')),
            outline_json=data.get('outline_json') or None,
            zip_url=data.get('zip_url') or None,
            final_video_url=data.get('final_video_url') or None,
            error=data.get('error') or None,
            created_at=data.get('created_at') or None,
            started_at=data.get('started_at') or None,
            completed_at=data.get('completed_at') or None,
        )

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage"""
        if self.total_lectures == 0:
            return 0.0
        return (self.completed_lectures / self.total_lectures) * 100

    @property
    def is_complete(self) -> bool:
        """Check if all lectures are done (completed or failed)"""
        return (self.completed_lectures + self.failed_lectures) >= self.total_lectures

    @property
    def is_partial_success(self) -> bool:
        """Check if some lectures succeeded and some failed"""
        return self.completed_lectures > 0 and self.failed_lectures > 0


@dataclass
class QueuedFinalizationJob:
    """
    Job for finalizing a course after all lectures are generated.

    Published to finalization_queue when all lectures complete.
    """
    course_job_id: str
    user_id: str

    # Optional: Force finalization even with failed lectures
    force_finalization: bool = False

    # Quiz generation config
    generate_quizzes: bool = True
    quiz_config: Optional[Dict] = None

    # Output config
    create_zip: bool = True
    create_combined_video: bool = False

    # Metadata
    created_at: Optional[str] = None
    priority: int = 5

    def to_json(self) -> str:
        """Serialize to JSON for queue message"""
        data = asdict(self)
        if not data.get('created_at'):
            data['created_at'] = datetime.utcnow().isoformat()
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'QueuedFinalizationJob':
        """Deserialize from JSON queue message"""
        data = json.loads(json_str)
        # Filter to only known fields for backward compatibility
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# Type aliases for Redis keys
def get_course_key(course_job_id: str) -> str:
    """Get Redis key for course progress"""
    return f"course:{course_job_id}"


def get_course_lectures_key(course_job_id: str) -> str:
    """Get Redis key for course lecture results"""
    return f"course:{course_job_id}:lectures"


def get_lecture_key(course_job_id: str, lecture_id: str) -> str:
    """Get Redis key for a specific lecture result"""
    return f"course:{course_job_id}:lectures:{lecture_id}"
