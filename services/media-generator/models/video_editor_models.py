"""
Video Editor Models

Data models for video editing, timeline management, and segment handling.
Phase 3: User Video Editing feature.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class SegmentType(str, Enum):
    """Types of timeline segments"""
    GENERATED = "generated"      # AI-generated course video
    USER_VIDEO = "user_video"    # User uploaded video (webcam, screen recording)
    USER_AUDIO = "user_audio"    # User uploaded audio
    SLIDE = "slide"              # Custom slide/image
    TRANSITION = "transition"    # Transition effect between segments
    OVERLAY = "overlay"          # Overlay (logo, watermark, text)


class TransitionType(str, Enum):
    """Video transition types"""
    NONE = "none"
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


class SegmentStatus(str, Enum):
    """Segment processing status"""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class ProjectStatus(str, Enum):
    """Video project status"""
    DRAFT = "draft"
    EDITING = "editing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class AudioTrack(BaseModel):
    """Audio track within a segment"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_url: str = Field(..., description="URL to audio file")
    volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Volume multiplier")
    start_time: float = Field(default=0.0, ge=0.0, description="Start time in segment (seconds)")
    duration: Optional[float] = Field(None, description="Duration (None = full length)")
    fade_in: float = Field(default=0.0, ge=0.0, description="Fade in duration (seconds)")
    fade_out: float = Field(default=0.0, ge=0.0, description="Fade out duration (seconds)")
    is_muted: bool = Field(default=False)


class VideoSegment(BaseModel):
    """
    A segment in the video timeline.
    Can be a generated lecture, user video, slide, or transition.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = Field(..., description="Parent project ID")

    # Segment type and source
    segment_type: SegmentType = Field(..., description="Type of segment")
    source_url: Optional[str] = Field(None, description="URL to source video/image")
    source_lecture_id: Optional[str] = Field(None, description="Reference to course lecture if generated")

    # Timeline position
    order: int = Field(..., description="Order in timeline")
    start_time: float = Field(..., ge=0.0, description="Start time in project timeline (seconds)")
    duration: float = Field(..., gt=0.0, description="Duration of segment (seconds)")

    # Trim settings (for cutting source video)
    trim_start: float = Field(default=0.0, ge=0.0, description="Trim from start of source (seconds)")
    trim_end: Optional[float] = Field(None, description="Trim from end of source (seconds)")

    # Audio settings
    audio_tracks: List[AudioTrack] = Field(default_factory=list)
    original_audio_volume: float = Field(default=1.0, ge=0.0, le=2.0)
    is_audio_muted: bool = Field(default=False)

    # Visual settings
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    scale: float = Field(default=1.0, ge=0.1, le=3.0, description="Scale factor")
    position_x: float = Field(default=0.5, ge=0.0, le=1.0, description="X position (0-1)")
    position_y: float = Field(default=0.5, ge=0.0, le=1.0, description="Y position (0-1)")
    rotation: float = Field(default=0.0, description="Rotation in degrees")

    # Transitions
    transition_in: TransitionType = Field(default=TransitionType.NONE)
    transition_in_duration: float = Field(default=0.5, ge=0.0, le=3.0)
    transition_out: TransitionType = Field(default=TransitionType.NONE)
    transition_out_duration: float = Field(default=0.5, ge=0.0, le=3.0)

    # Status
    status: SegmentStatus = Field(default=SegmentStatus.PENDING)
    error_message: Optional[str] = Field(None)

    # Metadata
    title: Optional[str] = Field(None, description="Display title")
    thumbnail_url: Optional[str] = Field(None)
    original_filename: Optional[str] = Field(None)
    file_size_bytes: int = Field(default=0)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "seg-001",
                "project_id": "proj-001",
                "segment_type": "generated",
                "source_url": "/videos/lecture_001.mp4",
                "order": 0,
                "start_time": 0.0,
                "duration": 120.0,
                "status": "ready"
            }
        }


class TextOverlay(BaseModel):
    """Text overlay on video"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="Text content")
    font_family: str = Field(default="Arial")
    font_size: int = Field(default=32, ge=8, le=200)
    font_color: str = Field(default="#FFFFFF")
    background_color: Optional[str] = Field(None, description="Background color (None = transparent)")
    position_x: float = Field(default=0.5, ge=0.0, le=1.0)
    position_y: float = Field(default=0.9, ge=0.0, le=1.0)
    start_time: float = Field(..., ge=0.0, description="When to show (seconds)")
    duration: float = Field(..., gt=0.0, description="How long to show (seconds)")
    animation: str = Field(default="none", description="Animation type: none, fade, slide")


class ImageOverlay(BaseModel):
    """Image overlay (logo, watermark)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_url: str = Field(..., description="URL to overlay image")
    position_x: float = Field(default=0.95, ge=0.0, le=1.0)
    position_y: float = Field(default=0.05, ge=0.0, le=1.0)
    scale: float = Field(default=0.1, ge=0.01, le=1.0)
    opacity: float = Field(default=0.8, ge=0.0, le=1.0)
    start_time: Optional[float] = Field(None, description="None = show entire video")
    duration: Optional[float] = Field(None)


class VideoProject(BaseModel):
    """
    A video editing project.
    Contains timeline with segments and project settings.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="Owner user ID")
    course_id: Optional[str] = Field(None, description="Associated course ID")
    course_job_id: Optional[str] = Field(None, description="Original course generation job ID")

    # Project info
    title: str = Field(..., description="Project title")
    description: Optional[str] = Field(None)

    # Timeline
    segments: List[VideoSegment] = Field(default_factory=list)
    total_duration: float = Field(default=0.0, description="Total timeline duration (seconds)")

    # Overlays
    text_overlays: List[TextOverlay] = Field(default_factory=list)
    image_overlays: List[ImageOverlay] = Field(default_factory=list)

    # Output settings
    output_resolution: str = Field(default="1920x1080", description="Output resolution")
    output_fps: int = Field(default=30, ge=24, le=60)
    output_format: str = Field(default="mp4")
    output_quality: str = Field(default="high", description="low, medium, high")

    # Background audio
    background_music_url: Optional[str] = Field(None)
    background_music_volume: float = Field(default=0.3, ge=0.0, le=1.0)

    # Status
    status: ProjectStatus = Field(default=ProjectStatus.DRAFT)
    render_progress: float = Field(default=0.0, ge=0.0, le=100.0)
    render_message: Optional[str] = Field(None)
    output_url: Optional[str] = Field(None, description="Final rendered video URL")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    rendered_at: Optional[datetime] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "proj-001",
                "user_id": "user-123",
                "course_id": "course-456",
                "title": "Python Fundamentals - Edited",
                "status": "draft",
                "total_duration": 3600.0
            }
        }


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateProjectRequest(BaseModel):
    """Request to create a new video project"""
    user_id: str = Field(..., description="User ID")
    course_id: Optional[str] = Field(None, description="Course ID to import from")
    course_job_id: Optional[str] = Field(None, description="Course job ID to import from")
    title: str = Field(..., description="Project title")
    description: Optional[str] = Field(None)

    # If importing from course, auto-populate segments
    import_course_videos: bool = Field(default=True)


class CreateProjectResponse(BaseModel):
    """Response after creating a project"""
    project_id: str
    title: str
    status: ProjectStatus
    segment_count: int
    message: str


class AddSegmentRequest(BaseModel):
    """Request to add a segment to timeline"""
    segment_type: SegmentType
    source_url: Optional[str] = Field(None, description="URL for user uploads")
    source_lecture_id: Optional[str] = Field(None, description="For generated segments")

    # Position (optional - will append if not specified)
    insert_after_segment_id: Optional[str] = Field(None)

    # Trim settings
    trim_start: float = Field(default=0.0)
    trim_end: Optional[float] = Field(None)

    # For slides
    slide_image_url: Optional[str] = Field(None)
    slide_duration: float = Field(default=5.0)

    # Title
    title: Optional[str] = Field(None)


class UpdateSegmentRequest(BaseModel):
    """Request to update a segment"""
    # Trim
    trim_start: Optional[float] = Field(None)
    trim_end: Optional[float] = Field(None)

    # Audio
    original_audio_volume: Optional[float] = Field(None)
    is_audio_muted: Optional[bool] = Field(None)

    # Visual
    opacity: Optional[float] = Field(None)
    scale: Optional[float] = Field(None)
    position_x: Optional[float] = Field(None)
    position_y: Optional[float] = Field(None)

    # Transitions
    transition_in: Optional[TransitionType] = Field(None)
    transition_in_duration: Optional[float] = Field(None)
    transition_out: Optional[TransitionType] = Field(None)
    transition_out_duration: Optional[float] = Field(None)

    # Title
    title: Optional[str] = Field(None)


class ReorderSegmentsRequest(BaseModel):
    """Request to reorder segments"""
    segment_ids: List[str] = Field(..., description="Segment IDs in new order")


class RenderProjectRequest(BaseModel):
    """Request to render the final video"""
    output_resolution: Optional[str] = Field(None, description="Override resolution")
    output_fps: Optional[int] = Field(None)
    output_quality: Optional[str] = Field(None)
    include_watermark: bool = Field(default=False)
    watermark_url: Optional[str] = Field(None)


class ProjectListResponse(BaseModel):
    """Response listing projects"""
    projects: List[VideoProject]
    total: int
    page: int
    page_size: int


class UploadSegmentResponse(BaseModel):
    """Response after uploading a user video segment"""
    segment_id: str
    status: SegmentStatus
    source_url: str
    duration: float
    thumbnail_url: Optional[str]
    message: str
