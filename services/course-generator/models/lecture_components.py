"""
Lecture Components Models

Defines models for editable lecture components, enabling:
- Partial course success (courses complete even if some lectures fail)
- Lecture editing (modify slides, voiceover, diagrams)
- Selective regeneration (regenerate single slide or entire lecture)
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class SlideType(str, Enum):
    """Types of slides in a lecture"""
    TITLE = "title"
    CONTENT = "content"
    CODE = "code"
    CODE_DEMO = "code_demo"
    DIAGRAM = "diagram"
    SPLIT = "split"
    TERMINAL = "terminal"
    CONCLUSION = "conclusion"
    MEDIA = "media"  # New type for user-inserted media (image/video)


class MediaType(str, Enum):
    """Types of media for media slides"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ComponentStatus(str, Enum):
    """Status of a component"""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    EDITED = "edited"


class CodeBlockComponent(BaseModel):
    """An editable code block within a slide"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    language: str = Field(..., description="Programming language")
    code: str = Field(..., description="The code content")
    filename: Optional[str] = Field(None, description="Optional filename to display")
    highlight_lines: List[int] = Field(default_factory=list, description="Lines to highlight")
    execution_order: int = Field(default=0, description="Order for execution in demos")
    expected_output: Optional[str] = Field(None, description="Expected output for validation")
    actual_output: Optional[str] = Field(None, description="Actual execution output")
    show_line_numbers: bool = Field(default=True, description="Show line numbers")


class SlideComponent(BaseModel):
    """
    An editable slide component within a lecture.
    Contains all data needed to regenerate or modify a single slide.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    index: int = Field(default=0, description="Position in the lecture")
    type: SlideType = Field(..., description="Type of slide")
    status: ComponentStatus = Field(default=ComponentStatus.COMPLETED)

    # Content
    title: Optional[str] = Field(None, description="Slide title")
    subtitle: Optional[str] = Field(None, description="Optional subtitle")
    content: Optional[str] = Field(None, description="Text content (markdown supported)")
    bullet_points: List[str] = Field(default_factory=list, description="Bullet points")
    code_blocks: List[CodeBlockComponent] = Field(default_factory=list, description="Code blocks")

    # Voiceover
    voiceover_text: str = Field(default="", description="Text for voiceover narration")

    # Timing
    duration: float = Field(default=10.0, description="Duration in seconds")
    transition: str = Field(default="fade", description="Transition effect")

    # Diagram specific
    diagram_type: Optional[str] = Field(None, description="flowchart, architecture, process, comparison, hierarchy")
    diagram_data: Optional[Dict[str, Any]] = Field(None, description="Diagram configuration data")

    # Generated assets
    image_url: Optional[str] = Field(None, description="Generated slide image URL")
    animation_url: Optional[str] = Field(None, description="Typing animation video URL (for code slides)")

    # Media slide specific (for user-inserted media)
    media_type: Optional[MediaType] = Field(None, description="Type of media for media slides")
    media_url: Optional[str] = Field(None, description="URL of uploaded media (image/video)")
    media_thumbnail_url: Optional[str] = Field(None, description="Thumbnail for video media")
    media_original_filename: Optional[str] = Field(None, description="Original filename of uploaded media")

    # Edit tracking
    is_edited: bool = Field(default=False, description="Whether this slide has been manually edited")
    edited_at: Optional[datetime] = Field(None, description="When the slide was last edited")
    edited_fields: List[str] = Field(default_factory=list, description="Which fields were edited")

    # Error info
    error: Optional[str] = Field(None, description="Error message if failed")

    def mark_edited(self, fields: List[str]):
        """Mark slide as edited with specific fields"""
        self.is_edited = True
        self.edited_at = datetime.utcnow()
        self.edited_fields = list(set(self.edited_fields + fields))
        self.status = ComponentStatus.EDITED


class VoiceoverComponent(BaseModel):
    """
    Voiceover component for a lecture.
    Can be regenerated from slide voiceover_text or replaced with custom audio.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: ComponentStatus = Field(default=ComponentStatus.COMPLETED)

    # Generated voiceover
    audio_url: Optional[str] = Field(None, description="Generated/uploaded audio URL")
    duration_seconds: float = Field(default=0.0, description="Audio duration")

    # TTS settings (for regeneration)
    voice_id: str = Field(default="alloy", description="Voice ID for TTS")
    voice_settings: Dict[str, Any] = Field(default_factory=dict, description="Voice settings (speed, pitch, etc.)")

    # Combined text from all slides (for regeneration)
    full_text: str = Field(default="", description="Combined voiceover text from all slides")

    # Custom audio
    is_custom_audio: bool = Field(default=False, description="Whether user uploaded custom audio")
    original_filename: Optional[str] = Field(None, description="Original filename if custom")

    # Edit tracking
    is_edited: bool = Field(default=False)
    edited_at: Optional[datetime] = Field(None)

    # Error info
    error: Optional[str] = Field(None)


class LectureComponents(BaseModel):
    """
    Complete editable components of a lecture.
    Stored after successful generation to enable editing and regeneration.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lecture_id: str = Field(..., description="Parent lecture ID")
    job_id: str = Field(..., description="Parent course job ID")

    # Slides (ordered)
    slides: List[SlideComponent] = Field(default_factory=list, description="Editable slides")

    # Voiceover
    voiceover: Optional[VoiceoverComponent] = Field(None, description="Lecture voiceover")

    # Timing
    total_duration: float = Field(default=0.0, description="Total lecture duration in seconds")

    # Generation metadata (for regeneration)
    generation_params: Dict[str, Any] = Field(default_factory=dict, description="Original generation parameters")
    presentation_job_id: Optional[str] = Field(None, description="Presentation-generator job ID")

    # Final output
    video_url: Optional[str] = Field(None, description="Final composed video URL")

    # Status
    status: ComponentStatus = Field(default=ComponentStatus.COMPLETED)
    is_edited: bool = Field(default=False, description="Whether any component has been edited")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Error info
    error: Optional[str] = Field(None)

    def get_slide(self, slide_id: str) -> Optional[SlideComponent]:
        """Get a slide by ID"""
        return next((s for s in self.slides if s.id == slide_id), None)

    def update_slide(self, slide_id: str, updates: Dict[str, Any]) -> Optional[SlideComponent]:
        """Update a slide and mark it as edited"""
        slide = self.get_slide(slide_id)
        if slide:
            edited_fields = []
            for key, value in updates.items():
                if hasattr(slide, key):
                    setattr(slide, key, value)
                    edited_fields.append(key)
            if edited_fields:
                slide.mark_edited(edited_fields)
                self.is_edited = True
                self.updated_at = datetime.utcnow()
        return slide

    def recalculate_duration(self):
        """Recalculate total duration from slides"""
        self.total_duration = sum(s.duration for s in self.slides)

    def get_combined_voiceover_text(self) -> str:
        """Get combined voiceover text from all slides"""
        return " ".join(s.voiceover_text for s in self.slides if s.voiceover_text)

    def reorder_slide(self, slide_id: str, new_index: int) -> bool:
        """Move a slide to a new position"""
        # Find the slide
        slide = self.get_slide(slide_id)
        if not slide:
            return False

        old_index = slide.index
        if old_index == new_index:
            return True  # No change needed

        # Clamp new_index to valid range
        new_index = max(0, min(new_index, len(self.slides) - 1))

        # Remove slide from current position
        self.slides.remove(slide)

        # Insert at new position
        self.slides.insert(new_index, slide)

        # Update all indices
        for i, s in enumerate(self.slides):
            s.index = i

        self.is_edited = True
        self.updated_at = datetime.utcnow()
        return True

    def delete_slide(self, slide_id: str) -> Optional[SlideComponent]:
        """Remove a slide from the lecture"""
        slide = self.get_slide(slide_id)
        if not slide:
            return None

        self.slides.remove(slide)

        # Update indices
        for i, s in enumerate(self.slides):
            s.index = i

        self.recalculate_duration()
        self.is_edited = True
        self.updated_at = datetime.utcnow()
        return slide

    def insert_slide(self, slide: SlideComponent, after_slide_id: Optional[str] = None) -> SlideComponent:
        """Insert a new slide after the specified slide (or at beginning if None)"""
        if after_slide_id:
            after_slide = self.get_slide(after_slide_id)
            if after_slide:
                insert_index = after_slide.index + 1
            else:
                insert_index = len(self.slides)  # Append at end if not found
        else:
            insert_index = 0  # Insert at beginning

        # Ensure unique ID
        if not slide.id:
            slide.id = str(uuid.uuid4())[:8]

        slide.index = insert_index
        self.slides.insert(insert_index, slide)

        # Update indices for slides after the inserted one
        for i, s in enumerate(self.slides):
            s.index = i

        self.recalculate_duration()
        self.is_edited = True
        self.updated_at = datetime.utcnow()
        return slide


# =============================================================================
# API Request/Response Models
# =============================================================================

class UpdateSlideRequest(BaseModel):
    """Request to update a slide"""
    title: Optional[str] = None
    subtitle: Optional[str] = None
    content: Optional[str] = None
    bullet_points: Optional[List[str]] = None
    voiceover_text: Optional[str] = None
    duration: Optional[float] = None
    code_blocks: Optional[List[CodeBlockComponent]] = None
    diagram_type: Optional[str] = None
    diagram_data: Optional[Dict[str, Any]] = None


class RegenerateSlideRequest(BaseModel):
    """Request to regenerate a slide"""
    regenerate_image: bool = Field(default=True, description="Regenerate slide image")
    regenerate_animation: bool = Field(default=False, description="Regenerate typing animation (code slides)")
    use_edited_content: bool = Field(default=True, description="Use edited content or regenerate from scratch")


class RegenerateLectureRequest(BaseModel):
    """Request to regenerate entire lecture"""
    use_edited_components: bool = Field(default=True, description="Keep edited components, regenerate others")
    regenerate_voiceover: bool = Field(default=True, description="Regenerate voiceover audio")
    voice_id: Optional[str] = Field(None, description="Voice ID for new voiceover")


class RegenerateVoiceoverRequest(BaseModel):
    """Request to regenerate voiceover only"""
    voice_id: Optional[str] = Field(None, description="Voice ID (uses original if not specified)")
    voice_settings: Optional[Dict[str, Any]] = Field(None, description="Voice settings override")


class UploadCustomAudioRequest(BaseModel):
    """Request to upload custom audio for voiceover"""
    # Actual file uploaded via multipart/form-data
    replace_existing: bool = Field(default=True, description="Replace existing voiceover")


class RecomposeVideoRequest(BaseModel):
    """Request to recompose video from current components"""
    quality: str = Field(default="1080p", pattern=r"^(720p|1080p|4k)$", description="Render quality: 720p, 1080p, 4k")
    include_transitions: bool = Field(default=True)


class ReorderSlideRequest(BaseModel):
    """Request to reorder a slide"""
    new_index: int = Field(..., description="New position index for the slide")


class DeleteSlideRequest(BaseModel):
    """Request to delete a slide"""
    # No additional fields needed, slide_id comes from URL path
    pass


class InsertMediaRequest(BaseModel):
    """Request to insert a new media slide"""
    media_type: MediaType = Field(..., description="Type of media: image, video")
    insert_after_slide_id: Optional[str] = Field(None, description="Insert after this slide (None = at beginning)")
    title: Optional[str] = Field(None, description="Optional title for the slide")
    voiceover_text: Optional[str] = Field(None, description="Optional voiceover text")
    duration: float = Field(default=5.0, description="Duration in seconds")
    # Media URL will be set after upload
    media_url: Optional[str] = Field(None, description="URL if already uploaded")


class UploadMediaToSlideRequest(BaseModel):
    """Request to upload media to an existing slide"""
    media_type: MediaType = Field(..., description="Type of media: image, video")
    replace_existing: bool = Field(default=True, description="Replace existing media if present")


class LectureComponentsResponse(BaseModel):
    """Response with lecture components"""
    lecture_id: str
    job_id: str
    status: ComponentStatus
    slides: List[SlideComponent]
    voiceover: Optional[VoiceoverComponent]
    total_duration: float
    video_url: Optional[str]
    is_edited: bool
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None


class SlideComponentResponse(BaseModel):
    """Response with a single slide component"""
    slide: SlideComponent
    lecture_id: str
    message: str = ""


class RegenerateResponse(BaseModel):
    """Response for regeneration requests"""
    success: bool
    message: str
    job_id: Optional[str] = Field(None, description="Background job ID if async")
    result: Optional[Any] = Field(None, description="Result if sync")


# =============================================================================
# Database Models (for PostgreSQL persistence)
# =============================================================================

class LectureComponentsDB(BaseModel):
    """
    Database schema for lecture components.
    Used for lazy loading - only loads full components when editing.
    """
    id: str = Field(..., description="Primary key")
    lecture_id: str = Field(..., description="Foreign key to lecture")
    job_id: str = Field(..., description="Foreign key to course job")

    # JSON columns for complex data
    slides_json: str = Field(..., description="JSON serialized slides")
    voiceover_json: Optional[str] = Field(None, description="JSON serialized voiceover")
    generation_params_json: str = Field(default="{}", description="JSON serialized generation params")

    # Scalar columns
    total_duration: float = Field(default=0.0)
    video_url: Optional[str] = None
    presentation_job_id: Optional[str] = None
    status: str = Field(default="completed")
    is_edited: bool = Field(default=False)
    error: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_lecture_components(self) -> LectureComponents:
        """Convert DB model to domain model"""
        import json
        slides_data = json.loads(self.slides_json) if self.slides_json else []
        voiceover_data = json.loads(self.voiceover_json) if self.voiceover_json else None
        generation_params = json.loads(self.generation_params_json) if self.generation_params_json else {}

        return LectureComponents(
            id=self.id,
            lecture_id=self.lecture_id,
            job_id=self.job_id,
            slides=[SlideComponent(**s) for s in slides_data],
            voiceover=VoiceoverComponent(**voiceover_data) if voiceover_data else None,
            generation_params=generation_params,
            total_duration=self.total_duration,
            video_url=self.video_url,
            presentation_job_id=self.presentation_job_id,
            status=ComponentStatus(self.status),
            is_edited=self.is_edited,
            error=self.error,
            created_at=self.created_at,
            updated_at=self.updated_at
        )

    @classmethod
    def from_lecture_components(cls, components: LectureComponents) -> "LectureComponentsDB":
        """Convert domain model to DB model"""
        import json
        return cls(
            id=components.id,
            lecture_id=components.lecture_id,
            job_id=components.job_id,
            slides_json=json.dumps([s.model_dump() for s in components.slides], default=str),
            voiceover_json=json.dumps(components.voiceover.model_dump(), default=str) if components.voiceover else None,
            generation_params_json=json.dumps(components.generation_params, default=str),
            total_duration=components.total_duration,
            video_url=components.video_url,
            presentation_job_id=components.presentation_job_id,
            status=components.status.value,
            is_edited=components.is_edited,
            error=components.error,
            created_at=components.created_at,
            updated_at=components.updated_at
        )
