"""
Base Agent class and common types for the multi-agent system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict
from enum import Enum
from datetime import datetime


class SyncStatus(str, Enum):
    """Synchronization status for a scene"""
    PENDING = "pending"
    IN_SYNC = "in_sync"
    OUT_OF_SYNC = "out_of_sync"
    FAILED = "failed"


@dataclass
class TimingCue:
    """A timing cue that links audio to visual events"""
    timestamp: float  # seconds from scene start
    event_type: str  # "show_text", "show_code", "highlight", "show_output", etc.
    target: str  # What to show/highlight
    duration: Optional[float] = None  # How long the event lasts
    description: str = ""  # Human-readable description


@dataclass
class WordTimestamp:
    """Timestamp for a single word in the audio"""
    word: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class AudioResult:
    """Result from audio generation"""
    audio_url: str
    duration: float
    word_timestamps: List[WordTimestamp]
    transcript: str


@dataclass
class VisualElement:
    """A visual element with timing"""
    element_type: str  # "image", "animation", "text_overlay"
    file_path: str
    url: str
    start_time: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenePackage:
    """Complete package for a single scene, ready for composition"""
    scene_id: str
    scene_index: int

    # Content
    title: Optional[str]
    content_type: str  # "title", "content", "code", "code_demo", "conclusion"

    # Audio
    audio_url: str
    audio_duration: float
    word_timestamps: List[WordTimestamp]

    # Visual
    visual_elements: List[VisualElement]
    primary_visual_url: str  # Main image or video

    # Timing
    total_duration: float
    timing_cues: List[TimingCue]

    # Sync status
    sync_status: SyncStatus
    sync_score: float  # 0-1, how well synced
    sync_issues: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class SceneState(TypedDict):
    """State for processing a single scene"""
    # Input
    scene_id: str
    scene_index: int
    slide_data: Dict[str, Any]
    style: str
    job_id: str

    # Planning
    planned_content: Optional[Dict[str, Any]]
    timing_cues: List[Dict[str, Any]]

    # Audio
    audio_result: Optional[Dict[str, Any]]

    # Visual
    visual_elements: List[Dict[str, Any]]
    primary_visual: Optional[Dict[str, Any]]

    # Animation
    animation_result: Optional[Dict[str, Any]]

    # Validation
    sync_status: str
    sync_score: float
    sync_issues: List[str]

    # Control
    iteration: int
    max_iterations: int
    errors: List[str]
    warnings: List[str]

    # Output
    scene_package: Optional[Dict[str, Any]]


class MainState(TypedDict):
    """State for the main orchestrator"""
    # Input
    request: Dict[str, Any]
    job_id: str

    # Script
    script: Optional[Dict[str, Any]]
    slides: List[Dict[str, Any]]

    # Scene processing
    scene_packages: List[Dict[str, Any]]
    current_scene_index: int

    # Final output
    output_video_url: Optional[str]
    total_duration: float

    # Control
    phase: str
    errors: List[str]
    warnings: List[str]

    # Metadata
    started_at: str
    completed_at: Optional[str]


@dataclass
class AgentResult:
    """Standard result from any agent"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> AgentResult:
        """Execute the agent's task"""
        pass

    def log(self, message: str):
        """Log a message with agent name prefix"""
        print(f"[{self.name}] {message}", flush=True)
