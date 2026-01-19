"""
Multi-Agent System for Video Generation

This module implements a scene-by-scene multi-agent architecture where each slide
is processed as an independent unit with perfect audio-visual synchronization.

Agents:
- ScenePlannerAgent: Plans content and timing cues for a single scene
- AudioAgent: Generates TTS with precise word timestamps
- VisualSyncAgent: Aligns visual elements to audio timestamps
- AnimationAgent: Creates animations timed to audio
- SceneValidatorAgent: Verifies sync and triggers regeneration if needed
- CompositorAgent: Assembles final video from pre-synced scenes
"""

from .base_agent import (
    BaseAgent,
    AgentResult,
    SceneState,
    MainState,
    TimingCue,
    WordTimestamp,
    AudioResult,
    VisualElement,
    ScenePackage,
    SyncStatus,
)
from .scene_planner import ScenePlannerAgent
from .audio_agent import AudioAgent
from .visual_sync_agent import VisualSyncAgent
from .animation_agent import AnimationAgent
from .scene_validator import SceneValidatorAgent
from .compositor_agent import CompositorAgent
from .scene_graph import create_scene_graph, create_initial_scene_state, compiled_scene_graph
from .main_orchestrator import (
    create_main_graph,
    create_initial_main_state,
    compiled_main_graph,
    generate_presentation_video
)

__all__ = [
    # Base types
    "BaseAgent",
    "AgentResult",
    "SceneState",
    "MainState",
    "TimingCue",
    "WordTimestamp",
    "AudioResult",
    "VisualElement",
    "ScenePackage",
    "SyncStatus",
    # Agents
    "ScenePlannerAgent",
    "AudioAgent",
    "VisualSyncAgent",
    "AnimationAgent",
    "SceneValidatorAgent",
    "CompositorAgent",
    # Graphs
    "create_scene_graph",
    "create_initial_scene_state",
    "compiled_scene_graph",
    "create_main_graph",
    "create_initial_main_state",
    "compiled_main_graph",
    # Entry point
    "generate_presentation_video",
]
