"""
Scene Subgraph

LangGraph subgraph that processes a single scene through the multi-agent pipeline.
Each scene goes through: Plan -> Audio -> Visual -> Animation -> Validate
with potential regeneration loops if sync fails.
"""

import os
from typing import Any, Dict, Annotated, Literal
from langgraph.graph import StateGraph, END

from .base_agent import SceneState, SyncStatus, ScenePackage
from .scene_planner import ScenePlannerAgent
from .audio_agent import AudioAgent
from .visual_sync_agent import VisualSyncAgent
from .animation_agent import AnimationAgent
from .scene_validator import SceneValidatorAgent


def create_scene_graph() -> StateGraph:
    """Create the scene processing subgraph"""

    # Initialize agents
    scene_planner = ScenePlannerAgent()
    audio_agent = AudioAgent()
    visual_sync_agent = VisualSyncAgent()
    animation_agent = AnimationAgent()
    scene_validator = SceneValidatorAgent()

    # Define the graph
    graph = StateGraph(SceneState)

    # Node: Plan the scene
    async def plan_scene(state: SceneState) -> Dict[str, Any]:
        """Plan the scene with timing cues"""
        result = await scene_planner.execute({
            "slide_data": state.get("slide_data", {}),
            "scene_index": state.get("scene_index", 0),
            "job_id": state.get("job_id", "")
        })

        if result.success:
            return {
                "planned_content": result.data.get("planned_content", {}),
                "timing_cues": result.data.get("timing_cues", []),
                "voiceover_text": result.data.get("voiceover_text", ""),
                "estimated_duration": result.data.get("estimated_duration", 10)
            }
        else:
            return {
                "errors": state.get("errors", []) + result.errors
            }

    # Node: Generate audio with timestamps
    async def generate_audio(state: SceneState) -> Dict[str, Any]:
        """Generate audio with word-level timestamps"""
        voiceover_text = state.get("voiceover_text") or state.get("slide_data", {}).get("voiceover_text", "")
        content_language = state.get("content_language", "en")

        result = await audio_agent.execute({
            "voiceover_text": voiceover_text,
            "scene_index": state.get("scene_index", 0),
            "job_id": state.get("job_id", ""),
            "content_language": content_language
        })

        if result.success:
            return {
                "audio_result": {
                    "audio_url": result.data.get("audio_url", ""),
                    "duration": result.data.get("audio_duration", 0),
                    "word_timestamps": result.data.get("word_timestamps", []),
                    "transcript": result.data.get("transcript", "")
                }
            }
        else:
            return {
                "errors": state.get("errors", []) + result.errors
            }

    # Node: Sync visuals to audio
    async def sync_visuals(state: SceneState) -> Dict[str, Any]:
        """Align visual elements to audio timestamps and generate slide image"""
        audio_result = state.get("audio_result", {})

        result = await visual_sync_agent.execute({
            "slide_data": state.get("slide_data", {}),
            "word_timestamps": audio_result.get("word_timestamps", []),
            "timing_cues": state.get("timing_cues", []),
            "audio_duration": audio_result.get("duration", 0),
            "scene_index": state.get("scene_index", 0),
            "job_id": state.get("job_id", ""),
            "style": state.get("style", "dark")
        })

        if result.success:
            return {
                "visual_elements": result.data.get("visual_elements", []),
                "sync_map": result.data.get("sync_map", {}),
                "primary_visual": {
                    "url": result.data.get("primary_visual_url", ""),
                    "type": result.data.get("primary_visual_type", "image")
                }
            }
        else:
            return {
                "errors": state.get("errors", []) + result.errors
            }

    # Node: Create animations
    async def create_animations(state: SceneState) -> Dict[str, Any]:
        """Create animations timed to audio"""
        audio_result = state.get("audio_result", {})

        result = await animation_agent.execute({
            "slide_data": state.get("slide_data", {}),
            "visual_elements": state.get("visual_elements", []),
            "sync_map": state.get("sync_map", {}),
            "word_timestamps": audio_result.get("word_timestamps", []),
            "audio_duration": audio_result.get("duration", 0),
            "scene_index": state.get("scene_index", 0),
            "job_id": state.get("job_id", "")
        })

        if result.success:
            return {
                "animation_result": {
                    "animations": result.data.get("animations", []),
                    "scene_animation": result.data.get("scene_animation", {})
                }
            }
        else:
            return {
                "warnings": state.get("warnings", []) + ["Animation creation failed, using fallback"]
            }

    # Node: Validate sync
    async def validate_sync(state: SceneState) -> Dict[str, Any]:
        """Validate audio-visual synchronization"""
        audio_result = state.get("audio_result", {})
        animation_result = state.get("animation_result", {})

        result = await scene_validator.execute({
            "scene_index": state.get("scene_index", 0),
            "word_timestamps": audio_result.get("word_timestamps", []),
            "visual_elements": state.get("visual_elements", []),
            "sync_map": state.get("sync_map", {}),
            "animations": animation_result.get("animations", []),
            "audio_duration": audio_result.get("duration", 0),
            "timing_cues": state.get("timing_cues", []),
            "slide_data": state.get("slide_data", {}),
            "iteration": state.get("iteration", 0)
        })

        return {
            "sync_status": result.data.get("sync_status", SyncStatus.PENDING.value),
            "sync_score": result.data.get("sync_score", 0),
            "sync_issues": result.data.get("issues", []),
            "iteration": state.get("iteration", 0) + 1
        }

    # Node: Build scene package
    async def build_package(state: SceneState) -> Dict[str, Any]:
        """Build the final scene package"""
        audio_result = state.get("audio_result", {})
        visual_elements = state.get("visual_elements", [])
        primary_visual_data = state.get("primary_visual", {})

        # Get primary visual URL from state or find from elements
        primary_visual = primary_visual_data.get("url", "")
        if not primary_visual:
            for ve in visual_elements:
                if ve.get("url"):
                    primary_visual = ve.get("url")
                    break
                elif ve.get("file_path"):
                    primary_visual = ve.get("file_path")
                    break

        scene_package = {
            "scene_id": state.get("scene_id", f"scene_{state.get('scene_index', 0)}"),
            "scene_index": state.get("scene_index", 0),
            "title": state.get("slide_data", {}).get("title", ""),
            "content_type": state.get("slide_data", {}).get("type", "content"),
            "audio_url": audio_result.get("audio_url", ""),
            "audio_duration": audio_result.get("duration", 0),
            "word_timestamps": audio_result.get("word_timestamps", []),
            "visual_elements": visual_elements,
            "primary_visual_url": primary_visual,
            "total_duration": audio_result.get("duration", 0),
            "timing_cues": state.get("timing_cues", []),
            "sync_status": state.get("sync_status", SyncStatus.PENDING.value),
            "sync_score": state.get("sync_score", 0),
            "sync_issues": state.get("sync_issues", []),
            "animations": state.get("animation_result", {}).get("animations", [])
        }

        return {
            "scene_package": scene_package
        }

    # Routing function
    def should_regenerate(state: SceneState) -> Literal["regenerate", "build_package"]:
        """Decide if regeneration is needed"""
        sync_status = state.get("sync_status", SyncStatus.PENDING.value)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        if sync_status == SyncStatus.OUT_OF_SYNC.value and iteration < max_iterations:
            return "regenerate"
        return "build_package"

    # Add nodes
    graph.add_node("plan_scene", plan_scene)
    graph.add_node("generate_audio", generate_audio)
    graph.add_node("sync_visuals", sync_visuals)
    graph.add_node("create_animations", create_animations)
    graph.add_node("validate_sync", validate_sync)
    graph.add_node("build_package", build_package)

    # Add edges (linear flow with validation loop)
    graph.add_edge("plan_scene", "generate_audio")
    graph.add_edge("generate_audio", "sync_visuals")
    graph.add_edge("sync_visuals", "create_animations")
    graph.add_edge("create_animations", "validate_sync")

    # Conditional edge: regenerate or proceed to package
    graph.add_conditional_edges(
        "validate_sync",
        should_regenerate,
        {
            "regenerate": "sync_visuals",  # Go back to visual sync
            "build_package": "build_package"
        }
    )

    graph.add_edge("build_package", END)

    # Set entry point
    graph.set_entry_point("plan_scene")

    return graph


def create_initial_scene_state(
    slide_data: Dict[str, Any],
    scene_index: int,
    job_id: str,
    style: str = "modern",
    content_language: str = "en"
) -> SceneState:
    """Create initial state for scene processing"""
    return {
        "scene_id": f"scene_{scene_index}",
        "scene_index": scene_index,
        "slide_data": slide_data,
        "style": style,
        "job_id": job_id,
        "content_language": content_language,
        "planned_content": None,
        "timing_cues": [],
        "audio_result": None,
        "visual_elements": [],
        "primary_visual": None,
        "animation_result": None,
        "sync_status": SyncStatus.PENDING.value,
        "sync_score": 0,
        "sync_issues": [],
        "iteration": 0,
        "max_iterations": 3,
        "errors": [],
        "warnings": [],
        "scene_package": None
    }


# Compile the graph
scene_graph = create_scene_graph()
compiled_scene_graph = scene_graph.compile()
