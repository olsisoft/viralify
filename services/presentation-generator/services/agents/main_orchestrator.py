"""
Main Orchestrator

The main LangGraph workflow that orchestrates the entire video generation process.
Processes scenes in parallel and composes the final video.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from langgraph.graph import StateGraph, END

from .base_agent import MainState, SyncStatus
from .scene_graph import compiled_scene_graph, create_initial_scene_state
from .compositor_agent import CompositorAgent


def create_main_graph() -> StateGraph:
    """Create the main orchestration graph"""

    compositor = CompositorAgent()

    # Define the graph
    graph = StateGraph(MainState)

    # Node: Initialize job
    async def initialize(state: MainState) -> Dict[str, Any]:
        """Initialize the video generation job"""
        print(f"[ORCHESTRATOR] Initializing job {state.get('job_id')}", flush=True)

        request = state.get("request", {})
        slides = request.get("slides", [])

        return {
            "slides": slides,
            "scene_packages": [],
            "current_scene_index": 0,
            "phase": "processing_scenes",
            "started_at": datetime.utcnow().isoformat(),
            "total_duration": 0
        }

    # Node: Process all scenes in parallel
    async def process_scenes(state: MainState) -> Dict[str, Any]:
        """Process all scenes through the scene subgraph in parallel"""
        slides = state.get("slides", [])
        job_id = state.get("job_id", "unknown")
        request = state.get("request", {})
        style = request.get("style", "modern")
        content_language = request.get("content_language", "en")
        voice_id = request.get("voice_id")  # User-selected voice

        print(f"[ORCHESTRATOR] Processing {len(slides)} scenes in parallel (language: {content_language}, voice: {voice_id or 'default'})", flush=True)

        # Create tasks for parallel processing
        async def process_single_scene(slide: Dict[str, Any], index: int) -> Dict[str, Any]:
            """Process a single scene through the scene graph"""
            try:
                initial_state = create_initial_scene_state(
                    slide_data=slide,
                    scene_index=index,
                    job_id=job_id,
                    style=style,
                    content_language=content_language,
                    voice_id=voice_id,  # Pass user-selected voice
                )

                # Run the scene graph
                final_state = await compiled_scene_graph.ainvoke(initial_state)

                scene_package = final_state.get("scene_package", {})
                print(f"[ORCHESTRATOR] Scene {index} complete: sync_score={scene_package.get('sync_score', 0):.2f}", flush=True)

                return scene_package

            except Exception as e:
                print(f"[ORCHESTRATOR] Scene {index} failed: {e}", flush=True)
                # Return minimal scene package on failure
                return {
                    "scene_id": f"scene_{index}",
                    "scene_index": index,
                    "title": slide.get("title", ""),
                    "content_type": slide.get("type", "content"),
                    "sync_status": SyncStatus.FAILED.value,
                    "sync_score": 0,
                    "errors": [str(e)]
                }

        # Process all scenes in parallel with concurrency limit
        max_concurrent = int(os.getenv("MAX_CONCURRENT_SCENES", "3"))
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(slide: Dict[str, Any], index: int):
            async with semaphore:
                return await process_single_scene(slide, index)

        # Create tasks
        tasks = [
            process_with_limit(slide, i)
            for i, slide in enumerate(slides)
        ]

        # Wait for all scenes to complete
        scene_packages = await asyncio.gather(*tasks)

        # Sort by scene index (in case order was lost)
        scene_packages = sorted(scene_packages, key=lambda x: x.get("scene_index", 0))

        # Calculate total duration
        total_duration = sum(
            sp.get("total_duration", sp.get("audio_duration", 0))
            for sp in scene_packages
        )

        # Collect warnings and errors
        all_errors = []
        all_warnings = []
        for sp in scene_packages:
            if sp.get("errors"):
                all_errors.extend(sp["errors"])
            if sp.get("sync_issues"):
                for issue in sp["sync_issues"]:
                    if issue.get("severity") == "critical":
                        all_warnings.append(issue.get("description", ""))

        print(f"[ORCHESTRATOR] All scenes processed. Total duration: {total_duration:.1f}s", flush=True)

        return {
            "scene_packages": scene_packages,
            "total_duration": total_duration,
            "phase": "composing",
            "errors": all_errors,
            "warnings": all_warnings
        }

    # Node: Compose final video
    async def compose_video(state: MainState) -> Dict[str, Any]:
        """Compose the final video from scene packages"""
        scene_packages = state.get("scene_packages", [])
        job_id = state.get("job_id", "unknown")
        title = state.get("request", {}).get("title", "presentation")

        print(f"[ORCHESTRATOR] Composing final video from {len(scene_packages)} scenes", flush=True)

        # Filter out failed scenes
        valid_packages = [
            sp for sp in scene_packages
            if sp.get("sync_status") != SyncStatus.FAILED.value or sp.get("audio_url")
        ]

        if not valid_packages:
            return {
                "phase": "failed",
                "errors": state.get("errors", []) + ["No valid scenes to compose"]
            }

        result = await compositor.execute({
            "scene_packages": valid_packages,
            "job_id": job_id,
            "title": title
        })

        if result.success:
            return {
                "output_video_url": result.data.get("output_url", ""),
                "total_duration": result.data.get("duration", 0),
                "phase": "completed",
                "completed_at": datetime.utcnow().isoformat()
            }
        else:
            return {
                "phase": "failed",
                "errors": state.get("errors", []) + result.errors
            }

    # Node: Finalize
    async def finalize(state: MainState) -> Dict[str, Any]:
        """Finalize the job and create output summary"""
        phase = state.get("phase", "unknown")
        job_id = state.get("job_id", "unknown")

        if phase == "completed":
            print(f"[ORCHESTRATOR] Job {job_id} completed successfully", flush=True)
        else:
            print(f"[ORCHESTRATOR] Job {job_id} finished with status: {phase}", flush=True)

        # Build summary
        scene_packages = state.get("scene_packages", [])
        sync_scores = [sp.get("sync_score", 0) for sp in scene_packages if sp.get("sync_score")]
        avg_sync_score = sum(sync_scores) / len(sync_scores) if sync_scores else 0

        return {
            "summary": {
                "job_id": job_id,
                "status": phase,
                "total_scenes": len(scene_packages),
                "total_duration": state.get("total_duration", 0),
                "average_sync_score": avg_sync_score,
                "output_url": state.get("output_video_url"),
                "started_at": state.get("started_at"),
                "completed_at": state.get("completed_at"),
                "errors": state.get("errors", []),
                "warnings": state.get("warnings", [])
            }
        }

    # Routing function
    def route_after_scenes(state: MainState) -> Literal["compose_video", "finalize"]:
        """Route after scene processing"""
        scene_packages = state.get("scene_packages", [])
        valid_count = len([
            sp for sp in scene_packages
            if sp.get("sync_status") != SyncStatus.FAILED.value or sp.get("audio_url")
        ])

        if valid_count > 0:
            return "compose_video"
        return "finalize"

    # Add nodes
    graph.add_node("initialize", initialize)
    graph.add_node("process_scenes", process_scenes)
    graph.add_node("compose_video", compose_video)
    graph.add_node("finalize", finalize)

    # Add edges
    graph.add_edge("initialize", "process_scenes")
    graph.add_conditional_edges(
        "process_scenes",
        route_after_scenes,
        {
            "compose_video": "compose_video",
            "finalize": "finalize"
        }
    )
    graph.add_edge("compose_video", "finalize")
    graph.add_edge("finalize", END)

    # Set entry point
    graph.set_entry_point("initialize")

    return graph


def create_initial_main_state(
    job_id: str,
    request: Dict[str, Any]
) -> MainState:
    """Create initial state for main orchestrator"""
    return {
        "request": request,
        "job_id": job_id,
        "script": None,
        "slides": [],
        "scene_packages": [],
        "current_scene_index": 0,
        "output_video_url": None,
        "total_duration": 0,
        "phase": "initializing",
        "errors": [],
        "warnings": [],
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None
    }


# Compile the main graph
main_graph = create_main_graph()
compiled_main_graph = main_graph.compile()


async def generate_presentation_video(
    job_id: str,
    slides: List[Dict[str, Any]],
    title: str = "Presentation",
    style: str = "modern",
    content_language: str = "en"
) -> Dict[str, Any]:
    """
    Main entry point for video generation.

    Args:
        job_id: Unique job identifier
        slides: List of slide data dictionaries
        title: Presentation title
        style: Visual style
        content_language: Language code for voiceover (en, fr, es, etc.)

    Returns:
        Final state with output video URL and summary
    """
    print(f"[ORCHESTRATOR] Starting video generation: {job_id}", flush=True)
    print(f"[ORCHESTRATOR] {len(slides)} slides, style={style}, language={content_language}", flush=True)

    request = {
        "slides": slides,
        "title": title,
        "style": style,
        "content_language": content_language
    }

    initial_state = create_initial_main_state(job_id, request)

    try:
        final_state = await compiled_main_graph.ainvoke(initial_state)
        return {
            "success": final_state.get("phase") == "completed",
            "output_url": final_state.get("output_video_url"),
            "duration": final_state.get("total_duration", 0),
            "summary": final_state.get("summary", {}),
            "errors": final_state.get("errors", []),
            "warnings": final_state.get("warnings", [])
        }

    except Exception as e:
        print(f"[ORCHESTRATOR] Fatal error: {e}", flush=True)
        return {
            "success": False,
            "output_url": None,
            "duration": 0,
            "summary": {"status": "failed", "job_id": job_id},
            "errors": [str(e)],
            "warnings": []
        }
