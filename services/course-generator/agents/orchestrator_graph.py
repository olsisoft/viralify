"""
Orchestrator Graph

Main LangGraph orchestrator that coordinates the hierarchical subgraphs.
This is the "chef d'orchestre" - a lightweight graph that delegates to specialized subgraphs.

Architecture:
    OrchestratorGraph (this file)
        ├── PlanningSubgraph (planning_graph.py)
        │   └── Handles curriculum planning
        └── ProductionSubgraph (production_graph.py)
            └── Handles per-lecture media production

Flow:
    validate_input -> run_planning -> iterate_lectures -> package_output -> END
                           |               |
                      (uses Planning    (uses Production
                       Subgraph)         Subgraph per lecture)

Benefits:
- Simple, linear main flow
- Complex logic isolated in subgraphs
- Easy to test and debug
- Ready for async/event-driven scaling
"""
import asyncio
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import StateGraph, END

from agents.state import (
    OrchestratorState,
    PlanningState,
    ProductionState,
    LecturePlan,
    create_orchestrator_state,
    create_planning_state_from_orchestrator,
    create_production_state_for_lecture,
    merge_planning_result_to_orchestrator,
    merge_production_result_to_orchestrator,
    PlanningStatus,
    ProductionStatus,
)
from agents.planning_graph import get_planning_graph
from agents.production_graph import get_production_graph
from agents.input_validator import InputValidatorAgent


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_PARALLEL_LECTURES = int(os.getenv("MAX_PARALLEL_LECTURES", "3"))
OUTPUT_DIR = Path(os.getenv("COURSE_OUTPUT_DIR", "/app/output"))


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

async def validate_input(state: OrchestratorState) -> OrchestratorState:
    """
    Node: Validate all input parameters.

    Uses InputValidatorAgent to check required fields and values.
    """
    print(f"[ORCHESTRATOR] Validating input for job: {state.get('job_id', 'Unknown')}", flush=True)

    state["current_stage"] = "validating"

    try:
        agent = InputValidatorAgent()

        # Convert orchestrator state to validation format
        validation_state = {
            "topic": state.get("topic"),
            "description": state.get("description"),
            "profile_category": state.get("profile_category"),
            "difficulty_start": state.get("difficulty_start"),
            "difficulty_end": state.get("difficulty_end"),
            "content_language": state.get("content_language"),
            "target_audience": state.get("target_audience"),
            "structure": state.get("structure") or {
                "total_duration_minutes": state.get("total_duration_minutes", 60),
                "number_of_sections": state.get("number_of_sections", 4),
                "lectures_per_section": state.get("lectures_per_section", 3),
            },
        }

        result = await agent.process(validation_state)

        if result.get("input_validated", False):
            state["input_validated"] = True
            state["validation_errors"] = []
            print("[ORCHESTRATOR] Input validation passed", flush=True)
        else:
            state["input_validated"] = False
            errors = result.get("input_validation_errors", [])
            state["validation_errors"] = [e.get("message", "Unknown error") for e in errors]
            print(f"[ORCHESTRATOR] Input validation failed: {state['validation_errors']}", flush=True)

    except Exception as e:
        print(f"[ORCHESTRATOR] Validation error: {e}", flush=True)
        state["input_validated"] = False
        state["validation_errors"] = [str(e)]

    return state


async def run_planning(state: OrchestratorState) -> OrchestratorState:
    """
    Node: Run the Planning subgraph.

    Delegates to PlanningSubgraph for:
    - Pedagogical analysis
    - Outline generation
    - Lecture plan preparation
    """
    print(f"[ORCHESTRATOR] Starting planning for: {state.get('topic', 'Unknown')}", flush=True)

    state["current_stage"] = "planning"

    # Create planning state from orchestrator state
    planning_state = create_planning_state_from_orchestrator(state)

    # Run planning subgraph
    planning_graph = get_planning_graph()

    try:
        result = await planning_graph.ainvoke(planning_state)

        # Merge results back
        state = merge_planning_result_to_orchestrator(state, result)

        if state.get("planning_completed"):
            print(f"[ORCHESTRATOR] Planning completed: {state.get('total_lectures', 0)} lectures planned", flush=True)
        else:
            print(f"[ORCHESTRATOR] Planning failed: {state.get('errors', [])}", flush=True)

    except Exception as e:
        print(f"[ORCHESTRATOR] Planning error: {e}", flush=True)
        state["planning_completed"] = False
        state["errors"] = state.get("errors", []) + [f"Planning failed: {str(e)}"]

    return state


async def iterate_lectures(state: OrchestratorState) -> OrchestratorState:
    """
    Node: Iterate through lectures, running Production subgraph for each.

    Uses semaphore for bounded parallelism.
    """
    total_lectures = state.get("total_lectures", 0)
    print(f"[ORCHESTRATOR] Starting lecture production ({total_lectures} lectures)", flush=True)

    state["current_stage"] = "producing"

    # Get progress callback from state if available
    progress_callback = state.get("_progress_callback")

    # Report initial production status
    if progress_callback:
        progress_callback(
            stage="producing",
            completed=0,
            total=total_lectures,
            errors=[],
        )

    lecture_plans = state.get("lecture_plans", [])

    if not lecture_plans:
        print("[ORCHESTRATOR] No lecture plans to produce", flush=True)
        return state

    # Create semaphore for bounded parallelism
    semaphore = asyncio.Semaphore(MAX_PARALLEL_LECTURES)
    production_graph = get_production_graph()

    # Shared counter for progress tracking
    completed_count = [0]  # Use list for mutable reference in closure

    async def produce_lecture(lecture_plan: LecturePlan) -> ProductionState:
        """Produce a single lecture with semaphore"""
        async with semaphore:
            print(f"[ORCHESTRATOR] Producing: {lecture_plan.get('title', 'Unknown')} "
                  f"({lecture_plan.get('position', 0)}/{lecture_plan.get('total_lectures', 0)})", flush=True)

            # Create production state
            production_state = create_production_state_for_lecture(state, lecture_plan)

            try:
                # Run production subgraph
                result = await production_graph.ainvoke(production_state)

                # Update progress after each lecture completes
                completed_count[0] += 1
                if progress_callback:
                    progress_callback(
                        stage="producing",
                        completed=completed_count[0],
                        total=total_lectures,
                        errors=[],
                    )

                return result
            except Exception as e:
                print(f"[ORCHESTRATOR] Production error for {lecture_plan.get('lecture_id')}: {e}", flush=True)
                production_state["status"] = ProductionStatus.FAILED
                production_state["last_media_error"] = str(e)

                # Still update progress on failure
                completed_count[0] += 1
                if progress_callback:
                    progress_callback(
                        stage="producing",
                        completed=completed_count[0],
                        total=total_lectures,
                        errors=[str(e)],
                    )

                return production_state

    # Run all lectures (with bounded parallelism via semaphore)
    tasks = [produce_lecture(lp) for lp in lecture_plans]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge all results
    for result in results:
        if isinstance(result, Exception):
            state["errors"] = state.get("errors", []) + [str(result)]
        else:
            state = merge_production_result_to_orchestrator(state, result)

    completed = len(state.get("lectures_completed", []))
    failed = len(state.get("lectures_failed", []))
    skipped = len(state.get("lectures_skipped", []))

    print(f"[ORCHESTRATOR] Production complete: {completed} succeeded, "
          f"{failed} failed, {skipped} skipped", flush=True)

    return state


def _resolve_video_url(url: str) -> str:
    """
    Resolve video URL to a downloadable URL.

    Handles:
    - Internal Docker hostnames (presentation-generator:8006, media-generator:8004)
    - Local file paths (/tmp/...)
    - External URLs (https://...)
    """
    if not url:
        return url

    # Get service URLs from environment
    presentation_url = os.getenv("PRESENTATION_GENERATOR_URL", "http://presentation-generator:8006")
    media_url = os.getenv("MEDIA_GENERATOR_URL", "http://media-generator:8004")

    # If it's already using correct internal hostnames, return as-is
    if "presentation-generator" in url or "media-generator" in url:
        return url

    # Convert localhost URLs to internal Docker hostnames
    if "localhost:8006" in url or "127.0.0.1:8006" in url:
        url = url.replace("localhost:8006", "presentation-generator:8006")
        url = url.replace("127.0.0.1:8006", "presentation-generator:8006")
    elif "localhost:8004" in url or "127.0.0.1:8004" in url:
        url = url.replace("localhost:8004", "media-generator:8004")
        url = url.replace("127.0.0.1:8004", "media-generator:8004")

    # Handle local file paths (convert to HTTP URL)
    if url.startswith("/tmp/presentations/"):
        relative = url.replace("/tmp/presentations/", "")
        url = f"{presentation_url}/files/presentations/{relative}"
    elif url.startswith("/tmp/viralify/videos/"):
        relative = url.replace("/tmp/viralify/videos/", "")
        url = f"{media_url}/files/videos/{relative}"

    return url


async def package_output(state: OrchestratorState) -> OrchestratorState:
    """
    Node: Package all generated videos into a ZIP file.

    Creates downloadable archive of course materials.
    Handles internal Docker URLs properly.
    """
    print("[ORCHESTRATOR] Packaging output", flush=True)

    state["current_stage"] = "packaging"

    video_urls = state.get("video_urls", {})

    if not video_urls:
        print("[ORCHESTRATOR] No videos to package", flush=True)
        state["final_status"] = "failed" if state.get("lectures_failed") else "partial"
        return state

    job_id = state.get("job_id", "unknown")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = OUTPUT_DIR / f"course_{job_id}.zip"
    downloaded_count = 0
    failed_count = 0

    try:
        import httpx

        # Use longer timeout for video downloads
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for lecture_id, video_url in video_urls.items():
                    if not video_url:
                        continue

                    try:
                        # Resolve URL to handle internal Docker hostnames
                        resolved_url = _resolve_video_url(video_url)
                        print(f"[ORCHESTRATOR] Downloading: {lecture_id} from {resolved_url}", flush=True)

                        # Download video with retries
                        for attempt in range(3):
                            try:
                                response = await client.get(resolved_url)
                                response.raise_for_status()
                                break
                            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                                if attempt < 2:
                                    print(f"[ORCHESTRATOR] Retry {attempt + 1} for {lecture_id}: {e}", flush=True)
                                    await asyncio.sleep(2 ** attempt)
                                else:
                                    raise

                        # Add to ZIP
                        filename = f"{lecture_id}.mp4"
                        zf.writestr(filename, response.content)
                        downloaded_count += 1

                        print(f"[ORCHESTRATOR] Added to ZIP: {filename} ({len(response.content) / 1024 / 1024:.1f} MB)", flush=True)

                    except Exception as e:
                        failed_count += 1
                        print(f"[ORCHESTRATOR] Failed to download {lecture_id}: {e}", flush=True)

        # Check if we have any videos
        if downloaded_count == 0:
            print(f"[ORCHESTRATOR] No videos downloaded, ZIP not created", flush=True)
            state["output_zip_url"] = None
        else:
            # Store the local file path (FileResponse will serve it)
            state["output_zip_url"] = str(zip_path)
            print(f"[ORCHESTRATOR] ZIP created: {state['output_zip_url']} ({downloaded_count} videos, {failed_count} failed)", flush=True)

    except Exception as e:
        print(f"[ORCHESTRATOR] Packaging error: {e}", flush=True)
        state["errors"] = state.get("errors", []) + [f"Packaging failed: {str(e)}"]

    return state


async def finalize(state: OrchestratorState) -> OrchestratorState:
    """
    Node: Finalize the job and determine final status.
    """
    print("[ORCHESTRATOR] Finalizing job", flush=True)

    state["current_stage"] = "done"
    state["completed_at"] = datetime.utcnow().isoformat()

    completed = len(state.get("lectures_completed", []))
    failed = len(state.get("lectures_failed", []))
    total = state.get("total_lectures", 0)

    if completed == total:
        state["final_status"] = "success"
    elif completed > 0:
        state["final_status"] = "partial"
    else:
        state["final_status"] = "failed"

    print(f"[ORCHESTRATOR] Job finalized: {state['final_status']} "
          f"({completed}/{total} lectures)", flush=True)

    return state


async def handle_validation_failure(state: OrchestratorState) -> OrchestratorState:
    """Node: Handle validation failures"""
    state["current_stage"] = "done"
    state["final_status"] = "failed"
    state["completed_at"] = datetime.utcnow().isoformat()
    print(f"[ORCHESTRATOR] Validation failed: {state.get('validation_errors', [])}", flush=True)
    return state


async def handle_planning_failure(state: OrchestratorState) -> OrchestratorState:
    """Node: Handle planning failures"""
    state["current_stage"] = "done"
    state["final_status"] = "failed"
    state["completed_at"] = datetime.utcnow().isoformat()
    print(f"[ORCHESTRATOR] Planning failed: {state.get('errors', [])}", flush=True)
    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_validation(
    state: OrchestratorState
) -> Literal["run_planning", "validation_failed"]:
    """Route based on validation result"""
    if state.get("input_validated", False):
        return "run_planning"
    return "validation_failed"


def route_after_planning(
    state: OrchestratorState
) -> Literal["iterate_lectures", "planning_failed"]:
    """Route based on planning result"""
    if state.get("planning_completed", False) and state.get("lecture_plans"):
        return "iterate_lectures"
    return "planning_failed"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

class CourseOrchestrator:
    """
    Main orchestrator for course generation.

    This is the entry point that coordinates:
    1. Input validation
    2. Planning (via PlanningSubgraph)
    3. Production (via ProductionSubgraph per lecture)
    4. Output packaging
    """

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the orchestrator graph"""
        workflow = StateGraph(OrchestratorState)

        # Add nodes
        workflow.add_node("validate_input", validate_input)
        workflow.add_node("validation_failed", handle_validation_failure)
        workflow.add_node("run_planning", run_planning)
        workflow.add_node("planning_failed", handle_planning_failure)
        workflow.add_node("iterate_lectures", iterate_lectures)
        workflow.add_node("package_output", package_output)
        workflow.add_node("finalize", finalize)

        # Set entry point
        workflow.set_entry_point("validate_input")

        # Edges with routing
        workflow.add_conditional_edges(
            "validate_input",
            route_after_validation,
            {
                "run_planning": "run_planning",
                "validation_failed": "validation_failed",
            }
        )

        workflow.add_conditional_edges(
            "run_planning",
            route_after_planning,
            {
                "iterate_lectures": "iterate_lectures",
                "planning_failed": "planning_failed",
            }
        )

        # Linear flow after successful routing
        workflow.add_edge("iterate_lectures", "package_output")
        workflow.add_edge("package_output", "finalize")

        # Terminal nodes
        workflow.add_edge("validation_failed", END)
        workflow.add_edge("planning_failed", END)
        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def run(
        self,
        job_id: str,
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> OrchestratorState:
        """
        Run the complete course generation workflow.

        Args:
            job_id: Unique job identifier
            progress_callback: Optional callback for progress updates
            **kwargs: Request parameters

        Returns:
            Final orchestrator state
        """
        print(f"[ORCHESTRATOR] Starting job: {job_id}", flush=True)
        print(f"[ORCHESTRATOR] Topic: {kwargs.get('topic', 'Unknown')}", flush=True)

        # Create initial state
        initial_state = create_orchestrator_state(job_id, **kwargs)

        # Store callback in state so produce_lectures can use it
        if progress_callback:
            initial_state["_progress_callback"] = progress_callback

        try:
            # Run the graph with streaming for progress updates
            final_state = initial_state

            async for event in self.graph.astream(initial_state):
                # Extract current node and state
                node_name = list(event.keys())[0]
                current_state = event[node_name]
                final_state = current_state

                # Report progress
                if progress_callback:
                    progress_callback(
                        stage=current_state.get("current_stage", "unknown"),
                        completed=len(current_state.get("lectures_completed", [])),
                        total=current_state.get("total_lectures", 0),
                        errors=current_state.get("errors", []),
                    )

            return final_state

        except Exception as e:
            print(f"[ORCHESTRATOR] Error: {e}", flush=True)
            initial_state["errors"] = [str(e)]
            initial_state["final_status"] = "failed"
            initial_state["completed_at"] = datetime.utcnow().isoformat()
            return initial_state

    async def run_planning_only(
        self,
        job_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run only the planning phase (for preview).

        Returns the outline without generating videos.
        """
        print(f"[ORCHESTRATOR] Running planning only for: {kwargs.get('topic', 'Unknown')}", flush=True)

        initial_state = create_orchestrator_state(job_id, **kwargs)

        # Run validation
        state = await validate_input(initial_state)

        if not state.get("input_validated"):
            return {
                "success": False,
                "errors": state.get("validation_errors", []),
            }

        # Run planning
        state = await run_planning(state)

        if state.get("planning_completed"):
            return {
                "success": True,
                "outline": state.get("outline"),
                "lecture_plans": state.get("lecture_plans"),
                "total_lectures": state.get("total_lectures"),
                "content_preferences": state.get("content_preferences"),
            }
        else:
            return {
                "success": False,
                "errors": state.get("errors", []),
            }


# =============================================================================
# SINGLETON AND FACTORY
# =============================================================================

_orchestrator_instance: Optional[CourseOrchestrator] = None


def get_course_orchestrator() -> CourseOrchestrator:
    """Get the singleton course orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = CourseOrchestrator()
    return _orchestrator_instance


def create_course_orchestrator() -> CourseOrchestrator:
    """Factory function to create a new orchestrator instance"""
    return CourseOrchestrator()
