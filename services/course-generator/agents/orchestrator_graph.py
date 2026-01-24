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
from agents.production_graph import (
    get_production_graph,
    register_lecture_progress_callback,
    unregister_lecture_progress_callback,
)
from agents.input_validator import InputValidatorAgent
from services.coherence_service import get_coherence_service

# Global registry for progress callbacks (indexed by job_id)
# This is needed because LangGraph state doesn't preserve callable objects
_progress_callbacks: Dict[str, callable] = {}


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


async def check_coherence(state: OrchestratorState) -> OrchestratorState:
    """
    Node: Check pedagogical coherence of the planned course.

    Phase 2: Validates that:
    - Prerequisites are met before each lecture
    - Concepts are introduced in logical order
    - There are no large knowledge gaps

    Also enriches lectures with coherence metadata.
    """
    print(f"[ORCHESTRATOR] Checking coherence for job: {state.get('job_id', 'Unknown')}", flush=True)

    state["current_stage"] = "checking_coherence"

    outline = state.get("outline")
    if not outline:
        print("[ORCHESTRATOR] No outline to check coherence", flush=True)
        state["coherence_checked"] = True
        state["coherence_score"] = 0.0
        return state

    try:
        coherence_service = get_coherence_service()

        # Check coherence
        result = await coherence_service.check_coherence(outline, verbose=True)

        state["coherence_checked"] = True
        state["coherence_score"] = result.score
        state["coherence_issues"] = [
            {
                "type": issue.issue_type,
                "severity": issue.severity,
                "lecture": issue.lecture_title,
                "description": issue.description,
                "suggestion": issue.suggestion,
            }
            for issue in result.issues
        ]

        # Enrich outline with coherence metadata if coherent enough
        if result.score >= 50.0:
            enriched_outline = await coherence_service.enrich_outline_with_coherence(outline)
            state["outline"] = enriched_outline
            print(f"[ORCHESTRATOR] Coherence check passed (score: {result.score:.0f}/100)", flush=True)
        else:
            print(f"[ORCHESTRATOR] Coherence score low ({result.score:.0f}/100), proceeding anyway", flush=True)

    except Exception as e:
        print(f"[ORCHESTRATOR] Coherence check error: {e}, proceeding anyway", flush=True)
        state["coherence_checked"] = True
        state["coherence_score"] = 50.0  # Assume neutral score on error
        state["coherence_issues"] = []

    return state


async def iterate_lectures(state: OrchestratorState) -> OrchestratorState:
    """
    Node: Iterate through lectures, running Production subgraph for each.

    Uses semaphore for bounded parallelism.
    """
    total_lectures = state.get("total_lectures", 0)
    print(f"[ORCHESTRATOR] Starting lecture production ({total_lectures} lectures)", flush=True)

    # Get progress callback and send outline before starting production
    job_id = state.get("job_id", "")
    progress_callback = _progress_callbacks.get(job_id)

    # Send outline to main.py so it can be set on the job BEFORE lecture updates
    if progress_callback:
        progress_callback(
            stage="outline_ready",
            completed=0,
            total=total_lectures,
            in_progress=0,
            current_lectures=[],
            errors=[],
            outline_data=state.get("outline"),
        )

    state["current_stage"] = "producing"

    # Get progress callback from global registry
    job_id = state.get("job_id", "")
    progress_callback = _progress_callbacks.get(job_id)
    print(f"[ORCHESTRATOR] Progress callback for job {job_id}: {'found' if progress_callback else 'NOT FOUND'}", flush=True)

    lecture_plans = state.get("lecture_plans", [])

    if not lecture_plans:
        print("[ORCHESTRATOR] No lecture plans to produce", flush=True)
        return state

    # Create semaphore for bounded parallelism
    semaphore = asyncio.Semaphore(MAX_PARALLEL_LECTURES)
    production_graph = get_production_graph()

    # Shared counters for progress tracking (use lists for mutable reference in closure)
    completed_count = [0]
    in_progress_count = [0]
    in_progress_titles = []  # Track which lectures are in progress

    # Report initial production status (all lectures pending)
    if progress_callback:
        # First, mark all lectures as pending
        for lp in lecture_plans:
            progress_callback(
                stage="producing",
                completed=0,
                total=total_lectures,
                in_progress=0,
                current_lectures=[],
                errors=[],
                lecture_update={
                    "lecture_id": lp.get("lecture_id", ""),
                    "title": lp.get("title", ""),
                    "status": "pending",
                    "current_stage": None,
                    "progress_percent": 0.0,
                },
            )

    async def produce_lecture(lecture_plan: LecturePlan) -> ProductionState:
        """Produce a single lecture with semaphore"""
        lecture_title = lecture_plan.get('title', 'Unknown')
        lecture_id = lecture_plan.get('lecture_id', '')

        async with semaphore:
            # Track lecture starting
            in_progress_count[0] += 1
            in_progress_titles.append(lecture_title)

            print(f"[ORCHESTRATOR] Producing: {lecture_title} "
                  f"({lecture_plan.get('position', 0)}/{lecture_plan.get('total_lectures', 0)})", flush=True)

            # Report that this lecture started
            if progress_callback:
                progress_callback(
                    stage="producing",
                    completed=completed_count[0],
                    total=total_lectures,
                    in_progress=in_progress_count[0],
                    current_lectures=list(in_progress_titles),
                    errors=[],
                    lecture_update={
                        "lecture_id": lecture_id,
                        "title": lecture_title,
                        "status": "generating",
                        "current_stage": "starting",
                        "progress_percent": 0.0,
                    },
                )

            # Register a lecture-specific progress callback for real-time updates
            def lecture_progress_handler(stage: str, progress: float, status: str):
                """Forward lecture progress to main callback"""
                if progress_callback:
                    progress_callback(
                        stage="producing",
                        completed=completed_count[0],
                        total=total_lectures,
                        in_progress=in_progress_count[0],
                        current_lectures=list(in_progress_titles),
                        errors=[],
                        lecture_update={
                            "lecture_id": lecture_id,
                            "title": lecture_title,
                            "status": status,
                            "current_stage": stage,
                            "progress_percent": progress,
                        },
                    )

            # Register the callback so production_graph can use it
            register_lecture_progress_callback(lecture_id, lecture_progress_handler)

            # Create production state
            production_state = create_production_state_for_lecture(state, lecture_plan)

            try:
                # Run production subgraph
                result = await production_graph.ainvoke(production_state)

                # Get video URL from result
                video_url = None
                media_result = result.get("media_result", {})
                if media_result:
                    video_url = media_result.get("video_url")

                # Update progress after lecture completes
                completed_count[0] += 1
                in_progress_count[0] -= 1
                if lecture_title in in_progress_titles:
                    in_progress_titles.remove(lecture_title)

                if progress_callback:
                    progress_callback(
                        stage="producing",
                        completed=completed_count[0],
                        total=total_lectures,
                        in_progress=in_progress_count[0],
                        current_lectures=list(in_progress_titles),
                        errors=[],
                        lecture_update={
                            "lecture_id": lecture_id,
                            "title": lecture_title,
                            "status": "completed",
                            "current_stage": "completed",
                            "progress_percent": 100.0,
                            "video_url": video_url,
                        },
                    )

                return result
            except Exception as e:
                print(f"[ORCHESTRATOR] Production error for {lecture_plan.get('lecture_id')}: {e}", flush=True)
                production_state["status"] = ProductionStatus.FAILED
                production_state["last_media_error"] = str(e)

                # Still update progress on failure
                completed_count[0] += 1
                in_progress_count[0] -= 1
                if lecture_title in in_progress_titles:
                    in_progress_titles.remove(lecture_title)

                if progress_callback:
                    progress_callback(
                        stage="producing",
                        completed=completed_count[0],
                        total=total_lectures,
                        in_progress=in_progress_count[0],
                        current_lectures=list(in_progress_titles),
                        errors=[str(e)],
                        lecture_update={
                            "lecture_id": lecture_id,
                            "title": lecture_title,
                            "status": "failed",
                            "current_stage": "failed",
                            "progress_percent": 0.0,
                            "error": str(e),
                        },
                    )

                return production_state
            finally:
                # Always unregister the callback
                unregister_lecture_progress_callback(lecture_id)

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
) -> Literal["check_coherence", "planning_failed"]:
    """Route based on planning result"""
    if state.get("planning_completed", False) and state.get("lecture_plans"):
        return "check_coherence"  # Go to coherence check
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
        workflow.add_node("check_coherence", check_coherence)  # Phase 2: Coherence check
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
                "check_coherence": "check_coherence",  # Go to coherence check instead of iterate_lectures
                "planning_failed": "planning_failed",
            }
        )

        # Linear flow after coherence check
        workflow.add_edge("check_coherence", "iterate_lectures")

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

        # Store callback in global registry (LangGraph doesn't preserve callables in state)
        if progress_callback:
            _progress_callbacks[job_id] = progress_callback
            print(f"[ORCHESTRATOR] Registered progress callback for job {job_id}", flush=True)

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

        finally:
            # Clean up callback from registry
            if job_id in _progress_callbacks:
                del _progress_callbacks[job_id]
                print(f"[ORCHESTRATOR] Cleaned up progress callback for job {job_id}", flush=True)

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
