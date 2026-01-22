"""
Production Subgraph

LangGraph subgraph for single lecture media production.
This subgraph handles the complex production pipeline with recovery loops.

Responsibilities:
1. Script writing (if not provided)
2. Code generation and review loop
3. Media generation (video)
4. Error recovery (script simplification, animation reduction)

Flow:
    write_script -> generate_code -> review_code -> generate_media -> END
                         ^              |                  |
                         |    (bad)     |      (failure)   |
                         +<-------------+                  v
                                              handle_failure
                                                    |
                                    +---------------+---------------+
                                    |               |               |
                               simplify       reduce_anim        skip
                                    |               |               |
                                    +-------+-------+               |
                                            |                       |
                                            v                       |
                                     generate_media                 |
                                            |                       |
                                            +------------->---------+
                                                                    |
                                                                   END
"""
import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import StateGraph, END

from agents.state import (
    ProductionState,
    ProductionStatus,
    RecoveryStrategy,
    CodeBlockInfo,
    GeneratedCodeBlock,
    MediaResult,
)
from agents.code_expert import CodeExpertAgent
from agents.code_reviewer import CodeReviewerAgent
from agents.script_simplifier import ScriptSimplifierAgent


# =============================================================================
# SERVICE CLIENTS
# =============================================================================

class MediaGeneratorClient:
    """Client for interacting with the media generator service"""

    def __init__(self):
        self.base_url = os.getenv("PRESENTATION_GENERATOR_URL", "http://presentation-generator:8006")
        self.timeout = float(os.getenv("MEDIA_GENERATION_TIMEOUT", "600"))  # 10 minutes default

    async def generate_lecture_video(
        self,
        lecture_plan: Dict[str, Any],
        voiceover_script: str,
        code_blocks: List[GeneratedCodeBlock],
        settings: Dict[str, Any],
    ) -> MediaResult:
        """
        Generate video for a lecture.

        This calls the presentation-generator service.
        """
        import httpx

        lecture_id = lecture_plan.get("lecture_id", "unknown")

        # Build request payload
        payload = {
            "title": lecture_plan.get("title", "Untitled Lecture"),
            "topic": lecture_plan.get("description", ""),
            "voiceover_text": voiceover_script,
            "code_blocks": [
                {
                    "language": cb.get("language", "python"),
                    "code": cb.get("code", ""),
                    "explanation": cb.get("explanation", ""),
                }
                for cb in code_blocks
            ],
            "settings": {
                "voice_id": settings.get("voice_id", "default"),
                "style": settings.get("style", "modern"),
                "typing_speed": settings.get("typing_speed", "natural"),
                "show_typing_animation": not settings.get("animations_disabled", False),
                "include_avatar": settings.get("include_avatar", False),
                "avatar_id": settings.get("avatar_id"),
                "language": settings.get("content_language", "en"),
            },
            "lecture_metadata": {
                "position": lecture_plan.get("position", 1),
                "total_lectures": lecture_plan.get("total_lectures", 1),
                "objectives": lecture_plan.get("objectives", []),
            }
        }

        print(f"[PRODUCTION] Submitting media generation for: {lecture_plan.get('title')}", flush=True)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
                # Submit job
                response = await client.post(
                    f"{self.base_url}/api/v1/presentations/generate",
                    json=payload
                )
                response.raise_for_status()
                job_data = response.json()
                job_id = job_data.get("job_id")

                print(f"[PRODUCTION] Job submitted: {job_id}", flush=True)

                # Poll for completion
                result = await self._poll_job(client, job_id, lecture_id)
                return result

        except httpx.TimeoutException as e:
            print(f"[PRODUCTION] Timeout generating media: {e}", flush=True)
            return MediaResult(
                lecture_id=lecture_id,
                video_url=None,
                thumbnail_url=None,
                duration_seconds=0,
                error=f"Timeout: {str(e)}",
                job_id=None,
            )
        except Exception as e:
            print(f"[PRODUCTION] Error generating media: {e}", flush=True)
            return MediaResult(
                lecture_id=lecture_id,
                video_url=None,
                thumbnail_url=None,
                duration_seconds=0,
                error=str(e),
                job_id=None,
            )

    async def _poll_job(
        self,
        client: "httpx.AsyncClient",
        job_id: str,
        lecture_id: str,
        max_wait: float = 1200.0,  # 20 minutes
        poll_interval: float = 5.0,
    ) -> MediaResult:
        """Poll for job completion"""
        start_time = datetime.utcnow()

        while True:
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > max_wait:
                return MediaResult(
                    lecture_id=lecture_id,
                    video_url=None,
                    thumbnail_url=None,
                    duration_seconds=0,
                    error=f"Job {job_id} timed out after {max_wait}s",
                    job_id=job_id,
                )

            try:
                response = await client.get(
                    f"{self.base_url}/api/v1/presentations/jobs/{job_id}"
                )

                if response.status_code == 404:
                    # Job not found, might be temporary
                    await asyncio.sleep(poll_interval)
                    continue

                response.raise_for_status()
                job_status = response.json()

                status = job_status.get("status", "unknown")

                if status == "completed":
                    return MediaResult(
                        lecture_id=lecture_id,
                        video_url=job_status.get("video_url"),
                        thumbnail_url=job_status.get("thumbnail_url"),
                        duration_seconds=job_status.get("duration", 0),
                        error=None,
                        job_id=job_id,
                    )
                elif status == "failed":
                    return MediaResult(
                        lecture_id=lecture_id,
                        video_url=None,
                        thumbnail_url=None,
                        duration_seconds=0,
                        error=job_status.get("error", "Unknown error"),
                        job_id=job_id,
                    )

                # Still processing
                await asyncio.sleep(poll_interval)

            except Exception as e:
                print(f"[PRODUCTION] Poll error (will retry): {e}", flush=True)
                await asyncio.sleep(poll_interval)


# Global client instance
_media_client = None


def get_media_client() -> MediaGeneratorClient:
    """Get the media generator client"""
    global _media_client
    if _media_client is None:
        _media_client = MediaGeneratorClient()
    return _media_client


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

async def write_script(state: ProductionState) -> ProductionState:
    """
    Node: Write voiceover script if not already provided.

    Uses lecture plan objectives to generate a coherent script.
    """
    print(f"[PRODUCTION] Writing script for: {state.get('lecture_plan', {}).get('title', 'Unknown')}", flush=True)

    state["status"] = ProductionStatus.WRITING_SCRIPT

    lecture_plan = state.get("lecture_plan", {})
    existing_script = state.get("voiceover_script") or lecture_plan.get("voiceover_script")

    if existing_script:
        state["voiceover_script"] = existing_script
        print("[PRODUCTION] Using existing script", flush=True)
        return state

    # Generate script from objectives
    from openai import AsyncOpenAI

    client = AsyncOpenAI(timeout=120.0, max_retries=2)

    objectives = lecture_plan.get("objectives", [])
    title = lecture_plan.get("title", "Untitled")
    description = lecture_plan.get("description", "")
    language = state.get("content_language", "en")

    prompt = f"""Write a voiceover script for an educational video lecture.

TITLE: {title}
DESCRIPTION: {description}

LEARNING OBJECTIVES:
{chr(10).join(f"- {obj}" for obj in objectives)}

REQUIREMENTS:
1. Write in {language.upper()} language
2. Use clear, conversational tone
3. Cover all learning objectives
4. Duration: approximately {lecture_plan.get('duration_seconds', 300) // 60} minutes
5. Include smooth transitions between topics
6. End with a brief summary

Write ONLY the script text, no stage directions or metadata."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert educational content writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
        )

        script = response.choices[0].message.content.strip()
        state["voiceover_script"] = script
        state["script_complexity_score"] = 5  # Default mid-complexity

        print(f"[PRODUCTION] Script generated: {len(script)} chars", flush=True)

    except Exception as e:
        print(f"[PRODUCTION] Script generation failed: {e}", flush=True)
        # Use description as fallback
        state["voiceover_script"] = description or f"In this lecture, we will cover: {title}"
        state["errors"] = state.get("errors", []) + [f"Script generation failed: {str(e)}"]

    return state


async def generate_code(state: ProductionState) -> ProductionState:
    """
    Node: Generate code for pending code blocks.

    Processes one code block at a time through the expert agent.
    """
    pending = state.get("pending_code_blocks", [])

    if not pending:
        print("[PRODUCTION] No code blocks to generate", flush=True)
        return state

    state["status"] = ProductionStatus.GENERATING_CODE

    # Get next pending block
    current_block = pending[0]
    state["current_code_block"] = current_block
    state["pending_code_blocks"] = pending[1:]

    print(f"[PRODUCTION] Generating code for: {current_block.get('concept', 'Unknown')}", flush=True)

    try:
        agent = CodeExpertAgent()

        result = await agent.generate_code(
            concept=current_block.get("concept", ""),
            language=current_block.get("language", "python"),
            persona_level=current_block.get("persona_level", "intermediate"),
            context=state.get("lecture_plan", {}).get("description", ""),
        )

        if result.success:
            generated = GeneratedCodeBlock(
                concept=current_block.get("concept", ""),
                language=current_block.get("language", "python"),
                code=result.data.get("code_block", ""),
                explanation=result.data.get("explanation", ""),
                expected_output=result.data.get("expected_output"),
                review_status="pending",
                quality_score=0,
                retry_count=0,
            )

            # Add to list for review
            blocks = state.get("generated_code_blocks", [])
            blocks.append(generated)
            state["generated_code_blocks"] = blocks

            print(f"[PRODUCTION] Code generated: {len(generated['code'])} chars", flush=True)
        else:
            state["errors"] = state.get("errors", []) + result.errors

    except Exception as e:
        print(f"[PRODUCTION] Code generation failed: {e}", flush=True)
        state["errors"] = state.get("errors", []) + [f"Code generation failed: {str(e)}"]

    return state


async def review_code(state: ProductionState) -> ProductionState:
    """
    Node: Review generated code blocks.

    Uses CodeReviewerAgent to assess quality and correctness.
    """
    blocks = state.get("generated_code_blocks", [])

    # Find pending reviews
    pending_review = [b for b in blocks if b.get("review_status") == "pending"]

    if not pending_review:
        print("[PRODUCTION] No code blocks pending review", flush=True)
        return state

    state["status"] = ProductionStatus.REVIEWING_CODE

    # Review each pending block
    agent = CodeReviewerAgent()

    for block in pending_review:
        print(f"[PRODUCTION] Reviewing code for: {block.get('concept', 'Unknown')}", flush=True)

        try:
            result = await agent.review_code(
                code=block.get("code", ""),
                language=block.get("language", "python"),
                concept=block.get("concept", ""),
                persona_level=state.get("lecture_plan", {}).get("difficulty", "intermediate"),
            )

            if result.success:
                status = result.data.get("status", "approved")
                block["review_status"] = status
                block["quality_score"] = result.data.get("quality_score", 7)

                if status == "rejected":
                    block["retry_count"] = block.get("retry_count", 0) + 1

                print(f"[PRODUCTION] Code review: {status}, score: {block['quality_score']}", flush=True)
            else:
                block["review_status"] = "approved"  # Approve on failure to avoid loops
                block["quality_score"] = 5

        except Exception as e:
            print(f"[PRODUCTION] Code review failed: {e}", flush=True)
            block["review_status"] = "approved"  # Approve on failure
            block["quality_score"] = 5

    state["generated_code_blocks"] = blocks
    state["code_review_iterations"] = state.get("code_review_iterations", 0) + 1

    return state


async def refine_code(state: ProductionState) -> ProductionState:
    """
    Node: Refine rejected code blocks.

    Called when code review rejects a block.
    """
    blocks = state.get("generated_code_blocks", [])

    state["status"] = ProductionStatus.REFINING_CODE

    # Find rejected blocks that haven't exceeded retry limit
    max_iterations = state.get("max_code_iterations", 3)

    for block in blocks:
        if block.get("review_status") == "rejected" and block.get("retry_count", 0) < max_iterations:
            print(f"[PRODUCTION] Refining code for: {block.get('concept', 'Unknown')}", flush=True)

            try:
                agent = CodeExpertAgent()
                result = await agent.refine_code(
                    original_code=block.get("code", ""),
                    feedback="Code needs improvement for educational clarity",
                    language=block.get("language", "python"),
                    persona_level=state.get("lecture_plan", {}).get("difficulty", "intermediate"),
                )

                if result.success:
                    block["code"] = result.data.get("code_block", block["code"])
                    block["explanation"] = result.data.get("explanation", block["explanation"])
                    block["review_status"] = "pending"  # Re-review

                    print(f"[PRODUCTION] Code refined", flush=True)

            except Exception as e:
                print(f"[PRODUCTION] Code refinement failed: {e}", flush=True)
                block["review_status"] = "approved"  # Accept to move on

    state["generated_code_blocks"] = blocks

    return state


async def generate_media(state: ProductionState) -> ProductionState:
    """
    Node: Generate video media for the lecture.

    Calls the presentation-generator service.
    """
    lecture_plan = state.get("lecture_plan", {})
    print(f"[PRODUCTION] Generating media for: {lecture_plan.get('title', 'Unknown')}", flush=True)

    state["status"] = ProductionStatus.GENERATING_MEDIA
    state["media_generation_attempts"] = state.get("media_generation_attempts", 0) + 1

    # Prepare code blocks (only approved ones)
    code_blocks = [
        b for b in state.get("generated_code_blocks", [])
        if b.get("review_status") == "approved"
    ]

    # Build settings
    settings = {
        "voice_id": state.get("voice_id", "default"),
        "style": state.get("style", "modern"),
        "typing_speed": state.get("typing_speed", "natural"),
        "animations_disabled": state.get("animations_disabled", False),
        "include_avatar": state.get("include_avatar", False),
        "avatar_id": state.get("avatar_id"),
        "content_language": state.get("content_language", "en"),
    }

    # Generate
    client = get_media_client()
    result = await client.generate_lecture_video(
        lecture_plan=lecture_plan,
        voiceover_script=state.get("voiceover_script", ""),
        code_blocks=code_blocks,
        settings=settings,
    )

    state["media_result"] = result
    state["media_job_id"] = result.get("job_id")

    if result.get("error"):
        state["last_media_error"] = result["error"]
        print(f"[PRODUCTION] Media generation failed: {result['error']}", flush=True)
    else:
        state["status"] = ProductionStatus.COMPLETED
        state["completed_at"] = datetime.utcnow().isoformat()
        print(f"[PRODUCTION] Media generation completed: {result.get('video_url')}", flush=True)

    return state


async def handle_media_failure(state: ProductionState) -> ProductionState:
    """
    Node: Handle media generation failures.

    Analyzes the error and decides on recovery strategy.
    """
    last_error = state.get("last_media_error", "").lower()
    attempts = state.get("media_generation_attempts", 0)
    max_attempts = state.get("max_media_attempts", 3)
    recovery_attempts = state.get("recovery_attempts", 0)
    max_recovery = state.get("max_recovery_attempts", 2)

    print(f"[PRODUCTION] Handling media failure. Attempt {attempts}/{max_attempts}, "
          f"Recovery {recovery_attempts}/{max_recovery}", flush=True)
    print(f"[PRODUCTION] Error: {state.get('last_media_error', 'Unknown')}", flush=True)

    # Check if we should give up
    if recovery_attempts >= max_recovery:
        state["recovery_strategy"] = RecoveryStrategy.SKIP
        print("[PRODUCTION] Max recovery attempts reached, skipping", flush=True)
        return state

    # Analyze error and choose strategy
    if "timeout" in last_error or "ffmpeg" in last_error:
        # Timeout or FFmpeg errors -> reduce complexity
        if not state.get("animations_disabled"):
            state["recovery_strategy"] = RecoveryStrategy.REDUCE_ANIMATIONS
        else:
            state["recovery_strategy"] = RecoveryStrategy.SIMPLIFY_SCRIPT
        print(f"[PRODUCTION] Strategy: {state['recovery_strategy']}", flush=True)

    elif "voiceover" in last_error or "audio" in last_error:
        # Audio errors -> simplify script
        state["recovery_strategy"] = RecoveryStrategy.SIMPLIFY_SCRIPT
        print("[PRODUCTION] Strategy: simplify_script (audio error)", flush=True)

    elif attempts >= max_attempts:
        # Too many attempts -> skip
        state["recovery_strategy"] = RecoveryStrategy.SKIP
        print("[PRODUCTION] Strategy: skip (max attempts)", flush=True)

    else:
        # Unknown error -> simple retry
        state["recovery_strategy"] = RecoveryStrategy.RETRY
        print("[PRODUCTION] Strategy: retry", flush=True)

    return state


async def simplify_script_node(state: ProductionState) -> ProductionState:
    """Node: Simplify script for recovery"""
    agent = ScriptSimplifierAgent()
    return await agent.process(state)


async def mark_skipped(state: ProductionState) -> ProductionState:
    """Node: Mark lecture as skipped"""
    state["status"] = ProductionStatus.SKIPPED
    state["completed_at"] = datetime.utcnow().isoformat()
    print(f"[PRODUCTION] Lecture skipped: {state.get('lecture_plan', {}).get('title', 'Unknown')}", flush=True)
    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_code_review(
    state: ProductionState
) -> Literal["generate_code", "refine_code", "generate_media"]:
    """Route based on code review results"""
    pending = state.get("pending_code_blocks", [])
    blocks = state.get("generated_code_blocks", [])
    max_iterations = state.get("max_code_iterations", 3)

    # Check for rejected blocks that need refinement
    rejected = [b for b in blocks if b.get("review_status") == "rejected" and b.get("retry_count", 0) < max_iterations]

    if rejected:
        return "refine_code"

    # Check if there are more blocks to generate
    if pending:
        return "generate_code"

    # All done, proceed to media
    return "generate_media"


def route_after_refinement(
    state: ProductionState
) -> Literal["review_code", "generate_media"]:
    """Route after code refinement"""
    blocks = state.get("generated_code_blocks", [])

    # Check if any blocks need re-review
    pending_review = [b for b in blocks if b.get("review_status") == "pending"]

    if pending_review:
        return "review_code"

    return "generate_media"


def route_after_media(
    state: ProductionState
) -> Literal["handle_failure", "end"]:
    """Route based on media generation result"""
    result = state.get("media_result")

    if result and result.get("error"):
        return "handle_failure"

    return "end"


def route_after_failure_handling(
    state: ProductionState
) -> Literal["simplify_script", "generate_media", "skip"]:
    """Route based on recovery strategy"""
    strategy = state.get("recovery_strategy")

    if strategy == RecoveryStrategy.SKIP:
        return "skip"
    elif strategy in [RecoveryStrategy.SIMPLIFY_SCRIPT, RecoveryStrategy.REDUCE_ANIMATIONS]:
        return "simplify_script"
    else:
        # RETRY
        return "generate_media"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_production_subgraph() -> StateGraph:
    """
    Build the Production subgraph.

    Flow:
        write_script -> generate_code -> review_code -> generate_media
                            ^               |                |
                            |    (bad)      |     (failure)  |
                            +<--------------+                v
                                                      handle_failure
                                                            |
                                         +------------------+------------------+
                                         |                  |                  |
                                    simplify            retry               skip
                                         |                  |                  |
                                         +--------+---------+                  |
                                                  |                            |
                                                  v                            |
                                           generate_media                      |
                                                  |                            |
                                                  +---------------->-----------+
                                                                               |
                                                                              END
    """
    workflow = StateGraph(ProductionState)

    # Add nodes
    workflow.add_node("write_script", write_script)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("review_code", review_code)
    workflow.add_node("refine_code", refine_code)
    workflow.add_node("generate_media", generate_media)
    workflow.add_node("handle_failure", handle_media_failure)
    workflow.add_node("simplify_script", simplify_script_node)
    workflow.add_node("skip", mark_skipped)

    # Set entry point
    workflow.set_entry_point("write_script")

    # Main flow
    workflow.add_edge("write_script", "generate_code")

    # Code generation/review loop
    workflow.add_conditional_edges(
        "generate_code",
        lambda s: "review_code" if s.get("generated_code_blocks") else "generate_media",
        {
            "review_code": "review_code",
            "generate_media": "generate_media",
        }
    )

    workflow.add_conditional_edges(
        "review_code",
        route_after_code_review,
        {
            "generate_code": "generate_code",
            "refine_code": "refine_code",
            "generate_media": "generate_media",
        }
    )

    workflow.add_conditional_edges(
        "refine_code",
        route_after_refinement,
        {
            "review_code": "review_code",
            "generate_media": "generate_media",
        }
    )

    # Media generation with failure handling
    workflow.add_conditional_edges(
        "generate_media",
        route_after_media,
        {
            "handle_failure": "handle_failure",
            "end": END,
        }
    )

    # Recovery routing
    workflow.add_conditional_edges(
        "handle_failure",
        route_after_failure_handling,
        {
            "simplify_script": "simplify_script",
            "generate_media": "generate_media",
            "skip": "skip",
        }
    )

    # After simplification, retry media generation
    workflow.add_edge("simplify_script", "generate_media")

    # Skip goes to end
    workflow.add_edge("skip", END)

    return workflow.compile()


# =============================================================================
# SINGLETON
# =============================================================================

_production_graph_instance = None


def get_production_graph():
    """Get the singleton production subgraph instance"""
    global _production_graph_instance
    if _production_graph_instance is None:
        _production_graph_instance = build_production_subgraph()
    return _production_graph_instance
