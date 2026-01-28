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
from services.http_client import ResilientHTTPClient, RetryConfig


# =============================================================================
# LANGUAGE NAMES MAPPING
# =============================================================================

LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
}


# =============================================================================
# SERVICE CLIENTS
# =============================================================================

class MediaGeneratorClient:
    """
    Client for interacting with the presentation-generator service.

    This client properly builds the topic prompt and API payload
    to match what presentation-generator expects.
    """

    # Configuration constants - tuned for performance + reliability
    MAX_WAIT_PER_LECTURE = 1200.0  # 20 minutes
    POLL_INTERVAL_MIN = 2.0
    POLL_INTERVAL_MAX = 15.0
    MAX_CONSECUTIVE_ERRORS = 15
    RETRY_BACKOFF_BASE = 5.0

    def __init__(self):
        self.base_url = os.getenv("PRESENTATION_GENERATOR_URL", "http://presentation-generator:8006")
        self.timeout = float(os.getenv("MEDIA_GENERATION_TIMEOUT", "90"))  # 90s for HTTP requests

        # Use resilient HTTP client with retry logic for Docker network issues
        retry_config = RetryConfig(
            max_retries=5,
            initial_delay=2.0,
            max_delay=30.0,
            exponential_base=2.0,
        )
        self.http_client = ResilientHTTPClient(
            self.base_url,
            timeout=self.timeout,
            retry_config=retry_config,
        )

    def _build_topic_prompt(
        self,
        lecture_plan: Dict[str, Any],
        voiceover_script: str,
        code_blocks: List[Dict[str, Any]],
        settings: Dict[str, Any],
    ) -> str:
        """
        Build a detailed topic prompt for the presentation-generator.

        This matches the format used by CourseCompositor._generate_single_lecture().
        """
        title = lecture_plan.get("title", "Untitled Lecture")
        description = lecture_plan.get("description", "")
        objectives = lecture_plan.get("objectives", [])
        difficulty = lecture_plan.get("difficulty", "intermediate")
        duration = lecture_plan.get("duration_seconds", 300)
        position = lecture_plan.get("position", 1)
        total = lecture_plan.get("total_lectures", 1)
        section_title = lecture_plan.get("section_title", "")
        section_description = lecture_plan.get("section_description", "")
        course_title = lecture_plan.get("course_title", "")
        target_audience = lecture_plan.get("target_audience", "")

        content_language = settings.get("content_language", "en")
        programming_language = settings.get("programming_language", "python")
        language_name = LANGUAGE_NAMES.get(content_language, content_language)

        # Build lesson elements text
        elements_text = []
        lesson_elements = settings.get("lesson_elements", {})

        if lesson_elements.get("concept_intro", True):
            elements_text.append("- Start with a concept introduction slide explaining the theory")
        if lesson_elements.get("diagram_schema", True):
            elements_text.append("- Include visual diagrams or schemas to illustrate concepts (MANDATORY: at least 1-2 diagrams)")
        if lesson_elements.get("code_typing", True):
            elements_text.append(f"- Show code with typing animation (CODE_DEMO slides) - IMPORTANT: Include 2-4 code examples in {programming_language}")
            elements_text.append("- Each code example should build progressively from simple to more complex")
            elements_text.append("- Include comments in the code to explain key concepts")
        if lesson_elements.get("code_execution", False):
            elements_text.append("- Execute code and show the output (include expected output)")
        if lesson_elements.get("voiceover_explanation", True):
            elements_text.append("- Include detailed voiceover explanation during code")
        if lesson_elements.get("curriculum_slide", True):
            elements_text.append("- Start with a curriculum slide showing course position")

        elements_str = "\n".join(elements_text) if elements_text else "- Standard presentation elements"

        # Build code blocks section
        code_section = ""
        if code_blocks:
            code_section = "\n\nCODE EXAMPLES TO INCLUDE:\n"
            for i, cb in enumerate(code_blocks, 1):
                code_section += f"\n--- Code Example {i}: {cb.get('concept', 'Example')} ---\n"
                code_section += f"```{cb.get('language', programming_language)}\n{cb.get('code', '')}\n```\n"
                if cb.get('explanation'):
                    code_section += f"Explanation: {cb['explanation']}\n"

        # Build voiceover section if provided
        voiceover_section = ""
        if voiceover_script:
            voiceover_section = f"""

VOICEOVER SCRIPT (use this as the narration):
{voiceover_script}
"""

        return f"""Create a video presentation for Lecture {position}/{total} in the course "{course_title}".

**CRITICAL: ALL CONTENT MUST BE IN {language_name.upper()}**
- All titles, subtitles, and text content must be in {language_name}
- All voiceover narration text must be in {language_name}
- All bullet points and explanations must be in {language_name}
- Code comments SHOULD be in {language_name} for educational clarity
- Only code syntax/keywords remain in the programming language

COURSE CONTEXT:
- Course: {course_title}
- Target Audience: {target_audience}
- Content Language: {language_name} (code: {content_language})

SECTION: {section_title}
{section_description}

LECTURE: {title}
{description}

LEARNING OBJECTIVES:
{chr(10).join(f'- {obj}' for obj in objectives) if objectives else '- Understand the core concepts presented'}

DIFFICULTY LEVEL: {difficulty}

TARGET DURATION: {duration} seconds

LESSON ELEMENTS TO INCLUDE:
{elements_str}
{code_section}
{voiceover_section}
SLIDE STRUCTURE:
1. CURRICULUM - Show this lecture's position in the course
2. Follow with requested elements in logical order
3. End with a conclusion summarizing key takeaways

PROGRAMMING LANGUAGE/TOOLS: {programming_language}

IMPORTANT REQUIREMENTS:
- This is lecture {position} of {total} in the course
- **LANGUAGE: Write ALL content in {language_name}** - this is MANDATORY
- STRICTLY MATCH the {difficulty} difficulty level
- CODE REQUIREMENT: Include MULTIPLE code examples (minimum 2-3) that progressively build understanding
- Each code example should demonstrate a specific concept from the learning objectives
- DIAGRAM REQUIREMENT: Include at least 1-2 visual diagrams/schemas to illustrate complex concepts
- Voiceover should be engaging and educational, explaining the code line by line (in {language_name})
- Focus on the specific learning objectives listed above
- After each code block, pause to allow learner comprehension"""

    async def generate_lecture_video(
        self,
        lecture_plan: Dict[str, Any],
        voiceover_script: str,
        code_blocks: List[Dict[str, Any]],
        settings: Dict[str, Any],
        progress_callback: Optional[callable] = None,
    ) -> MediaResult:
        """
        Generate video for a lecture via presentation-generator.

        Builds the proper API payload matching what presentation-generator expects.
        """
        lecture_id = lecture_plan.get("lecture_id", "unknown")
        title = lecture_plan.get("title", "Untitled")

        # Build the detailed topic prompt
        topic_prompt = self._build_topic_prompt(
            lecture_plan=lecture_plan,
            voiceover_script=voiceover_script,
            code_blocks=code_blocks,
            settings=settings,
        )

        # Build presentation request matching CourseCompositor format
        programming_language = settings.get("programming_language", "python")
        content_language = settings.get("content_language", "en")

        # Get RAG context from settings (already fetched by course-generator)
        rag_context = settings.get("rag_context")

        presentation_request = {
            "topic": topic_prompt,
            "language": content_language,  # Human language for content (required)
            "programming_language": programming_language,  # Programming language for code (optional)
            "content_language": content_language,  # Human language for content
            "duration": lecture_plan.get("duration_seconds", 300),
            "style": settings.get("style", "modern"),
            "include_avatar": settings.get("include_avatar", False),
            "avatar_id": settings.get("avatar_id"),
            "voice_id": settings.get("voice_id", "default"),
            "execute_code": settings.get("lesson_elements", {}).get("code_execution", False),
            "show_typing_animation": not settings.get("animations_disabled", False),
            "typing_speed": settings.get("typing_speed", "natural"),
            "target_audience": lecture_plan.get("target_audience", ""),
            "enable_visuals": settings.get("lesson_elements", {}).get("diagram_schema", True),
            "visual_style": settings.get("style", "dark"),
            # Pass RAG context to presentation-generator (avoids warning about missing documents)
            "rag_context": rag_context if rag_context else None,
        }

        print(f"[PRODUCTION] Submitting video generation for: {title}", flush=True)
        print(f"[PRODUCTION] Settings: language={programming_language}, content={content_language}, "
              f"duration={presentation_request['duration']}s", flush=True)

        try:
            # Submit job using resilient client
            # Use v3 endpoint which includes VoiceoverEnforcer for proper video duration
            response = await self.http_client.post(
                "/api/v1/presentations/generate/v3",
                json=presentation_request
            )

            if response.status_code != 200:
                error_text = response.text
                print(f"[PRODUCTION] Failed to start generation: {error_text}", flush=True)
                return MediaResult(
                    lecture_id=lecture_id,
                    video_url=None,
                    thumbnail_url=None,
                    duration_seconds=0,
                    error=f"Failed to start generation: {error_text}",
                    job_id=None,
                )

            job_data = response.json()
            job_id = job_data.get("job_id")
            print(f"[PRODUCTION] Presentation job started: {job_id}", flush=True)

            # Poll for completion with adaptive intervals
            result = await self._poll_job(job_id, lecture_id, progress_callback)
            return result

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

    def _get_adaptive_poll_interval(self, elapsed: float, progress: float) -> float:
        """
        Calculate adaptive polling interval based on elapsed time and progress.
        """
        # Base interval increases with time (logarithmic growth)
        time_factor = min(elapsed / 60.0, 5.0)
        base_interval = self.POLL_INTERVAL_MIN + (time_factor * 2.0)

        # If progress is high (>80%), poll more frequently
        if progress > 80:
            base_interval = self.POLL_INTERVAL_MIN
        elif progress > 50:
            base_interval = min(base_interval, 5.0)

        return min(base_interval, self.POLL_INTERVAL_MAX)

    async def _poll_job(
        self,
        job_id: str,
        lecture_id: str,
        progress_callback: Optional[callable] = None,
    ) -> MediaResult:
        """
        Poll presentation-generator until job completes.

        Uses adaptive polling intervals and resilient HTTP client.
        """
        start_time = asyncio.get_event_loop().time()
        consecutive_errors = 0
        last_progress_log = 0
        last_callback_progress = -1  # Track last progress sent to callback
        current_progress = 0.0

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.MAX_WAIT_PER_LECTURE:
                return MediaResult(
                    lecture_id=lecture_id,
                    video_url=None,
                    thumbnail_url=None,
                    duration_seconds=0,
                    error=f"Job {job_id} timed out after {self.MAX_WAIT_PER_LECTURE/60:.0f} minutes",
                    job_id=job_id,
                )

            try:
                response = await self.http_client.get(
                    f"/api/v1/presentations/jobs/{job_id}"
                )

                if response.status_code == 404:
                    consecutive_errors += 1
                    if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        return MediaResult(
                            lecture_id=lecture_id,
                            video_url=None,
                            thumbnail_url=None,
                            duration_seconds=0,
                            error=f"Job {job_id} not found after {consecutive_errors} attempts",
                            job_id=job_id,
                        )
                    await asyncio.sleep(self._get_adaptive_poll_interval(elapsed, current_progress))
                    continue

                if response.status_code != 200:
                    consecutive_errors += 1
                    backoff_delay = min(
                        self.RETRY_BACKOFF_BASE * (2 ** min(consecutive_errors - 1, 4)),
                        60.0
                    )
                    if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        return MediaResult(
                            lecture_id=lecture_id,
                            video_url=None,
                            thumbnail_url=None,
                            duration_seconds=0,
                            error=f"Too many errors polling job: {response.text}",
                            job_id=job_id,
                        )
                    await asyncio.sleep(backoff_delay)
                    continue

                # Success - reset error count
                consecutive_errors = 0
                job_data = response.json()

                status = job_data.get("status", "unknown")
                # Note: presentation-generator returns "phase", not "current_stage"
                current_stage = job_data.get("phase") or job_data.get("current_stage") or status
                current_progress = float(job_data.get("progress", 0))

                # Send progress update via callback (only if progress changed significantly)
                if progress_callback and int(current_progress) != last_callback_progress:
                    last_callback_progress = int(current_progress)
                    try:
                        progress_callback(
                            stage=current_stage,
                            progress=current_progress,
                            status="generating",
                        )
                    except Exception as e:
                        print(f"[PRODUCTION] Progress callback error: {e}", flush=True)

                # Log progress periodically (every 30 seconds)
                current_time = int(elapsed)
                if current_time - last_progress_log >= 30:
                    last_progress_log = current_time
                    remaining = self.MAX_WAIT_PER_LECTURE - elapsed
                    print(f"[PRODUCTION] Job {job_id}: {current_stage} ({current_progress:.0f}%) - "
                          f"{remaining/60:.1f}min remaining", flush=True)

                if status == "completed":
                    video_url = job_data.get("output_url") or job_data.get("video_url")
                    if not video_url:
                        # Sometimes output_url takes a moment - retry briefly
                        if consecutive_errors < 3:
                            consecutive_errors += 1
                            print(f"[PRODUCTION] Job {job_id} completed but no URL yet, retrying...", flush=True)
                            await asyncio.sleep(5.0)
                            continue
                        return MediaResult(
                            lecture_id=lecture_id,
                            video_url=None,
                            thumbnail_url=None,
                            duration_seconds=0,
                            error="Job completed but no output URL",
                            job_id=job_id,
                        )

                    print(f"[PRODUCTION] Job {job_id} completed: {video_url}", flush=True)
                    return MediaResult(
                        lecture_id=lecture_id,
                        video_url=video_url,
                        thumbnail_url=job_data.get("thumbnail_url"),
                        duration_seconds=job_data.get("duration", 0),
                        error=None,
                        job_id=job_id,
                    )

                if status == "failed":
                    error = job_data.get("error", "Unknown error")
                    print(f"[PRODUCTION] Job {job_id} failed: {error}", flush=True)
                    return MediaResult(
                        lecture_id=lecture_id,
                        video_url=None,
                        thumbnail_url=None,
                        duration_seconds=0,
                        error=f"Presentation generation failed: {error}",
                        job_id=job_id,
                    )

                # Still processing - use adaptive interval
                await asyncio.sleep(self._get_adaptive_poll_interval(elapsed, current_progress))

            except Exception as e:
                consecutive_errors += 1
                print(f"[PRODUCTION] Poll error {consecutive_errors}: {e}", flush=True)

                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    return MediaResult(
                        lecture_id=lecture_id,
                        video_url=None,
                        thumbnail_url=None,
                        duration_seconds=0,
                        error=f"Connection lost after {consecutive_errors} errors: {str(e)}",
                        job_id=job_id,
                    )

                await asyncio.sleep(self._get_adaptive_poll_interval(elapsed, current_progress))


# Global client instance
_media_client = None

# Global registry for lecture progress callbacks (indexed by lecture_id)
# This allows passing progress updates from _poll_job back to the orchestrator
_lecture_progress_callbacks: Dict[str, callable] = {}


def register_lecture_progress_callback(lecture_id: str, callback: callable):
    """Register a progress callback for a lecture"""
    _lecture_progress_callbacks[lecture_id] = callback


def unregister_lecture_progress_callback(lecture_id: str):
    """Unregister a progress callback for a lecture"""
    if lecture_id in _lecture_progress_callbacks:
        del _lecture_progress_callbacks[lecture_id]


def get_lecture_progress_callback(lecture_id: str) -> Optional[callable]:
    """Get the progress callback for a lecture"""
    return _lecture_progress_callbacks.get(lecture_id)


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

    Calls the presentation-generator service with properly formatted request.
    Uses checkpointing to skip already completed lectures (for crash recovery).
    """
    from services.lecture_checkpoint import checkpoint_service

    lecture_plan = state.get("lecture_plan", {})
    title = lecture_plan.get("title", "Unknown")
    lecture_id = lecture_plan.get("lecture_id", "unknown")
    job_id = state.get("job_id", "unknown")

    print(f"[PRODUCTION] Generating media for: {title} (lecture_id={lecture_id})", flush=True)

    # === CHECKPOINTING: Skip if already completed ===
    if await checkpoint_service.is_completed(job_id, lecture_id):
        existing_url = await checkpoint_service.get_video_url(job_id, lecture_id)
        print(f"[PRODUCTION] CHECKPOINT HIT: {lecture_id} already completed, skipping", flush=True)
        print(f"[PRODUCTION] Using cached video: {existing_url}", flush=True)

        state["media_result"] = {
            "lecture_id": lecture_id,
            "video_url": existing_url,
            "error": None,
            "from_checkpoint": True
        }
        state["status"] = ProductionStatus.COMPLETED
        state["completed_at"] = datetime.utcnow().isoformat()
        return state

    # Mark as in-progress for tracking
    await checkpoint_service.mark_in_progress(job_id, lecture_id)

    state["status"] = ProductionStatus.GENERATING_MEDIA
    state["media_generation_attempts"] = state.get("media_generation_attempts", 0) + 1

    # Prepare code blocks (only approved ones)
    code_blocks = [
        b for b in state.get("generated_code_blocks", [])
        if b.get("review_status") == "approved"
    ]

    # Build comprehensive settings for MediaGeneratorClient
    settings = {
        # Voice and style
        "voice_id": state.get("voice_id", "default"),
        "style": state.get("style", "modern"),
        "typing_speed": state.get("typing_speed", "natural"),
        "animations_disabled": state.get("animations_disabled", False),

        # Avatar
        "include_avatar": state.get("include_avatar", False),
        "avatar_id": state.get("avatar_id"),

        # Languages
        "content_language": state.get("content_language", "en"),
        "programming_language": state.get("programming_language", "python"),

        # Lesson elements configuration
        "lesson_elements": {
            "concept_intro": state.get("lesson_elements", {}).get("concept_intro", True),
            "diagram_schema": state.get("lesson_elements", {}).get("diagram_schema", True),
            "code_typing": state.get("lesson_elements", {}).get("code_typing", True),
            "code_execution": state.get("lesson_elements", {}).get("code_execution", False),
            "voiceover_explanation": state.get("lesson_elements", {}).get("voiceover_explanation", True),
            "curriculum_slide": state.get("lesson_elements", {}).get("curriculum_slide", True),
        },

        # RAG context from source documents (passed to presentation-generator)
        "rag_context": state.get("rag_context"),
    }

    # Ensure lecture_plan has all required context for prompt building
    enriched_plan = {
        **lecture_plan,
        "course_title": state.get("course_title", lecture_plan.get("course_title", "")),
        "target_audience": state.get("target_audience", lecture_plan.get("target_audience", "")),
        "section_title": state.get("section_title", lecture_plan.get("section_title", "")),
        "section_description": state.get("section_description", lecture_plan.get("section_description", "")),
    }

    # Get progress callback from registry (if registered by orchestrator)
    lecture_id = lecture_plan.get("lecture_id", "unknown")
    progress_callback = get_lecture_progress_callback(lecture_id)

    # Generate via the client
    client = get_media_client()
    result = await client.generate_lecture_video(
        lecture_plan=enriched_plan,
        voiceover_script=state.get("voiceover_script", ""),
        code_blocks=code_blocks,
        settings=settings,
        progress_callback=progress_callback,
    )

    state["media_result"] = result
    state["media_job_id"] = result.get("job_id")

    if result.get("error"):
        state["last_media_error"] = result["error"]
        print(f"[PRODUCTION] Media generation failed: {result['error']}", flush=True)

        # Mark as failed in checkpoint (for tracking, not skipping on retry)
        await checkpoint_service.mark_failed(
            job_id, lecture_id,
            error=result["error"],
            retry_count=state.get("media_generation_attempts", 1)
        )
    else:
        state["status"] = ProductionStatus.COMPLETED
        state["completed_at"] = datetime.utcnow().isoformat()
        video_url = result.get("video_url")
        print(f"[PRODUCTION] Media generation completed: {video_url}", flush=True)

        # === CHECKPOINT: Mark as completed for future recovery ===
        await checkpoint_service.mark_completed(
            job_id, lecture_id,
            video_url=video_url,
            duration_seconds=result.get("duration_seconds", 0),
            metadata={"job_id": result.get("job_id")}
        )

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
