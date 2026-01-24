"""
Course Generation Graph

LangGraph-based orchestrator for the multi-agent course generation system.
This graph coordinates all agents in the proper sequence with conditional
routing based on configuration and intermediate results.

Flow:
    validate_input -> review_config -> pedagogical_analysis -> plan_course
                                                                   |
                                                                   v
    finalize <-- (loop) <-- generate_media <-- route_production_loop
"""
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END

from agents.base import (
    CourseGenerationState,
    AgentStatus,
    create_initial_state,
)
from agents.input_validator import InputValidatorAgent
from agents.technical_reviewer import TechnicalReviewerAgent
from agents.code_expert import CodeExpertAgent
from agents.code_reviewer import CodeReviewerAgent
from agents.pedagogical_graph import get_pedagogical_agent

# Models only - services are imported lazily to avoid circular imports
from models.course_models import (
    PreviewOutlineRequest,
    CourseContext,
    ProfileCategory,
    DifficultyLevel,
    CourseStructureConfig,
)
from services.rag_threshold_validator import (
    validate_rag_threshold,
    RAGMode,
    RAGThresholdResult,
)


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

async def validate_input(state: CourseGenerationState) -> CourseGenerationState:
    """Node: Validate all input parameters"""
    agent = InputValidatorAgent()
    return await agent.process(state)


async def review_config(state: CourseGenerationState) -> CourseGenerationState:
    """Node: Review configuration and enrich prompts"""
    agent = TechnicalReviewerAgent()
    return await agent.process(state)


async def run_pedagogical_analysis(state: CourseGenerationState) -> CourseGenerationState:
    """Node: Run pedagogical analysis and planning"""
    agent = get_pedagogical_agent()

    # Convert state fields to pedagogical agent format
    result = await agent.plan_from_scratch(
        topic=state.get("topic", ""),
        description=state.get("description"),
        category=state.get("profile_category", "education"),
        difficulty_start=state.get("difficulty_start", "beginner"),
        difficulty_end=state.get("difficulty_end", "intermediate"),
        target_language=state.get("content_language", "en"),
        target_audience=state.get("target_audience", "general learners"),
        num_sections=state.get("structure", {}).get("number_of_sections", 4),
        lectures_per_section=state.get("structure", {}).get("lectures_per_section", 3),
        duration_minutes=state.get("structure", {}).get("total_duration_minutes", 60),
        rag_context=state.get("rag_context"),
        document_ids=state.get("document_ids"),
        quiz_enabled=state.get("quiz_config", {}).get("enabled", True),
        quiz_frequency=state.get("quiz_config", {}).get("frequency", "per_section"),
    )

    # Update state with pedagogical analysis results
    state["detected_persona"] = result.get("detected_persona", "student")
    state["topic_complexity"] = result.get("topic_complexity", "intermediate")
    state["requires_code"] = result.get("requires_code", False)
    state["requires_diagrams"] = result.get("requires_diagrams", True)
    state["content_preferences"] = result.get("content_preferences", {})
    state["recommended_elements"] = result.get("recommended_elements", [])
    state["rag_images"] = result.get("rag_images", [])

    # Set persona level based on difficulty
    diff_start = state.get("difficulty_start", "beginner").lower()
    state["persona_level"] = diff_start

    print(f"[PEDAGOGICAL] Analysis complete: persona={state['detected_persona']}, "
          f"complexity={state['topic_complexity']}, requires_code={state['requires_code']}", flush=True)

    return state


async def check_rag_threshold(state: CourseGenerationState) -> CourseGenerationState:
    """
    Node: Validate RAG context meets minimum thresholds.

    This prevents generation when source documents are insufficient,
    avoiding hallucinations from the LLM compensating with its own knowledge.

    Thresholds:
    - < 500 tokens: BLOCKED - refuse to generate
    - 500-2000 tokens: PARTIAL - generate with warning
    - > 2000 tokens: FULL - optimal RAG mode
    """
    rag_context = state.get("rag_context")
    document_ids = state.get("document_ids", [])
    topic = state.get("topic", "Unknown topic")

    # Validate threshold
    result: RAGThresholdResult = validate_rag_threshold(
        rag_context=rag_context,
        document_ids=document_ids,
        topic=topic,
    )

    # Store result in state
    state["rag_mode"] = result.mode.value
    state["rag_token_count"] = result.token_count
    state["rag_threshold_passed"] = result.is_sufficient

    if result.warning_message:
        state["rag_warning"] = result.warning_message
        print(f"[RAG_CHECK] WARNING: {result.warning_message}", flush=True)

    if result.error_message:
        state["rag_error"] = result.error_message
        state["errors"] = state.get("errors", []) + [result.error_message]
        print(f"[RAG_CHECK] BLOCKED: {result.error_message}", flush=True)

    # Log the result
    if result.mode == RAGMode.FULL:
        print(f"[RAG_CHECK] FULL mode: {result.token_count} tokens (sufficient)", flush=True)
    elif result.mode == RAGMode.PARTIAL:
        print(f"[RAG_CHECK] PARTIAL mode: {result.token_count} tokens (limited)", flush=True)
    elif result.mode == RAGMode.NONE:
        print(f"[RAG_CHECK] NONE mode: No documents provided (standard generation)", flush=True)
    elif result.mode == RAGMode.BLOCKED:
        print(f"[RAG_CHECK] BLOCKED: {result.token_count} tokens (insufficient)", flush=True)

    return state


async def handle_insufficient_rag(state: CourseGenerationState) -> CourseGenerationState:
    """
    Node: Handle insufficient RAG context.

    Called when RAG threshold check fails (BLOCKED mode).
    Sets appropriate error state and prepares user feedback.
    """
    error = state.get("rag_error", "Insufficient source documents")
    token_count = state.get("rag_token_count", 0)

    state["final_status"] = "insufficient_sources"
    state["completed_at"] = datetime.utcnow().isoformat()

    print(f"[INSUFFICIENT_RAG] Generation blocked. Tokens: {token_count}", flush=True)
    print(f"[INSUFFICIENT_RAG] User should provide more comprehensive documents.", flush=True)

    # Add helpful suggestions
    suggestions = [
        "Upload more documents covering the topic",
        "Ensure documents contain text (not just images)",
        "Check that documents are relevant to the requested topic",
        "Consider using PDFs or Word documents with more content",
    ]
    state["rag_suggestions"] = suggestions

    return state


async def plan_course(state: CourseGenerationState) -> CourseGenerationState:
    """
    Node: Generate the course outline and structure.

    Uses CoursePlanner to create sections and lectures based on:
    - Topic and description
    - Pedagogical analysis results
    - Structure configuration
    """
    # Lazy import to avoid circular dependency
    from services.course_planner import CoursePlanner

    planner = CoursePlanner()

    print(f"[GRAPH] Generating outline for topic: {state['topic']}", flush=True)

    try:
        # Map profile category to enum
        category_str = state.get("profile_category", "education").lower()
        try:
            category = ProfileCategory(category_str)
        except ValueError:
            category = ProfileCategory.EDUCATION

        # Build context
        context = CourseContext(
            category=category,
            profile_niche=state.get("topic"),
            profile_audience_level=state.get("target_audience", "Beginner"),
            profile_tone="Educational"
        )

        # Build structure config
        structure = state.get("structure", {})
        structure_config = CourseStructureConfig(
            total_duration_minutes=structure.get("total_duration_minutes", 10),
            number_of_sections=structure.get("number_of_sections", 3),
            lectures_per_section=structure.get("lectures_per_section", 2),
            random_structure=False
        )

        # Map difficulty levels
        diff_start_str = state.get("difficulty_start", "beginner").lower()
        diff_end_str = state.get("difficulty_end", "intermediate").lower()
        try:
            diff_start = DifficultyLevel(diff_start_str)
        except ValueError:
            diff_start = DifficultyLevel.BEGINNER
        try:
            diff_end = DifficultyLevel(diff_end_str)
        except ValueError:
            diff_end = DifficultyLevel.INTERMEDIATE

        # Build request
        request = PreviewOutlineRequest(
            topic=state["topic"],
            description=state.get("description"),
            difficulty_start=diff_start,
            difficulty_end=diff_end,
            structure=structure_config,
            context=context,
            rag_context=state.get("rag_context"),
            language=state.get("content_language", "en"),
            document_ids=state.get("document_ids", []),
        )

        # Generate outline
        outline = await planner.generate_outline(request)

        # Flatten outline into list of lectures for sequential processing
        lectures_flat: List[Dict[str, Any]] = []
        for section in outline.sections:
            for lecture in section.lectures:
                lecture_dict = lecture.model_dump() if hasattr(lecture, "model_dump") else lecture.dict()
                lecture_dict["section_id"] = section.id if hasattr(section, "id") else f"section_{section.order}"
                lecture_dict["section_title"] = section.title
                lecture_dict["section_description"] = section.description
                lecture_dict["status"] = "pending"
                lecture_dict["video_url"] = None
                lecture_dict["error"] = None
                lectures_flat.append(lecture_dict)

        # Update state
        state["outline"] = outline.model_dump() if hasattr(outline, "model_dump") else outline.dict()
        state["outline_validated"] = True
        state["lectures"] = lectures_flat
        state["current_lecture_index"] = 0
        state["output_videos"] = []

        print(f"[GRAPH] Plan created: {len(lectures_flat)} lectures to generate.", flush=True)

    except Exception as e:
        print(f"[GRAPH] Planning failed: {e}", flush=True)
        state["errors"] = state.get("errors", []) + [f"Planning failed: {str(e)}"]
        state["lectures"] = []

    return state


async def generate_lecture_media(state: CourseGenerationState) -> CourseGenerationState:
    """
    Node: Generate script and video for the current lecture.

    Uses CourseCompositor to generate video content via presentation-generator.
    """
    idx = state.get("current_lecture_index", 0)
    lectures = state.get("lectures", [])

    if idx >= len(lectures):
        print(f"[GRAPH] No more lectures to process (index {idx} >= {len(lectures)})", flush=True)
        return state

    current_lecture = lectures[idx]
    print(f"[GRAPH] Processing lecture {idx + 1}/{len(lectures)}: {current_lecture.get('title', 'Unknown')}", flush=True)

    try:
        # Lazy import to avoid circular dependency
        from services.course_compositor import CourseCompositor

        # Initialize compositor
        compositor = CourseCompositor()

        # Build a minimal job and request for the compositor
        from models.course_models import (
            CourseJob,
            CourseStage,
            GenerateCourseRequest,
            Lecture,
            Section,
            LessonElementConfig,
        )

        # Create a Lecture model from the dict
        lecture_model = Lecture(
            id=current_lecture.get("id", f"lecture_{idx}"),
            title=current_lecture.get("title", f"Lecture {idx + 1}"),
            description=current_lecture.get("description", ""),
            objectives=current_lecture.get("objectives", []),
            difficulty=DifficultyLevel(current_lecture.get("difficulty", "intermediate").lower()),
            duration_seconds=current_lecture.get("duration_seconds", 300),
            order=current_lecture.get("order", idx),
        )

        # Create a Section model
        section_model = Section(
            id=current_lecture.get("section_id", "section_0"),
            title=current_lecture.get("section_title", "Section"),
            description=current_lecture.get("section_description", ""),
            order=0,
            lectures=[lecture_model],
        )

        # Get outline
        outline_dict = state.get("outline", {})

        # Build lesson elements
        lesson_elements_config = state.get("lesson_elements", {})
        lesson_elements = LessonElementConfig(
            concept_intro=lesson_elements_config.get("concept_intro", True),
            diagram_schema=lesson_elements_config.get("diagram_schema", True),
            code_typing=lesson_elements_config.get("code_typing", True),
            code_execution=lesson_elements_config.get("code_execution", False),
            voiceover_explanation=lesson_elements_config.get("voiceover_explanation", True),
            curriculum_slide=lesson_elements_config.get("curriculum_slide", True),
        )

        # Build a minimal request
        # Include rag_context so compositor can pass it to presentation-generator
        rag_context = state.get("rag_context")

        request = GenerateCourseRequest(
            profile_id=state.get("profile_id", "default"),
            topic=state.get("topic", "Unknown Topic"),
            style=state.get("style", "modern"),
            include_avatar=state.get("include_avatar", False),
            avatar_id=state.get("avatar_id"),
            voice_id=state.get("voice_id", "default"),
            typing_speed=state.get("typing_speed", "natural"),
            lesson_elements=lesson_elements,
            rag_context=rag_context,  # Pass RAG context to compositor
        )

        # Reconstruct outline for compositor
        from models.course_models import CourseOutline

        outline_model = CourseOutline(
            title=outline_dict.get("title", state.get("topic", "")),
            description=outline_dict.get("description", ""),
            target_audience=outline_dict.get("target_audience", state.get("target_audience", "")),
            language=outline_dict.get("language", state.get("content_language", "en")),
            difficulty_start=DifficultyLevel(outline_dict.get("difficulty_start", "beginner").lower()),
            difficulty_end=DifficultyLevel(outline_dict.get("difficulty_end", "intermediate").lower()),
            total_duration_minutes=outline_dict.get("total_duration_minutes", 60),
            sections=[section_model],
            total_lectures=len(lectures),
        )

        # Generate the single lecture
        video_url = await compositor._generate_single_lecture(
            lecture=lecture_model,
            section=section_model,
            outline=outline_model,
            request=request,
            position=idx + 1,
            total=len(lectures),
            job_id=state.get("job_id", "unknown"),
        )

        # Update lecture status
        current_lecture["video_url"] = video_url
        current_lecture["status"] = "completed"

        # Add to output videos
        output_videos = state.get("output_videos", [])
        output_videos.append(video_url)
        state["output_videos"] = output_videos

        print(f"[GRAPH] Lecture {idx + 1} completed: {video_url}", flush=True)

    except Exception as e:
        print(f"[GRAPH] Error generating lecture {idx + 1}: {e}", flush=True)
        current_lecture["status"] = "failed"
        current_lecture["error"] = str(e)
        state["errors"] = state.get("errors", []) + [f"Lecture {idx + 1} failed: {str(e)}"]

    # Update lecture in state and move to next
    state["lectures"][idx] = current_lecture
    state["current_lecture_index"] = idx + 1

    return state


async def generate_code_block(state: CourseGenerationState) -> CourseGenerationState:
    """Node: Generate code for current code block"""
    agent = CodeExpertAgent()
    return await agent.process(state)


async def review_code_block(state: CourseGenerationState) -> CourseGenerationState:
    """Node: Review generated code"""
    agent = CodeReviewerAgent()
    return await agent.process(state)


async def refine_code_block(state: CourseGenerationState) -> CourseGenerationState:
    """Node: Refine rejected code based on feedback"""
    code_block = state.get("current_code_block")

    if not code_block:
        return state

    agent = CodeExpertAgent()

    # Get retry prompt from code block
    retry_prompt = code_block.get("retry_prompt", "")
    original_code = code_block.get("refined_code") or code_block.get("raw_code", "")

    result = await agent.refine_code(
        original_code=original_code,
        feedback=retry_prompt,
        language=code_block.get("language", "python"),
        persona_level=code_block.get("persona_level", "intermediate"),
    )

    if result.success:
        code_block["refined_code"] = result.data.get("code_block", "")
        code_block["expected_output"] = result.data.get("expected_output", "")
        code_block["improvements_made"] = result.data.get("improvements_made", [])

    state["current_code_block"] = code_block
    return state


async def finalize_generation(state: CourseGenerationState) -> CourseGenerationState:
    """Node: Finalize the generation process"""
    state["completed_at"] = datetime.utcnow().isoformat()

    # Lecture statistics
    lectures = state.get("lectures", [])
    total_lectures = len(lectures)
    completed_lectures = sum(1 for l in lectures if l.get("status") == "completed")
    failed_lectures = sum(1 for l in lectures if l.get("status") == "failed")

    # Code block statistics
    code_approved = state.get("code_blocks_approved", 0)
    code_rejected = state.get("code_blocks_rejected", 0)
    code_total = state.get("code_blocks_processed", 0)

    # Output videos
    output_videos = state.get("output_videos", [])

    print(f"[FINALIZE] Generation complete.", flush=True)
    print(f"[FINALIZE] Lectures: {completed_lectures}/{total_lectures} completed, "
          f"{failed_lectures} failed", flush=True)
    print(f"[FINALIZE] Videos generated: {len(output_videos)}", flush=True)

    if code_total > 0:
        print(f"[FINALIZE] Code blocks: {code_approved} approved, "
              f"{code_rejected} rejected out of {code_total} processed.", flush=True)

    # Determine final status
    if completed_lectures == total_lectures and total_lectures > 0:
        state["final_status"] = "success"
    elif completed_lectures > 0:
        state["final_status"] = "partial_success"
    elif total_lectures == 0:
        state["final_status"] = "no_lectures"
    else:
        state["final_status"] = "failed"

    return state


async def handle_validation_failure(state: CourseGenerationState) -> CourseGenerationState:
    """Node: Handle validation failures"""
    errors = state.get("input_validation_errors", [])
    error_messages = [e.get("message", "Unknown error") for e in errors]

    state["errors"] = state.get("errors", []) + error_messages

    print(f"[VALIDATION_FAILED] {len(error_messages)} validation errors", flush=True)
    for msg in error_messages[:5]:
        print(f"  - {msg}", flush=True)

    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_validation(
    state: CourseGenerationState
) -> Literal["review_config", "validation_failed"]:
    """Route based on validation result"""
    if state.get("input_validated", False):
        return "review_config"
    return "validation_failed"


def route_after_rag_check(
    state: CourseGenerationState
) -> Literal["plan_course", "insufficient_rag"]:
    """
    Route based on RAG threshold check result.

    If RAG mode is BLOCKED, route to insufficient_rag handler.
    Otherwise, continue to plan_course.
    """
    rag_mode = state.get("rag_mode", "none")

    if rag_mode == "blocked":
        return "insufficient_rag"

    # FULL, PARTIAL, or NONE modes all continue to planning
    return "plan_course"


def route_after_code_review(
    state: CourseGenerationState
) -> Literal["finalize", "refine_code", "generate_next"]:
    """Route based on code review result"""
    code_block = state.get("current_code_block")

    if not code_block:
        return "finalize"

    status = code_block.get("review_status", "")

    if status == "approved":
        # Check if there are more code blocks to process
        # (This would be managed by the orchestrator)
        return "finalize"

    elif status == "rejected":
        retry_count = code_block.get("retry_count", 0)
        max_retries = code_block.get("max_retries", 3)

        if retry_count < max_retries and code_block.get("retry_needed"):
            return "refine_code"
        else:
            # Max retries reached, move on
            return "finalize"

    return "finalize"


def route_after_refinement(
    state: CourseGenerationState
) -> Literal["review_code", "finalize"]:
    """Route after code refinement"""
    code_block = state.get("current_code_block")

    if code_block and code_block.get("refined_code"):
        return "review_code"

    return "finalize"


def route_production_loop(
    state: CourseGenerationState
) -> Literal["generate_media", "finalize"]:
    """
    Decide whether to continue generating media or finish.

    Routes to:
    - generate_media: If there are more lectures to process
    - finalize: If all lectures are done or no lectures exist
    """
    idx = state.get("current_lecture_index", 0)
    lectures = state.get("lectures", [])
    total = len(lectures)

    if idx < total:
        return "generate_media"

    return "finalize"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

class CourseGenerationGraph:
    """
    Main orchestrator graph for course generation.

    This graph coordinates:
    1. Input validation
    2. Technical configuration review
    3. Pedagogical analysis
    4. Code generation and review loop
    5. Final output preparation
    """

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Flow:
            validate_input -> review_config -> pedagogical_analysis
                                                      |
                                                      v
                                              check_rag_threshold
                                                /           \
                                    (blocked) /             \ (ok)
                                             v               v
                                    insufficient_rag     plan_course
                                             |               |
                                             v               v
                                           END     route_production_loop
                                                    /           \
                                                   v             v
                                           generate_media --> finalize --> END
        """
        workflow = StateGraph(CourseGenerationState)

        # --- NODES ---
        workflow.add_node("validate_input", validate_input)
        workflow.add_node("validation_failed", handle_validation_failure)
        workflow.add_node("review_config", review_config)
        workflow.add_node("pedagogical_analysis", run_pedagogical_analysis)

        # RAG threshold check node (NEW)
        workflow.add_node("check_rag_threshold", check_rag_threshold)
        workflow.add_node("insufficient_rag", handle_insufficient_rag)

        # Planning and production nodes
        workflow.add_node("plan_course", plan_course)
        workflow.add_node("generate_media", generate_lecture_media)

        workflow.add_node("finalize", finalize_generation)

        # Code generation loop nodes (optional, for individual code blocks)
        workflow.add_node("generate_code", generate_code_block)
        workflow.add_node("review_code", review_code_block)
        workflow.add_node("refine_code", refine_code_block)

        # --- EDGES ---
        workflow.set_entry_point("validate_input")

        # Validation routing
        workflow.add_conditional_edges(
            "validate_input",
            route_after_validation,
            {
                "review_config": "review_config",
                "validation_failed": "validation_failed",
            }
        )
        workflow.add_edge("validation_failed", END)

        # Sequential pipeline: config -> analysis -> RAG check
        workflow.add_edge("review_config", "pedagogical_analysis")

        # After analysis, check RAG threshold before planning
        workflow.add_edge("pedagogical_analysis", "check_rag_threshold")

        # RAG threshold routing: sufficient -> plan, blocked -> insufficient_rag
        workflow.add_conditional_edges(
            "check_rag_threshold",
            route_after_rag_check,
            {
                "plan_course": "plan_course",
                "insufficient_rag": "insufficient_rag",
            }
        )

        # Insufficient RAG leads to END
        workflow.add_edge("insufficient_rag", END)

        # Enter production loop after planning
        workflow.add_conditional_edges(
            "plan_course",
            route_production_loop,
            {
                "generate_media": "generate_media",
                "finalize": "finalize",  # Skip if 0 lectures
            }
        )

        # Production loop: generate_media -> check for more -> repeat or finalize
        workflow.add_conditional_edges(
            "generate_media",
            route_production_loop,
            {
                "generate_media": "generate_media",  # More lectures? Loop back.
                "finalize": "finalize",              # Done? Finalize.
            }
        )

        # Code generation loop (for individual code block processing)
        workflow.add_edge("generate_code", "review_code")
        workflow.add_conditional_edges(
            "review_code",
            route_after_code_review,
            {
                "finalize": "finalize",
                "refine_code": "refine_code",
                "generate_next": "finalize",
            }
        )
        workflow.add_conditional_edges(
            "refine_code",
            route_after_refinement,
            {
                "review_code": "review_code",
                "finalize": "finalize",
            }
        )

        # Terminal node
        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def run(
        self,
        job_id: str,
        topic: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the course generation graph.

        Args:
            job_id: Unique job identifier
            topic: Course topic
            **kwargs: Additional state fields

        Returns:
            Final state after graph execution
        """
        initial_state = create_initial_state(job_id, topic, **kwargs)

        print(f"[GRAPH] Starting course generation for job: {job_id}", flush=True)
        print(f"[GRAPH] Topic: {topic}", flush=True)

        try:
            result = await self.graph.ainvoke(initial_state)
            return result
        except Exception as e:
            print(f"[GRAPH] Error: {e}", flush=True)
            initial_state["errors"] = [str(e)]
            return initial_state

    async def process_code_block(
        self,
        state: CourseGenerationState,
        concept: str,
        language: str,
        raw_code: Optional[str] = None,
    ) -> CourseGenerationState:
        """
        Process a single code block through the generation/review loop.

        This is a convenience method for processing code blocks outside
        the main graph flow.

        Args:
            state: Current state
            concept: Concept to demonstrate
            language: Programming language
            raw_code: Optional raw code to review (skip generation)

        Returns:
            Updated state with processed code block
        """
        # Set up the code block in state
        state["current_code_block"] = {
            "raw_code": raw_code or "",
            "refined_code": None,
            "language": language,
            "concept": concept,
            "persona_level": state.get("persona_level", "intermediate"),
            "complexity_score": 0,
            "review_status": "pending",
            "rejection_reasons": [],
            "execution_result": None,
            "retry_count": 0,
            "max_retries": 3,
        }

        # If no raw code, generate it
        if not raw_code:
            state = await generate_code_block(state)

        # Review loop
        max_iterations = 4  # 1 initial + 3 retries
        for _ in range(max_iterations):
            state = await review_code_block(state)

            code_block = state.get("current_code_block", {})
            status = code_block.get("review_status")

            if status == "approved":
                break
            elif status == "rejected" and code_block.get("retry_needed"):
                state = await refine_code_block(state)
            else:
                break

        return state


# =============================================================================
# FACTORY AND SINGLETON
# =============================================================================

_graph_instance: Optional[CourseGenerationGraph] = None


def get_course_generation_graph() -> CourseGenerationGraph:
    """Get the singleton course generation graph instance"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = CourseGenerationGraph()
    return _graph_instance


def create_course_generation_graph() -> CourseGenerationGraph:
    """Factory function to create a new graph instance"""
    return CourseGenerationGraph()
