"""
Planning Subgraph

LangGraph subgraph for curriculum planning and outline generation.
This subgraph is isolated from production concerns.

Responsibilities:
1. Pedagogical analysis (detect persona, complexity, content preferences)
2. Outline generation (sections, lectures, elements)
3. Structure validation and enforcement
4. Prepare lecture plans for production

Flow:
    analyze_context -> generate_outline -> enforce_structure -> prepare_lectures -> END
"""
import json
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END

from agents.state import (
    PlanningState,
    PlanningStatus,
    LecturePlan,
    CodeBlockInfo,
)
from agents.pedagogical_graph import get_pedagogical_agent
# CoursePlanner is imported lazily inside generate_outline to avoid circular import
from models.course_models import (
    PreviewOutlineRequest,
    CourseOutline,
    CourseStructureConfig,
    ProfileCategory,
    DifficultyLevel,
)


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

async def analyze_context(state: PlanningState) -> PlanningState:
    """
    Node: Analyze topic and context using PedagogicalAgent.

    Determines:
    - Target persona and complexity
    - Content preferences (code vs diagrams vs theory)
    - Recommended lesson elements
    - RAG image extraction (if documents provided)
    """
    print(f"[PLANNING] Analyzing context for: {state.get('topic', 'Unknown')}", flush=True)

    state["status"] = PlanningStatus.ANALYZING

    try:
        agent = get_pedagogical_agent()

        result = await agent.plan_from_scratch(
            topic=state.get("topic", ""),
            description=state.get("description"),
            category=state.get("profile_category", "education"),
            difficulty_start=state.get("difficulty_start", "beginner"),
            difficulty_end=state.get("difficulty_end", "intermediate"),
            target_language=state.get("content_language", "en"),
            target_audience=state.get("target_audience", "general learners"),
            num_sections=state.get("structure", {}).get("number_of_sections", state.get("number_of_sections", 4)),
            lectures_per_section=state.get("structure", {}).get("lectures_per_section", state.get("lectures_per_section", 3)),
            duration_minutes=state.get("structure", {}).get("total_duration_minutes", state.get("total_duration_minutes", 60)),
            rag_context=state.get("rag_context"),
            document_ids=state.get("document_ids"),
            quiz_enabled=state.get("quiz_enabled", True),
            quiz_frequency=state.get("quiz_frequency", "per_section"),
        )

        # Update state with analysis results
        state["detected_persona"] = result.get("detected_persona", "student")
        state["topic_complexity"] = result.get("topic_complexity", "intermediate")
        state["domain_keywords"] = result.get("domain_keywords", [])
        state["requires_code"] = result.get("requires_code", False)
        state["requires_diagrams"] = result.get("requires_diagrams", True)
        state["requires_hands_on"] = result.get("requires_hands_on", False)
        state["content_preferences"] = result.get("content_preferences", {})
        state["recommended_elements"] = result.get("recommended_elements", [])
        state["rag_images"] = result.get("rag_images", [])

        print(f"[PLANNING] Analysis complete: persona={state['detected_persona']}, "
              f"complexity={state['topic_complexity']}, "
              f"requires_code={state['requires_code']}", flush=True)

    except Exception as e:
        print(f"[PLANNING] Analysis failed: {e}", flush=True)
        state["errors"] = state.get("errors", []) + [f"Pedagogical analysis failed: {str(e)}"]
        # Set defaults on failure
        state["detected_persona"] = "student"
        state["topic_complexity"] = "intermediate"
        state["requires_code"] = state.get("profile_category") == "tech"
        state["requires_diagrams"] = True
        state["content_preferences"] = {"theory": 0.5, "code": 0.3, "diagram": 0.5}

    return state


async def generate_outline(state: PlanningState) -> PlanningState:
    """
    Node: Generate course outline using CoursePlanner.

    Creates sections and lectures based on:
    - Topic and description
    - Pedagogical analysis results
    - Structure configuration
    - RAG context (if available)
    """
    print(f"[PLANNING] Generating outline for: {state.get('topic', 'Unknown')}", flush=True)

    state["status"] = PlanningStatus.GENERATING_OUTLINE

    try:
        # Lazy import to avoid circular dependency
        from services.course_planner import CoursePlanner

        planner = CoursePlanner()

        # Map profile category
        profile_map = {
            "tech": ProfileCategory.TECH,
            "business": ProfileCategory.BUSINESS,
            "creative": ProfileCategory.CREATIVE,
            "health": ProfileCategory.HEALTH,
            "education": ProfileCategory.EDUCATION,
            "lifestyle": ProfileCategory.LIFESTYLE,
        }
        profile = profile_map.get(
            state.get("profile_category", "education").lower(),
            ProfileCategory.EDUCATION
        )

        # Map difficulty
        diff_map = {
            "beginner": DifficultyLevel.BEGINNER,
            "intermediate": DifficultyLevel.INTERMEDIATE,
            "advanced": DifficultyLevel.ADVANCED,
            "expert": DifficultyLevel.EXPERT,
        }
        diff_start = diff_map.get(
            state.get("difficulty_start", "beginner").lower(),
            DifficultyLevel.BEGINNER
        )
        diff_end = diff_map.get(
            state.get("difficulty_end", "intermediate").lower(),
            DifficultyLevel.INTERMEDIATE
        )

        # Build structure config
        structure_dict = state.get("structure", {})
        structure_config = CourseStructureConfig(
            total_duration_minutes=structure_dict.get("total_duration_minutes", state.get("total_duration_minutes", 60)),
            number_of_sections=structure_dict.get("number_of_sections", state.get("number_of_sections", 4)),
            lectures_per_section=structure_dict.get("lectures_per_section", state.get("lectures_per_section", 3)),
            random_structure=structure_dict.get("random_structure", False),
        )

        # Build request with correct field names
        request = PreviewOutlineRequest(
            topic=state.get("topic", ""),
            description=state.get("description"),
            difficulty_start=diff_start,
            difficulty_end=diff_end,
            structure=structure_config,
            language=state.get("content_language", "en"),
            rag_context=state.get("rag_context"),
            document_ids=state.get("document_ids", []),
        )

        # Generate outline
        outline: CourseOutline = await planner.generate_outline(request)

        # Serialize and store
        state["outline"] = outline.dict() if hasattr(outline, "dict") else outline.model_dump()
        state["sections"] = [
            s.dict() if hasattr(s, "dict") else s.model_dump()
            for s in outline.sections
        ]
        state["total_lectures"] = outline.total_lectures

        print(f"[PLANNING] Outline generated: {len(outline.sections)} sections, "
              f"{outline.total_lectures} lectures", flush=True)

    except Exception as e:
        print(f"[PLANNING] Outline generation failed: {e}", flush=True)
        state["errors"] = state.get("errors", []) + [f"Outline generation failed: {str(e)}"]
        state["status"] = PlanningStatus.FAILED

    return state


async def enforce_structure(state: PlanningState) -> PlanningState:
    """
    Node: Enforce curriculum structure constraints.

    Validates and adjusts:
    - Duration constraints per lecture
    - Difficulty progression
    - Element distribution
    - Quiz placement
    """
    print("[PLANNING] Enforcing structure constraints", flush=True)

    state["status"] = PlanningStatus.ENFORCING_STRUCTURE

    if not state.get("outline"):
        state["warnings"] = state.get("warnings", []) + ["No outline to enforce structure on"]
        return state

    outline = state["outline"]
    sections = state.get("sections", [])

    # Validate total duration
    total_duration = state.get("total_duration_minutes", 60) * 60  # Convert to seconds
    current_duration = sum(
        lecture.get("duration_seconds", 0)
        for section in sections
        for lecture in section.get("lectures", [])
    )

    if current_duration > total_duration * 1.2:  # 20% tolerance
        state["warnings"] = state.get("warnings", []) + [
            f"Total duration ({current_duration}s) exceeds target ({total_duration}s) by >20%"
        ]

    # Validate difficulty progression
    diff_order = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    start_level = diff_order.get(state.get("difficulty_start", "beginner").lower(), 1)
    end_level = diff_order.get(state.get("difficulty_end", "intermediate").lower(), 2)

    for section in sections:
        for lecture in section.get("lectures", []):
            lecture_diff = lecture.get("difficulty", "intermediate").lower()
            lecture_level = diff_order.get(lecture_diff, 2)

            if lecture_level < start_level or lecture_level > end_level:
                state["warnings"] = state.get("warnings", []) + [
                    f"Lecture '{lecture.get('title', 'Unknown')}' difficulty "
                    f"'{lecture_diff}' outside range [{state.get('difficulty_start')}-{state.get('difficulty_end')}]"
                ]

    print(f"[PLANNING] Structure enforcement complete. Warnings: {len(state.get('warnings', []))}", flush=True)

    return state


async def prepare_lecture_plans(state: PlanningState) -> PlanningState:
    """
    Node: Prepare LecturePlan objects for production.

    Transforms the outline into production-ready lecture plans that include:
    - Code block specifications
    - Diagram descriptions
    - Element assignments
    - Position metadata
    - Course and section context for prompt building
    """
    print("[PLANNING] Preparing lecture plans for production", flush=True)

    if not state.get("sections"):
        state["status"] = PlanningStatus.FAILED
        state["errors"] = state.get("errors", []) + ["No sections available for lecture planning"]
        return state

    sections = state["sections"]
    outline = state.get("outline", {})
    lecture_plans: List[LecturePlan] = []
    position = 0
    total_lectures = state.get("total_lectures", 0)

    # Calculate duration per lecture from total course duration
    # FIX: Don't rely on LLM-generated duration_seconds, calculate from user's requested total
    total_duration_minutes = state.get("total_duration_minutes", 60)
    duration_per_lecture_seconds = (total_duration_minutes * 60) // max(total_lectures, 1)
    # Ensure minimum 60 seconds, maximum 30 minutes per lecture
    duration_per_lecture_seconds = max(60, min(duration_per_lecture_seconds, 1800))

    print(f"[PLANNING] Duration calc: {total_duration_minutes}min total / {total_lectures} lectures = {duration_per_lecture_seconds}s per lecture", flush=True)

    requires_code = state.get("requires_code", False)
    programming_language = state.get("programming_language", "python")
    content_preferences = state.get("content_preferences", {})

    # Get course-level context
    course_title = outline.get("title", state.get("topic", ""))
    target_audience = outline.get("target_audience", state.get("target_audience", "general learners"))

    for section in sections:
        section_id = section.get("id", f"section_{len(lecture_plans)}")
        section_title = section.get("title", f"Section {section.get('order', 0) + 1}")
        section_description = section.get("description", "")

        for lecture in section.get("lectures", []):
            position += 1

            # Extract code blocks if requires_code
            code_blocks: List[CodeBlockInfo] = []
            if requires_code:
                # Check lecture elements for code
                elements = lecture.get("lesson_elements", [])
                if any(e in ["code_demo", "code_typing", "code_execution"] for e in elements):
                    # Create code block spec from lecture objectives
                    for i, objective in enumerate(lecture.get("objectives", [])[:2]):  # Max 2 code blocks
                        code_blocks.append(CodeBlockInfo(
                            concept=objective,
                            language=programming_language or "python",
                            description=f"Code demonstrating: {objective}",
                            persona_level=state.get("difficulty_start", "intermediate"),
                            complexity_target=3,  # Mid complexity
                        ))

            # Extract diagram descriptions
            diagram_descriptions: List[str] = []
            if state.get("requires_diagrams", True):
                elements = lecture.get("lesson_elements", [])
                if any(e in ["diagram_schema", "architecture_diagram", "flowchart"] for e in elements):
                    diagram_descriptions.append(
                        f"Diagram illustrating: {lecture.get('title', 'concept')}"
                    )

            lecture_plan = LecturePlan(
                lecture_id=lecture.get("id", f"lecture_{position}"),
                section_id=section_id,
                title=lecture.get("title", f"Lecture {position}"),
                description=lecture.get("description", ""),
                objectives=lecture.get("objectives", []),
                difficulty=lecture.get("difficulty", "intermediate"),
                # FIX: Use calculated duration from total course duration, not LLM-generated
                duration_seconds=duration_per_lecture_seconds,
                position=position,
                total_lectures=total_lectures,
                # Section context for prompt building
                section_title=section_title,
                section_description=section_description,
                # Course context for prompt building
                course_title=course_title,
                target_audience=target_audience,
                # Content specifications
                requires_code=len(code_blocks) > 0,
                requires_diagrams=len(diagram_descriptions) > 0,
                code_blocks=code_blocks,
                diagram_descriptions=diagram_descriptions,
                voiceover_script=lecture.get("voiceover_script"),
                lesson_elements=lecture.get("lesson_elements", []),
            )

            lecture_plans.append(lecture_plan)

    state["lecture_plans"] = lecture_plans
    state["status"] = PlanningStatus.COMPLETED

    print(f"[PLANNING] Prepared {len(lecture_plans)} lecture plans", flush=True)

    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_analysis(state: PlanningState) -> Literal["generate_outline", "failed"]:
    """Route based on analysis result"""
    if state.get("status") == PlanningStatus.FAILED:
        return "failed"
    return "generate_outline"


def route_after_outline(state: PlanningState) -> Literal["enforce_structure", "failed"]:
    """Route based on outline generation result"""
    if state.get("status") == PlanningStatus.FAILED or not state.get("outline"):
        return "failed"
    return "enforce_structure"


async def handle_planning_failure(state: PlanningState) -> PlanningState:
    """Node: Handle planning failures"""
    state["status"] = PlanningStatus.FAILED
    print(f"[PLANNING] Planning failed with errors: {state.get('errors', [])}", flush=True)
    return state


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_planning_subgraph() -> StateGraph:
    """
    Build the Planning subgraph.

    Flow:
        analyze_context -> generate_outline -> enforce_structure -> prepare_lectures
                    ↓              ↓
                 failed         failed
    """
    workflow = StateGraph(PlanningState)

    # Add nodes
    workflow.add_node("analyze_context", analyze_context)
    workflow.add_node("generate_outline", generate_outline)
    workflow.add_node("enforce_structure", enforce_structure)
    workflow.add_node("prepare_lectures", prepare_lecture_plans)
    workflow.add_node("failed", handle_planning_failure)

    # Set entry point
    workflow.set_entry_point("analyze_context")

    # Add edges
    workflow.add_conditional_edges(
        "analyze_context",
        route_after_analysis,
        {
            "generate_outline": "generate_outline",
            "failed": "failed",
        }
    )

    workflow.add_conditional_edges(
        "generate_outline",
        route_after_outline,
        {
            "enforce_structure": "enforce_structure",
            "failed": "failed",
        }
    )

    workflow.add_edge("enforce_structure", "prepare_lectures")
    workflow.add_edge("prepare_lectures", END)
    workflow.add_edge("failed", END)

    return workflow.compile()


# =============================================================================
# SINGLETON
# =============================================================================

_planning_graph_instance = None


def get_planning_graph():
    """Get the singleton planning subgraph instance"""
    global _planning_graph_instance
    if _planning_graph_instance is None:
        _planning_graph_instance = build_planning_subgraph()
    return _planning_graph_instance
