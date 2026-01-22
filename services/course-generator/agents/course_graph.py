"""
Course Generation Graph

LangGraph-based orchestrator for the multi-agent course generation system.
This graph coordinates all agents in the proper sequence with conditional
routing based on configuration and intermediate results.
"""
from typing import Any, Dict, Literal, Optional
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

    # Summary statistics
    approved = state.get("code_blocks_approved", 0)
    rejected = state.get("code_blocks_rejected", 0)
    total = state.get("code_blocks_processed", 0)

    print(f"[FINALIZE] Generation complete. Code blocks: {approved} approved, "
          f"{rejected} rejected out of {total} processed.", flush=True)

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
        """Build the LangGraph workflow"""
        workflow = StateGraph(CourseGenerationState)

        # Add nodes
        workflow.add_node("validate_input", validate_input)
        workflow.add_node("validation_failed", handle_validation_failure)
        workflow.add_node("review_config", review_config)
        workflow.add_node("pedagogical_analysis", run_pedagogical_analysis)
        workflow.add_node("generate_code", generate_code_block)
        workflow.add_node("review_code", review_code_block)
        workflow.add_node("refine_code", refine_code_block)
        workflow.add_node("finalize", finalize_generation)

        # Set entry point
        workflow.set_entry_point("validate_input")

        # Add conditional edges
        workflow.add_conditional_edges(
            "validate_input",
            route_after_validation,
            {
                "review_config": "review_config",
                "validation_failed": "validation_failed",
            }
        )

        # Linear edges
        workflow.add_edge("validation_failed", END)
        workflow.add_edge("review_config", "pedagogical_analysis")
        workflow.add_edge("pedagogical_analysis", "finalize")  # For now, skip to finalize

        # Code generation loop (used when processing code blocks)
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
