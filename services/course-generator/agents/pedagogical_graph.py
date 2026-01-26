"""
Pedagogical Agent Graph

LangGraph workflow for intelligent course planning.
"""
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END

from agents.pedagogical_state import PedagogicalAgentState
from agents.pedagogical_nodes import (
    analyze_context,
    fetch_rag_images,
    adapt_for_profile,
    suggest_elements,
    plan_quizzes,
    validate_language,
    validate_structure,
    refine_outline,
    should_refine,
    finalize_plan,
)
from models.course_models import (
    CourseOutline,
    PreviewOutlineRequest,
    ProfileCategory,
    DifficultyLevel,
)


class PedagogicalAgent:
    """
    LangGraph-based pedagogical agent for intelligent course planning.

    This agent analyzes the course topic, adapts content for the learner profile,
    suggests appropriate lesson elements, plans quizzes, and validates the
    pedagogical quality of the course structure.
    """

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create the graph with our state type
        workflow = StateGraph(PedagogicalAgentState)

        # Add nodes
        workflow.add_node("analyze_context", analyze_context)
        workflow.add_node("fetch_rag_images", fetch_rag_images)
        workflow.add_node("adapt_for_profile", adapt_for_profile)
        workflow.add_node("suggest_elements", suggest_elements)
        workflow.add_node("plan_quizzes", plan_quizzes)
        workflow.add_node("validate_language", validate_language)
        workflow.add_node("validate_structure", validate_structure)
        workflow.add_node("refine_outline", refine_outline)  # Feedback loop node
        workflow.add_node("finalize_plan", finalize_plan)

        # Define the workflow edges
        workflow.set_entry_point("analyze_context")

        # Main flow
        workflow.add_edge("analyze_context", "fetch_rag_images")
        workflow.add_edge("fetch_rag_images", "adapt_for_profile")
        workflow.add_edge("adapt_for_profile", "suggest_elements")
        workflow.add_edge("suggest_elements", "plan_quizzes")
        workflow.add_edge("plan_quizzes", "validate_language")
        workflow.add_edge("validate_language", "validate_structure")

        # FEEDBACK LOOP: Conditional routing after validation
        # If validation fails and attempts remain → refine_outline
        # If validation passes or max attempts → finalize_plan
        workflow.add_conditional_edges(
            "validate_structure",
            should_refine,
            {
                "refine": "refine_outline",
                "finalize": "finalize_plan",
            }
        )

        # After refinement, go back to validation
        workflow.add_edge("refine_outline", "validate_structure")

        # Final step
        workflow.add_edge("finalize_plan", END)

        return workflow.compile()

    async def enhance_outline(
        self,
        outline: CourseOutline,
        request: PreviewOutlineRequest,
    ) -> Dict[str, Any]:
        """
        Enhance an existing outline with pedagogical intelligence.

        Args:
            outline: The base course outline to enhance
            request: The original generation request

        Returns:
            Dict with enhanced outline and metadata
        """
        # Build initial state from request and outline
        initial_state: PedagogicalAgentState = {
            "topic": request.topic,
            "description": request.description,
            "profile_category": request.context.category if request.context else ProfileCategory.EDUCATION,
            "difficulty_start": request.difficulty_start,
            "difficulty_end": request.difficulty_end,
            "target_language": getattr(request, 'language', 'en'),
            "target_audience": outline.target_audience,
            "structure_sections": request.structure.number_of_sections,
            "structure_lectures_per_section": request.structure.lectures_per_section,
            "total_duration_minutes": request.structure.total_duration_minutes,
            "rag_context": request.rag_context,
            "document_ids": request.document_ids or [],
            "quiz_enabled": True,  # Always enabled
            "quiz_frequency": "per_section",  # Default
            "outline": outline,
            "errors": [],
            # Feedback loop control
            "refinement_attempts": 0,
            "max_refinement_attempts": 2,  # Allow up to 2 refinement cycles
            "refinement_history": [],
        }

        print(f"[AGENT] Starting pedagogical enhancement for: {request.topic}", flush=True)

        # Run the graph
        try:
            result = await self.graph.ainvoke(initial_state)

            # Include refinement info in metadata
            metadata = result.get("generation_metadata", {})
            metadata["refinement_attempts"] = result.get("refinement_attempts", 0)
            metadata["refinement_history"] = result.get("refinement_history", [])

            return {
                "outline": result.get("final_outline", outline),
                "metadata": metadata,
                "element_mapping": result.get("element_mapping", {}),
                "quiz_placement": result.get("quiz_placement", []),
                "validation_result": result.get("validation_result", {}),
                "errors": result.get("errors", []),
            }

        except Exception as e:
            print(f"[AGENT] Error in pedagogical agent: {e}", flush=True)
            return {
                "outline": outline,  # Return original outline
                "metadata": {"error": str(e)},
                "element_mapping": {},
                "quiz_placement": [],
                "validation_result": {},
                "errors": [str(e)],
            }

    async def plan_from_scratch(
        self,
        topic: str,
        description: Optional[str] = None,
        category: ProfileCategory = ProfileCategory.EDUCATION,
        difficulty_start: DifficultyLevel = DifficultyLevel.BEGINNER,
        difficulty_end: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
        target_language: str = "en",
        target_audience: str = "general learners",
        num_sections: int = 4,
        lectures_per_section: int = 3,
        duration_minutes: int = 60,
        rag_context: Optional[str] = None,
        document_ids: Optional[list] = None,
        quiz_enabled: bool = True,
        quiz_frequency: str = "per_section",
    ) -> Dict[str, Any]:
        """
        Create a pedagogical plan from scratch (without pre-existing outline).

        This is useful for analysis and planning before outline generation.

        Returns analysis results that can inform the outline generation.
        """
        initial_state: PedagogicalAgentState = {
            "topic": topic,
            "description": description,
            "profile_category": category,
            "difficulty_start": difficulty_start,
            "difficulty_end": difficulty_end,
            "target_language": target_language,
            "target_audience": target_audience,
            "structure_sections": num_sections,
            "structure_lectures_per_section": lectures_per_section,
            "total_duration_minutes": duration_minutes,
            "rag_context": rag_context,
            "document_ids": document_ids or [],
            "quiz_enabled": quiz_enabled,
            "quiz_frequency": quiz_frequency,
            "outline": None,  # No outline yet
            "errors": [],
        }

        print(f"[AGENT] Planning from scratch for: {topic}", flush=True)

        try:
            # Run only the analysis and adaptation nodes
            # (skip suggest_elements and later nodes that need an outline)
            context_result = await analyze_context(initial_state)
            initial_state.update(context_result)

            rag_result = await fetch_rag_images(initial_state)
            initial_state.update(rag_result)

            profile_result = await adapt_for_profile(initial_state)
            initial_state.update(profile_result)

            return {
                "detected_persona": initial_state.get("detected_persona"),
                "topic_complexity": initial_state.get("topic_complexity"),
                "requires_code": initial_state.get("requires_code"),
                "requires_diagrams": initial_state.get("requires_diagrams"),
                "content_preferences": initial_state.get("content_preferences"),
                "recommended_elements": initial_state.get("recommended_elements"),
                "rag_images": initial_state.get("rag_images"),
                "errors": initial_state.get("errors", []),
            }

        except Exception as e:
            print(f"[AGENT] Error in planning: {e}", flush=True)
            return {
                "error": str(e),
                "detected_persona": "student",
                "topic_complexity": "intermediate",
                "requires_code": False,
                "requires_diagrams": True,
                "content_preferences": {},
                "recommended_elements": [],
            }


def create_pedagogical_agent() -> PedagogicalAgent:
    """Factory function to create a pedagogical agent instance"""
    return PedagogicalAgent()


# Singleton instance for reuse
_agent_instance: Optional[PedagogicalAgent] = None


def get_pedagogical_agent() -> PedagogicalAgent:
    """Get the singleton pedagogical agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PedagogicalAgent()
    return _agent_instance
