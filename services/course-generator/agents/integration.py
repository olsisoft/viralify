"""
Multi-Agent Integration Module

Integrates the multi-agent course generation system with the existing
FastAPI service. Provides functions for validation, prompt enrichment,
and code quality management.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.base import (
    CourseGenerationState,
    create_initial_state,
    AgentStatus,
)
from agents.input_validator import InputValidatorAgent
from agents.technical_reviewer import TechnicalReviewerAgent
from agents.code_expert import CodeExpertAgent
from agents.code_reviewer import CodeReviewerAgent
from agents.course_graph import get_course_generation_graph


class MultiAgentOrchestrator:
    """
    Orchestrates multi-agent validation and enrichment for course generation.

    This class provides a bridge between the existing course generation flow
    and the new multi-agent system. It can be used incrementally:

    1. validate_and_enrich() - Run validation and prompt enrichment only
    2. process_code_block() - Generate and review a single code block
    3. run_full_pipeline() - Run the complete multi-agent graph
    """

    def __init__(self):
        self.input_validator = InputValidatorAgent()
        self.technical_reviewer = TechnicalReviewerAgent()
        self.code_expert = CodeExpertAgent()
        self.code_reviewer = CodeReviewerAgent()
        self.graph = get_course_generation_graph()

    async def validate_and_enrich(
        self,
        job_id: str,
        topic: str,
        description: Optional[str] = None,
        profile_category: str = "education",
        difficulty_start: str = "beginner",
        difficulty_end: str = "intermediate",
        content_language: str = "en",
        programming_language: str = "python",
        target_audience: Optional[str] = None,
        structure: Optional[Dict[str, Any]] = None,
        lesson_elements: Optional[Dict[str, bool]] = None,
        quiz_config: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
        rag_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run validation and technical review only.

        This is useful for integrating with existing flows where you want
        multi-agent validation without replacing the entire pipeline.

        Returns:
            Dictionary containing:
            - validated: bool - Whether validation passed
            - validation_errors: List[Dict] - Any validation errors
            - enriched_state: Dict - State with prompt enrichments
            - code_expert_prompt: str - Enriched prompt for code generation
            - warnings: List[str] - Non-blocking warnings
            - suggestions: List[str] - Improvement suggestions
        """
        # Create initial state
        state = create_initial_state(
            job_id=job_id,
            topic=topic,
            description=description,
            profile_category=profile_category,
            difficulty_start=difficulty_start,
            difficulty_end=difficulty_end,
            content_language=content_language,
            programming_language=programming_language,
            target_audience=target_audience,
            structure=structure or {"number_of_sections": 4, "lectures_per_section": 3},
            lesson_elements=lesson_elements or {},
            quiz_config=quiz_config or {"enabled": True, "frequency": "per_section"},
            document_ids=document_ids,
            rag_context=rag_context,
        )

        # Step 1: Input Validation
        print(f"[MULTI-AGENT:{job_id}] Running input validation...", flush=True)
        state = await self.input_validator.process(state)

        if not state.get("input_validated", False):
            errors = state.get("input_validation_errors", [])
            print(f"[MULTI-AGENT:{job_id}] Validation FAILED: {len(errors)} errors", flush=True)
            return {
                "validated": False,
                "validation_errors": errors,
                "enriched_state": state,
                "code_expert_prompt": None,
                "warnings": state.get("warnings", []),
                "suggestions": [],
            }

        print(f"[MULTI-AGENT:{job_id}] Validation PASSED", flush=True)

        # Step 2: Technical Review / Prompt Enrichment
        print(f"[MULTI-AGENT:{job_id}] Running technical review...", flush=True)
        state = await self.technical_reviewer.process(state)

        print(f"[MULTI-AGENT:{job_id}] Prompt enrichment complete", flush=True)

        return {
            "validated": True,
            "validation_errors": [],
            "enriched_state": dict(state),
            "code_expert_prompt": state.get("code_expert_prompt"),
            "prompt_enrichments": state.get("prompt_enrichments", {}),
            "warnings": state.get("warnings", []),
            "suggestions": state.get("config_suggestions", []),
        }

    async def process_code_block(
        self,
        concept: str,
        language: str,
        persona_level: str,
        rag_context: Optional[str] = None,
        code_expert_prompt: Optional[str] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate and review a single code block.

        This runs the code through the generation -> review -> refine loop
        until it's approved or max retries are reached.

        Args:
            concept: The concept to demonstrate in code
            language: Programming language
            persona_level: Learner level (beginner/intermediate/advanced/expert)
            rag_context: Optional RAG context for code generation
            code_expert_prompt: Optional custom system prompt
            max_retries: Maximum refinement attempts

        Returns:
            Dictionary containing:
            - approved: bool - Whether code was approved
            - code: str - The final code
            - explanation: str - Code explanation for voiceover
            - expected_output: str - Expected terminal output
            - complexity_score: int - Code complexity (1-10)
            - quality_score: int - Review quality score (1-10)
            - patterns_used: List[str] - Design patterns used
            - rejection_reasons: List[str] - Why code was rejected (if any)
            - iterations: int - Number of generation/review cycles
        """
        print(f"[CODE-BLOCK] Processing code for: {concept[:50]}...", flush=True)

        # Create minimal state for code processing
        state: CourseGenerationState = {
            "job_id": f"code_block_{datetime.utcnow().timestamp()}",
            "topic": concept,
            "persona_level": persona_level,
            "programming_language": language,
            "rag_context": rag_context,
            "code_expert_prompt": code_expert_prompt,
            "current_code_block": {
                "concept": concept,
                "language": language,
                "persona_level": persona_level,
                "raw_code": "",
                "refined_code": None,
                "complexity_score": 0,
                "review_status": "pending",
                "rejection_reasons": [],
                "retry_count": 0,
                "max_retries": max_retries,
            },
            "code_blocks_approved": 0,
            "code_blocks_rejected": 0,
            "code_blocks_processed": 0,
        }

        iterations = 0

        # Generation -> Review -> Refine loop
        for i in range(max_retries + 1):
            iterations = i + 1

            if i == 0:
                # First iteration: Generate code
                print(f"[CODE-BLOCK] Iteration {iterations}: Generating code...", flush=True)
                state = await self.code_expert.process(state)
            else:
                # Subsequent iterations: Refine based on feedback
                print(f"[CODE-BLOCK] Iteration {iterations}: Refining code...", flush=True)
                code_block = state.get("current_code_block", {})

                result = await self.code_expert.refine_code(
                    original_code=code_block.get("refined_code") or code_block.get("raw_code", ""),
                    feedback=code_block.get("retry_prompt", ""),
                    language=language,
                    persona_level=persona_level,
                )

                if result.success:
                    code_block["refined_code"] = result.data.get("code_block", "")
                    code_block["expected_output"] = result.data.get("expected_output", "")
                    state["current_code_block"] = code_block

            # Review the code
            print(f"[CODE-BLOCK] Iteration {iterations}: Reviewing code...", flush=True)
            state = await self.code_reviewer.process(state)

            code_block = state.get("current_code_block", {})
            status = code_block.get("review_status")

            if status == "approved":
                print(f"[CODE-BLOCK] Code APPROVED after {iterations} iteration(s)", flush=True)
                break
            elif status == "rejected":
                if not code_block.get("retry_needed") or i >= max_retries:
                    print(f"[CODE-BLOCK] Code REJECTED (max retries reached)", flush=True)
                    break
                print(f"[CODE-BLOCK] Code rejected, will retry...", flush=True)

        # Build result
        code_block = state.get("current_code_block", {})

        return {
            "approved": code_block.get("review_status") == "approved",
            "code": code_block.get("refined_code") or code_block.get("raw_code", ""),
            "explanation": code_block.get("explanation", ""),
            "expected_output": code_block.get("expected_output", ""),
            "complexity_score": code_block.get("complexity_score", 0),
            "quality_score": code_block.get("quality_score", 0),
            "patterns_used": code_block.get("patterns_used", []),
            "rejection_reasons": code_block.get("rejection_reasons", []),
            "iterations": iterations,
        }

    async def run_full_pipeline(
        self,
        job_id: str,
        topic: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete multi-agent pipeline via LangGraph.

        This uses the CourseGenerationGraph to run all agents
        in the proper sequence with conditional routing.

        Args:
            job_id: Unique job identifier
            topic: Course topic
            **kwargs: Additional state fields

        Returns:
            Final state after graph execution
        """
        print(f"[MULTI-AGENT:{job_id}] Running full pipeline...", flush=True)

        result = await self.graph.run(job_id=job_id, topic=topic, **kwargs)

        print(f"[MULTI-AGENT:{job_id}] Pipeline complete", flush=True)

        return result


# Singleton instance
_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    """Get the singleton orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator


async def validate_course_config(
    job_id: str,
    topic: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to validate and enrich course configuration.

    This is the recommended entry point for integrating multi-agent
    validation into existing flows.
    """
    orchestrator = get_multi_agent_orchestrator()
    return await orchestrator.validate_and_enrich(job_id=job_id, topic=topic, **kwargs)


async def generate_quality_code(
    concept: str,
    language: str,
    persona_level: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to generate quality-reviewed code.

    This runs the full code generation -> review -> refine loop.
    """
    orchestrator = get_multi_agent_orchestrator()
    return await orchestrator.process_code_block(
        concept=concept,
        language=language,
        persona_level=persona_level,
        **kwargs
    )
