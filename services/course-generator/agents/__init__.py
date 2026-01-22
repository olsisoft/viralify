"""
Multi-Agent Course Generation System

This module provides a scalable, LangGraph-based multi-agent architecture
for intelligent course generation. The system includes:

- InputValidatorAgent: Validates all frontend configuration choices
- TechnicalReviewerAgent: Enriches prompts with configuration requirements
- PedagogicalAgent: Analyzes topic and plans curriculum
- CodeExpertAgent: Generates production-quality code
- CodeReviewerAgent: Validates code quality before inclusion
- CourseGenerationGraph: Orchestrates all agents

Usage:
    from agents import get_course_generation_graph

    graph = get_course_generation_graph()
    result = await graph.run(
        job_id="job_123",
        topic="Python Web Development with FastAPI",
        content_language="fr",
        programming_language="python",
        ...
    )
"""

# Base classes and state
from agents.base import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentResult,
    CourseGenerationState,
    CodeBlockState,
    LectureState,
    SlideState,
    ValidationError,
    create_initial_state,
)

# Individual agents
from agents.input_validator import (
    InputValidatorAgent,
    create_input_validator,
)
from agents.technical_reviewer import (
    TechnicalReviewerAgent,
    create_technical_reviewer,
)
from agents.code_expert import (
    CodeExpertAgent,
    create_code_expert,
)
from agents.code_reviewer import (
    CodeReviewerAgent,
    create_code_reviewer,
)

# Pedagogical agent (existing)
from agents.pedagogical_graph import (
    PedagogicalAgent,
    create_pedagogical_agent,
    get_pedagogical_agent,
)

# Main orchestration graph
from agents.course_graph import (
    CourseGenerationGraph,
    get_course_generation_graph,
    create_course_generation_graph,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentType",
    "AgentStatus",
    "AgentResult",
    "CourseGenerationState",
    "CodeBlockState",
    "LectureState",
    "SlideState",
    "ValidationError",
    "create_initial_state",

    # Agents
    "InputValidatorAgent",
    "create_input_validator",
    "TechnicalReviewerAgent",
    "create_technical_reviewer",
    "CodeExpertAgent",
    "create_code_expert",
    "CodeReviewerAgent",
    "create_code_reviewer",
    "PedagogicalAgent",
    "create_pedagogical_agent",
    "get_pedagogical_agent",

    # Orchestration
    "CourseGenerationGraph",
    "get_course_generation_graph",
    "create_course_generation_graph",
]
