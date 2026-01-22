"""
Multi-Agent Course Generation System

This module provides a scalable, LangGraph-based multi-agent architecture
for intelligent course generation. The system uses a hierarchical subgraph
architecture for isolation and maintainability:

ARCHITECTURE:
    OrchestratorGraph (main)
        ├── PlanningSubgraph
        │   └── Handles curriculum planning, pedagogical analysis
        └── ProductionSubgraph (per lecture)
            └── Handles code generation, media production, recovery

AGENTS:
- InputValidatorAgent: Validates all frontend configuration choices
- TechnicalReviewerAgent: Enriches prompts with configuration requirements
- PedagogicalAgent: Analyzes topic and plans curriculum
- CodeExpertAgent: Generates production-quality code
- CodeReviewerAgent: Validates code quality before inclusion
- ScriptSimplifierAgent: Simplifies scripts for error recovery

USAGE:
    from agents import get_course_orchestrator

    orchestrator = get_course_orchestrator()
    result = await orchestrator.run(
        job_id="job_123",
        topic="Python Web Development with FastAPI",
        content_language="fr",
        programming_language="python",
        ...
    )

LEGACY SUPPORT:
    The old CourseGenerationGraph is still available for backward compatibility.
"""

# Base classes and state (legacy)
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

# New hierarchical state definitions
from agents.state import (
    OrchestratorState,
    PlanningState,
    ProductionState,
    RecoveryStrategy,
    ProductionStatus,
    PlanningStatus,
    LecturePlan,
    CodeBlockInfo,
    GeneratedCodeBlock,
    MediaResult,
    create_orchestrator_state,
    create_planning_state_from_orchestrator,
    create_production_state_for_lecture,
    merge_planning_result_to_orchestrator,
    merge_production_result_to_orchestrator,
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

# Main orchestration graph (legacy)
from agents.course_graph import (
    CourseGenerationGraph,
    get_course_generation_graph,
    create_course_generation_graph,
)

# NEW: Hierarchical subgraphs
from agents.planning_graph import (
    build_planning_subgraph,
    get_planning_graph,
)
from agents.production_graph import (
    build_production_subgraph,
    get_production_graph,
)
from agents.orchestrator_graph import (
    CourseOrchestrator,
    get_course_orchestrator,
    create_course_orchestrator,
)

# NEW: Script simplifier agent
from agents.script_simplifier import (
    ScriptSimplifierAgent,
    get_script_simplifier,
)

# Integration utilities
from agents.integration import (
    MultiAgentOrchestrator,
    get_multi_agent_orchestrator,
    validate_course_config,
    generate_quality_code,
)

__all__ = [
    # Base classes (legacy)
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

    # NEW: Hierarchical states
    "OrchestratorState",
    "PlanningState",
    "ProductionState",
    "RecoveryStrategy",
    "ProductionStatus",
    "PlanningStatus",
    "LecturePlan",
    "CodeBlockInfo",
    "GeneratedCodeBlock",
    "MediaResult",
    "create_orchestrator_state",
    "create_planning_state_from_orchestrator",
    "create_production_state_for_lecture",
    "merge_planning_result_to_orchestrator",
    "merge_production_result_to_orchestrator",

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
    "ScriptSimplifierAgent",
    "get_script_simplifier",

    # Orchestration (legacy)
    "CourseGenerationGraph",
    "get_course_generation_graph",
    "create_course_generation_graph",

    # NEW: Hierarchical orchestration
    "CourseOrchestrator",
    "get_course_orchestrator",
    "create_course_orchestrator",
    "build_planning_subgraph",
    "get_planning_graph",
    "build_production_subgraph",
    "get_production_graph",

    # Integration
    "MultiAgentOrchestrator",
    "get_multi_agent_orchestrator",
    "validate_course_config",
    "generate_quality_code",
]
