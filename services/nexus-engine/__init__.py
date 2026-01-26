"""
NEXUS - Neural Execution & Understanding Synthesis

Algorithme innovant de génération de code pédagogique contextuel
avec décomposition cognitive unique et orchestration multi-agents.

Usage:
    from nexus import generate_code, NexusRequest
    
    # Simple
    result = generate_code(
        project_description="une plateforme e-commerce",
        skill_level="intermediate",
        language="python",
        provider="groq",
        api_key="your-api-key"
    )
    
    # Avancé
    from nexus import NEXUSPipeline, NexusRequest, create_nexus_pipeline
    
    pipeline = create_nexus_pipeline(provider="groq", api_key="...")
    
    request = NexusRequest(
        project_description="une plateforme e-commerce",
        lesson_context="Leçon sur l'architecture backend",
        skill_level="intermediate",
        language="python",
        target_audience=TargetAudience.DEVELOPER,
        allocated_time_seconds=300,
    )
    
    response = pipeline.generate(request)
    
    # Accéder aux segments
    for segment in response.get_segments_ordered():
        print(f"{segment.filename}: {segment.line_count} lines")
        print(segment.code)
"""

__version__ = "1.0.0"
__author__ = "NEXUS Team"
__codename__ = "Neural Execution & Understanding Synthesis"

# Core
from core.pipeline import (
    NEXUSPipeline,
    create_nexus_pipeline,
    generate_code,
    NexusConfig,
    PipelineProgress,
)

# Models
from models.data_models import (
    # Request/Response
    NexusRequest,
    NexusResponse,
    
    # Architecture DNA
    ArchitectureDNA,
    DomainEntity,
    EntityRelation,
    BusinessFlow,
    
    # Cognitive Blueprint
    CognitiveBlueprint,
    CognitiveStep,
    
    # Code Segments
    CodeSegment,
    ExecutionResult,
    
    # Enums
    TargetAudience,
    CodeVerbosity,
    ExecutionMode,
    ComponentType,
    PatternType,
)

# Engines (pour utilisation avancée)
from engines.cognitive_decomposition import CognitiveDecompositionAlgorithm
from engines.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    AgentContext,
    AgentRole,
    BaseAgent,
    ArchitectAgent,
    CoderAgent,
    ReviewerAgent,
    ExecutorAgent,
    NarratorAgent,
)
from engines.temporal_synthesizer import (
    TemporalCodeSynthesizer,
    TimeBudget,
    TimeSlot,
)

# Providers
from providers.llm_provider import (
    BaseLLMProvider,
    LLMConfig,
    LLMProvider,
    create_llm_provider,
    groq_provider,
    openai_provider,
    anthropic_provider,
    ollama_provider,
)


__all__ = [
    # Version
    "__version__",
    "__codename__",
    
    # Core
    "NEXUSPipeline",
    "create_nexus_pipeline",
    "generate_code",
    "NexusConfig",
    "PipelineProgress",
    
    # Models - Request/Response
    "NexusRequest",
    "NexusResponse",
    
    # Models - Architecture DNA
    "ArchitectureDNA",
    "DomainEntity",
    "EntityRelation",
    "BusinessFlow",
    
    # Models - Cognitive Blueprint
    "CognitiveBlueprint",
    "CognitiveStep",
    
    # Models - Code
    "CodeSegment",
    "ExecutionResult",
    
    # Models - Enums
    "TargetAudience",
    "CodeVerbosity",
    "ExecutionMode",
    "ComponentType",
    "PatternType",
    
    # Engines
    "CognitiveDecompositionAlgorithm",
    "MultiAgentOrchestrator",
    "AgentContext",
    "AgentRole",
    "TemporalCodeSynthesizer",
    "TimeBudget",
    "TimeSlot",
    
    # Agents
    "BaseAgent",
    "ArchitectAgent",
    "CoderAgent",
    "ReviewerAgent",
    "ExecutorAgent",
    "NarratorAgent",
    
    # Providers
    "BaseLLMProvider",
    "LLMConfig",
    "LLMProvider",
    "create_llm_provider",
    "groq_provider",
    "openai_provider",
    "anthropic_provider",
    "ollama_provider",
]
