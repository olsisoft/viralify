"""
RAG Enforcement Module

Enforces RAG compliance through:
- Citation validation (inline [REF:X] citations)
- Sentence-level verification (each sentence grounded in sources)
- Retry loop with increasing strictness

Usage:
    from services.rag_enforcement import RAGEnforcer, EnforcementConfig

    enforcer = RAGEnforcer()
    result = await enforcer.enforce(
        generator_func=my_generator,
        sources=source_chunks,
        topic="Apache Kafka"
    )

    if result.is_compliant:
        print("Content is RAG-compliant!")
    else:
        print(f"Compliance: {result.overall_score:.0%}")
"""

from .models import (
    # Enums
    FactStatus,
    ComplianceLevel,

    # Data classes
    Citation,
    SentenceScore,
    CitationReport,
    SentenceReport,
    EnforcementResult,
    EnforcementConfig,
)

from .citation_validator import CitationValidator
from .sentence_verifier import SentenceVerifier, AsyncSentenceVerifier
from .rag_enforcer import (
    RAGEnforcer,
    AsyncRAGEnforcer,
    RAGComplianceError,
    create_enforcer,
    verify_content,
)

__all__ = [
    # Enums
    "FactStatus",
    "ComplianceLevel",

    # Models
    "Citation",
    "SentenceScore",
    "CitationReport",
    "SentenceReport",
    "EnforcementResult",
    "EnforcementConfig",

    # Validators
    "CitationValidator",
    "SentenceVerifier",
    "AsyncSentenceVerifier",

    # Enforcers
    "RAGEnforcer",
    "AsyncRAGEnforcer",
    "RAGComplianceError",

    # Convenience functions
    "create_enforcer",
    "verify_content",
]
