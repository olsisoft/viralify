"""
RAG (Retrieval-Augmented Generation) Module

This module provides document processing, storage, and retrieval capabilities
for course generation. It implements weighted multi-source RAG with:
- Document parsing and security scanning
- Vector search with pgvector
- Cross-encoder re-ranking
- Token-aware context building

Usage:
    from services.rag import RAGService, WeightedMultiSourceRAG

    rag_service = RAGService()
    context = await rag_service.get_context_for_course_generation(
        topic="Apache Kafka",
        document_ids=["doc_123"],
        user_id="user_456"
    )
"""

# Models
from .models import (
    DocumentRelevanceScore,
    WeightedRAGResult,
)

# Algorithms
from .algorithms import (
    WeightedMultiSourceRAG,
    KeywordExtractor,
    TokenAllocator,
)

# Storage
from .storage import (
    DocumentRepositoryPg,
    RAGDocumentStorage,
)

# Processors
from .processors import (
    StructureExtractor,
    AIStructureGenerator,
)

# Retrieval
from .retrieval import (
    ContextBuilder,
    ChunkPrioritizer,
    ImageRetriever,
)

# Services
from .services import RAGService

# Prompts
from .prompts import (
    BasePromptBuilder,
    DocumentSummaryPromptBuilder,
    StructureExtractionPromptBuilder,
)

__all__ = [
    # Models
    "DocumentRelevanceScore",
    "WeightedRAGResult",
    # Algorithms
    "WeightedMultiSourceRAG",
    "KeywordExtractor",
    "TokenAllocator",
    # Storage
    "DocumentRepositoryPg",
    "RAGDocumentStorage",
    # Processors
    "StructureExtractor",
    "AIStructureGenerator",
    # Retrieval
    "ContextBuilder",
    "ChunkPrioritizer",
    "ImageRetriever",
    # Services
    "RAGService",
    # Prompts
    "BasePromptBuilder",
    "DocumentSummaryPromptBuilder",
    "StructureExtractionPromptBuilder",
]
