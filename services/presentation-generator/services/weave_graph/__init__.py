"""
WeaveGraph - Semantic Concept Graph for RAG Enhancement

A graph-based system that discovers relationships between concepts
extracted from documents, enabling smarter query expansion and
cross-language matching.

Usage:
    from services.weave_graph import (
        WeaveGraphBuilder,
        build_weave_graph,
        get_weave_graph_store
    )

    # Build graph from documents
    documents = [
        {"id": "doc1", "content": "Apache Kafka is a message broker..."},
        {"id": "doc2", "content": "Les consumers Kafka lisent les messages..."}
    ]

    stats = await build_weave_graph(documents, user_id="user123")

    # Query expansion
    builder = WeaveGraphBuilder()
    expansion = await builder.expand_query("message queue", user_id="user123")
    print(expansion.expanded_terms)
    # ['message queue', 'kafka', 'consumer', 'broker', 'file d'attente']
"""

from .models import (
    ConceptNode,
    ConceptEdge,
    WeaveGraph,
    WeaveGraphStats,
    ConceptCluster,
    QueryExpansion,
    RelationType,
    ConceptSource
)

from .concept_extractor import (
    ConceptExtractor,
    ExtractionConfig
)

from .pgvector_store import (
    WeaveGraphPgVectorStore,
    get_weave_graph_store
)

from .graph_builder import (
    WeaveGraphBuilder,
    GraphBuilderConfig,
    build_weave_graph,
    get_weave_graph_builder,
    clear_processed_documents_cache
)

from .resonance_matcher import (
    ResonanceMatcher,
    ResonanceConfig,
    ResonanceResult,
    ResonanceVerifier,
    create_resonance_matcher,
    propagate_resonance
)

from .compound_detector import (
    CompoundTermDetector,
    CompoundTermResult,
    CompoundDetectorConfig,
    PMICalculator,
    PMIConfig,
    SemanticFilter,
    detect_compound_terms
)


__all__ = [
    # Models
    'ConceptNode',
    'ConceptEdge',
    'WeaveGraph',
    'WeaveGraphStats',
    'ConceptCluster',
    'QueryExpansion',
    'RelationType',
    'ConceptSource',

    # Extractor
    'ConceptExtractor',
    'ExtractionConfig',

    # Store
    'WeaveGraphPgVectorStore',
    'get_weave_graph_store',

    # Builder
    'WeaveGraphBuilder',
    'GraphBuilderConfig',
    'build_weave_graph',
    'get_weave_graph_builder',
    'clear_processed_documents_cache',

    # Resonance Matcher (Phase 3)
    'ResonanceMatcher',
    'ResonanceConfig',
    'ResonanceResult',
    'ResonanceVerifier',
    'create_resonance_matcher',
    'propagate_resonance',

    # Compound Term Detector (ML-based)
    'CompoundTermDetector',
    'CompoundTermResult',
    'CompoundDetectorConfig',
    'PMICalculator',
    'PMIConfig',
    'SemanticFilter',
    'detect_compound_terms',
]
