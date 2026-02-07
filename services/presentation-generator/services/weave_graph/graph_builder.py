"""
Graph Builder for WeaveGraph

Builds edges between concepts based on:
- Embedding similarity (E5-large)
- Co-occurrence in documents (PMI)
- Hierarchy (TECH_HIERARCHY)
- Cross-language translation detection

Supports multi-factor edge weighting combining all signals.
"""

import os
import asyncio
from typing import List, Dict, Optional, Tuple, Set, ClassVar
from dataclasses import dataclass, field
import numpy as np

from .models import (
    ConceptNode, ConceptEdge, WeaveGraph, QueryExpansion,
    RelationType, ConceptSource, WeaveGraphStats
)
from .concept_extractor import ConceptExtractor, ExtractionConfig
from .pgvector_store import WeaveGraphPgVectorStore, get_weave_graph_store
from .edge_weight_calculator import (
    EdgeWeightCalculator,
    EdgeWeightConfig,
    EdgeWeight,
    CooccurrenceCalculator,
    HierarchyResolver
)

# Import embedding engine from sync module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sync.embedding_engine import EmbeddingEngineFactory, EmbeddingEngineBase


@dataclass
class GraphBuilderConfig:
    """Configuration for graph building"""
    # Edge thresholds (used when multi-factor is disabled)
    similarity_threshold: float = 0.70      # Minimum similarity for edge
    translation_threshold: float = 0.80     # Higher threshold for cross-language

    # Graph limits
    max_edges_per_concept: int = 10         # Limit connections
    min_concept_frequency: int = 1          # Minimum occurrences
    batch_size: int = 50                    # Embedding batch size

    # Embedding
    embedding_backend: str = "e5-large"     # E5-large for multilingual

    # Multi-factor edge weighting (NEW)
    use_multi_factor_weights: bool = True   # Enable co-occurrence + hierarchy + embedding
    cooccurrence_weight: float = 0.4        # Weight for PMI co-occurrence
    hierarchy_weight: float = 0.3           # Weight for TECH_HIERARCHY
    embedding_weight: float = 0.3           # Weight for embedding similarity
    min_edge_weight: float = 0.15           # Minimum combined weight for edge
    cooccurrence_window_size: int = 1       # 1 = same chunk, 2 = adjacent chunks


class WeaveGraphBuilder:
    """
    Builds and maintains the WeaveGraph.

    Orchestrates:
    1. Concept extraction from documents
    2. Embedding generation with E5-large
    3. Edge creation based on similarity
    4. Graph storage in pgvector

    Uses singleton pattern to avoid:
    - Multiple embedding engine loads (E5-Large is 2GB)
    - Database connection pool exhaustion
    - Redundant document processing
    """

    # Singleton instance
    _instance: Optional['WeaveGraphBuilder'] = None
    _initialized: bool = False

    # Track processed documents to avoid re-processing
    _processed_documents: Set[str] = set()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - return existing instance if available"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config: Optional[GraphBuilderConfig] = None,
        store: Optional[WeaveGraphPgVectorStore] = None
    ):
        # Skip re-initialization for singleton
        if WeaveGraphBuilder._initialized:
            return

        self.config = config or GraphBuilderConfig()
        self.store = store or get_weave_graph_store()
        self.extractor = ConceptExtractor()
        self._embedding_engine: Optional[EmbeddingEngineBase] = None

        # Initialize EdgeWeightCalculator for multi-factor weighting
        self._edge_weight_calculator: Optional[EdgeWeightCalculator] = None
        if self.config.use_multi_factor_weights:
            edge_config = EdgeWeightConfig(
                cooccurrence_weight=self.config.cooccurrence_weight,
                hierarchy_weight=self.config.hierarchy_weight,
                embedding_weight=self.config.embedding_weight,
                window_size=self.config.cooccurrence_window_size,
                min_edge_weight=self.config.min_edge_weight,
            )
            self._edge_weight_calculator = EdgeWeightCalculator(edge_config)
            print("[WEAVE_GRAPH] Multi-factor edge weighting enabled", flush=True)

        WeaveGraphBuilder._initialized = True
        print("[WEAVE_GRAPH] Builder initialized (singleton)", flush=True)

    def _get_embedding_engine(self) -> EmbeddingEngineBase:
        """Lazy-load embedding engine"""
        if self._embedding_engine is None:
            # Use unified env var, fallback to legacy, then config
            backend = (
                os.getenv("EMBEDDING_BACKEND") or
                os.getenv("RAG_EMBEDDING_BACKEND") or
                self.config.embedding_backend
            )
            self._embedding_engine = EmbeddingEngineFactory.create(backend)
            print(f"[WEAVE_GRAPH] Using embedding engine: {self._embedding_engine.name}", flush=True)
        return self._embedding_engine

    async def build_from_documents(
        self,
        documents: List[Dict],
        user_id: str,
        rebuild: bool = False
    ) -> WeaveGraphStats:
        """
        Build WeaveGraph from a list of documents.

        Args:
            documents: List of dicts with 'id' and 'content' keys
            user_id: User ID for graph ownership
            rebuild: If True, delete existing graph first

        Returns:
            Graph statistics
        """
        print(f"[WEAVE_GRAPH] Building graph from {len(documents)} documents", flush=True)

        if rebuild:
            await self.store.delete_user_graph(user_id)

        # Step 1: Extract concepts from all documents and collect chunks
        all_concepts: Dict[str, ConceptNode] = {}
        all_chunks: List[str] = []  # Collect chunks for co-occurrence training

        for doc in documents:
            doc_id = doc.get('id', 'unknown')
            content = doc.get('content', '')

            if not content:
                continue

            # Collect content chunks for co-occurrence training
            all_chunks.append(content)

            concepts = self.extractor.extract_concepts(
                content,
                document_id=doc_id,
                existing_concepts=all_concepts
            )

            for concept in concepts:
                if concept.canonical_name in all_concepts:
                    # Merge
                    existing = all_concepts[concept.canonical_name]
                    existing.frequency += concept.frequency
                    existing.source_document_ids.extend(concept.source_document_ids)
                else:
                    all_concepts[concept.canonical_name] = concept

        print(f"[WEAVE_GRAPH] Extracted {len(all_concepts)} unique concepts", flush=True)

        if not all_concepts:
            return WeaveGraphStats()

        # Step 2: Generate embeddings
        concepts_list = list(all_concepts.values())
        await self._generate_embeddings(concepts_list)

        # Step 2.5: Train co-occurrence calculator if multi-factor enabled
        if self._edge_weight_calculator and all_chunks:
            print(f"[WEAVE_GRAPH] Training co-occurrence on {len(all_chunks)} chunks...", flush=True)
            self._edge_weight_calculator.train_cooccurrence(all_chunks, self.extractor)

            # Set embeddings for the calculator
            embeddings_dict = {
                c.canonical_name: c.embedding
                for c in concepts_list
                if c.embedding
            }
            self._edge_weight_calculator.set_embeddings(embeddings_dict)
            print(f"[WEAVE_GRAPH] Co-occurrence trained, {len(embeddings_dict)} embeddings set", flush=True)

        # Step 3: Store concepts
        concept_ids = {}
        for concept in concepts_list:
            if concept.embedding:
                cid = await self.store.store_concept(concept, user_id)
                concept_ids[concept.canonical_name] = cid
                concept.id = cid

        print(f"[WEAVE_GRAPH] Stored {len(concept_ids)} concepts with embeddings", flush=True)

        # Step 4: Build edges based on similarity (or multi-factor weights)
        edges = await self._build_similarity_edges(concepts_list, user_id)

        # Step 5: Store edges
        edge_count = await self.store.store_edges_batch(edges, user_id)
        print(f"[WEAVE_GRAPH] Created {edge_count} edges", flush=True)

        # Step 6: Get stats
        stats = await self.store.get_user_graph_stats(user_id)

        return WeaveGraphStats(
            total_concepts=stats['total_concepts'],
            total_edges=stats['total_edges'],
            languages=stats['languages'],
            top_concepts=[c[0] for c in stats['top_concepts']]
        )

    async def _generate_embeddings(self, concepts: List[ConceptNode]) -> None:
        """Generate embeddings for concepts in batches"""
        engine = self._get_embedding_engine()

        # Prepare texts for embedding
        texts = [c.name for c in concepts]

        # Batch embedding
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_concepts = concepts[i:i + self.config.batch_size]

            try:
                embeddings = engine.embed_batch(batch)

                for concept, embedding in zip(batch_concepts, embeddings):
                    concept.embedding = embedding.tolist()

            except Exception as e:
                print(f"[WEAVE_GRAPH] Embedding batch failed: {e}", flush=True)

        embedded_count = sum(1 for c in concepts if c.embedding)
        print(f"[WEAVE_GRAPH] Generated {embedded_count}/{len(concepts)} embeddings", flush=True)

    async def _build_similarity_edges(
        self,
        concepts: List[ConceptNode],
        user_id: str
    ) -> List[ConceptEdge]:
        """Build edges based on embedding similarity or multi-factor weights"""
        edges = []

        # Filter concepts with embeddings
        embedded_concepts = [c for c in concepts if c.embedding]

        if len(embedded_concepts) < 2:
            return edges

        # Use multi-factor edge weighting if enabled and calculator is ready
        if self._edge_weight_calculator:
            return await self._build_multi_factor_edges(embedded_concepts, user_id)

        # Fallback: Use simple embedding similarity
        return await self._build_embedding_only_edges(embedded_concepts, user_id)

    async def _build_multi_factor_edges(
        self,
        concepts: List[ConceptNode],
        user_id: str
    ) -> List[ConceptEdge]:
        """Build edges using multi-factor weights (co-occurrence + hierarchy + embedding)"""
        edges = []

        print(f"[WEAVE_GRAPH] Building multi-factor edges for {len(concepts)} concepts...", flush=True)

        # Get concept names for the calculator
        concept_names = [c.canonical_name for c in concepts]

        # Build weighted edges using EdgeWeightCalculator
        weighted_edges = self._edge_weight_calculator.build_weighted_edges(
            concept_names,
            max_edges_per_concept=self.config.max_edges_per_concept
        )

        # Create a mapping from canonical_name to ConceptNode for quick lookup
        concept_map = {c.canonical_name: c for c in concepts}

        for edge_weight in weighted_edges:
            concept_a = concept_map.get(edge_weight.concept_a)
            concept_b = concept_map.get(edge_weight.concept_b)

            if not concept_a or not concept_b:
                continue

            if not concept_a.id or not concept_b.id:
                continue

            # Determine relationship type based on factors
            is_cross_language = concept_a.language != concept_b.language
            if is_cross_language:
                rel_type = RelationType.TRANSLATION
            elif edge_weight.hierarchy_score > 0.5:
                rel_type = RelationType.PART_OF
            else:
                rel_type = RelationType.SIMILAR

            edge = ConceptEdge(
                source_id=concept_a.id,
                target_id=concept_b.id,
                relation_type=rel_type,
                weight=edge_weight.combined_weight,
                bidirectional=True
            )
            edges.append(edge)

        # Log weight distribution
        if edges:
            weights = [e.weight for e in edges]
            avg_weight = sum(weights) / len(weights)
            print(f"[WEAVE_GRAPH] Multi-factor edges: {len(edges)}, avg weight: {avg_weight:.3f}", flush=True)

        return edges

    async def _build_embedding_only_edges(
        self,
        concepts: List[ConceptNode],
        user_id: str
    ) -> List[ConceptEdge]:
        """Build edges based on embedding similarity only (fallback mode)"""
        edges = []
        engine = self._get_embedding_engine()

        print(f"[WEAVE_GRAPH] Computing similarities for {len(concepts)} concepts...", flush=True)

        # Compute pairwise similarities
        # For efficiency, we'll only compare each concept with its nearest neighbors

        for i, concept_a in enumerate(concepts):
            if not concept_a.embedding:
                continue

            candidates = []

            for j, concept_b in enumerate(concepts):
                if i >= j or not concept_b.embedding:  # Avoid duplicates
                    continue

                # Compute cosine similarity
                similarity = engine.similarity(
                    np.array(concept_a.embedding),
                    np.array(concept_b.embedding)
                )

                # Determine relationship type
                is_cross_language = concept_a.language != concept_b.language
                threshold = self.config.translation_threshold if is_cross_language else self.config.similarity_threshold

                if similarity >= threshold:
                    rel_type = RelationType.TRANSLATION if is_cross_language else RelationType.SIMILAR

                    candidates.append((concept_b, similarity, rel_type))

            # Sort by similarity and take top N
            candidates.sort(key=lambda x: x[1], reverse=True)

            for concept_b, similarity, rel_type in candidates[:self.config.max_edges_per_concept]:
                edge = ConceptEdge(
                    source_id=concept_a.id,
                    target_id=concept_b.id,
                    relation_type=rel_type,
                    weight=similarity,
                    bidirectional=True
                )
                edges.append(edge)

        print(f"[WEAVE_GRAPH] Built {len(edges)} similarity edges", flush=True)
        return edges

    async def add_document(
        self,
        document_id: str,
        content: str,
        user_id: str
    ) -> int:
        """
        Add a single document to an existing graph.

        Returns number of new concepts added.
        Skips processing if document was already processed in this session.
        """
        # Create a cache key based on document_id and content hash
        content_hash = hash(content[:1000]) if content else 0  # Hash first 1000 chars
        cache_key = f"{document_id}_{user_id}_{content_hash}"

        # Check if already processed (avoid re-generating 500 embeddings)
        if cache_key in WeaveGraphBuilder._processed_documents:
            print(f"[WEAVE_GRAPH] ✓ Skipping already processed document: {document_id[:20]}...", flush=True)
            return 0

        # Mark as being processed
        WeaveGraphBuilder._processed_documents.add(cache_key)
        print(f"[WEAVE_GRAPH] Processing new document: {document_id[:20]}...", flush=True)

        # Extract concepts
        concepts = self.extractor.extract_concepts(content, document_id=document_id)

        if not concepts:
            return 0

        # Generate embeddings
        await self._generate_embeddings(concepts)

        # Update EdgeWeightCalculator if multi-factor is enabled
        if self._edge_weight_calculator:
            # Train with new document chunk
            self._edge_weight_calculator.train_cooccurrence([content], self.extractor)

            # Update embeddings
            new_embeddings = {
                c.canonical_name: c.embedding
                for c in concepts
                if c.embedding
            }
            self._edge_weight_calculator.set_embeddings(new_embeddings)

        # Store concepts
        new_count = 0
        for concept in concepts:
            if concept.embedding:
                cid = await self.store.store_concept(concept, user_id)
                concept.id = cid
                new_count += 1

        # Build edges with existing concepts
        for concept in concepts:
            if not concept.embedding or not concept.id:
                continue

            # Find similar concepts in the store
            similar = await self.store.find_similar_concepts(
                concept.embedding, user_id,
                limit=self.config.max_edges_per_concept,
                threshold=self.config.similarity_threshold
            )

            for other_concept, similarity in similar:
                if other_concept.id == concept.id:
                    continue

                # Calculate edge weight
                if self._edge_weight_calculator:
                    # Use multi-factor weight
                    edge_weight = self._edge_weight_calculator.calculate_weight(
                        concept.canonical_name,
                        other_concept.canonical_name
                    )
                    weight = edge_weight.combined_weight

                    # Skip if below threshold
                    if weight < self.config.min_edge_weight:
                        continue

                    # Determine relation type
                    if edge_weight.hierarchy_score > 0.5:
                        rel_type = RelationType.PART_OF
                    else:
                        rel_type = RelationType.SIMILAR
                else:
                    # Use embedding similarity only
                    weight = similarity
                    rel_type = RelationType.SIMILAR

                edge = ConceptEdge(
                    source_id=concept.id,
                    target_id=other_concept.id,
                    relation_type=rel_type,
                    weight=weight,
                    bidirectional=True
                )
                await self.store.store_edge(edge, user_id)

        return new_count

    async def expand_query(
        self,
        query: str,
        user_id: str,
        max_expansions: int = 10
    ) -> QueryExpansion:
        """
        Expand a query using the graph.

        Given a query, find related concepts and return
        expanded terms for better retrieval.
        """
        print(f"[WEAVE_GRAPH] expand_query: starting...", flush=True)
        engine = self._get_embedding_engine()

        # Extract terms from query
        print(f"[WEAVE_GRAPH] expand_query: extracting concepts...", flush=True)
        query_concepts = self.extractor.extract_concepts(query)
        query_terms = [c.name for c in query_concepts] if query_concepts else [query]
        print(f"[WEAVE_GRAPH] expand_query: extracted {len(query_terms)} terms", flush=True)

        # Embed the full query for similarity search
        # (Safe now - running in main async loop, not a thread)
        print(f"[WEAVE_GRAPH] expand_query: embedding query...", flush=True)
        query_embedding = engine.embed(query).tolist()
        print(f"[WEAVE_GRAPH] expand_query: embedding done", flush=True)

        # Find similar concepts by embedding
        print(f"[WEAVE_GRAPH] expand_query: finding similar concepts in DB...", flush=True)
        similar_concepts = await self.store.find_similar_concepts(
            query_embedding, user_id,
            limit=max_expansions,
            threshold=0.5
        )
        print(f"[WEAVE_GRAPH] expand_query: found {len(similar_concepts)} similar concepts", flush=True)

        # Expand using graph edges
        print(f"[WEAVE_GRAPH] expand_query: expanding via graph edges...", flush=True)
        term_expansions = await self.store.expand_query(query_terms, user_id, max_expansions)
        print(f"[WEAVE_GRAPH] expand_query: expansion done", flush=True)

        # Combine results
        expanded_terms = set()
        expansion_paths = {}
        languages = set()

        # Add terms from query
        for term in query_terms:
            expanded_terms.add(term)

        # Add similar concepts
        for concept, similarity in similar_concepts:
            expanded_terms.add(concept.name)
            languages.add(concept.language)

        # Add graph expansions
        for original, expansions in term_expansions.items():
            expansion_paths[original] = expansions
            for exp in expansions:
                expanded_terms.add(exp)

        return QueryExpansion(
            original_query=query,
            expanded_terms=list(expanded_terms),
            expansion_paths=expansion_paths,
            total_weight=sum(s for _, s in similar_concepts),
            languages_covered=languages
        )


# Singleton getter
def get_weave_graph_builder() -> WeaveGraphBuilder:
    """
    Get the singleton WeaveGraphBuilder instance.

    This ensures:
    - Single E5-Large model load (2GB)
    - Shared embedding cache
    - No duplicate document processing
    """
    return WeaveGraphBuilder()


def clear_processed_documents_cache():
    """
    Clear the processed documents cache.

    Call this when you want to force re-processing of documents,
    e.g., after a major update to the documents.
    """
    count = len(WeaveGraphBuilder._processed_documents)
    WeaveGraphBuilder._processed_documents.clear()
    print(f"[WEAVE_GRAPH] Cleared processed documents cache ({count} entries)", flush=True)


# Convenience function
async def build_weave_graph(
    documents: List[Dict],
    user_id: str,
    rebuild: bool = False
) -> WeaveGraphStats:
    """
    Build a WeaveGraph from documents.

    Convenience function for common use case.
    """
    builder = get_weave_graph_builder()
    return await builder.build_from_documents(documents, user_id, rebuild)
