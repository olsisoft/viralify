"""
Graph Builder for WeaveGraph

Builds edges between concepts based on:
- Embedding similarity (E5-large)
- Co-occurrence in documents
- Cross-language translation detection
"""

import os
import asyncio
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np

from .models import (
    ConceptNode, ConceptEdge, WeaveGraph, QueryExpansion,
    RelationType, ConceptSource, WeaveGraphStats
)
from .concept_extractor import ConceptExtractor, ExtractionConfig
from .pgvector_store import WeaveGraphPgVectorStore, get_weave_graph_store

# Import embedding engine from sync module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sync.embedding_engine import EmbeddingEngineFactory, EmbeddingEngineBase


@dataclass
class GraphBuilderConfig:
    """Configuration for graph building"""
    similarity_threshold: float = 0.70      # Minimum similarity for edge
    translation_threshold: float = 0.80     # Higher threshold for cross-language
    max_edges_per_concept: int = 10         # Limit connections
    min_concept_frequency: int = 1          # Minimum occurrences
    batch_size: int = 50                    # Embedding batch size
    embedding_backend: str = "e5-large"     # E5-large for multilingual


class WeaveGraphBuilder:
    """
    Builds and maintains the WeaveGraph.

    Orchestrates:
    1. Concept extraction from documents
    2. Embedding generation with E5-large
    3. Edge creation based on similarity
    4. Graph storage in pgvector
    """

    def __init__(
        self,
        config: Optional[GraphBuilderConfig] = None,
        store: Optional[WeaveGraphPgVectorStore] = None
    ):
        self.config = config or GraphBuilderConfig()
        self.store = store or get_weave_graph_store()
        self.extractor = ConceptExtractor()
        self._embedding_engine: Optional[EmbeddingEngineBase] = None

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

        # Step 1: Extract concepts from all documents
        all_concepts: Dict[str, ConceptNode] = {}

        for doc in documents:
            doc_id = doc.get('id', 'unknown')
            content = doc.get('content', '')

            if not content:
                continue

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

        # Step 3: Store concepts
        concept_ids = {}
        for concept in concepts_list:
            if concept.embedding:
                cid = await self.store.store_concept(concept, user_id)
                concept_ids[concept.canonical_name] = cid
                concept.id = cid

        print(f"[WEAVE_GRAPH] Stored {len(concept_ids)} concepts with embeddings", flush=True)

        # Step 4: Build edges based on similarity
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
        """Build edges based on embedding similarity"""
        edges = []
        engine = self._get_embedding_engine()

        # Filter concepts with embeddings
        embedded_concepts = [c for c in concepts if c.embedding]

        if len(embedded_concepts) < 2:
            return edges

        print(f"[WEAVE_GRAPH] Computing similarities for {len(embedded_concepts)} concepts...", flush=True)

        # Compute pairwise similarities
        # For efficiency, we'll only compare each concept with its nearest neighbors

        for i, concept_a in enumerate(embedded_concepts):
            if not concept_a.embedding:
                continue

            edge_count = 0
            candidates = []

            for j, concept_b in enumerate(embedded_concepts):
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
        """
        # Extract concepts
        concepts = self.extractor.extract_concepts(content, document_id=document_id)

        if not concepts:
            return 0

        # Generate embeddings
        await self._generate_embeddings(concepts)

        # Store concepts
        new_count = 0
        for concept in concepts:
            if concept.embedding:
                cid = await self.store.store_concept(concept, user_id)
                concept.id = cid
                new_count += 1

        # Build edges with existing concepts
        # Get existing concepts to compare with
        existing = []
        for concept in concepts:
            if concept.embedding:
                similar = await self.store.find_similar_concepts(
                    concept.embedding, user_id,
                    limit=self.config.max_edges_per_concept,
                    threshold=self.config.similarity_threshold
                )
                for other_concept, similarity in similar:
                    if other_concept.id != concept.id:
                        edge = ConceptEdge(
                            source_id=concept.id,
                            target_id=other_concept.id,
                            relation_type=RelationType.SIMILAR,
                            weight=similarity,
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

        # Also embed the full query for similarity search
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
    builder = WeaveGraphBuilder()
    return await builder.build_from_documents(documents, user_id, rebuild)
