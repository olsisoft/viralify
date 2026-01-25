"""
Resonance Matcher for WeaveGraph

Propagates matching scores through the concept graph,
allowing related concepts to "resonate" and boost matching.

Algorithm:
1. Direct matches get resonance = 1.0
2. Neighbors get resonance = edge_weight × decay^depth
3. Propagation continues until max_depth or min_resonance threshold

Example:
    Query: "Kafka consumer"

    [consumer] (1.0) ──edge(0.85)──> [Kafka] (0.85)
                                        │
                                   edge(0.70)
                                        ↓
                               [message broker] (0.60)
"""

import asyncio
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .models import ConceptNode, ConceptEdge, WeaveGraph, RelationType


@dataclass
class ResonanceConfig:
    """Configuration for resonance propagation"""
    decay_factor: float = 0.7           # Score decay per hop
    max_depth: int = 3                  # Maximum propagation depth
    min_resonance: float = 0.10         # Minimum score to propagate
    boost_translation: float = 1.2      # Boost for cross-language edges
    boost_synonym: float = 1.1          # Boost for synonym edges
    max_resonating_concepts: int = 50   # Limit total resonating concepts


@dataclass
class ResonanceResult:
    """Result of resonance propagation"""
    # Concept ID -> resonance score (0-1)
    scores: Dict[str, float] = field(default_factory=dict)

    # Concept ID -> depth at which it was reached
    depths: Dict[str, int] = field(default_factory=dict)

    # Concept ID -> path from original match
    paths: Dict[str, List[str]] = field(default_factory=dict)

    # Statistics
    direct_matches: int = 0
    propagated_matches: int = 0
    total_resonance: float = 0.0
    max_depth_reached: int = 0

    def get_top_concepts(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N concepts by resonance score"""
        sorted_items = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]

    def get_concepts_above_threshold(self, threshold: float = 0.3) -> List[str]:
        """Get all concepts with resonance above threshold"""
        return [cid for cid, score in self.scores.items() if score >= threshold]


class ResonanceMatcher:
    """
    Propagates matching scores through the WeaveGraph.

    When a concept matches directly, its neighbors in the graph
    receive a "resonance" score based on edge weight and depth.
    This allows semantically related concepts to contribute to
    matching even if they weren't directly mentioned.
    """

    def __init__(self, config: Optional[ResonanceConfig] = None):
        self.config = config or ResonanceConfig()

    def propagate(
        self,
        matched_concept_ids: List[str],
        graph: WeaveGraph,
        initial_scores: Optional[Dict[str, float]] = None
    ) -> ResonanceResult:
        """
        Propagate resonance from matched concepts through the graph.

        Args:
            matched_concept_ids: IDs of directly matched concepts
            graph: The WeaveGraph to propagate through
            initial_scores: Optional custom initial scores (default: 1.0 for all)

        Returns:
            ResonanceResult with scores for all resonating concepts
        """
        result = ResonanceResult()

        if not matched_concept_ids:
            return result

        # Initialize scores for direct matches
        scores = initial_scores.copy() if initial_scores else {}
        for cid in matched_concept_ids:
            if cid not in scores:
                scores[cid] = 1.0
            result.depths[cid] = 0
            result.paths[cid] = [cid]

        result.direct_matches = len(matched_concept_ids)

        # Build adjacency map for efficient traversal
        adjacency = self._build_adjacency_map(graph)

        # BFS-style propagation with decay
        current_frontier = set(matched_concept_ids)
        visited = set(matched_concept_ids)

        for depth in range(1, self.config.max_depth + 1):
            next_frontier = set()

            for concept_id in current_frontier:
                current_score = scores.get(concept_id, 0)

                if current_score < self.config.min_resonance:
                    continue

                # Get neighbors
                neighbors = adjacency.get(concept_id, [])

                for neighbor_id, edge_weight, relation_type in neighbors:
                    if neighbor_id in visited:
                        # Update score if new path is better
                        new_score = self._compute_resonance(
                            current_score, edge_weight, depth, relation_type
                        )
                        if new_score > scores.get(neighbor_id, 0):
                            scores[neighbor_id] = new_score
                        continue

                    # Compute resonance score
                    resonance = self._compute_resonance(
                        current_score, edge_weight, depth, relation_type
                    )

                    if resonance >= self.config.min_resonance:
                        scores[neighbor_id] = resonance
                        result.depths[neighbor_id] = depth
                        result.paths[neighbor_id] = result.paths.get(concept_id, []) + [neighbor_id]
                        next_frontier.add(neighbor_id)
                        visited.add(neighbor_id)
                        result.propagated_matches += 1
                        result.max_depth_reached = max(result.max_depth_reached, depth)

            current_frontier = next_frontier

            # Check if we've reached the limit
            if len(scores) >= self.config.max_resonating_concepts:
                break

            # Stop if no more propagation possible
            if not next_frontier:
                break

        # Finalize result
        result.scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        result.total_resonance = sum(scores.values())

        return result

    def _compute_resonance(
        self,
        parent_score: float,
        edge_weight: float,
        depth: int,
        relation_type: RelationType
    ) -> float:
        """Compute resonance score for a neighbor concept"""
        # Base resonance: parent score × edge weight × decay
        resonance = parent_score * edge_weight * (self.config.decay_factor ** depth)

        # Apply relation type boosts
        if relation_type == RelationType.TRANSLATION:
            resonance *= self.config.boost_translation
        elif relation_type == RelationType.SYNONYM:
            resonance *= self.config.boost_synonym

        # Clamp to [0, 1]
        return min(1.0, max(0.0, resonance))

    def _build_adjacency_map(
        self,
        graph: WeaveGraph
    ) -> Dict[str, List[Tuple[str, float, RelationType]]]:
        """Build adjacency map from graph edges"""
        adjacency = defaultdict(list)

        for edge in graph.edges:
            # Add forward edge
            adjacency[edge.source_id].append(
                (edge.target_id, edge.weight, edge.relation_type)
            )

            # Add reverse edge if bidirectional
            if edge.bidirectional:
                adjacency[edge.target_id].append(
                    (edge.source_id, edge.weight, edge.relation_type)
                )

        return dict(adjacency)

    def match_with_resonance(
        self,
        query_terms: List[str],
        source_terms: List[str],
        graph: WeaveGraph,
        embedding_engine=None
    ) -> Tuple[ResonanceResult, float]:
        """
        Match query terms against source terms using resonance propagation.

        Args:
            query_terms: Terms from generated content
            source_terms: Terms from source documents
            graph: The WeaveGraph
            embedding_engine: Optional embedding engine for similarity matching

        Returns:
            Tuple of (ResonanceResult, coverage_boost)
        """
        # Step 1: Find direct concept matches in graph
        matched_ids = []
        matched_scores = {}

        source_set = set(t.lower() for t in source_terms)

        for term in query_terms:
            term_lower = term.lower()

            # Try to find concept in graph
            concept = graph.find_concept_by_name(term)

            if concept:
                # Check if concept or its aliases appear in source
                if concept.canonical_name in source_set or term_lower in source_set:
                    matched_ids.append(concept.id)
                    matched_scores[concept.id] = 1.0
                elif any(alias.lower() in source_set for alias in concept.aliases):
                    matched_ids.append(concept.id)
                    matched_scores[concept.id] = 0.9  # Slightly lower for alias match

        if not matched_ids:
            return ResonanceResult(), 0.0

        # Step 2: Propagate resonance
        result = self.propagate(matched_ids, graph, matched_scores)

        # Step 3: Calculate coverage boost
        # Boost is based on how many additional concepts resonate
        if result.propagated_matches > 0:
            # More propagation = more semantic coverage discovered
            propagation_ratio = min(1.0, result.propagated_matches / 20)  # Cap at 20
            avg_resonance = result.total_resonance / (result.direct_matches + result.propagated_matches)

            # Boost formula: up to 15% based on propagation quality
            coverage_boost = min(0.15, propagation_ratio * avg_resonance * 0.20)
        else:
            coverage_boost = 0.0

        return result, coverage_boost

    async def match_with_resonance_async(
        self,
        query_terms: List[str],
        source_terms: List[str],
        graph: WeaveGraph,
        embedding_engine=None
    ) -> Tuple[ResonanceResult, float]:
        """Async version of match_with_resonance"""
        # The actual computation is CPU-bound, so we just wrap it
        return self.match_with_resonance(
            query_terms, source_terms, graph, embedding_engine
        )


class ResonanceVerifier:
    """
    High-level interface for using resonance matching in RAG verification.

    Combines WeaveGraph query expansion with resonance propagation
    for comprehensive semantic matching.
    """

    def __init__(
        self,
        matcher: Optional[ResonanceMatcher] = None,
        config: Optional[ResonanceConfig] = None
    ):
        self.matcher = matcher or ResonanceMatcher(config)

    async def verify_with_resonance(
        self,
        generated_concepts: List[str],
        source_concepts: List[str],
        graph: WeaveGraph,
        base_coverage: float = 0.0
    ) -> Dict:
        """
        Verify generated content against source using resonance matching.

        Args:
            generated_concepts: Concepts extracted from generated content
            source_concepts: Concepts from source documents
            graph: The WeaveGraph
            base_coverage: Base coverage score from other methods

        Returns:
            Dict with resonance verification results
        """
        result, boost = await self.matcher.match_with_resonance_async(
            generated_concepts,
            source_concepts,
            graph
        )

        # Compute final coverage with resonance boost
        final_coverage = min(1.0, base_coverage + boost)

        # Get top resonating concepts for explanation
        top_concepts = result.get_top_concepts(10)

        return {
            "base_coverage": base_coverage,
            "resonance_boost": boost,
            "final_coverage": final_coverage,
            "direct_matches": result.direct_matches,
            "propagated_matches": result.propagated_matches,
            "max_depth_reached": result.max_depth_reached,
            "total_resonance": result.total_resonance,
            "top_resonating_concepts": [
                {"id": cid, "score": score, "depth": result.depths.get(cid, 0)}
                for cid, score in top_concepts
            ]
        }


# Convenience functions
def create_resonance_matcher(
    decay: float = 0.7,
    max_depth: int = 3,
    min_resonance: float = 0.10
) -> ResonanceMatcher:
    """Create a configured ResonanceMatcher"""
    config = ResonanceConfig(
        decay_factor=decay,
        max_depth=max_depth,
        min_resonance=min_resonance
    )
    return ResonanceMatcher(config)


def propagate_resonance(
    matched_ids: List[str],
    graph: WeaveGraph,
    decay: float = 0.7,
    max_depth: int = 3
) -> Dict[str, float]:
    """
    Simple function to propagate resonance through a graph.

    Returns dict of concept_id -> resonance_score
    """
    matcher = create_resonance_matcher(decay, max_depth)
    result = matcher.propagate(matched_ids, graph)
    return result.scores
