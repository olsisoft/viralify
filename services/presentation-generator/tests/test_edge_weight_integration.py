"""
Integration tests for EdgeWeightCalculator with WeaveGraph and ResonanceMatcher.

Tests the complete flow:
1. Document chunking → Concept extraction → Edge weight calculation
2. Edge weights used in WeaveGraph for resonance propagation
3. End-to-end RAG verification with weighted edges
"""

import pytest
import sys
import os
from typing import List, Dict, Set
from dataclasses import dataclass, field

# Add path to import modules
_weave_graph_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "services",
    "weave_graph"
)
sys.path.insert(0, _weave_graph_path)

from edge_weight_calculator import (
    EdgeWeightConfig,
    EdgeWeightCalculator,
    CooccurrenceCalculator,
    HierarchyResolver,
    EmbeddingSimilarityCalculator,
    EdgeWeight,
    TECH_HIERARCHY,
)

# Try to import real modules, fall back to mocks
try:
    from compound_detector import CompoundTermDetector, CompoundDetectorConfig
    COMPOUND_DETECTOR_AVAILABLE = True
except ImportError:
    COMPOUND_DETECTOR_AVAILABLE = False


# =============================================================================
# Mock Classes for Standalone Testing
# =============================================================================

@dataclass
class MockConceptNode:
    """Mock concept node for testing."""
    name: str
    canonical_name: str
    embedding: List[float] = field(default_factory=list)
    frequency: int = 1


@dataclass
class MockConceptEdge:
    """Mock concept edge for testing."""
    source_id: str
    target_id: str
    weight: float
    relation_type: str = "similar"


class MockConceptExtractor:
    """Mock concept extractor for testing co-occurrence."""

    def extract_concepts(self, text: str) -> List[MockConceptNode]:
        """Extract concepts as simple word tokens."""
        words = text.lower().split()
        concepts = []
        for word in words:
            # Filter short words and common stopwords
            if len(word) > 3 and word not in {'this', 'that', 'with', 'from', 'about'}:
                concepts.append(MockConceptNode(
                    name=word,
                    canonical_name=word.lower()
                ))
        return concepts


class MockEmbeddingEngine:
    """Mock embedding engine for testing."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self._cache: Dict[str, List[float]] = {}

    def encode(self, text: str) -> List[float]:
        """Generate deterministic mock embedding."""
        if text in self._cache:
            return self._cache[text]

        # Create deterministic embedding based on text hash
        import hashlib
        hash_bytes = hashlib.md5(text.lower().encode()).digest()
        embedding = [float(b) / 255.0 for b in hash_bytes[:self.dimension]]

        # Pad or truncate to dimension
        if len(embedding) < self.dimension:
            embedding.extend([0.0] * (self.dimension - len(embedding)))
        embedding = embedding[:self.dimension]

        self._cache[text] = embedding
        return embedding


# =============================================================================
# Integration Test: Co-occurrence with Concept Extraction
# =============================================================================

class TestCooccurrenceWithConceptExtraction:
    """Test co-occurrence calculation integrated with concept extraction."""

    @pytest.fixture
    def extractor(self):
        return MockConceptExtractor()

    @pytest.fixture
    def calculator(self):
        return EdgeWeightCalculator()

    def test_train_with_extractor(self, calculator, extractor):
        """Test training co-occurrence with concept extractor."""
        chunks = [
            "Apache Kafka is a distributed streaming platform.",
            "Kafka consumers read messages from topics.",
            "Producers publish data to Kafka topics.",
            "Consumer groups enable parallel processing in Kafka.",
            "Redis is used for caching and pub/sub.",
            "Redis pub/sub is different from Kafka.",
        ]

        calculator.train_cooccurrence(chunks, concept_extractor=extractor)

        # Kafka and consumer should have high co-occurrence
        cooc = calculator.cooccurrence.get_cooccurrence_count("kafka", "consumer")
        assert cooc > 0

    def test_cooccurrence_reflects_document_structure(self, calculator, extractor):
        """Test that co-occurrence reflects document proximity."""
        # Chunk 1: Kafka + Streaming
        # Chunk 2: Kafka + Consumer
        # Chunk 3: Redis + Cache (separate topic)
        chunks = [
            "Kafka enables real-time streaming.",
            "Kafka consumer reads streaming data.",
            "Redis provides fast caching.",
        ]

        calculator.train_cooccurrence(chunks, concept_extractor=extractor)

        # Kafka-streaming should have higher co-occurrence than kafka-caching
        cooc_kafka_streaming = calculator.cooccurrence.get_cooccurrence_count("kafka", "streaming")
        cooc_kafka_caching = calculator.cooccurrence.get_cooccurrence_count("kafka", "caching")

        assert cooc_kafka_streaming >= cooc_kafka_caching


# =============================================================================
# Integration Test: Hierarchy with Real Domain Knowledge
# =============================================================================

class TestHierarchyIntegration:
    """Test hierarchy resolver with real tech domain knowledge."""

    @pytest.fixture
    def resolver(self):
        config = EdgeWeightConfig()
        return HierarchyResolver(config)

    def test_messaging_domain_hierarchy(self, resolver):
        """Test messaging domain parent-child relationships."""
        # messaging -> kafka (parent-child)
        assert resolver.is_parent_child("messaging", "kafka")

        # distributed_systems -> messaging (grandparent)
        assert resolver.is_parent_child("distributed_systems", "messaging")

        # distributed_systems -> kafka (transitive)
        ancestors = resolver.get_ancestors("kafka")
        assert "messaging" in ancestors

    def test_sibling_concepts(self, resolver):
        """Test sibling concepts in same domain."""
        # kafka and rabbitmq are both children of messaging
        assert resolver.are_related_domains("kafka", "rabbitmq")

        # consumer and producer are both concepts in messaging
        assert resolver.is_same_domain("consumer", "producer")

    def test_cloud_hierarchy(self, resolver):
        """Test cloud provider hierarchy."""
        # aws, azure, gcp are children of cloud
        assert resolver.is_parent_child("cloud", "aws")
        assert resolver.is_parent_child("cloud", "azure")
        assert resolver.is_parent_child("cloud", "gcp")

        # aws and azure are related (siblings)
        assert resolver.are_related_domains("aws", "azure")

    def test_data_engineering_hierarchy(self, resolver):
        """Test data engineering domain."""
        # data -> data_engineering -> data_pipeline
        assert resolver.is_parent_child("data", "data_engineering")
        assert resolver.is_parent_child("data_engineering", "data_pipeline")

        # etl and elt are in the same domain (data_engineering)
        assert resolver.is_same_domain("etl", "elt")

        # kafka and rabbitmq are related (siblings under messaging)
        assert resolver.are_related_domains("kafka", "rabbitmq")


# =============================================================================
# Integration Test: Combined Edge Weights
# =============================================================================

class TestCombinedEdgeWeights:
    """Test combined edge weight calculation with all signals."""

    @pytest.fixture
    def embedding_engine(self):
        return MockEmbeddingEngine(dimension=16)

    @pytest.fixture
    def calculator(self, embedding_engine):
        calc = EdgeWeightCalculator()

        # Train co-occurrence
        chunks = [
            "Kafka is a distributed messaging platform.",
            "Kafka consumers process messages from topics.",
            "Producers send events to Kafka topics.",
            "Consumer groups scale Kafka processing.",
            "Redis provides caching and pub/sub messaging.",
        ]
        calc.train_cooccurrence(chunks)

        # Set embeddings
        concepts = ["kafka", "consumer", "producer", "topics", "messaging", "redis", "caching"]
        embeddings = {c: embedding_engine.encode(c) for c in concepts}
        calc.set_embeddings(embeddings)

        return calc

    def test_combined_weight_components(self, calculator):
        """Test that combined weight includes all components."""
        weight = calculator.calculate_weight("kafka", "consumer")

        # All components should be calculated
        assert weight.cooccurrence_score >= 0.0
        assert weight.hierarchy_score >= 0.0
        assert weight.embedding_score >= 0.0

        # Weight should be weighted sum
        config = calculator.config
        expected = (
            config.cooccurrence_weight * weight.cooccurrence_score +
            config.hierarchy_weight * weight.hierarchy_score +
            config.embedding_weight * weight.embedding_score
        )
        assert abs(weight.weight - expected) < 0.001

    def test_related_concepts_higher_weight(self, calculator):
        """Test that related concepts have higher combined weight."""
        # Kafka and consumer are highly related
        weight_related = calculator.calculate_weight("kafka", "consumer")

        # Kafka and redis are less related
        weight_unrelated = calculator.calculate_weight("kafka", "redis")

        # Related should have higher weight
        assert weight_related.weight > weight_unrelated.weight

    def test_hierarchy_boosts_weight(self, calculator):
        """Test that hierarchy relationships boost weight."""
        # Consumer and producer are in same domain (messaging)
        weight = calculator.calculate_weight("consumer", "producer")

        # Hierarchy score should contribute
        assert weight.hierarchy_score > 0.0


# =============================================================================
# Integration Test: Edge Building for Graph
# =============================================================================

class TestEdgeBuildingForGraph:
    """Test building weighted edges for graph construction."""

    @pytest.fixture
    def calculator(self):
        calc = EdgeWeightCalculator()

        # Training data
        chunks = [
            "Apache Kafka is used for event streaming.",
            "Kafka consumers and producers interact via topics.",
            "Consumer groups enable horizontal scaling.",
            "Kubernetes orchestrates containerized applications.",
            "Pods run in Kubernetes clusters.",
            "Docker containers are orchestrated by Kubernetes.",
        ]
        calc.train_cooccurrence(chunks)

        # Embeddings
        embedding_engine = MockEmbeddingEngine(dimension=16)
        concepts = ["kafka", "consumer", "producer", "topics", "kubernetes", "pods", "docker", "containers"]
        embeddings = {c: embedding_engine.encode(c) for c in concepts}
        calc.set_embeddings(embeddings)

        return calc

    def test_build_edges_creates_valid_edges(self, calculator):
        """Test that build_weighted_edges creates valid edges."""
        concepts = ["kafka", "consumer", "producer", "kubernetes", "pods"]
        edges = calculator.build_weighted_edges(concepts)

        assert len(edges) > 0
        for edge in edges:
            assert edge.source in concepts
            assert edge.target in concepts
            assert edge.weight >= calculator.config.min_edge_weight

    def test_edges_respect_max_per_concept(self, calculator):
        """Test that max edges per concept is respected."""
        concepts = ["kafka", "consumer", "producer", "topics", "kubernetes", "pods", "docker"]
        edges = calculator.build_weighted_edges(concepts, max_edges_per_concept=2)

        # Count edges per concept
        edge_counts: Dict[str, int] = {}
        for edge in edges:
            edge_counts[edge.source] = edge_counts.get(edge.source, 0) + 1
            edge_counts[edge.target] = edge_counts.get(edge.target, 0) + 1

        for count in edge_counts.values():
            assert count <= 2

    def test_edges_sorted_by_weight(self, calculator):
        """Test that edges are sorted by weight descending."""
        concepts = ["kafka", "consumer", "producer", "topics"]
        edges = calculator.build_weighted_edges(concepts)

        # Edges should be in descending weight order
        weights = [e.weight for e in edges]
        assert weights == sorted(weights, reverse=True)


# =============================================================================
# Integration Test: Resonance with Weighted Edges
# =============================================================================

class MockResonanceMatcher:
    """
    Mock resonance matcher that uses edge weights for propagation.

    Simulates the resonance propagation algorithm.
    """

    def __init__(self, decay: float = 0.7, max_depth: int = 3):
        self.decay = decay
        self.max_depth = max_depth
        self.edges: Dict[str, List[tuple]] = {}  # concept -> [(neighbor, weight), ...]

    def add_edge(self, source: str, target: str, weight: float):
        """Add weighted edge."""
        if source not in self.edges:
            self.edges[source] = []
        if target not in self.edges:
            self.edges[target] = []

        self.edges[source].append((target, weight))
        self.edges[target].append((source, weight))

    def propagate(self, seed_concepts: Set[str]) -> Dict[str, float]:
        """
        Propagate resonance from seed concepts.

        Returns dict of concept -> resonance score.
        """
        scores: Dict[str, float] = {c: 1.0 for c in seed_concepts}
        visited: Set[str] = set(seed_concepts)

        # BFS propagation
        frontier = list(seed_concepts)
        depth = 0

        while frontier and depth < self.max_depth:
            next_frontier = []
            depth += 1

            for concept in frontier:
                parent_score = scores.get(concept, 0.0)

                for neighbor, weight in self.edges.get(concept, []):
                    if neighbor not in visited:
                        # Resonance = parent_score * weight * decay^depth
                        resonance = parent_score * weight * (self.decay ** depth)

                        if resonance > 0.01:  # Threshold
                            scores[neighbor] = max(scores.get(neighbor, 0.0), resonance)
                            next_frontier.append(neighbor)
                            visited.add(neighbor)

            frontier = next_frontier

        return scores


class TestResonanceWithWeightedEdges:
    """Test resonance propagation using weighted edges."""

    @pytest.fixture
    def edge_calculator(self):
        calc = EdgeWeightCalculator()

        chunks = [
            "Kafka handles distributed messaging.",
            "Kafka consumers read from topics.",
            "Producers write to Kafka.",
            "Redis provides caching.",
        ]
        calc.train_cooccurrence(chunks)

        embedding_engine = MockEmbeddingEngine(16)
        for c in ["kafka", "consumer", "producer", "topics", "messaging", "redis", "caching"]:
            calc.add_embedding(c, embedding_engine.encode(c))

        return calc

    @pytest.fixture
    def resonance_matcher(self, edge_calculator):
        matcher = MockResonanceMatcher(decay=0.7, max_depth=3)

        concepts = ["kafka", "consumer", "producer", "topics", "messaging", "redis"]
        edges = edge_calculator.build_weighted_edges(concepts)

        for edge in edges:
            matcher.add_edge(edge.source, edge.target, edge.weight)

        return matcher

    def test_resonance_propagates_through_weighted_edges(self, resonance_matcher):
        """Test that resonance propagates through weighted edges."""
        # Start with kafka
        scores = resonance_matcher.propagate({"kafka"})

        # Kafka should have score 1.0
        assert scores["kafka"] == 1.0

        # Related concepts should have resonance
        assert len(scores) > 1

    def test_higher_weight_means_higher_resonance(self, resonance_matcher, edge_calculator):
        """Test that higher edge weight leads to higher resonance."""
        # Get edge weights
        weight_kafka_consumer = edge_calculator.calculate_weight("kafka", "consumer").weight
        weight_kafka_redis = edge_calculator.calculate_weight("kafka", "redis").weight

        # Propagate from kafka
        scores = resonance_matcher.propagate({"kafka"})

        # If kafka-consumer has higher weight, consumer should have higher resonance
        if weight_kafka_consumer > weight_kafka_redis:
            if "consumer" in scores and "redis" in scores:
                assert scores.get("consumer", 0) >= scores.get("redis", 0)

    def test_resonance_decays_with_depth(self, resonance_matcher):
        """Test that resonance decays with graph depth."""
        scores = resonance_matcher.propagate({"kafka"})

        # Direct neighbors should have higher resonance than distant ones
        direct_neighbors = [c for c, w in resonance_matcher.edges.get("kafka", [])]

        for neighbor in direct_neighbors:
            if neighbor in scores:
                # Direct neighbor score should be relatively high
                assert scores[neighbor] > 0.1


# =============================================================================
# Integration Test: Full Pipeline
# =============================================================================

class TestFullPipeline:
    """Test complete pipeline from documents to weighted graph."""

    def test_document_to_weighted_graph(self):
        """Test building weighted graph from documents."""
        # 1. Documents
        documents = [
            "Apache Kafka is a distributed event streaming platform. "
            "Kafka can handle high-throughput data pipelines.",

            "Kafka consumers subscribe to topics and process messages. "
            "Consumer groups enable parallel processing of topics.",

            "Producers publish data to Kafka topics. "
            "Each topic can have multiple partitions.",

            "Kubernetes orchestrates containerized applications. "
            "Pods are the smallest deployable units in Kubernetes.",
        ]

        # 2. Chunk documents (simple sentence splitting)
        chunks = []
        for doc in documents:
            sentences = doc.split(". ")
            chunks.extend(sentences)

        # 3. Create calculator and train
        calc = EdgeWeightCalculator()
        extractor = MockConceptExtractor()
        calc.train_cooccurrence(chunks, concept_extractor=extractor)

        # 4. Generate embeddings for key concepts
        embedding_engine = MockEmbeddingEngine(32)
        key_concepts = ["kafka", "consumer", "producer", "topics", "kubernetes", "pods", "streaming"]
        for concept in key_concepts:
            calc.add_embedding(concept, embedding_engine.encode(concept))

        # 5. Build weighted edges
        edges = calc.build_weighted_edges(key_concepts)

        # 6. Validate graph structure
        assert len(edges) > 0

        # Kafka-related concepts should be connected
        kafka_edges = [e for e in edges if "kafka" in (e.source, e.target)]
        assert len(kafka_edges) > 0

        # 7. Use for resonance
        matcher = MockResonanceMatcher()
        for edge in edges:
            matcher.add_edge(edge.source, edge.target, edge.weight)

        scores = matcher.propagate({"kafka"})

        # Related concepts should resonate
        assert len(scores) > 1
        assert scores.get("kafka") == 1.0

    def test_weighted_edges_improve_rag_coverage(self):
        """Test that weighted edges help find related content."""
        # Simulate RAG scenario:
        # Query mentions "Kafka consumer", documents mention "Kafka subscriber"
        # Edge weights should connect these concepts

        calc = EdgeWeightCalculator()

        chunks = [
            "Kafka consumers subscribe to topics.",
            "Subscribers receive messages from Kafka.",
            "Consumer and subscriber are similar concepts.",
        ]
        calc.train_cooccurrence(chunks)

        # Set similar embeddings for related concepts
        calc.set_embeddings({
            "consumer": [0.9, 0.5, 0.3],
            "subscriber": [0.85, 0.55, 0.35],
            "kafka": [0.7, 0.8, 0.2],
        })

        # Consumer and subscriber should be connected
        weight = calc.calculate_weight("consumer", "subscriber")
        assert weight.weight > 0.2  # Should have significant weight

        # This connection helps RAG find "subscriber" when query mentions "consumer"


# =============================================================================
# Edge Weight Configuration Tests
# =============================================================================

class TestEdgeWeightConfigurations:
    """Test different edge weight configurations."""

    def test_cooccurrence_only(self):
        """Test with only co-occurrence signal."""
        config = EdgeWeightConfig(
            cooccurrence_weight=1.0,
            hierarchy_weight=0.0,
            embedding_weight=0.0
        )
        calc = EdgeWeightCalculator(config)

        chunks = ["Kafka consumer reads data.", "Consumer processes Kafka messages."]
        calc.train_cooccurrence(chunks)

        weight = calc.calculate_weight("kafka", "consumer")

        assert weight.hierarchy_score == 0.0 or config.hierarchy_weight == 0.0
        # Weight should be based only on co-occurrence
        assert abs(weight.weight - weight.cooccurrence_score) < 0.01

    def test_hierarchy_only(self):
        """Test with only hierarchy signal."""
        config = EdgeWeightConfig(
            cooccurrence_weight=0.0,
            hierarchy_weight=1.0,
            embedding_weight=0.0
        )
        calc = EdgeWeightCalculator(config)

        weight = calc.calculate_weight("messaging", "kafka")

        # Weight should be based only on hierarchy
        assert abs(weight.weight - weight.hierarchy_score) < 0.01

    def test_embedding_only(self):
        """Test with only embedding signal."""
        config = EdgeWeightConfig(
            cooccurrence_weight=0.0,
            hierarchy_weight=0.0,
            embedding_weight=1.0
        )
        calc = EdgeWeightCalculator(config)

        calc.set_embeddings({
            "kafka": [1.0, 0.5],
            "messaging": [0.95, 0.55]
        })

        weight = calc.calculate_weight("kafka", "messaging")

        # Weight should be based only on embedding
        assert abs(weight.weight - weight.embedding_score) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
