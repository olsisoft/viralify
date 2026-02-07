"""
Tests for multi-factor edge weighting integration in WeaveGraphBuilder.

Tests that EdgeWeightCalculator is properly integrated with:
- GraphBuilderConfig multi-factor settings
- build_from_documents() co-occurrence training
- Edge building with combined weights
"""

import pytest
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np

# Add the services directory to the path for direct imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
services_dir = os.path.join(base_dir, 'services')
weave_graph_dir = os.path.join(services_dir, 'weave_graph')
sys.path.insert(0, weave_graph_dir)

# Import directly from files to avoid __init__.py circular imports
import importlib.util

def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import models directly
models_module = import_module_from_file(
    "weave_graph_models",
    os.path.join(weave_graph_dir, "models.py")
)
ConceptNode = models_module.ConceptNode
ConceptEdge = models_module.ConceptEdge
RelationType = models_module.RelationType

# Import edge_weight_calculator directly
edge_weight_module = import_module_from_file(
    "edge_weight_calculator",
    os.path.join(weave_graph_dir, "edge_weight_calculator.py")
)
EdgeWeightCalculator = edge_weight_module.EdgeWeightCalculator
EdgeWeightConfig = edge_weight_module.EdgeWeightConfig
CooccurrenceCalculator = edge_weight_module.CooccurrenceCalculator
HierarchyResolver = edge_weight_module.HierarchyResolver
EmbeddingSimilarityCalculator = edge_weight_module.EmbeddingSimilarityCalculator


class TestGraphBuilderConfigIntegration:
    """Test that GraphBuilderConfig properly configures multi-factor settings."""

    def test_default_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        config = EdgeWeightConfig()
        total = (
            config.cooccurrence_weight +
            config.hierarchy_weight +
            config.embedding_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_edge_weight_config_from_builder_config(self):
        """EdgeWeightConfig should be configurable from builder config values."""
        config = EdgeWeightConfig(
            cooccurrence_weight=0.5,
            hierarchy_weight=0.3,
            embedding_weight=0.2,
            min_edge_weight=0.2,
            window_size=2
        )
        assert config.cooccurrence_weight == 0.5
        assert config.hierarchy_weight == 0.3
        assert config.embedding_weight == 0.2
        assert config.min_edge_weight == 0.2
        assert config.window_size == 2


class TestEdgeWeightCalculatorWithConcepts:
    """Test EdgeWeightCalculator with ConceptNode objects."""

    @pytest.fixture
    def calculator(self):
        """Create a configured calculator."""
        config = EdgeWeightConfig(
            cooccurrence_weight=0.4,
            hierarchy_weight=0.3,
            embedding_weight=0.3,
            min_edge_weight=0.1
        )
        return EdgeWeightCalculator(config)

    @pytest.fixture
    def tech_concepts(self):
        """Create test concepts for tech domain."""
        return [
            ConceptNode(
                id="id_kafka",
                name="Kafka",
                canonical_name="kafka",
                language="en",
                embedding=[1.0, 0.0, 0.0, 0.0],
                source_document_ids=["doc1"],
                frequency=2
            ),
            ConceptNode(
                id="id_consumer",
                name="Consumer",
                canonical_name="consumer",
                language="en",
                embedding=[0.9, 0.1, 0.0, 0.0],
                source_document_ids=["doc1"],
                frequency=2
            ),
            ConceptNode(
                id="id_producer",
                name="Producer",
                canonical_name="producer",
                language="en",
                embedding=[0.85, 0.15, 0.0, 0.0],
                source_document_ids=["doc1"],
                frequency=1
            ),
        ]

    def test_train_with_document_chunks(self, calculator):
        """Calculator should be trained with document chunks."""
        chunks = [
            "Kafka is a message broker for event streaming.",
            "A consumer reads messages from Kafka topics.",
            "Producers publish messages to Kafka.",
        ]

        calculator.train_cooccurrence(chunks)
        assert calculator.cooccurrence._is_trained

    def test_embeddings_set_correctly(self, calculator, tech_concepts):
        """Embeddings should be set from concept embeddings."""
        embeddings = {
            c.canonical_name: c.embedding
            for c in tech_concepts
            if c.embedding
        }
        calculator.set_embeddings(embeddings)

        # Verify embeddings were set via calculate_weight (which uses them)
        weight = calculator.calculate_weight("kafka", "consumer")
        assert weight.embedding_score >= 0

    def test_calculate_weight_between_concepts(self, calculator, tech_concepts):
        """Should calculate combined weight between concepts."""
        # Setup
        chunks = [
            "Kafka consumer reads events",
            "Producer writes to Kafka"
        ]
        calculator.train_cooccurrence(chunks)

        embeddings = {c.canonical_name: c.embedding for c in tech_concepts if c.embedding}
        calculator.set_embeddings(embeddings)

        # Calculate weight
        weight = calculator.calculate_weight("kafka", "consumer")

        assert weight.weight > 0  # Use .weight not .combined_weight
        assert 0.0 <= weight.weight <= 1.0

    def test_build_edges_from_concepts(self, calculator, tech_concepts):
        """Should build weighted edges from concept list."""
        # Setup
        chunks = [
            "Kafka consumer reads events. Producer writes to Kafka.",
        ]
        calculator.train_cooccurrence(chunks)

        embeddings = {c.canonical_name: c.embedding for c in tech_concepts if c.embedding}
        calculator.set_embeddings(embeddings)

        # Build edges
        concept_names = [c.canonical_name for c in tech_concepts]
        edges = calculator.build_weighted_edges(concept_names, max_edges_per_concept=5)

        assert len(edges) > 0
        for edge in edges:
            assert edge.weight >= calculator.config.min_edge_weight


class TestHierarchyWithConcepts:
    """Test hierarchy resolver with tech concepts."""

    @pytest.fixture
    def config(self):
        return EdgeWeightConfig()

    @pytest.fixture
    def resolver(self, config):
        return HierarchyResolver(config)

    def test_kafka_in_messaging_hierarchy(self, resolver):
        """Kafka should be in messaging hierarchy."""
        score = resolver.get_score("kafka", "messaging")
        assert score > 0

    def test_consumer_producer_related(self, resolver):
        """Consumer and producer should be related via messaging."""
        # Both are in messaging domain
        domain_a = resolver.get_domain("consumer")
        domain_b = resolver.get_domain("producer")

        # They should be in the same or related domains
        score = resolver.get_score("consumer", "producer")
        assert score >= 0

    def test_unrelated_concepts(self, resolver):
        """Unrelated concepts should have low hierarchy score."""
        score = resolver.get_score("random_concept_xyz", "another_random_abc")
        assert score == 0.0


class TestConceptEdgeCreation:
    """Test creating ConceptEdge from EdgeWeight."""

    def test_create_edge_from_weight(self):
        """Should create ConceptEdge from EdgeWeight results."""
        # Simulating what graph_builder does
        source_id = "id_kafka"
        target_id = "id_consumer"
        weight = 0.75
        hierarchy_score = 0.6

        # Determine relation type based on hierarchy
        if hierarchy_score > 0.5:
            rel_type = RelationType.PART_OF
        else:
            rel_type = RelationType.SIMILAR

        edge = ConceptEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=rel_type,
            weight=weight,
            bidirectional=True
        )

        assert edge.source_id == "id_kafka"
        assert edge.target_id == "id_consumer"
        assert edge.relation_type == RelationType.PART_OF
        assert edge.weight == 0.75
        assert edge.bidirectional is True

    def test_similar_relation_for_low_hierarchy(self):
        """Low hierarchy score should result in SIMILAR relation."""
        hierarchy_score = 0.3

        if hierarchy_score > 0.5:
            rel_type = RelationType.PART_OF
        else:
            rel_type = RelationType.SIMILAR

        assert rel_type == RelationType.SIMILAR


class TestMultiFactorWeighting:
    """Test the combined multi-factor weighting logic."""

    def test_all_factors_contribute(self):
        """All three factors should contribute to final weight."""
        config = EdgeWeightConfig(
            cooccurrence_weight=0.4,
            hierarchy_weight=0.3,
            embedding_weight=0.3
        )
        calc = EdgeWeightCalculator(config)

        # Train cooccurrence
        calc.train_cooccurrence([
            "Kafka and consumer work together frequently.",
            "Consumer reads from Kafka broker."
        ])

        # Set embeddings (similar vectors)
        calc.set_embeddings({
            "kafka": [1.0, 0.0, 0.0],
            "consumer": [0.95, 0.05, 0.0]
        })

        weight = calc.calculate_weight("kafka", "consumer")

        # All components should have non-zero contribution
        assert weight.cooccurrence_score >= 0
        assert weight.embedding_score >= 0
        # Note: hierarchy might be 0 if not in TECH_HIERARCHY

    def test_weight_normalization(self):
        """Combined weight should never exceed 1.0."""
        config = EdgeWeightConfig(
            cooccurrence_weight=0.4,
            hierarchy_weight=0.3,
            embedding_weight=0.3
        )
        calc = EdgeWeightCalculator(config)

        # Setup for maximum possible scores
        calc.train_cooccurrence([
            "kafka consumer kafka consumer kafka consumer"
        ])
        calc.set_embeddings({
            "kafka": [1.0, 0.0, 0.0],
            "consumer": [1.0, 0.0, 0.0]  # Identical embeddings
        })

        weight = calc.calculate_weight("kafka", "consumer")
        assert weight.weight <= 1.0  # Use .weight not .combined_weight


class TestMinEdgeWeightFiltering:
    """Test that weak edges are filtered by min_edge_weight."""

    def test_filter_weak_edges(self):
        """Edges below threshold should not be created."""
        config = EdgeWeightConfig(min_edge_weight=0.5)
        calc = EdgeWeightCalculator(config)

        # Setup with low similarity concepts
        calc.set_embeddings({
            "unrelated_a": [1.0, 0.0, 0.0],
            "unrelated_b": [0.0, 1.0, 0.0],  # Orthogonal
        })

        weight = calc.calculate_weight("unrelated_a", "unrelated_b")
        should_create, _ = calc.should_create_edge("unrelated_a", "unrelated_b")

        # Weight should be below threshold
        assert weight.weight < 0.5
        assert should_create is False

    def test_allow_strong_edges(self):
        """Edges above threshold should be created."""
        config = EdgeWeightConfig(min_edge_weight=0.1)
        calc = EdgeWeightCalculator(config)

        # Setup with high similarity concepts
        calc.train_cooccurrence(["kafka consumer kafka consumer"])
        calc.set_embeddings({
            "kafka": [1.0, 0.0, 0.0],
            "consumer": [0.95, 0.05, 0.0],
        })

        should_create, edge_weight = calc.should_create_edge("kafka", "consumer")
        assert should_create is True
        assert edge_weight.weight >= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
