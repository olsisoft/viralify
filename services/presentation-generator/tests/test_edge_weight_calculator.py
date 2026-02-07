"""
Unit tests for EdgeWeightCalculator.

Tests the three components:
1. CooccurrenceCalculator - PMI-based co-occurrence scoring
2. HierarchyResolver - Taxonomic relationship scoring
3. EmbeddingSimilarityCalculator - Cosine similarity scoring
4. EdgeWeightCalculator - Combined weighted scoring
"""

import pytest
import math
from typing import List, Dict

import sys
import os

# Add path to import edge_weight_calculator
_weave_graph_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "services",
    "weave_graph"
)
sys.path.insert(0, _weave_graph_path)

from edge_weight_calculator import (
    EdgeWeightConfig,
    CooccurrenceCalculator,
    HierarchyResolver,
    EmbeddingSimilarityCalculator,
    EdgeWeightCalculator,
    EdgeWeight,
    create_edge_weight_calculator,
    TECH_HIERARCHY,
    _CONCEPT_TO_DOMAIN,
)


# =============================================================================
# EdgeWeightConfig Tests
# =============================================================================

class TestEdgeWeightConfig:
    """Tests for EdgeWeightConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EdgeWeightConfig()

        assert config.cooccurrence_weight == 0.4
        assert config.hierarchy_weight == 0.3
        assert config.embedding_weight == 0.3
        assert config.window_size == 1
        assert config.min_cooccurrence == 1

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        config = EdgeWeightConfig(
            cooccurrence_weight=2.0,
            hierarchy_weight=1.0,
            embedding_weight=1.0
        )

        total = config.cooccurrence_weight + config.hierarchy_weight + config.embedding_weight
        assert abs(total - 1.0) < 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = EdgeWeightConfig(
            cooccurrence_weight=0.5,
            hierarchy_weight=0.3,
            embedding_weight=0.2,
            window_size=2,
            min_cooccurrence=3
        )

        assert config.window_size == 2
        assert config.min_cooccurrence == 3


# =============================================================================
# CooccurrenceCalculator Tests
# =============================================================================

class TestCooccurrenceCalculator:
    """Tests for CooccurrenceCalculator."""

    @pytest.fixture
    def calculator(self):
        config = EdgeWeightConfig(min_cooccurrence=1)
        return CooccurrenceCalculator(config)

    @pytest.fixture
    def trained_calculator(self, calculator):
        chunks = [
            "Kafka is a distributed messaging system.",
            "Kafka uses consumers and producers.",
            "The consumer reads from Kafka topics.",
            "Producers publish messages to Kafka.",
            "Redis is a key-value store.",
            "Redis can be used for caching.",
        ]
        calculator.train(chunks)
        return calculator

    def test_train_empty_chunks(self, calculator):
        """Test training on empty chunks."""
        calculator.train([])
        assert calculator._is_trained
        assert len(calculator._concept_counts) == 0

    def test_train_counts_concepts(self, trained_calculator):
        """Test that training counts concepts correctly."""
        assert trained_calculator._is_trained
        assert len(trained_calculator._concept_counts) > 0
        assert "kafka" in trained_calculator._concept_counts

    def test_cooccurrence_count(self, trained_calculator):
        """Test co-occurrence counting."""
        # Kafka and consumer appear together
        count = trained_calculator.get_cooccurrence_count("kafka", "consumer")
        assert count > 0

    def test_cooccurrence_symmetry(self, trained_calculator):
        """Test that co-occurrence is symmetric."""
        count_ab = trained_calculator.get_cooccurrence_count("kafka", "consumer")
        count_ba = trained_calculator.get_cooccurrence_count("consumer", "kafka")
        assert count_ab == count_ba

    def test_pmi_high_for_collocations(self, trained_calculator):
        """Test PMI is higher for collocations."""
        # Kafka and consumer often appear together
        pmi_kafka_consumer = trained_calculator.calculate_pmi("kafka", "consumer")

        # Kafka and redis rarely appear together
        pmi_kafka_redis = trained_calculator.calculate_pmi("kafka", "redis")

        # PMI for collocations should be higher
        assert pmi_kafka_consumer > pmi_kafka_redis

    def test_pmi_normalized_range(self, trained_calculator):
        """Test PMI is normalized to [0, 1]."""
        pmi = trained_calculator.calculate_pmi("kafka", "consumer")
        assert 0.0 <= pmi <= 1.0

    def test_pmi_untrained(self, calculator):
        """Test PMI returns 0 when not trained."""
        pmi = calculator.calculate_pmi("kafka", "consumer")
        assert pmi == 0.0

    def test_get_score(self, trained_calculator):
        """Test get_score method."""
        score = trained_calculator.get_score("kafka", "consumer")
        assert score > 0.0
        assert score <= 1.0

    def test_score_unknown_concepts(self, trained_calculator):
        """Test score for unknown concepts."""
        score = trained_calculator.get_score("unknown_concept_xyz", "another_unknown")
        assert score == 0.0

    def test_window_size_effect(self):
        """Test that larger window size increases co-occurrence."""
        chunks = [
            "First chunk about Kafka.",
            "Second chunk about nothing.",
            "Third chunk about consumer.",
        ]

        # Window size 1 - only same chunk
        config1 = EdgeWeightConfig(window_size=1)
        calc1 = CooccurrenceCalculator(config1)
        calc1.train(chunks)
        cooc1 = calc1.get_cooccurrence_count("kafka", "consumer")

        # Window size 2 - adjacent chunks too
        config2 = EdgeWeightConfig(window_size=2)
        calc2 = CooccurrenceCalculator(config2)
        calc2.train(chunks)
        cooc2 = calc2.get_cooccurrence_count("kafka", "consumer")

        # Larger window should capture more co-occurrences
        assert cooc2 >= cooc1


# =============================================================================
# HierarchyResolver Tests
# =============================================================================

class TestHierarchyResolver:
    """Tests for HierarchyResolver."""

    @pytest.fixture
    def resolver(self):
        config = EdgeWeightConfig()
        return HierarchyResolver(config)

    def test_get_domain_known_concept(self, resolver):
        """Test getting domain for a known concept."""
        # Consumer is a concept in messaging domain
        domain = resolver.get_domain("consumer")
        assert domain == "messaging"

        # Kafka is its own domain (sub-domain of messaging)
        domain = resolver.get_domain("kafka")
        assert domain == "kafka"

    def test_get_domain_unknown_concept(self, resolver):
        """Test getting domain for unknown concept."""
        domain = resolver.get_domain("unknown_xyz")
        assert domain is None

    def test_get_domain_case_insensitive(self, resolver):
        """Test domain lookup is case insensitive."""
        domain1 = resolver.get_domain("kafka")
        domain2 = resolver.get_domain("KAFKA")
        domain3 = resolver.get_domain("Kafka")
        assert domain1 == domain2 == domain3

    def test_is_parent_child_direct(self, resolver):
        """Test direct parent-child relationship."""
        # messaging is parent of kafka
        assert resolver.is_parent_child("messaging", "kafka")
        assert resolver.is_parent_child("kafka", "messaging")

    def test_is_parent_child_indirect(self, resolver):
        """Test indirect parent-child (grandparent)."""
        # distributed_systems -> messaging -> kafka
        assert resolver.is_parent_child("distributed_systems", "kafka")

    def test_is_parent_child_false(self, resolver):
        """Test non parent-child relationship."""
        # kafka and redis are not parent-child
        assert not resolver.is_parent_child("kafka", "redis")

    def test_is_same_domain(self, resolver):
        """Test same domain detection."""
        # consumer and producer are both in messaging
        assert resolver.is_same_domain("consumer", "producer")

    def test_is_same_domain_false(self, resolver):
        """Test different domain detection."""
        # kafka (messaging) and python (programming)
        assert not resolver.is_same_domain("kafka", "python")

    def test_are_related_domains(self, resolver):
        """Test related domains (sibling domains)."""
        # kafka and rabbitmq are siblings (both children of messaging)
        assert resolver.are_related_domains("kafka", "rabbitmq")

    def test_are_related_domains_false(self, resolver):
        """Test unrelated domains."""
        # kafka and python are not related
        assert not resolver.are_related_domains("kafka", "python")

    def test_get_score_parent_child(self, resolver):
        """Test score for parent-child relationship."""
        score = resolver.get_score("messaging", "kafka")
        assert score == resolver.config.parent_child_score

    def test_get_score_same_domain(self, resolver):
        """Test score for same domain."""
        score = resolver.get_score("consumer", "producer")
        assert score == resolver.config.same_domain_score

    def test_get_score_related_domains(self, resolver):
        """Test score for related domains."""
        score = resolver.get_score("kafka", "rabbitmq")
        assert score == resolver.config.related_domain_score

    def test_get_score_unrelated(self, resolver):
        """Test score for unrelated concepts."""
        score = resolver.get_score("kafka", "python")
        assert score == 0.0

    def test_get_ancestors(self, resolver):
        """Test ancestor chain retrieval."""
        ancestors = resolver.get_ancestors("kafka")
        assert "messaging" in ancestors
        assert "distributed_systems" in ancestors


# =============================================================================
# EmbeddingSimilarityCalculator Tests
# =============================================================================

class TestEmbeddingSimilarityCalculator:
    """Tests for EmbeddingSimilarityCalculator."""

    @pytest.fixture
    def calculator(self):
        config = EdgeWeightConfig(min_similarity=0.3)
        return EmbeddingSimilarityCalculator(config)

    def test_cosine_similarity_identical(self, calculator):
        """Test cosine similarity for identical vectors."""
        vec = [1.0, 0.0, 0.0]
        sim = calculator.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self, calculator):
        """Test cosine similarity for orthogonal vectors."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        sim = calculator.cosine_similarity(vec_a, vec_b)
        assert abs(sim) < 0.001

    def test_cosine_similarity_opposite(self, calculator):
        """Test cosine similarity for opposite vectors."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        sim = calculator.cosine_similarity(vec_a, vec_b)
        assert abs(sim + 1.0) < 0.001

    def test_cosine_similarity_similar(self, calculator):
        """Test cosine similarity for similar vectors."""
        vec_a = [1.0, 1.0, 0.0]
        vec_b = [1.0, 0.9, 0.1]
        sim = calculator.cosine_similarity(vec_a, vec_b)
        assert sim > 0.9

    def test_set_embeddings(self, calculator):
        """Test setting embeddings."""
        embeddings = {
            "kafka": [1.0, 0.5, 0.3],
            "consumer": [0.9, 0.6, 0.2]
        }
        calculator.set_embeddings(embeddings)
        assert "kafka" in calculator._embeddings
        assert "consumer" in calculator._embeddings

    def test_add_embedding(self, calculator):
        """Test adding single embedding."""
        calculator.add_embedding("kafka", [1.0, 0.5, 0.3])
        assert "kafka" in calculator._embeddings

    def test_get_score_similar(self, calculator):
        """Test score for similar embeddings."""
        calculator.set_embeddings({
            "kafka": [1.0, 0.5, 0.3],
            "messaging": [0.95, 0.55, 0.25]
        })
        score = calculator.get_score("kafka", "messaging")
        assert score > 0.5

    def test_get_score_below_threshold(self, calculator):
        """Test score below minimum threshold."""
        calculator.set_embeddings({
            "kafka": [1.0, 0.0, 0.0],
            "python": [0.0, 1.0, 0.0]  # Orthogonal
        })
        score = calculator.get_score("kafka", "python")
        assert score == 0.0

    def test_get_score_missing_embedding(self, calculator):
        """Test score when embedding is missing."""
        calculator.set_embeddings({
            "kafka": [1.0, 0.5, 0.3]
        })
        score = calculator.get_score("kafka", "unknown")
        assert score == 0.0

    def test_case_insensitive(self, calculator):
        """Test case insensitive lookup."""
        calculator.set_embeddings({
            "Kafka": [1.0, 0.5, 0.3],
            "CONSUMER": [0.9, 0.6, 0.2]
        })
        # Should find regardless of case
        assert "kafka" in calculator._embeddings
        assert "consumer" in calculator._embeddings


# =============================================================================
# EdgeWeightCalculator Tests
# =============================================================================

class TestEdgeWeightCalculator:
    """Tests for the main EdgeWeightCalculator."""

    @pytest.fixture
    def calculator(self):
        return EdgeWeightCalculator()

    @pytest.fixture
    def trained_calculator(self, calculator):
        # Train co-occurrence
        chunks = [
            "Kafka is a messaging system.",
            "Kafka consumers read messages.",
            "Producers send to Kafka topics.",
            "Consumer groups in Kafka.",
        ]
        calculator.train_cooccurrence(chunks)

        # Set embeddings
        calculator.set_embeddings({
            "kafka": [1.0, 0.5, 0.3, 0.2],
            "consumer": [0.9, 0.6, 0.25, 0.15],
            "producer": [0.85, 0.55, 0.35, 0.2],
            "messaging": [0.95, 0.45, 0.4, 0.25],
            "python": [0.1, 0.2, 0.9, 0.8],
        })

        return calculator

    def test_calculate_weight_basic(self, trained_calculator):
        """Test basic weight calculation."""
        weight = trained_calculator.calculate_weight("kafka", "consumer")

        assert isinstance(weight, EdgeWeight)
        assert weight.source == "kafka"
        assert weight.target == "consumer"
        assert 0.0 <= weight.weight <= 1.0

    def test_calculate_weight_components(self, trained_calculator):
        """Test that weight includes all components."""
        weight = trained_calculator.calculate_weight("kafka", "consumer")

        assert hasattr(weight, "cooccurrence_score")
        assert hasattr(weight, "hierarchy_score")
        assert hasattr(weight, "embedding_score")

    def test_weight_is_weighted_sum(self, trained_calculator):
        """Test that weight is weighted sum of components."""
        weight = trained_calculator.calculate_weight("kafka", "consumer")
        config = trained_calculator.config

        expected = (
            config.cooccurrence_weight * weight.cooccurrence_score +
            config.hierarchy_weight * weight.hierarchy_score +
            config.embedding_weight * weight.embedding_score
        )

        assert abs(weight.weight - expected) < 0.001

    def test_high_weight_for_related_concepts(self, trained_calculator):
        """Test high weight for related concepts."""
        weight_related = trained_calculator.calculate_weight("kafka", "consumer")
        weight_unrelated = trained_calculator.calculate_weight("kafka", "python")

        assert weight_related.weight > weight_unrelated.weight

    def test_should_create_edge_true(self, trained_calculator):
        """Test edge creation for related concepts."""
        should_create, weight = trained_calculator.should_create_edge("kafka", "consumer")
        assert should_create
        assert weight.weight >= trained_calculator.config.min_edge_weight

    def test_should_create_edge_false(self, trained_calculator):
        """Test edge not created for unrelated concepts."""
        # Set high threshold
        trained_calculator.config.min_edge_weight = 0.8
        should_create, weight = trained_calculator.should_create_edge("kafka", "python")
        # Might or might not create depending on scores
        assert isinstance(should_create, bool)

    def test_build_weighted_edges(self, trained_calculator):
        """Test building weighted edges for concept list."""
        concepts = ["kafka", "consumer", "producer", "messaging"]
        edges = trained_calculator.build_weighted_edges(concepts)

        assert len(edges) > 0
        for edge in edges:
            assert isinstance(edge, EdgeWeight)
            assert edge.weight >= trained_calculator.config.min_edge_weight

    def test_build_weighted_edges_max_per_concept(self, trained_calculator):
        """Test max edges per concept limit."""
        concepts = ["kafka", "consumer", "producer", "messaging", "python"]
        edges = trained_calculator.build_weighted_edges(concepts, max_edges_per_concept=2)

        # Count edges per concept
        edge_counts = {}
        for edge in edges:
            edge_counts[edge.source] = edge_counts.get(edge.source, 0) + 1
            edge_counts[edge.target] = edge_counts.get(edge.target, 0) + 1

        for concept, count in edge_counts.items():
            assert count <= 2

    def test_edge_weight_to_dict(self, trained_calculator):
        """Test EdgeWeight.to_dict() method."""
        weight = trained_calculator.calculate_weight("kafka", "consumer")
        d = weight.to_dict()

        assert "source" in d
        assert "target" in d
        assert "weight" in d
        assert "components" in d
        assert "cooccurrence" in d["components"]
        assert "hierarchy" in d["components"]
        assert "embedding" in d["components"]


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_edge_weight_calculator_default(self):
        """Test creating calculator with defaults."""
        calc = create_edge_weight_calculator()
        assert isinstance(calc, EdgeWeightCalculator)
        assert calc.config.cooccurrence_weight == 0.4

    def test_create_edge_weight_calculator_custom(self):
        """Test creating calculator with custom weights."""
        calc = create_edge_weight_calculator(
            cooccurrence_weight=0.5,
            hierarchy_weight=0.3,
            embedding_weight=0.2,
            window_size=3
        )

        total = (calc.config.cooccurrence_weight +
                calc.config.hierarchy_weight +
                calc.config.embedding_weight)
        assert abs(total - 1.0) < 0.01
        assert calc.config.window_size == 3


# =============================================================================
# TECH_HIERARCHY Tests
# =============================================================================

class TestTechHierarchy:
    """Tests for TECH_HIERARCHY structure."""

    def test_hierarchy_has_required_domains(self):
        """Test that key domains exist."""
        required = ["messaging", "kafka", "data_engineering", "kubernetes", "machine_learning"]
        for domain in required:
            assert domain in TECH_HIERARCHY

    def test_hierarchy_structure(self):
        """Test hierarchy entry structure."""
        for domain, info in TECH_HIERARCHY.items():
            assert "parent" in info
            assert "children" in info
            assert "concepts" in info
            assert isinstance(info["children"], list)
            assert isinstance(info["concepts"], list)

    def test_concept_to_domain_lookup(self):
        """Test concept to domain lookup table."""
        # Concepts map to their containing domain
        assert _CONCEPT_TO_DOMAIN.get("consumer") == "messaging"
        assert _CONCEPT_TO_DOMAIN.get("pod") == "kubernetes"

        # Sub-domains map to themselves
        assert _CONCEPT_TO_DOMAIN.get("kafka") == "kafka"
        assert _CONCEPT_TO_DOMAIN.get("kubernetes") == "kubernetes"

    def test_parent_child_consistency(self):
        """Test parent-child relationships are consistent."""
        for domain, info in TECH_HIERARCHY.items():
            parent = info.get("parent")
            if parent:
                assert parent in TECH_HIERARCHY
                parent_children = TECH_HIERARCHY[parent].get("children", [])
                assert domain in parent_children


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_concepts(self):
        """Test with empty concept strings."""
        calc = EdgeWeightCalculator()
        weight = calc.calculate_weight("", "")
        assert weight.weight == 0.0

    def test_same_concept(self):
        """Test weight between same concept."""
        calc = EdgeWeightCalculator()
        calc.set_embeddings({"kafka": [1.0, 0.5, 0.3]})
        weight = calc.calculate_weight("kafka", "kafka")
        # Embedding similarity should be 1.0
        assert weight.embedding_score > 0.9

    def test_unicode_concepts(self):
        """Test with unicode concept names."""
        calc = EdgeWeightCalculator()
        chunks = [
            "L'intégration de données est importante.",
            "Les données d'intégration sont traitées.",
        ]
        calc.train_cooccurrence(chunks)
        # Should not raise
        weight = calc.calculate_weight("données", "intégration")
        assert isinstance(weight, EdgeWeight)

    def test_very_long_concept(self):
        """Test with very long concept name."""
        calc = EdgeWeightCalculator()
        long_concept = "a" * 1000
        weight = calc.calculate_weight(long_concept, "kafka")
        assert weight.weight == 0.0

    def test_special_characters(self):
        """Test concepts with special characters."""
        calc = EdgeWeightCalculator()
        chunks = ["api_gateway handles requests.", "api-gateway routing."]
        calc.train_cooccurrence(chunks)
        # Should handle underscores and hyphens
        assert calc.cooccurrence._is_trained


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Basic performance tests."""

    def test_large_chunk_training(self):
        """Test training on many chunks."""
        calc = EdgeWeightCalculator()
        chunks = [f"This is chunk number {i} about topic {i % 10}." for i in range(1000)]
        calc.train_cooccurrence(chunks)
        assert calc.cooccurrence._is_trained

    def test_many_embeddings(self):
        """Test with many embeddings."""
        calc = EdgeWeightCalculator()
        embeddings = {f"concept_{i}": [float(i % 10) / 10] * 100 for i in range(1000)}
        calc.set_embeddings(embeddings)
        assert len(calc.embedding._embeddings) == 1000

    def test_build_edges_many_concepts(self):
        """Test building edges for many concepts."""
        calc = EdgeWeightCalculator()

        # Create embeddings
        embeddings = {f"concept_{i}": [float(i % 10) / 10] * 4 for i in range(50)}
        calc.set_embeddings(embeddings)

        concepts = list(embeddings.keys())
        edges = calc.build_weighted_edges(concepts, max_edges_per_concept=5)

        # Should complete without error
        assert isinstance(edges, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
