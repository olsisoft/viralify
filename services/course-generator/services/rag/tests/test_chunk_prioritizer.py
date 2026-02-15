"""
Unit Tests for ChunkPrioritizer

Tests chunk prioritization based on content importance markers.
"""

import pytest
from dataclasses import dataclass
from ..retrieval.chunk_prioritizer import (
    ChunkPrioritizer,
    PrioritizedChunk,
    prioritize_chunks,
    get_chunk_prioritizer,
)


@dataclass
class MockChunk:
    """Mock RAGChunkResult for testing."""
    content: str
    similarity_score: float = 0.5
    document_filename: str = "test.pdf"


class TestChunkPrioritizer:
    """Tests for ChunkPrioritizer class."""

    @pytest.fixture
    def prioritizer(self):
        """Create default prioritizer."""
        return ChunkPrioritizer()

    # ==========================================================================
    # Basic Prioritization Tests
    # ==========================================================================

    def test_prioritize_empty_list(self, prioritizer):
        """Test prioritizing empty list."""
        result = prioritizer.prioritize([])
        assert result == []

    def test_prioritize_single_chunk(self, prioritizer):
        """Test prioritizing single chunk."""
        chunk = MockChunk(content="Test content", similarity_score=0.8)

        result = prioritizer.prioritize([chunk])

        assert len(result) == 1
        assert result[0] is chunk

    def test_prioritize_sorts_by_similarity(self, prioritizer):
        """Test that chunks are sorted by similarity score."""
        chunks = [
            MockChunk(content="Low score", similarity_score=0.3),
            MockChunk(content="High score", similarity_score=0.9),
            MockChunk(content="Medium score", similarity_score=0.6),
        ]

        result = prioritizer.prioritize(chunks)

        # Should be sorted highest first
        assert result[0].similarity_score == 0.9
        assert result[1].similarity_score == 0.6
        assert result[2].similarity_score == 0.3

    # ==========================================================================
    # Definition Boost Tests
    # ==========================================================================

    def test_boost_definition_marker(self, prioritizer):
        """Test that definition marker boosts priority."""
        chunks = [
            MockChunk(
                content="Regular content",
                similarity_score=0.7,
            ),
            MockChunk(
                content="[contains: definition] Kafka is a distributed streaming platform",
                similarity_score=0.6,
            ),
        ]

        result = prioritizer.prioritize(chunks)

        # Definition chunk should be boosted above regular
        assert "definition" in result[0].content.lower()

    def test_boost_key_concept_marker(self, prioritizer):
        """Test that key concept marker boosts priority."""
        chunks = [
            MockChunk(
                content="Regular content",
                similarity_score=0.7,
            ),
            MockChunk(
                content="This is a key concept: partitions divide data",
                similarity_score=0.6,
            ),
        ]

        result = prioritizer.prioritize(chunks)

        assert "key concept" in result[0].content.lower()

    # ==========================================================================
    # Example Boost Tests
    # ==========================================================================

    def test_boost_example_marker(self, prioritizer):
        """Test that example marker boosts priority."""
        chunks = [
            MockChunk(
                content="Theory about Kafka",
                similarity_score=0.7,
            ),
            MockChunk(
                content="[contains: example] Here's how to create a topic",
                similarity_score=0.62,
            ),
        ]

        result = prioritizer.prioritize(chunks)

        assert "example" in result[0].content.lower()

    def test_boost_for_example_phrase(self, prioritizer):
        """Test boost for 'for example' phrase."""
        chunks = [
            MockChunk(
                content="General content",
                similarity_score=0.65,
            ),
            MockChunk(
                content="For example, you can configure retention like this",
                similarity_score=0.6,
            ),
        ]

        result = prioritizer.prioritize(chunks)

        assert "for example" in result[0].content.lower()

    # ==========================================================================
    # Code Boost Tests
    # ==========================================================================

    def test_boost_code_marker(self, prioritizer):
        """Test that code marker boosts priority."""
        chunks = [
            MockChunk(
                content="Description of the API",
                similarity_score=0.7,
            ),
            MockChunk(
                content="[content type: code] ```python\nproducer.send(msg)\n```",
                similarity_score=0.65,
            ),
        ]

        result = prioritizer.prioritize(chunks)

        assert "code" in result[0].content.lower()

    def test_boost_code_fence(self, prioritizer):
        """Test that code fence boosts priority."""
        chunks = [
            MockChunk(
                content="Some text description",
                similarity_score=0.7,
            ),
            MockChunk(
                content="Here's the code:\n```\nprint('hello')\n```",
                similarity_score=0.67,
            ),
        ]

        result = prioritizer.prioritize(chunks)

        assert "```" in result[0].content

    def test_boost_function_keyword(self, prioritizer):
        """Test that function keyword boosts priority."""
        chunks = [
            MockChunk(
                content="Description text",
                similarity_score=0.7,
            ),
            MockChunk(
                content="def process_message(msg): return msg.value",
                similarity_score=0.67,
            ),
        ]

        result = prioritizer.prioritize(chunks)

        assert "def " in result[0].content

    # ==========================================================================
    # Visual Boost Tests
    # ==========================================================================

    def test_boost_visual_marker(self, prioritizer):
        """Test that visual marker boosts priority."""
        chunks = [
            MockChunk(
                content="Text only content",
                similarity_score=0.7,
            ),
            MockChunk(
                content="[associated visuals: architecture diagram] The system consists of...",
                similarity_score=0.67,
            ),
        ]

        result = prioritizer.prioritize(chunks)

        assert "visuals" in result[0].content.lower()

    def test_boost_image_marker(self, prioritizer):
        """Test that image marker boosts priority."""
        chunks = [
            MockChunk(
                content="Plain text",
                similarity_score=0.7,
            ),
            MockChunk(
                content="[image: kafka-architecture.png] This diagram shows...",
                similarity_score=0.68,
            ),
        ]

        result = prioritizer.prioritize(chunks)

        assert "[image:" in result[0].content

    # ==========================================================================
    # Combined Boost Tests
    # ==========================================================================

    def test_multiple_boosts_stack(self, prioritizer):
        """Test that multiple boost types stack."""
        chunks = [
            MockChunk(
                content="Plain content",
                similarity_score=0.8,
            ),
            MockChunk(
                content="[contains: definition] [contains: example] def kafka_example(): pass",
                similarity_score=0.5,  # Lower base score
            ),
        ]

        result = prioritizer.prioritize(chunks)

        # Multiple boosts should overcome lower similarity
        assert "definition" in result[0].content.lower()

    # ==========================================================================
    # Prioritize With Scores Tests
    # ==========================================================================

    def test_prioritize_with_scores(self, prioritizer):
        """Test prioritize_with_scores returns PrioritizedChunk objects."""
        chunks = [
            MockChunk(
                content="[contains: definition] Test definition",
                similarity_score=0.6,
            ),
        ]

        result = prioritizer.prioritize_with_scores(chunks)

        assert len(result) == 1
        assert isinstance(result[0], PrioritizedChunk)
        assert result[0].chunk is chunks[0]
        assert result[0].priority_score > 0.6  # Boosted
        assert "definition" in result[0].boost_reasons

    def test_prioritize_with_scores_empty(self, prioritizer):
        """Test prioritize_with_scores with empty list."""
        result = prioritizer.prioritize_with_scores([])
        assert result == []

    def test_boost_reasons_tracked(self, prioritizer):
        """Test that boost reasons are tracked correctly."""
        chunk = MockChunk(
            content="[contains: definition] [contains: example] ```code```",
            similarity_score=0.5,
        )

        result = prioritizer.prioritize_with_scores([chunk])

        reasons = result[0].boost_reasons
        assert "definition" in reasons
        assert "example" in reasons
        assert "code" in reasons

    # ==========================================================================
    # Boost Summary Tests
    # ==========================================================================

    def test_get_boost_summary(self, prioritizer):
        """Test boost summary generation."""
        chunks = [
            MockChunk(content="[contains: definition] Def", similarity_score=0.5),
            MockChunk(content="[contains: example] Ex", similarity_score=0.5),
            MockChunk(content="Plain content", similarity_score=0.5),
        ]

        prioritized = prioritizer.prioritize_with_scores(chunks)
        summary = prioritizer.get_boost_summary(prioritized)

        assert summary["total_chunks"] == 3
        assert summary["boosted_chunks"] == 2
        assert summary["boost_counts"]["definition"] == 1
        assert summary["boost_counts"]["example"] == 1

    def test_get_boost_summary_empty(self, prioritizer):
        """Test boost summary with no boosts."""
        chunks = [
            MockChunk(content="Plain content 1", similarity_score=0.5),
            MockChunk(content="Plain content 2", similarity_score=0.5),
        ]

        prioritized = prioritizer.prioritize_with_scores(chunks)
        summary = prioritizer.get_boost_summary(prioritized)

        assert summary["total_chunks"] == 2
        assert summary["boosted_chunks"] == 0

    # ==========================================================================
    # Module-level Functions Tests
    # ==========================================================================

    def test_prioritize_chunks_function(self):
        """Test module-level prioritize_chunks function."""
        chunks = [
            MockChunk(content="Low", similarity_score=0.3),
            MockChunk(content="High", similarity_score=0.9),
        ]

        result = prioritize_chunks(chunks)

        assert result[0].similarity_score == 0.9

    def test_get_chunk_prioritizer_singleton(self):
        """Test singleton behavior."""
        p1 = get_chunk_prioritizer()
        p2 = get_chunk_prioritizer()
        assert p1 is p2


class TestChunkPrioritizerEdgeCases:
    """Edge case tests for ChunkPrioritizer."""

    @pytest.fixture
    def prioritizer(self):
        return ChunkPrioritizer()

    def test_case_insensitive_markers(self, prioritizer):
        """Test that markers are case-insensitive."""
        chunks = [
            MockChunk(content="Plain", similarity_score=0.7),
            MockChunk(content="[CONTAINS: DEFINITION] Upper case", similarity_score=0.6),
        ]

        result = prioritizer.prioritize(chunks)

        # Should still boost uppercase markers
        assert "DEFINITION" in result[0].content

    def test_partial_marker_no_boost(self, prioritizer):
        """Test that partial markers don't trigger boost."""
        chunks = [
            MockChunk(content="definitions are important", similarity_score=0.6),
            MockChunk(content="No marker here", similarity_score=0.7),
        ]

        result = prioritizer.prioritize(chunks)

        # "definitions" shouldn't match "[contains: definition"
        assert result[0].content == "No marker here"

    def test_empty_content(self, prioritizer):
        """Test handling of empty content."""
        chunks = [
            MockChunk(content="", similarity_score=0.5),
            MockChunk(content="Has content", similarity_score=0.5),
        ]

        result = prioritizer.prioritize(chunks)

        assert len(result) == 2

    def test_equal_scores_preserve_order(self, prioritizer):
        """Test that equal scores preserve original order."""
        chunks = [
            MockChunk(content="First", similarity_score=0.5),
            MockChunk(content="Second", similarity_score=0.5),
            MockChunk(content="Third", similarity_score=0.5),
        ]

        result = prioritizer.prioritize(chunks)

        # Python's sort is stable, so order should be preserved
        contents = [c.content for c in result]
        assert contents == ["First", "Second", "Third"]
