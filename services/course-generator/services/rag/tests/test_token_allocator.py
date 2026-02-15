"""
Unit Tests for TokenAllocator

Tests token budget allocation based on document relevance scores.
"""

import pytest
from ..algorithms.token_allocator import (
    TokenAllocator,
    allocate_tokens,
    get_token_allocator,
)
from ..models.scoring import DocumentRelevanceScore


class TestTokenAllocator:
    """Tests for TokenAllocator class."""

    @pytest.fixture
    def allocator(self):
        """Create default allocator."""
        return TokenAllocator()

    @pytest.fixture
    def sample_scores(self):
        """Create sample document scores."""
        return [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="high_relevance.pdf",
                final_score=0.9,
            ),
            DocumentRelevanceScore(
                document_id="doc2",
                filename="medium_relevance.pdf",
                final_score=0.6,
            ),
            DocumentRelevanceScore(
                document_id="doc3",
                filename="low_relevance.pdf",
                final_score=0.3,
            ),
        ]

    # ==========================================================================
    # Basic Allocation Tests
    # ==========================================================================

    def test_allocate_proportional(self, allocator, sample_scores):
        """Test that allocation is proportional to scores."""
        total_budget = 1000
        result = allocator.allocate(sample_scores, total_budget)

        # Higher score should get more tokens
        high_score = next(s for s in result if s.document_id == "doc1")
        medium_score = next(s for s in result if s.document_id == "doc2")
        low_score = next(s for s in result if s.document_id == "doc3")

        assert high_score.allocated_tokens > medium_score.allocated_tokens
        assert medium_score.allocated_tokens > low_score.allocated_tokens

    def test_allocate_respects_budget(self, allocator, sample_scores):
        """Test that total allocation doesn't exceed budget."""
        total_budget = 1000
        result = allocator.allocate(sample_scores, total_budget)

        total_allocated = sum(s.allocated_tokens for s in result if s.allocated_tokens > 0)
        assert total_allocated <= total_budget

    def test_allocate_sets_contribution_percentage(self, allocator, sample_scores):
        """Test that contribution percentages are set."""
        result = allocator.allocate(sample_scores, 1000)

        for score in result:
            if score.allocated_tokens > 0:
                assert score.contribution_percentage > 0
                assert score.contribution_percentage <= 100

    def test_allocate_excludes_below_threshold(self, allocator):
        """Test that documents below threshold are excluded."""
        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="above_threshold.pdf",
                final_score=0.5,  # Above default 0.25
            ),
            DocumentRelevanceScore(
                document_id="doc2",
                filename="below_threshold.pdf",
                final_score=0.1,  # Below default 0.25
            ),
        ]

        result = allocator.allocate(scores, 1000)

        above = next(s for s in result if s.document_id == "doc1")
        below = next(s for s in result if s.document_id == "doc2")

        assert above.allocated_tokens > 0
        assert below.allocated_tokens == 0
        assert below.contribution_percentage == 0.0

    def test_allocate_empty_list(self, allocator):
        """Test allocation with empty list."""
        result = allocator.allocate([], 1000)
        assert result == []

    def test_allocate_all_below_threshold(self, allocator):
        """Test when all documents are below threshold."""
        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="low1.pdf",
                final_score=0.1,
            ),
            DocumentRelevanceScore(
                document_id="doc2",
                filename="low2.pdf",
                final_score=0.2,
            ),
        ]

        result = allocator.allocate(scores, 1000)

        # All should be excluded
        for score in result:
            assert score.allocated_tokens == 0

    # ==========================================================================
    # Minimum Allocation Tests
    # ==========================================================================

    def test_allocate_minimum_guaranteed(self, allocator):
        """Test that minimum allocation percentage is guaranteed."""
        # One high score, one low but above threshold
        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="high.pdf",
                final_score=0.9,
            ),
            DocumentRelevanceScore(
                document_id="doc2",
                filename="just_above.pdf",
                final_score=0.26,  # Just above threshold
            ),
        ]

        result = allocator.allocate(scores, 1000)

        low_score = next(s for s in result if s.document_id == "doc2")

        # Should get at least min_allocation_percent (10%)
        assert low_score.allocated_tokens >= 100  # 10% of 1000

    def test_allocate_minimum_causes_scaling(self, allocator):
        """Test that minimum allocations are scaled if they exceed budget."""
        # Many documents that each want minimum allocation
        scores = [
            DocumentRelevanceScore(
                document_id=f"doc{i}",
                filename=f"doc{i}.pdf",
                final_score=0.3,  # Just above threshold
            )
            for i in range(15)  # 15 docs each wanting 10% = 150%
        ]

        result = allocator.allocate(scores, 1000)

        total_allocated = sum(s.allocated_tokens for s in result)
        assert total_allocated <= 1000

    # ==========================================================================
    # Custom Configuration Tests
    # ==========================================================================

    def test_custom_threshold(self):
        """Test with custom relevance threshold."""
        allocator = TokenAllocator(min_relevance_threshold=0.5)

        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="above.pdf",
                final_score=0.6,
            ),
            DocumentRelevanceScore(
                document_id="doc2",
                filename="below.pdf",
                final_score=0.4,  # Below custom threshold
            ),
        ]

        result = allocator.allocate(scores, 1000)

        above = next(s for s in result if s.document_id == "doc1")
        below = next(s for s in result if s.document_id == "doc2")

        assert above.allocated_tokens > 0
        assert below.allocated_tokens == 0

    def test_custom_min_allocation(self):
        """Test with custom minimum allocation percentage."""
        allocator = TokenAllocator(min_allocation_percent=0.20)  # 20%

        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="doc1.pdf",
                final_score=0.9,
            ),
            DocumentRelevanceScore(
                document_id="doc2",
                filename="doc2.pdf",
                final_score=0.3,
            ),
        ]

        result = allocator.allocate(scores, 1000)

        low_score = next(s for s in result if s.document_id == "doc2")

        # Should get at least 20% before scaling
        assert low_score.contribution_percentage >= 15  # May be scaled down

    # ==========================================================================
    # Summary Tests
    # ==========================================================================

    def test_get_allocation_summary(self, allocator, sample_scores):
        """Test allocation summary generation."""
        allocator.allocate(sample_scores, 1000)
        summary = allocator.get_allocation_summary(sample_scores)

        assert summary["total_documents"] == 3
        assert summary["documents_included"] >= 1
        assert summary["documents_excluded"] >= 0
        assert summary["total_tokens_allocated"] > 0
        assert "allocations" in summary

    def test_summary_allocations_have_required_fields(self, allocator, sample_scores):
        """Test that summary allocations have all required fields."""
        allocator.allocate(sample_scores, 1000)
        summary = allocator.get_allocation_summary(sample_scores)

        for alloc in summary["allocations"]:
            assert "filename" in alloc
            assert "score" in alloc
            assert "tokens" in alloc
            assert "percentage" in alloc

    # ==========================================================================
    # Module-level Functions Tests
    # ==========================================================================

    def test_allocate_tokens_function(self):
        """Test module-level allocate_tokens function."""
        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="test.pdf",
                final_score=0.8,
            ),
        ]

        result = allocate_tokens(scores, 1000)
        assert result[0].allocated_tokens > 0

    def test_get_token_allocator_singleton(self):
        """Test that get_token_allocator returns same instance."""
        alloc1 = get_token_allocator()
        alloc2 = get_token_allocator()
        assert alloc1 is alloc2


class TestTokenAllocatorEdgeCases:
    """Edge case tests for TokenAllocator."""

    @pytest.fixture
    def allocator(self):
        return TokenAllocator()

    def test_single_document(self, allocator):
        """Test allocation with single document."""
        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="only.pdf",
                final_score=0.8,
            ),
        ]

        result = allocator.allocate(scores, 1000)

        assert result[0].allocated_tokens == 1000
        assert result[0].contribution_percentage == 100.0

    def test_equal_scores(self, allocator):
        """Test allocation when all scores are equal."""
        scores = [
            DocumentRelevanceScore(
                document_id=f"doc{i}",
                filename=f"doc{i}.pdf",
                final_score=0.5,
            )
            for i in range(4)
        ]

        result = allocator.allocate(scores, 1000)

        # All should get roughly equal allocation
        allocations = [s.allocated_tokens for s in result]
        assert max(allocations) - min(allocations) <= 50  # Small variance

    def test_very_small_budget(self, allocator):
        """Test allocation with very small budget."""
        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="doc1.pdf",
                final_score=0.8,
            ),
            DocumentRelevanceScore(
                document_id="doc2",
                filename="doc2.pdf",
                final_score=0.6,
            ),
        ]

        result = allocator.allocate(scores, 10)

        total = sum(s.allocated_tokens for s in result)
        assert total <= 10

    def test_zero_budget(self, allocator):
        """Test allocation with zero budget."""
        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="doc1.pdf",
                final_score=0.8,
            ),
        ]

        result = allocator.allocate(scores, 0)
        assert result[0].allocated_tokens == 0

    def test_score_at_threshold(self, allocator):
        """Test document exactly at threshold."""
        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="at_threshold.pdf",
                final_score=0.25,  # Exactly at default threshold
            ),
        ]

        result = allocator.allocate(scores, 1000)

        # Should be included (>= threshold)
        assert result[0].allocated_tokens > 0
