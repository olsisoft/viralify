"""
Unit Tests for Scoring Models

Tests DocumentRelevanceScore and WeightedRAGResult dataclasses.
"""

import pytest
from datetime import datetime
from ..models.scoring import DocumentRelevanceScore, WeightedRAGResult


class TestDocumentRelevanceScore:
    """Tests for DocumentRelevanceScore dataclass."""

    def test_create_minimal(self):
        """Test creating score with minimal fields."""
        score = DocumentRelevanceScore(
            document_id="doc123",
            filename="test.pdf",
        )

        assert score.document_id == "doc123"
        assert score.filename == "test.pdf"
        assert score.semantic_similarity == 0.0
        assert score.keyword_coverage == 0.0
        assert score.freshness_score == 1.0
        assert score.document_type_score == 1.0
        assert score.final_score == 0.0
        assert score.allocated_tokens == 0
        assert score.contribution_percentage == 0.0

    def test_create_full(self):
        """Test creating score with all fields."""
        now = datetime.utcnow()
        score = DocumentRelevanceScore(
            document_id="doc123",
            filename="test.pdf",
            semantic_similarity=0.85,
            keyword_coverage=0.70,
            freshness_score=0.90,
            document_type_score=1.0,
            final_score=0.82,
            matched_keywords=["kafka", "streaming"],
            document_type="pdf",
            created_at=now,
            allocated_tokens=500,
            contribution_percentage=25.0,
        )

        assert score.semantic_similarity == 0.85
        assert score.keyword_coverage == 0.70
        assert score.freshness_score == 0.90
        assert score.final_score == 0.82
        assert score.matched_keywords == ["kafka", "streaming"]
        assert score.document_type == "pdf"
        assert score.created_at == now
        assert score.allocated_tokens == 500
        assert score.contribution_percentage == 25.0

    def test_matched_keywords_default(self):
        """Test that matched_keywords defaults to empty list."""
        score = DocumentRelevanceScore(
            document_id="doc1",
            filename="test.pdf",
        )
        assert score.matched_keywords == []
        # Verify it's a new list, not shared
        score.matched_keywords.append("test")

        score2 = DocumentRelevanceScore(
            document_id="doc2",
            filename="test2.pdf",
        )
        assert score2.matched_keywords == []

    def test_repr(self):
        """Test string representation."""
        score = DocumentRelevanceScore(
            document_id="doc123",
            filename="test.pdf",
            final_score=0.85,
            allocated_tokens=500,
        )

        repr_str = repr(score)
        assert "test.pdf" in repr_str
        assert "0.85" in repr_str
        assert "500" in repr_str

    def test_score_immutability(self):
        """Test that scores can be modified (dataclass not frozen)."""
        score = DocumentRelevanceScore(
            document_id="doc1",
            filename="test.pdf",
            final_score=0.5,
        )

        score.final_score = 0.8
        score.allocated_tokens = 1000

        assert score.final_score == 0.8
        assert score.allocated_tokens == 1000


class TestWeightedRAGResult:
    """Tests for WeightedRAGResult dataclass."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample document scores."""
        return [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="doc1.pdf",
                final_score=0.9,
                allocated_tokens=600,
                contribution_percentage=60.0,
            ),
            DocumentRelevanceScore(
                document_id="doc2",
                filename="doc2.pdf",
                final_score=0.6,
                allocated_tokens=400,
                contribution_percentage=40.0,
            ),
        ]

    def test_create_minimal(self):
        """Test creating result with minimal fields."""
        result = WeightedRAGResult(
            combined_context="Test context",
            document_scores=[],
        )

        assert result.combined_context == "Test context"
        assert result.document_scores == []
        assert result.total_documents_provided == 0
        assert result.documents_included == 0
        assert result.documents_excluded == 0
        assert result.total_tokens_used == 0
        assert result.source_contributions == {}

    def test_create_full(self, sample_scores):
        """Test creating result with all fields."""
        result = WeightedRAGResult(
            combined_context="Combined content from sources",
            document_scores=sample_scores,
            total_documents_provided=5,
            documents_included=2,
            documents_excluded=3,
            total_tokens_used=1000,
            source_contributions={"doc1.pdf": 60.0, "doc2.pdf": 40.0},
        )

        assert len(result.document_scores) == 2
        assert result.total_documents_provided == 5
        assert result.documents_included == 2
        assert result.documents_excluded == 3
        assert result.total_tokens_used == 1000
        assert result.source_contributions["doc1.pdf"] == 60.0

    def test_get_top_sources_default(self, sample_scores):
        """Test get_top_sources with default n."""
        result = WeightedRAGResult(
            combined_context="test",
            document_scores=sample_scores,
            source_contributions={
                "doc1.pdf": 60.0,
                "doc2.pdf": 40.0,
            },
        )

        top = result.get_top_sources()
        assert len(top) == 2
        assert top[0] == "doc1.pdf"  # Highest contribution
        assert top[1] == "doc2.pdf"

    def test_get_top_sources_limited(self):
        """Test get_top_sources with limit."""
        result = WeightedRAGResult(
            combined_context="test",
            document_scores=[],
            source_contributions={
                "doc1.pdf": 50.0,
                "doc2.pdf": 30.0,
                "doc3.pdf": 20.0,
            },
        )

        top = result.get_top_sources(n=2)
        assert len(top) == 2
        assert top[0] == "doc1.pdf"
        assert top[1] == "doc2.pdf"

    def test_get_top_sources_empty(self):
        """Test get_top_sources with no contributions."""
        result = WeightedRAGResult(
            combined_context="test",
            document_scores=[],
            source_contributions={},
        )

        top = result.get_top_sources()
        assert top == []

    def test_repr(self, sample_scores):
        """Test string representation."""
        result = WeightedRAGResult(
            combined_context="test",
            document_scores=sample_scores,
            total_documents_provided=5,
            documents_included=2,
            total_tokens_used=1000,
        )

        repr_str = repr(result)
        assert "2/5" in repr_str or "included=2" in repr_str
        assert "1000" in repr_str

    def test_source_contributions_default(self):
        """Test that source_contributions defaults to empty dict."""
        result = WeightedRAGResult(
            combined_context="test",
            document_scores=[],
        )
        assert result.source_contributions == {}

        # Verify it's a new dict, not shared
        result.source_contributions["test.pdf"] = 50.0

        result2 = WeightedRAGResult(
            combined_context="test2",
            document_scores=[],
        )
        assert result2.source_contributions == {}


class TestScoringModelIntegration:
    """Integration tests for scoring models working together."""

    def test_scores_in_result(self):
        """Test DocumentRelevanceScore used in WeightedRAGResult."""
        scores = [
            DocumentRelevanceScore(
                document_id=f"doc{i}",
                filename=f"doc{i}.pdf",
                final_score=0.5 + i * 0.1,
                allocated_tokens=100 * (i + 1),
            )
            for i in range(3)
        ]

        result = WeightedRAGResult(
            combined_context="Combined content",
            document_scores=scores,
            documents_included=3,
            total_tokens_used=sum(s.allocated_tokens for s in scores),
        )

        assert len(result.document_scores) == 3
        assert result.total_tokens_used == 600  # 100 + 200 + 300

    def test_build_contributions_from_scores(self):
        """Test building source_contributions from scores."""
        scores = [
            DocumentRelevanceScore(
                document_id="doc1",
                filename="doc1.pdf",
                contribution_percentage=60.0,
            ),
            DocumentRelevanceScore(
                document_id="doc2",
                filename="doc2.pdf",
                contribution_percentage=40.0,
            ),
        ]

        contributions = {s.filename: s.contribution_percentage for s in scores}

        result = WeightedRAGResult(
            combined_context="test",
            document_scores=scores,
            source_contributions=contributions,
        )

        assert result.source_contributions["doc1.pdf"] == 60.0
        assert result.source_contributions["doc2.pdf"] == 40.0
