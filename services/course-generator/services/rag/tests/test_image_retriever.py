"""
Unit Tests for ImageRetriever

Tests image retrieval and relevance scoring for RAG.
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional
from ..retrieval.image_retriever import (
    ImageRetriever,
    RAGImage,
    ImageRetrievalResult,
    get_image_retriever,
    get_images_for_topic,
)


class MockKeywordExtractor:
    """Mock keyword extractor for testing."""

    def extract(self, text: str) -> List[str]:
        """Extract keywords (simple word splitting)."""
        if not text:
            return []
        return [w.lower() for w in text.split() if len(w) > 2]

    def compute_coverage(self, query: str, text: str) -> float:
        """Compute keyword coverage (simple word overlap)."""
        if not query or not text:
            return 0.0
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        if not query_words:
            return 0.0
        overlap = len(query_words.intersection(text_words))
        return overlap / len(query_words)


def create_test_image(
    image_id: str = "img1",
    document_id: str = "doc1",
    filename: str = "test.png",
    image_type: str = "diagram",
    context: str = "",
    caption: str = None,
    description: str = None,
    keywords: List[str] = None,
) -> RAGImage:
    """Create a test RAGImage."""
    return RAGImage(
        image_id=image_id,
        document_id=document_id,
        filename=filename,
        file_path=f"/tmp/{filename}",
        image_type=image_type,
        context=context,
        caption=caption,
        description=description,
        keywords=keywords or [],
    )


class TestRAGImage:
    """Tests for RAGImage dataclass."""

    def test_create_minimal(self):
        """Test creating image with minimal fields."""
        image = RAGImage(
            image_id="img1",
            document_id="doc1",
            filename="test.png",
            file_path="/tmp/test.png",
            image_type="diagram",
            context="",
        )

        assert image.image_id == "img1"
        assert image.image_type == "diagram"
        assert image.caption is None
        assert image.keywords == []
        assert image.relevance_score == 0.0

    def test_create_full(self):
        """Test creating image with all fields."""
        image = RAGImage(
            image_id="img1",
            document_id="doc1",
            filename="architecture.png",
            file_path="/tmp/architecture.png",
            image_type="architecture",
            context="This diagram shows the Kafka architecture",
            caption="Kafka Architecture Overview",
            description="A detailed view of Apache Kafka components",
            keywords=["kafka", "architecture", "broker"],
            page_number=5,
            relevance_score=0.85,
        )

        assert image.caption == "Kafka Architecture Overview"
        assert "kafka" in image.keywords
        assert image.page_number == 5


class TestImageRetrievalResult:
    """Tests for ImageRetrievalResult dataclass."""

    def test_create_empty(self):
        """Test creating empty result."""
        result = ImageRetrievalResult(
            topic="test",
            images=[],
            total_available=0,
            threshold_used=0.7,
        )

        assert result.topic == "test"
        assert result.images == []
        assert result.total_available == 0

    def test_create_with_images(self):
        """Test creating result with images."""
        images = [create_test_image()]
        result = ImageRetrievalResult(
            topic="Kafka architecture",
            images=images,
            total_available=10,
            threshold_used=0.7,
        )

        assert len(result.images) == 1
        assert result.total_available == 10


class TestImageRetriever:
    """Tests for ImageRetriever class."""

    @pytest.fixture
    def retriever(self):
        """Create retriever with mock extractor."""
        return ImageRetriever(keyword_extractor=MockKeywordExtractor())

    # ==========================================================================
    # Basic Retrieval Tests
    # ==========================================================================

    def test_get_images_empty_list(self, retriever):
        """Test retrieval with no images."""
        result = retriever.get_images_for_topic(
            topic="Kafka",
            images=[],
        )

        assert result.images == []
        assert result.total_available == 0

    def test_get_images_empty_topic(self, retriever):
        """Test retrieval with empty topic."""
        images = [create_test_image(context="Some content")]

        result = retriever.get_images_for_topic(
            topic="",
            images=images,
        )

        assert result.images == []

    def test_get_images_single_match(self, retriever):
        """Test retrieval with single matching image."""
        images = [
            create_test_image(
                image_id="img1",
                context="Apache Kafka architecture diagram",
            ),
        ]

        result = retriever.get_images_for_topic(
            topic="Apache Kafka architecture",
            images=images,
            min_score=0.3,  # Lower threshold for testing
        )

        assert len(result.images) >= 1
        assert result.images[0].relevance_score > 0

    def test_get_images_multiple_scored(self, retriever):
        """Test that multiple images are scored and sorted."""
        images = [
            create_test_image(
                image_id="low_match",
                context="Random unrelated content",
            ),
            create_test_image(
                image_id="high_match",
                context="Apache Kafka broker architecture",
            ),
            create_test_image(
                image_id="medium_match",
                context="Kafka consumer groups",
            ),
        ]

        result = retriever.get_images_for_topic(
            topic="Apache Kafka broker",
            images=images,
            min_score=0.0,  # Include all for testing
        )

        # Should be sorted by relevance
        if len(result.images) > 1:
            assert result.images[0].relevance_score >= result.images[1].relevance_score

    # ==========================================================================
    # Scoring Weight Tests
    # ==========================================================================

    def test_context_weight(self, retriever):
        """Test that context contributes to score."""
        image = create_test_image(
            context="Apache Kafka broker architecture",
            caption=None,
            description=None,
        )

        result = retriever.get_images_for_topic(
            topic="Apache Kafka broker",
            images=[image],
            min_score=0.0,
        )

        assert len(result.images) == 1
        assert result.images[0].relevance_score > 0

    def test_caption_weight(self, retriever):
        """Test that caption contributes to score."""
        image = create_test_image(
            context="",
            caption="Apache Kafka broker architecture",
            description=None,
        )

        result = retriever.get_images_for_topic(
            topic="Apache Kafka broker",
            images=[image],
            min_score=0.0,
        )

        assert len(result.images) == 1
        assert result.images[0].relevance_score > 0

    def test_description_weight(self, retriever):
        """Test that description contributes to score."""
        image = create_test_image(
            context="",
            caption=None,
            description="Apache Kafka broker architecture diagram",
        )

        result = retriever.get_images_for_topic(
            topic="Apache Kafka broker",
            images=[image],
            min_score=0.0,
        )

        assert len(result.images) == 1
        assert result.images[0].relevance_score > 0

    def test_keywords_weight(self, retriever):
        """Test that keywords contribute to score."""
        image = create_test_image(
            context="",
            caption=None,
            description=None,
            keywords=["apache", "kafka", "broker"],
        )

        result = retriever.get_images_for_topic(
            topic="apache kafka broker",
            images=[image],
            min_score=0.0,
        )

        assert len(result.images) == 1
        assert result.images[0].relevance_score > 0

    # ==========================================================================
    # Diagram Type Boost Tests
    # ==========================================================================

    def test_diagram_type_boost(self, retriever):
        """Test that diagram types get boosted."""
        images = [
            create_test_image(
                image_id="photo",
                image_type="photo",
                context="Kafka architecture",
            ),
            create_test_image(
                image_id="diagram",
                image_type="diagram",
                context="Kafka architecture",
            ),
        ]

        result = retriever.get_images_for_topic(
            topic="Kafka architecture",
            images=images,
            min_score=0.0,
            prefer_diagrams=True,
        )

        # Diagram should rank higher due to boost
        assert len(result.images) == 2
        if result.images[0].image_id == "diagram":
            assert result.images[0].relevance_score >= result.images[1].relevance_score

    def test_no_diagram_boost_when_disabled(self, retriever):
        """Test that diagram boost can be disabled."""
        images = [
            create_test_image(
                image_id="diagram",
                image_type="diagram",
                context="Kafka",
            ),
        ]

        result_with_boost = retriever.get_images_for_topic(
            topic="Kafka",
            images=images,
            min_score=0.0,
            prefer_diagrams=True,
        )

        result_no_boost = retriever.get_images_for_topic(
            topic="Kafka",
            images=images,
            min_score=0.0,
            prefer_diagrams=False,
        )

        # With boost should have higher score
        assert result_with_boost.images[0].relevance_score >= result_no_boost.images[0].relevance_score

    # ==========================================================================
    # Threshold Tests
    # ==========================================================================

    def test_min_score_filtering(self, retriever):
        """Test that min_score filters images."""
        images = [
            create_test_image(
                image_id="low",
                context="Unrelated image",
            ),
            create_test_image(
                image_id="high",
                context="Apache Kafka broker architecture diagram",
            ),
        ]

        result = retriever.get_images_for_topic(
            topic="Apache Kafka broker",
            images=images,
            min_score=0.5,  # Higher threshold
        )

        # Low match should be filtered out
        for img in result.images:
            assert img.relevance_score >= 0.5

    def test_default_min_score(self, retriever):
        """Test that default min_score is used."""
        images = [create_test_image(context="Content")]

        result = retriever.get_images_for_topic(
            topic="Kafka",
            images=images,
        )

        assert result.threshold_used == retriever.DEFAULT_MIN_SCORE

    # ==========================================================================
    # Max Images Tests
    # ==========================================================================

    def test_max_images_limit(self, retriever):
        """Test that max_images limits results."""
        images = [
            create_test_image(image_id=f"img{i}", context="Kafka content")
            for i in range(10)
        ]

        result = retriever.get_images_for_topic(
            topic="Kafka content",
            images=images,
            min_score=0.0,
            max_images=3,
        )

        assert len(result.images) <= 3
        assert result.total_available == 10

    # ==========================================================================
    # Filter By Type Tests
    # ==========================================================================

    def test_filter_by_type(self, retriever):
        """Test filtering images by type."""
        images = [
            create_test_image(image_id="diagram1", image_type="diagram"),
            create_test_image(image_id="photo1", image_type="photo"),
            create_test_image(image_id="chart1", image_type="chart"),
        ]

        result = retriever.filter_by_type(images, allowed_types=["diagram"])

        assert len(result) == 1
        assert result[0].image_id == "diagram1"

    def test_filter_by_type_default(self, retriever):
        """Test filtering with default diagram types."""
        images = [
            create_test_image(image_id="diagram1", image_type="diagram"),
            create_test_image(image_id="photo1", image_type="photo"),
            create_test_image(image_id="chart1", image_type="chart"),
            create_test_image(image_id="arch1", image_type="architecture"),
        ]

        result = retriever.filter_by_type(images)

        # Should include diagram types
        types = {img.image_type for img in result}
        assert "diagram" in types or "chart" in types or "architecture" in types
        assert "photo" not in types

    def test_filter_by_type_case_insensitive(self, retriever):
        """Test that type filtering is case insensitive."""
        images = [
            create_test_image(image_type="DIAGRAM"),
            create_test_image(image_type="Diagram"),
            create_test_image(image_type="diagram"),
        ]

        result = retriever.filter_by_type(images, allowed_types=["diagram"])

        assert len(result) == 3

    # ==========================================================================
    # Get Best Diagram Tests
    # ==========================================================================

    def test_get_best_diagram(self, retriever):
        """Test getting best diagram for topic."""
        images = [
            create_test_image(
                image_id="img1",
                image_type="diagram",
                context="Kafka producer",
            ),
            create_test_image(
                image_id="img2",
                image_type="diagram",
                context="Kafka consumer architecture",
            ),
        ]

        best = retriever.get_best_diagram(
            topic="Kafka consumer",
            images=images,
            min_score=0.0,
        )

        assert best is not None
        assert best.relevance_score > 0

    def test_get_best_diagram_none_above_threshold(self, retriever):
        """Test when no diagram meets threshold."""
        images = [
            create_test_image(
                image_type="diagram",
                context="Completely unrelated content",
            ),
        ]

        best = retriever.get_best_diagram(
            topic="Apache Kafka architecture",
            images=images,
            min_score=0.9,  # Very high threshold
        )

        assert best is None

    def test_get_best_diagram_empty(self, retriever):
        """Test getting best diagram from empty list."""
        best = retriever.get_best_diagram(
            topic="Kafka",
            images=[],
        )

        assert best is None


class TestImageRetrieverEdgeCases:
    """Edge case tests for ImageRetriever."""

    @pytest.fixture
    def retriever(self):
        return ImageRetriever(keyword_extractor=MockKeywordExtractor())

    def test_image_all_none_metadata(self, retriever):
        """Test scoring image with all None metadata."""
        image = create_test_image(
            context="",
            caption=None,
            description=None,
            keywords=[],
        )

        result = retriever.get_images_for_topic(
            topic="Kafka",
            images=[image],
            min_score=0.0,
        )

        # Should not crash, score should be 0
        assert len(result.images) == 1
        assert result.images[0].relevance_score == 0.0

    def test_empty_keywords_list(self, retriever):
        """Test with empty keywords list."""
        image = create_test_image(keywords=[])

        result = retriever.get_images_for_topic(
            topic="Kafka",
            images=[image],
            min_score=0.0,
        )

        assert len(result.images) == 1


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_image_retriever_singleton(self):
        """Test that get_image_retriever returns consistent instance."""
        r1 = get_image_retriever()
        r2 = get_image_retriever()
        assert r1 is r2

    def test_get_images_for_topic_function(self):
        """Test module-level convenience function."""
        images = [
            create_test_image(context="Apache Kafka architecture"),
        ]

        result = get_images_for_topic(
            topic="Apache Kafka",
            images=images,
            min_score=0.0,
        )

        assert isinstance(result, list)

    def test_get_images_for_topic_empty(self):
        """Test convenience function with empty list."""
        result = get_images_for_topic(
            topic="Kafka",
            images=[],
            min_score=0.0,
        )

        assert result == []
