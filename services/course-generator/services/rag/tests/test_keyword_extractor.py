"""
Unit Tests for KeywordExtractor

Tests keyword extraction, coverage computation, and similarity scoring.
"""

import pytest
from ..algorithms.keyword_extractor import (
    KeywordExtractor,
    extract_keywords,
    get_keyword_extractor,
)


class TestKeywordExtractor:
    """Tests for KeywordExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a default extractor."""
        return KeywordExtractor()

    # ==========================================================================
    # Basic Extraction Tests
    # ==========================================================================

    def test_extract_basic_english(self, extractor):
        """Test basic English keyword extraction."""
        text = "Apache Kafka is a distributed streaming platform"
        keywords = extractor.extract(text)

        assert "apache" in keywords
        assert "kafka" in keywords
        assert "distributed" in keywords
        assert "streaming" in keywords
        assert "platform" in keywords
        # Stopwords should be excluded
        assert "is" not in keywords
        assert "a" not in keywords

    def test_extract_basic_french(self, extractor):
        """Test basic French keyword extraction."""
        text = "Apache Kafka est une plateforme de streaming distribuée"
        keywords = extractor.extract(text)

        assert "apache" in keywords
        assert "kafka" in keywords
        assert "plateforme" in keywords
        assert "streaming" in keywords
        # French stopwords should be excluded
        assert "est" not in keywords
        assert "une" not in keywords
        assert "de" not in keywords

    def test_extract_empty_text(self, extractor):
        """Test extraction from empty text."""
        assert extractor.extract("") == []
        assert extractor.extract(None) == []

    def test_extract_only_stopwords(self, extractor):
        """Test extraction from text with only stopwords."""
        text = "the a an is are was were"
        keywords = extractor.extract(text)
        assert keywords == []

    def test_extract_short_words_excluded(self, extractor):
        """Test that words shorter than min_length are excluded."""
        text = "I am at it to go do if"
        keywords = extractor.extract(text)
        assert keywords == []

    def test_extract_preserves_order(self, extractor):
        """Test that extraction preserves first occurrence order."""
        text = "Kafka streams Kafka producer Kafka consumer"
        keywords = extractor.extract(text)

        # First occurrence of each keyword
        assert keywords.index("kafka") < keywords.index("streams")
        assert keywords.index("streams") < keywords.index("producer")
        assert keywords.index("producer") < keywords.index("consumer")

    def test_extract_unique_keywords(self, extractor):
        """Test that duplicates are removed."""
        text = "Kafka Kafka Kafka streams streams"
        keywords = extractor.extract(text)

        assert keywords.count("kafka") == 1
        assert keywords.count("streams") == 1

    def test_extract_with_max_keywords(self, extractor):
        """Test extraction with max_keywords limit."""
        text = "one two three four five six seven eight nine ten"
        keywords = extractor.extract(text, max_keywords=5)

        assert len(keywords) == 5
        assert keywords[0] == "one"

    def test_extract_accented_characters(self, extractor):
        """Test extraction with accented characters."""
        text = "résumé café naïve"
        keywords = extractor.extract(text)

        assert "résumé" in keywords
        assert "café" in keywords
        assert "naïve" in keywords

    def test_extract_mixed_case(self, extractor):
        """Test that extraction is case-insensitive."""
        text = "Kafka KAFKA kafka KaFkA"
        keywords = extractor.extract(text)

        assert len(keywords) == 1
        assert "kafka" in keywords

    # ==========================================================================
    # Custom Stopwords Tests
    # ==========================================================================

    def test_custom_stopwords(self):
        """Test extraction with custom stopwords."""
        custom = {"kafka", "apache"}
        extractor = KeywordExtractor(custom_stopwords=custom)

        text = "Apache Kafka streaming platform"
        keywords = extractor.extract(text)

        assert "apache" not in keywords
        assert "kafka" not in keywords
        assert "streaming" in keywords
        assert "platform" in keywords

    def test_custom_min_length(self):
        """Test extraction with custom min_length."""
        extractor = KeywordExtractor(min_length=5)

        text = "cat dog elephant"
        keywords = extractor.extract(text)

        assert "cat" not in keywords
        assert "dog" not in keywords
        assert "elephant" in keywords

    # ==========================================================================
    # Coverage Tests
    # ==========================================================================

    def test_compute_coverage_full(self, extractor):
        """Test coverage when all query keywords are found."""
        query = "Apache Kafka streaming"
        document = "Apache Kafka is a distributed streaming platform"

        coverage = extractor.compute_coverage(query, document)
        assert coverage == 1.0

    def test_compute_coverage_partial(self, extractor):
        """Test coverage when some keywords are found."""
        query = "Apache Kafka Redis"
        document = "Apache Kafka is a streaming platform"

        coverage = extractor.compute_coverage(query, document)
        # 2 out of 3 keywords found
        assert 0.6 <= coverage <= 0.7

    def test_compute_coverage_none(self, extractor):
        """Test coverage when no keywords are found."""
        query = "Redis MongoDB"
        document = "Apache Kafka is a streaming platform"

        coverage = extractor.compute_coverage(query, document)
        assert coverage == 0.0

    def test_compute_coverage_empty_query(self, extractor):
        """Test coverage with empty query."""
        coverage = extractor.compute_coverage("", "Some document text")
        assert coverage == 0.0

    def test_compute_coverage_empty_document(self, extractor):
        """Test coverage with empty document."""
        coverage = extractor.compute_coverage("Apache Kafka", "")
        assert coverage == 0.0

    def test_compute_coverage_truncates_long_documents(self, extractor):
        """Test that coverage truncates very long documents."""
        query = "Kafka"
        # Document longer than max_doc_chars
        document = "x " * 20000 + "Kafka"

        # Should still find Kafka if within limit
        coverage = extractor.compute_coverage(query, document, max_doc_chars=50000)
        assert coverage == 1.0

    # ==========================================================================
    # Similarity Tests
    # ==========================================================================

    def test_compute_similarity_identical(self, extractor):
        """Test similarity between identical texts."""
        text = "Apache Kafka streaming platform"
        similarity = extractor.compute_similarity(text, text)
        assert similarity == 1.0

    def test_compute_similarity_different(self, extractor):
        """Test similarity between completely different texts."""
        text1 = "Apache Kafka streaming"
        text2 = "Python Django framework"

        similarity = extractor.compute_similarity(text1, text2)
        assert similarity == 0.0

    def test_compute_similarity_partial(self, extractor):
        """Test similarity between partially overlapping texts."""
        text1 = "Apache Kafka streaming platform"
        text2 = "Apache Spark streaming engine"

        similarity = extractor.compute_similarity(text1, text2)
        # "apache" and "streaming" overlap
        assert 0.4 <= similarity <= 0.6

    def test_compute_similarity_empty(self, extractor):
        """Test similarity with empty text returns default."""
        similarity = extractor.compute_similarity("", "Some text")
        assert similarity == 0.5  # Default for no keywords

    # ==========================================================================
    # Module-level Functions Tests
    # ==========================================================================

    def test_extract_keywords_function(self):
        """Test the module-level extract_keywords function."""
        keywords = extract_keywords("Apache Kafka streaming")
        assert "apache" in keywords
        assert "kafka" in keywords

    def test_get_keyword_extractor_singleton(self):
        """Test that get_keyword_extractor returns same instance."""
        ext1 = get_keyword_extractor()
        ext2 = get_keyword_extractor()
        assert ext1 is ext2


class TestKeywordExtractorEdgeCases:
    """Edge case tests for KeywordExtractor."""

    @pytest.fixture
    def extractor(self):
        return KeywordExtractor()

    def test_extract_with_numbers(self, extractor):
        """Test extraction ignores numbers."""
        text = "Python 3.11 released in 2023"
        keywords = extractor.extract(text)

        assert "python" in keywords
        assert "released" in keywords
        assert "3.11" not in keywords
        assert "2023" not in keywords

    def test_extract_with_special_characters(self, extractor):
        """Test extraction handles special characters."""
        text = "Apache-Kafka, Docker/Kubernetes!"
        keywords = extractor.extract(text)

        # Words should be extracted despite special chars
        assert any("apache" in kw or "kafka" in kw for kw in keywords)
        assert any("docker" in kw or "kubernetes" in kw for kw in keywords)

    def test_extract_technical_terms(self, extractor):
        """Test extraction of technical terms."""
        text = "microservices API REST GraphQL gRPC"
        keywords = extractor.extract(text)

        assert "microservices" in keywords
        assert "graphql" in keywords
        assert "grpc" in keywords

    def test_coverage_case_insensitive(self, extractor):
        """Test that coverage is case-insensitive."""
        query = "KAFKA"
        document = "kafka is a streaming platform"

        coverage = extractor.compute_coverage(query, document)
        assert coverage == 1.0
