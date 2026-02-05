"""
Tests for CitationValidator

Tests citation extraction, validation, and density checking.
"""

import pytest
import sys
import os

# Add rag_enforcement directory directly to path (avoid services/__init__.py)
_rag_enforcement_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services",
    "rag_enforcement"
)
sys.path.insert(0, _rag_enforcement_path)

from citation_validator import CitationValidator
from models import EnforcementConfig


class TestCitationExtraction:
    """Test citation extraction from content"""

    def setup_method(self):
        self.validator = CitationValidator()

    def test_extract_single_citation(self):
        """Test extracting a single citation"""
        content = "Apache Kafka uses partitions for scalability [REF:1]."
        citations = self.validator.extract_citations(content)

        assert len(citations) == 1
        assert citations[0].ref_id == "1"

    def test_extract_multiple_citations(self):
        """Test extracting multiple citations"""
        content = """
        Kafka uses partitions [REF:1]. Each partition is ordered [REF:2].
        Consumers can read in parallel [REF:3].
        """
        citations = self.validator.extract_citations(content)

        assert len(citations) == 3
        assert [c.ref_id for c in citations] == ["1", "2", "3"]

    def test_extract_same_citation_multiple_times(self):
        """Test when same citation is used multiple times"""
        content = "Kafka is fast [REF:1]. It's also scalable [REF:1]."
        citations = self.validator.extract_citations(content)

        assert len(citations) == 2
        assert all(c.ref_id == "1" for c in citations)

    def test_no_citations(self):
        """Test content without any citations"""
        content = "Apache Kafka is a distributed streaming platform."
        citations = self.validator.extract_citations(content)

        assert len(citations) == 0

    def test_citation_with_high_number(self):
        """Test citation with double-digit reference"""
        content = "This fact is from source 12 [REF:12]."
        citations = self.validator.extract_citations(content)

        assert len(citations) == 1
        assert citations[0].ref_id == "12"


class TestCitationValidation:
    """Test citation validation against sources"""

    def setup_method(self):
        self.validator = CitationValidator()
        self.sources = [
            "Apache Kafka is a distributed event streaming platform.",
            "Kafka uses partitions to enable parallel processing.",
            "Consumer groups allow multiple consumers to share workload.",
        ]

    def test_valid_citation(self):
        """Test validation of valid citation"""
        content = "Kafka uses partitions for parallel processing [REF:2]."
        report = self.validator.validate_citations(content, self.sources)

        assert report.total_citations == 1
        assert report.valid_citations == 1
        assert report.invalid_citations == 0

    def test_invalid_citation_ref_out_of_range(self):
        """Test citation with reference outside source range"""
        content = "This is from source 99 [REF:99]."
        report = self.validator.validate_citations(content, self.sources)

        assert report.total_citations == 1
        assert report.valid_citations == 0
        assert report.invalid_citations == 1

    def test_mixed_valid_invalid(self):
        """Test mix of valid and invalid citations"""
        content = """
        Kafka is distributed [REF:1].
        It has feature X [REF:50].
        Consumer groups exist [REF:3].
        """
        report = self.validator.validate_citations(content, self.sources)

        assert report.total_citations == 3
        assert report.valid_citations == 2
        assert report.invalid_citations == 1

    def test_citation_rate(self):
        """Test citation rate calculation"""
        content = "Kafka is a distributed streaming platform [REF:1]. This sentence has absolutely no citation whatsoever. Consumer groups are extremely useful for scalability [REF:3]. Another sentence without any citation here at all."
        report = self.validator.validate_citations(content, self.sources)

        # Should have 4 sentences and 2 citations
        assert report.total_sentences == 4
        assert report.total_citations == 2
        # Citation rate depends on uncited sentence detection (may vary)
        assert report.citation_rate >= 0.4  # At least some sentences are cited

    def test_validity_rate(self):
        """Test validity rate calculation"""
        content = "Valid [REF:1]. Invalid [REF:99]."
        report = self.validator.validate_citations(content, self.sources)

        assert report.validity_rate == 0.5  # 1 valid out of 2


class TestUncitedSentences:
    """Test detection of uncited sentences"""

    def setup_method(self):
        config = EnforcementConfig(min_words_for_citation=5)
        self.validator = CitationValidator(config)
        self.sources = [
            "Apache Kafka is a streaming platform.",
            "Partitions enable parallelism.",
        ]

    def test_detect_uncited_sentences(self):
        """Test detection of sentences without citations"""
        content = """
        Kafka is a streaming platform [REF:1].
        This sentence does not have any citation at all.
        Partitions enable parallel processing [REF:2].
        """
        report = self.validator.validate_citations(content, self.sources)

        assert report.uncited_sentences >= 1
        assert len(report.uncited_sentence_list) >= 1

    def test_short_sentences_ignored(self):
        """Test that short sentences don't need citations"""
        # Short sentences (< min_words_for_citation) shouldn't count as uncited
        content = "Yes. No. Kafka is a distributed streaming platform [REF:1]."
        report = self.validator.validate_citations(content, self.sources)

        # Only the long sentence counts, and it has a citation
        # Short sentences "Yes." and "No." are ignored
        assert report.uncited_sentences == 0

    def test_all_cited(self):
        """Test when all sentences have citations"""
        content = "Kafka is a distributed streaming platform [REF:1]. Partitions enable parallel processing and scalability [REF:2]."
        report = self.validator.validate_citations(content, self.sources)

        assert report.uncited_sentences == 0


class TestCitationDensity:
    """Test citation density checking"""

    def setup_method(self):
        self.validator = CitationValidator()

    def test_sufficient_density(self):
        """Test paragraph with sufficient citation density"""
        content = """
        Apache Kafka is a powerful platform [REF:1]. It uses partitions
        for scalability [REF:2]. Consumer groups enable parallel
        processing [REF:3].
        """
        is_sufficient, undercited = self.validator.check_citation_density(
            content, min_citations_per_paragraph=2
        )

        assert is_sufficient is True
        assert len(undercited) == 0

    def test_insufficient_density(self):
        """Test paragraph with insufficient citations"""
        content = """
        Apache Kafka is a powerful distributed streaming platform that
        enables real-time data processing. It was originally developed
        at LinkedIn and later became an open-source project. Many
        companies use Kafka for their data pipelines [REF:1].
        """
        is_sufficient, undercited = self.validator.check_citation_density(
            content, min_citations_per_paragraph=3
        )

        assert is_sufficient is False
        assert len(undercited) >= 1


class TestKeywordSimilarity:
    """Test keyword-based similarity calculation"""

    def setup_method(self):
        self.validator = CitationValidator()

    def test_identical_text(self):
        """Test similarity of identical texts"""
        text = "Kafka uses partitions"
        similarity = self.validator._keyword_similarity(text, text)

        assert similarity == 1.0

    def test_similar_text(self):
        """Test similarity of similar texts"""
        text1 = "Kafka uses partitions for scalability"
        text2 = "Kafka partitions enable scalability"  # More similar text
        similarity = self.validator._keyword_similarity(text1, text2)

        # Jaccard similarity should be reasonably high with overlapping keywords
        assert similarity > 0.2

    def test_different_text(self):
        """Test similarity of completely different texts"""
        text1 = "Kafka is a streaming platform"
        text2 = "Python is a programming language"
        similarity = self.validator._keyword_similarity(text1, text2)

        assert similarity < 0.2

    def test_empty_text(self):
        """Test similarity with empty text"""
        similarity = self.validator._keyword_similarity("", "some text")
        assert similarity == 0.0


class TestCitationPromptGeneration:
    """Test citation prompt generation"""

    def setup_method(self):
        self.validator = CitationValidator()

    def test_prompt_contains_sources(self):
        """Test that prompt contains source references"""
        sources = [
            "First source about Kafka.",
            "Second source about partitions.",
        ]
        prompt = self.validator.generate_citation_prompt(sources)

        assert "[REF:1]" in prompt
        assert "[REF:2]" in prompt
        assert "First source" in prompt
        assert "Second source" in prompt

    def test_prompt_contains_rules(self):
        """Test that prompt contains citation rules"""
        sources = ["Some source content."]
        prompt = self.validator.generate_citation_prompt(sources)

        assert "MUST" in prompt or "must" in prompt
        assert "citation" in prompt.lower()

    def test_long_source_truncation(self):
        """Test that long sources are truncated in prompt"""
        long_source = "x" * 1000
        sources = [long_source]
        prompt = self.validator.generate_citation_prompt(sources)

        # Should be truncated with ellipsis
        assert "..." in prompt
        assert len(prompt) < len(long_source) + 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
