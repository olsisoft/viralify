"""
Tests for SentenceVerifier

Tests sentence-level verification against sources.
"""

import pytest
import sys
import os
import numpy as np

# Add rag_enforcement directory directly to path (avoid services/__init__.py)
_rag_enforcement_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services",
    "rag_enforcement"
)
sys.path.insert(0, _rag_enforcement_path)

from sentence_verifier import SentenceVerifier, AsyncSentenceVerifier
from models import EnforcementConfig, FactStatus


class TestSentenceSplitting:
    """Test sentence splitting functionality"""

    def setup_method(self):
        self.verifier = SentenceVerifier()

    def test_split_simple_sentences(self):
        """Test splitting simple sentences"""
        text = "First sentence. Second sentence. Third sentence."
        sentences = self.verifier._split_sentences(text)

        assert len(sentences) == 3

    def test_split_with_question_marks(self):
        """Test splitting with question marks"""
        text = "What is Kafka? It is a streaming platform. Why use it?"
        sentences = self.verifier._split_sentences(text)

        assert len(sentences) == 3

    def test_split_with_exclamation(self):
        """Test splitting with exclamation marks"""
        text = "Kafka is fast! It scales well. Amazing!"
        sentences = self.verifier._split_sentences(text)

        assert len(sentences) == 3

    def test_filter_short_sentences(self):
        """Test that very short sentences are filtered"""
        text = "Yes. This is a longer sentence that should remain."
        sentences = self.verifier._split_sentences(text)

        # "Yes." should be filtered out
        assert len(sentences) == 1
        assert "longer sentence" in sentences[0]


class TestSentenceVerification:
    """Test sentence verification against sources"""

    def setup_method(self):
        config = EnforcementConfig(sentence_similarity_threshold=0.4)
        self.verifier = SentenceVerifier(config)
        self.sources = [
            "Apache Kafka is a distributed event streaming platform used for high-performance data pipelines.",
            "Kafka uses partitions to enable parallel processing and horizontal scalability.",
            "Consumer groups in Kafka allow multiple consumers to share the workload efficiently.",
            "Producers write messages to Kafka topics which are divided into partitions.",
        ]

    def test_verify_grounded_sentence(self):
        """Test verification of a sentence that is clearly from sources"""
        content = "Kafka uses partitions for parallel processing."
        report = self.verifier.verify_sentences(content, self.sources)

        assert report.total_sentences == 1
        assert report.grounded_sentences == 1
        assert report.sentence_scores[0].is_grounded is True

    def test_verify_ungrounded_sentence(self):
        """Test verification of a sentence not in sources"""
        content = "Python is a programming language created by Guido van Rossum."
        report = self.verifier.verify_sentences(content, self.sources)

        assert report.total_sentences == 1
        assert report.ungrounded_sentences == 1
        assert report.sentence_scores[0].is_grounded is False

    def test_mixed_sentences(self):
        """Test content with both grounded and ungrounded sentences"""
        content = """
        Kafka uses partitions for scalability.
        Python was created in 1991 by Guido.
        Consumer groups enable parallel processing.
        """
        report = self.verifier.verify_sentences(content, self.sources)

        assert report.total_sentences == 3
        assert report.grounded_sentences >= 1
        assert report.ungrounded_sentences >= 1

    def test_grounding_rate(self):
        """Test grounding rate calculation"""
        content = """
        Kafka is a streaming platform.
        Unrelated sentence about cooking recipes.
        Partitions enable parallel processing.
        Another unrelated sentence about weather.
        """
        report = self.verifier.verify_sentences(content, self.sources)

        # Should have approximately 50% grounding rate
        assert 0.3 <= report.grounding_rate <= 0.7

    def test_average_similarity(self):
        """Test average similarity calculation"""
        content = """
        Kafka uses partitions for scalability.
        Consumer groups share workload.
        """
        report = self.verifier.verify_sentences(content, self.sources)

        assert report.average_similarity > 0.0
        assert report.average_similarity <= 1.0


class TestFactStatus:
    """Test fact status classification"""

    def setup_method(self):
        config = EnforcementConfig(sentence_similarity_threshold=0.5)
        self.verifier = SentenceVerifier(config)
        self.sources = [
            "Apache Kafka processes millions of messages per second.",
            "Kafka topics are divided into partitions for parallelism.",
        ]

    def test_supported_fact(self):
        """Test that matching sentence is marked as SUPPORTED"""
        content = "Kafka processes millions of messages per second."
        report = self.verifier.verify_sentences(content, self.sources)

        assert report.sentence_scores[0].fact_status == FactStatus.SUPPORTED

    def test_hallucination_detection(self):
        """Test that completely unrelated sentence is marked as HALLUCINATION"""
        content = "The recipe calls for two cups of flour and one egg."
        report = self.verifier.verify_sentences(content, self.sources)

        assert report.sentence_scores[0].fact_status == FactStatus.HALLUCINATION

    def test_unsupported_fact(self):
        """Test that partially related sentence is marked as UNSUPPORTED"""
        content = "Kafka was developed at LinkedIn in 2011."
        report = self.verifier.verify_sentences(content, self.sources)

        # This mentions Kafka but the specific fact isn't in sources
        status = report.sentence_scores[0].fact_status
        assert status in [FactStatus.UNSUPPORTED, FactStatus.HALLUCINATION]


class TestHallucinationCandidates:
    """Test hallucination candidate detection"""

    def setup_method(self):
        self.verifier = SentenceVerifier()
        self.sources = [
            "Kafka is a distributed streaming platform.",
            "It processes messages in real-time.",
        ]

    def test_get_hallucination_candidates(self):
        """Test getting hallucination candidates sorted by likelihood"""
        content = """
        Kafka is distributed.
        The weather is nice today.
        Messages are processed in real-time.
        Python is great for data science.
        """
        report = self.verifier.verify_sentences(content, self.sources)
        candidates = self.verifier.get_hallucination_candidates(report, max_candidates=5)

        # Candidates should be sorted by similarity (lowest first)
        if len(candidates) >= 2:
            assert candidates[0].similarity <= candidates[-1].similarity

        # Weather and Python sentences should be top candidates
        candidate_texts = [c.sentence.lower() for c in candidates]
        assert any("weather" in t or "python" in t for t in candidate_texts)

    def test_max_candidates_limit(self):
        """Test that max_candidates limit is respected"""
        content = """
        Sentence one about nothing. Sentence two about nothing.
        Sentence three about nothing. Sentence four about nothing.
        Sentence five about nothing. Sentence six about nothing.
        """
        report = self.verifier.verify_sentences(content, self.sources)
        candidates = self.verifier.get_hallucination_candidates(report, max_candidates=3)

        assert len(candidates) <= 3


class TestWithEmbeddings:
    """Test with mock embedding function"""

    def setup_method(self):
        # Simple mock embedding function
        def mock_embed(text: str) -> np.ndarray:
            # Create a simple hash-based embedding
            np.random.seed(hash(text.lower()[:50]) % 2**32)
            return np.random.randn(384).astype(np.float32)

        self.verifier = SentenceVerifier(embedding_func=mock_embed)
        self.sources = [
            "Apache Kafka is a streaming platform.",
            "Partitions enable parallel processing.",
        ]

    def test_with_embeddings(self):
        """Test verification using embedding function"""
        content = "Kafka is a streaming platform."
        report = self.verifier.verify_sentences(content, self.sources)

        assert report.total_sentences == 1
        # With hash-based mock embeddings, similarity might be any value
        # Just verify the mechanism works (similarity is computed)
        assert report.sentence_scores[0].similarity >= 0  # Can be 0 with random vectors

    def test_precompute_source_embeddings(self):
        """Test precomputing source embeddings"""
        self.verifier.precompute_source_embeddings(self.sources)

        assert self.verifier._source_embeddings is not None
        assert len(self.verifier._source_embeddings) == len(self.sources)


class TestKeywordSimilarity:
    """Test keyword-based similarity as fallback"""

    def setup_method(self):
        self.verifier = SentenceVerifier()

    def test_identical_text(self):
        """Test similarity of identical texts"""
        text = "Kafka uses partitions for scalability"
        similarity = self.verifier._keyword_similarity(text, text)

        assert similarity == 1.0

    def test_overlapping_keywords(self):
        """Test similarity with overlapping keywords"""
        text1 = "Kafka partitions enable scalability"
        text2 = "Apache Kafka uses partitions for scaling"
        similarity = self.verifier._keyword_similarity(text1, text2)

        assert similarity > 0.3

    def test_no_overlap(self):
        """Test similarity with no keyword overlap"""
        text1 = "Apache Kafka streaming platform"
        text2 = "Python programming language"
        similarity = self.verifier._keyword_similarity(text1, text2)

        assert similarity < 0.2


class TestFeedbackGeneration:
    """Test human-readable feedback generation"""

    def setup_method(self):
        self.verifier = SentenceVerifier()
        self.sources = ["Kafka is a streaming platform."]

    def test_generate_feedback(self):
        """Test feedback generation"""
        content = """
        Kafka is a streaming platform.
        The weather is sunny today.
        """
        report = self.verifier.verify_sentences(content, self.sources)
        feedback = self.verifier.generate_feedback(report)

        assert "Sentence Verification Report" in feedback
        assert "Total sentences" in feedback
        assert "Grounded" in feedback

    def test_feedback_includes_ungrounded(self):
        """Test that feedback includes ungrounded sentences"""
        content = "Python is a programming language for data science."
        report = self.verifier.verify_sentences(content, self.sources)
        feedback = self.verifier.generate_feedback(report)

        assert "UNGROUNDED" in feedback


class TestAsyncSentenceVerifier:
    """Test async version of sentence verifier"""

    @pytest.mark.asyncio
    async def test_async_verify(self):
        """Test async verification"""
        async def mock_async_embed(text: str) -> np.ndarray:
            np.random.seed(hash(text.lower()[:50]) % 2**32)
            return np.random.randn(384).astype(np.float32)

        verifier = AsyncSentenceVerifier(async_embedding_func=mock_async_embed)
        sources = ["Kafka is a streaming platform."]
        content = "Kafka processes messages."

        report = await verifier.verify_sentences_async(content, sources)

        assert report.total_sentences >= 1
        assert report.average_similarity >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
