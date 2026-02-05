"""
Integration tests for RAG Enforcement module.

Tests the complete enforcement pipeline:
1. Content generation with citation requirements
2. Citation validation against sources
3. Sentence-level verification
4. Retry loop with strictness escalation
5. Compliance scoring and reporting

Uses mock LLM responses to simulate realistic generation scenarios.
"""

import sys
import os
import asyncio
import pytest

# Add rag_enforcement directory to path
_rag_enforcement_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "services",
    "rag_enforcement"
)
sys.path.insert(0, _rag_enforcement_path)

from models import EnforcementConfig, EnforcementResult, ComplianceLevel, FactStatus
from citation_validator import CitationValidator
from sentence_verifier import SentenceVerifier
from rag_enforcer import RAGEnforcer, RAGComplianceError, create_enforcer, verify_content


# ============================================================================
# Test Data: Realistic Source Documents
# ============================================================================

KAFKA_SOURCES = [
    """Apache Kafka is a distributed event streaming platform capable of handling
    trillions of events a day. Originally developed at LinkedIn, Kafka is now
    used by thousands of companies for high-performance data pipelines, streaming
    analytics, and mission-critical applications.""",

    """Kafka uses a partitioned, replicated commit log architecture. Topics are
    divided into partitions, which are distributed across brokers in the cluster.
    Each partition is an ordered, immutable sequence of records that is
    continually appended to.""",

    """Consumer groups in Kafka allow multiple consumers to divide the work of
    consuming and processing records. Each consumer in a group is assigned a
    subset of the partitions, enabling horizontal scaling of consumption.""",

    """Kafka Connect is a framework for connecting Kafka with external systems
    such as databases, key-value stores, search indexes, and file systems.
    It provides scalable and reliable streaming data between systems.""",

    """Kafka Streams is a client library for building applications and microservices
    that process and analyze data stored in Kafka. It combines the simplicity of
    writing standard Java applications with the benefits of Kafka's server-side
    cluster technology.""",
]


# ============================================================================
# Mock Generator Functions (Simulating LLM Responses)
# ============================================================================

async def mock_good_generator(topic, sources, strictness, citation_prompt, **kwargs):
    """Simulates a well-behaved LLM that produces compliant content."""
    return """
    Apache Kafka is a distributed event streaming platform that can handle
    trillions of events daily [REF:1]. Originally developed at LinkedIn,
    it's now used by thousands of companies worldwide [REF:1].

    Kafka's architecture is based on a partitioned, replicated commit log [REF:2].
    Topics are divided into partitions distributed across brokers [REF:2].
    Each partition maintains an ordered, immutable sequence of records [REF:2].

    For scaling consumption, Kafka uses consumer groups [REF:3]. Each consumer
    in a group handles a subset of partitions, enabling horizontal scaling [REF:3].

    Kafka Connect provides a framework for integrating with external systems
    like databases and search indexes [REF:4]. Kafka Streams enables building
    processing applications directly in Java [REF:5].
    """


async def mock_hallucinating_generator(topic, sources, strictness, citation_prompt, **kwargs):
    """Simulates an LLM that produces hallucinated content."""
    return """
    Apache Kafka was invented by NASA in 1995 for space communication.
    It uses quantum computing for ultra-fast message processing.
    Kafka requires at least 1TB of RAM to run a single broker.
    The latest version supports telepathic data transmission.
    """


async def mock_improving_generator(topic, sources, strictness, citation_prompt, **kwargs):
    """Simulates an LLM that improves with stricter prompting."""
    # Track attempts via closure
    if not hasattr(mock_improving_generator, 'attempt'):
        mock_improving_generator.attempt = 0
    mock_improving_generator.attempt += 1

    if mock_improving_generator.attempt == 1:
        # First attempt: Mostly hallucinated
        return """
        Kafka is a messaging system developed by Microsoft.
        It runs on Windows Server exclusively.
        The maximum throughput is 100 messages per second.
        """
    elif mock_improving_generator.attempt == 2:
        # Second attempt: Partially compliant
        return """
        Apache Kafka is a distributed streaming platform [REF:1].
        It was developed at LinkedIn for high-performance pipelines [REF:1].
        Kafka can handle millions of events per second.
        Topics are partitioned for scalability [REF:2].
        """
    else:
        # Third attempt: Fully compliant
        return """
        Apache Kafka is a distributed event streaming platform [REF:1].
        It was originally developed at LinkedIn [REF:1].
        Kafka uses partitioned commit log architecture [REF:2].
        Consumer groups enable horizontal scaling [REF:3].
        """


async def mock_partial_citation_generator(topic, sources, strictness, citation_prompt, **kwargs):
    """Simulates content with some citations but not all statements cited."""
    return """
    Apache Kafka is a powerful streaming platform [REF:1].
    It handles trillions of events daily.
    The architecture uses partitions for scalability [REF:2].
    Consumer groups divide work among consumers.
    Kafka Connect integrates with external systems [REF:4].
    """


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndEnforcement:
    """Test complete enforcement pipeline."""

    @pytest.fixture(autouse=True)
    def reset_generator_state(self):
        """Reset generator state before each test."""
        if hasattr(mock_improving_generator, 'attempt'):
            del mock_improving_generator.attempt
        yield

    @pytest.mark.asyncio
    async def test_compliant_content_passes_first_attempt(self):
        """Test that well-formed content passes on first attempt."""
        # Use lower threshold because keyword similarity (without embeddings)
        # produces lower scores than semantic similarity
        config = EnforcementConfig(
            min_compliance_score=0.40,  # Realistic for keyword-only matching
            max_attempts=3,
            require_citations=True,
            sentence_similarity_threshold=0.30  # Lower threshold for keywords
        )
        enforcer = RAGEnforcer(config)

        result = await enforcer.enforce(
            generator_func=mock_good_generator,
            sources=KAFKA_SOURCES,
            topic="Apache Kafka"
        )

        assert result.is_compliant is True
        assert result.attempt_number == 1
        assert result.overall_score >= 0.40
        assert result.compliance_level == ComplianceLevel.COMPLIANT
        assert len(result.hallucinations) == 0

    @pytest.mark.asyncio
    async def test_hallucinated_content_fails(self):
        """Test that hallucinated content fails even after retries."""
        config = EnforcementConfig(
            min_compliance_score=0.80,
            max_attempts=2
        )
        enforcer = RAGEnforcer(config)

        with pytest.raises(RAGComplianceError) as excinfo:
            await enforcer.enforce(
                generator_func=mock_hallucinating_generator,
                sources=KAFKA_SOURCES,
                topic="Apache Kafka"
            )

        assert "Cannot achieve" in str(excinfo.value)
        assert excinfo.value.result is not None
        assert excinfo.value.result.overall_score < 0.80
        # Should have detected hallucinations
        assert len(excinfo.value.result.hallucinations) > 0 or \
               len(excinfo.value.result.ungrounded_facts) > 0

    @pytest.mark.asyncio
    async def test_improving_generator_succeeds_after_retry(self):
        """Test that content improves with stricter prompting."""
        config = EnforcementConfig(
            min_compliance_score=0.60,
            max_attempts=3
        )
        enforcer = RAGEnforcer(config)

        result = await enforcer.enforce(
            generator_func=mock_improving_generator,
            sources=KAFKA_SOURCES,
            topic="Apache Kafka"
        )

        # Should succeed, potentially after retries
        assert result.is_compliant is True
        # Check that we either succeeded first try or improved
        assert result.overall_score >= 0.60

    @pytest.mark.asyncio
    async def test_strictness_escalation(self):
        """Test that strictness escalates with each attempt."""
        config = EnforcementConfig(
            min_compliance_score=0.99,  # Impossibly high
            max_attempts=3
        )
        enforcer = RAGEnforcer(config)

        strictness_levels = []

        async def tracking_generator(topic, sources, strictness, citation_prompt, **kwargs):
            strictness_levels.append(strictness)
            return "Python programming language was created by Guido van Rossum at CWI in the Netherlands."

        try:
            await enforcer.enforce(
                generator_func=tracking_generator,
                sources=KAFKA_SOURCES,
                topic="Kafka"
            )
        except RAGComplianceError:
            pass

        assert len(strictness_levels) == 3
        assert strictness_levels[0] == "standard"
        assert strictness_levels[1] == "strict"
        assert strictness_levels[2] == "ultra_strict"


class TestCitationIntegration:
    """Test citation validation integration."""

    def test_citation_prompt_generation(self):
        """Test that citation prompts are properly generated."""
        validator = CitationValidator()
        prompt = validator.generate_citation_prompt(KAFKA_SOURCES[:3])

        # Should contain reference markers
        assert "[REF:1]" in prompt
        assert "[REF:2]" in prompt
        assert "[REF:3]" in prompt

        # Should contain source content (truncated)
        assert "Kafka" in prompt
        assert "partition" in prompt.lower()

        # Should contain instructions
        assert "MUST" in prompt or "must" in prompt

    def test_citation_extraction_and_validation(self):
        """Test citation extraction from realistic content."""
        validator = CitationValidator()

        content = """
        Apache Kafka is a distributed streaming platform [REF:1].
        It uses partitions for scalability [REF:2].
        Consumer groups enable parallel consumption [REF:3].
        This statement has no citation.
        Kafka Connect integrates with databases [REF:4].
        """

        report = validator.validate_citations(content, KAFKA_SOURCES)

        assert report.total_citations == 4
        assert report.valid_citations == 4  # All refs 1-4 are valid
        assert report.invalid_citations == 0

    def test_invalid_citation_detection(self):
        """Test detection of invalid citations."""
        validator = CitationValidator()

        content = """
        Kafka is fast [REF:1].
        This cites a non-existent source [REF:99].
        This also cites invalid source [REF:100].
        """

        report = validator.validate_citations(content, KAFKA_SOURCES)

        assert report.total_citations == 3
        assert report.valid_citations == 1  # Only REF:1 is valid
        assert report.invalid_citations == 2


class TestSentenceVerificationIntegration:
    """Test sentence verification integration."""

    def test_grounded_sentences_detection(self):
        """Test detection of sentences grounded in sources."""
        config = EnforcementConfig(sentence_similarity_threshold=0.3)
        verifier = SentenceVerifier(config)

        content = """
        Apache Kafka is a distributed event streaming platform.
        It was originally developed at LinkedIn.
        Topics are divided into partitions.
        """

        report = verifier.verify_sentences(content, KAFKA_SOURCES)

        # All sentences should be grounded (they match sources)
        assert report.grounded_sentences >= 2
        assert report.grounding_rate >= 0.5

    def test_hallucination_detection(self):
        """Test detection of hallucinated sentences."""
        config = EnforcementConfig(sentence_similarity_threshold=0.4)
        verifier = SentenceVerifier(config)

        content = """
        Kafka was invented by NASA for space communication.
        It requires quantum computers to run.
        The maximum throughput is one message per hour.
        """

        report = verifier.verify_sentences(content, KAFKA_SOURCES)

        # All sentences should be ungrounded
        assert report.ungrounded_sentences >= 2
        assert report.grounding_rate < 0.5

        # Check hallucination candidates
        candidates = verifier.get_hallucination_candidates(report)
        assert len(candidates) >= 1


class TestScoreIntegration:
    """Test score calculation integration."""

    def test_score_weights(self):
        """Test that score weights are applied correctly."""
        config = EnforcementConfig(
            citation_weight=0.4,
            grounding_weight=0.6,
            min_compliance_score=0.50
        )
        enforcer = RAGEnforcer(config)

        content = """
        Apache Kafka is a streaming platform [REF:1].
        Partitions enable parallel processing [REF:2].
        """

        result = enforcer.verify_only(content, KAFKA_SOURCES)

        # Verify weights are applied
        expected_score = (
            config.citation_weight * result.citation_score +
            config.grounding_weight * result.grounding_score
        )
        assert abs(result.overall_score - expected_score) < 0.01

    def test_compliance_level_thresholds(self):
        """Test compliance level determination."""
        # Test compliant level - use content with very similar wording to sources
        high_compliance_content = """
        Apache Kafka is a distributed event streaming platform capable of handling trillions of events [REF:1].
        Originally developed at LinkedIn, Kafka is now used by thousands of companies [REF:1].
        Topics are divided into partitions which are distributed across brokers [REF:2].
        Consumer groups allow multiple consumers to divide the work of consuming records [REF:3].
        """

        # Lower threshold for keyword-only matching (no embeddings)
        config = EnforcementConfig(
            min_compliance_score=0.40,
            sentence_similarity_threshold=0.25
        )
        enforcer = RAGEnforcer(config)
        result = enforcer.verify_only(high_compliance_content, KAFKA_SOURCES)

        # Should be compliant with content closely matching sources
        assert result.overall_score >= 0.40
        assert result.compliance_level == ComplianceLevel.COMPLIANT


class TestConvenienceFunctionsIntegration:
    """Test convenience function integration."""

    def test_create_enforcer_with_config(self):
        """Test creating an enforcer with custom config."""
        enforcer = create_enforcer(
            min_compliance=0.85,
            max_attempts=5,
            require_citations=False
        )

        assert enforcer.config.min_compliance_score == 0.85
        assert enforcer.config.max_attempts == 5
        assert enforcer.config.require_citations is False

    def test_verify_content_quick_check(self):
        """Test quick content verification."""
        content = """
        Apache Kafka is a distributed streaming platform.
        It uses partitions for scalability.
        """

        result = verify_content(content, KAFKA_SOURCES, min_compliance=0.50)

        assert isinstance(result, EnforcementResult)
        assert result.overall_score >= 0


class TestEdgeCasesIntegration:
    """Test edge cases in integration."""

    def test_empty_sources(self):
        """Test handling of empty sources."""
        enforcer = RAGEnforcer()
        content = "Some content about Kafka."

        result = enforcer.verify_only(content, [])

        # Should handle gracefully
        assert result is not None

    def test_very_long_content(self):
        """Test handling of long content."""
        config = EnforcementConfig(min_compliance_score=0.30)
        enforcer = RAGEnforcer(config)

        # Generate long content
        base_sentence = "Apache Kafka is a distributed streaming platform [REF:1]. "
        content = base_sentence * 50

        result = enforcer.verify_only(content, KAFKA_SOURCES)

        assert result is not None
        assert result.sentence_report.total_sentences > 10

    def test_multilingual_content(self):
        """Test handling of content in different languages."""
        config = EnforcementConfig(sentence_similarity_threshold=0.2)
        enforcer = RAGEnforcer(config)

        # French content about Kafka
        content = """
        Apache Kafka est une plateforme de streaming distribué [REF:1].
        Elle utilise des partitions pour la scalabilité [REF:2].
        Les groupes de consommateurs permettent le traitement parallèle [REF:3].
        """

        result = enforcer.verify_only(content, KAFKA_SOURCES)

        # Should handle gracefully (may have lower scores due to language mismatch)
        assert result is not None
        assert result.overall_score >= 0


class TestReportGeneration:
    """Test report generation."""

    def test_feedback_generation(self):
        """Test human-readable feedback generation."""
        config = EnforcementConfig(sentence_similarity_threshold=0.3)
        verifier = SentenceVerifier(config)

        content = """
        Kafka is a streaming platform.
        Python is a programming language.
        Partitions enable scalability.
        """

        report = verifier.verify_sentences(content, KAFKA_SOURCES)
        feedback = verifier.generate_feedback(report)

        assert "Sentence Verification Report" in feedback
        assert "Total sentences" in feedback
        assert "Grounded" in feedback

    def test_result_to_dict(self):
        """Test EnforcementResult serialization."""
        enforcer = RAGEnforcer()
        content = "Kafka is a streaming platform [REF:1]."

        result = enforcer.verify_only(content, KAFKA_SOURCES)
        result_dict = result.to_dict()

        assert "is_compliant" in result_dict
        assert "overall_score" in result_dict
        assert "compliance_level" in result_dict
        assert isinstance(result_dict["overall_score"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
