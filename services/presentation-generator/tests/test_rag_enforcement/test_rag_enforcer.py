"""
Tests for RAGEnforcer

Tests the main enforcement orchestrator with retry logic.
"""

import pytest
import asyncio
import sys
import os

# Add rag_enforcement directory directly to path (avoid services/__init__.py)
_rag_enforcement_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services",
    "rag_enforcement"
)
sys.path.insert(0, _rag_enforcement_path)

from rag_enforcer import (
    RAGEnforcer,
    AsyncRAGEnforcer,
    RAGComplianceError,
    create_enforcer,
    verify_content,
)
from models import (
    EnforcementConfig,
    EnforcementResult,
    ComplianceLevel,
)


class TestEnforcementConfig:
    """Test enforcement configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = EnforcementConfig()

        assert config.min_compliance_score == 0.90
        assert config.max_attempts == 3
        assert config.require_citations is True
        assert config.citation_weight + config.grounding_weight == 1.0

    def test_custom_config(self):
        """Test custom configuration"""
        config = EnforcementConfig(
            min_compliance_score=0.85,
            max_attempts=5,
            require_citations=False
        )

        assert config.min_compliance_score == 0.85
        assert config.max_attempts == 5
        assert config.require_citations is False


class TestVerifyOnly:
    """Test verify_only method (no generation)"""

    def setup_method(self):
        self.enforcer = RAGEnforcer()
        self.sources = [
            "Apache Kafka is a distributed event streaming platform.",
            "Kafka uses partitions for parallel processing and scalability.",
            "Consumer groups allow multiple consumers to share workload.",
        ]

    def test_compliant_content(self):
        """Test verification of compliant content"""
        content = """
        Kafka is a distributed streaming platform [REF:1].
        It uses partitions for parallel processing [REF:2].
        Consumer groups share the workload [REF:3].
        """
        result = self.enforcer.verify_only(content, self.sources)

        assert result.overall_score > 0.5
        assert result.citation_report is not None
        assert result.sentence_report is not None

    def test_non_compliant_content(self):
        """Test verification of non-compliant content"""
        # Use sentences >= 5 words to avoid auto-approval of short sentences
        content = """
        Python is a programming language created by Guido van Rossum.
        JavaScript runs in web browsers and enables interactivity.
        Go is great for concurrent programming and microservices.
        """
        result = self.enforcer.verify_only(content, self.sources)

        assert result.is_compliant is False
        assert result.overall_score < 0.7  # More realistic threshold
        assert result.compliance_level != ComplianceLevel.COMPLIANT

    def test_partial_compliance(self):
        """Test content with partial compliance"""
        content = """
        Kafka uses partitions for scalability [REF:2].
        Python is also a great language for data processing.
        Consumer groups are useful [REF:3].
        """
        result = self.enforcer.verify_only(content, self.sources)

        # Should have some grounding but not full compliance
        assert 0.3 < result.overall_score < 0.95

    def test_hallucinations_detected(self):
        """Test that hallucinations are detected and listed"""
        content = """
        Kafka was invented by NASA in 1999.
        It runs only on Windows servers.
        Kafka is a streaming platform [REF:1].
        """
        result = self.enforcer.verify_only(content, self.sources)

        # Should detect hallucinations
        assert len(result.hallucinations) > 0 or len(result.ungrounded_facts) > 0


class TestEnforceWithRetry:
    """Test enforce method with retry logic"""

    def setup_method(self):
        self.sources = [
            "Kafka is a streaming platform.",
            "Partitions enable parallelism.",
        ]

    @pytest.mark.asyncio
    async def test_enforce_success_first_attempt(self):
        """Test successful enforcement on first attempt"""
        config = EnforcementConfig(min_compliance_score=0.5, max_attempts=3)
        enforcer = RAGEnforcer(config)

        async def good_generator(topic, sources, strictness, citation_prompt, **kwargs):
            return """
            Kafka is a streaming platform [REF:1].
            Partitions enable parallel processing [REF:2].
            """

        result = await enforcer.enforce(
            generator_func=good_generator,
            sources=self.sources,
            topic="Kafka"
        )

        assert result.is_compliant is True
        assert result.attempt_number == 1

    @pytest.mark.asyncio
    async def test_enforce_success_after_retry(self):
        """Test successful enforcement after retry"""
        config = EnforcementConfig(min_compliance_score=0.6, max_attempts=3)
        enforcer = RAGEnforcer(config)

        attempt_count = [0]

        async def improving_generator(topic, sources, strictness, citation_prompt, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                # First attempt - bad content
                return "Python is a programming language."
            else:
                # Second attempt - good content
                return """
                Kafka is a streaming platform [REF:1].
                Partitions enable parallelism [REF:2].
                """

        result = await enforcer.enforce(
            generator_func=improving_generator,
            sources=self.sources,
            topic="Kafka"
        )

        assert result.is_compliant is True
        assert result.attempt_number == 2

    @pytest.mark.asyncio
    async def test_enforce_failure_after_max_attempts(self):
        """Test failure after max attempts"""
        config = EnforcementConfig(min_compliance_score=0.99, max_attempts=2)
        enforcer = RAGEnforcer(config)

        async def bad_generator(topic, sources, strictness, citation_prompt, **kwargs):
            # Use long sentences that are clearly unrelated to Kafka sources
            return "Python is a programming language that was created by Guido van Rossum. JavaScript runs in web browsers and enables interactive web pages. Go is a compiled language developed at Google for system programming."

        with pytest.raises(RAGComplianceError) as excinfo:
            await enforcer.enforce(
                generator_func=bad_generator,
                sources=self.sources,
                topic="Kafka"
            )

        assert "Cannot achieve" in str(excinfo.value)
        assert excinfo.value.result is not None

    @pytest.mark.asyncio
    async def test_enforce_with_strictness_escalation(self):
        """Test that strictness escalates with each attempt"""
        config = EnforcementConfig(min_compliance_score=0.99, max_attempts=3)
        enforcer = RAGEnforcer(config)

        strictness_levels = []

        async def tracking_generator(topic, sources, strictness, citation_prompt, **kwargs):
            strictness_levels.append(strictness)
            # Return long unrelated content to force retries
            return "Python is a programming language that was created by Guido van Rossum in the late 1980s. JavaScript runs in web browsers and enables interactive web functionality."

        try:
            await enforcer.enforce(
                generator_func=tracking_generator,
                sources=self.sources,
                topic="Kafka"
            )
        except RAGComplianceError:
            pass

        # Should have escalated through strictness levels
        assert "standard" in strictness_levels
        assert "strict" in strictness_levels
        assert "ultra_strict" in strictness_levels


class TestScoreCalculation:
    """Test score calculation"""

    def setup_method(self):
        self.enforcer = RAGEnforcer()
        self.sources = [
            "Kafka is a streaming platform.",
            "Partitions enable parallelism.",
        ]

    def test_citation_score_calculation(self):
        """Test citation score calculation"""
        content = """
        Kafka is streaming [REF:1].
        Partitions are parallel [REF:2].
        """
        result = self.enforcer.verify_only(content, self.sources)

        assert result.citation_score > 0.5
        assert result.citation_score <= 1.0

    def test_grounding_score_calculation(self):
        """Test grounding score calculation"""
        content = """
        Kafka is a streaming platform.
        Partitions enable parallel processing.
        """
        result = self.enforcer.verify_only(content, self.sources)

        assert result.grounding_score > 0.0
        assert result.grounding_score <= 1.0

    def test_combined_score_weights(self):
        """Test that combined score respects weights"""
        config = EnforcementConfig(
            citation_weight=0.3,
            grounding_weight=0.7,
            require_citations=True
        )
        enforcer = RAGEnforcer(config)

        content = "Kafka is streaming [REF:1]. Partitions are parallel [REF:2]."
        result = enforcer.verify_only(content, self.sources)

        expected = 0.3 * result.citation_score + 0.7 * result.grounding_score
        assert abs(result.overall_score - expected) < 0.01


class TestComplianceLevels:
    """Test compliance level determination"""

    def setup_method(self):
        self.sources = ["Kafka is streaming.", "Partitions enable parallelism."]

    def test_compliant_level(self):
        """Test COMPLIANT level for high scores"""
        config = EnforcementConfig(min_compliance_score=0.5)
        enforcer = RAGEnforcer(config)

        content = "Kafka is streaming [REF:1]. Partitions enable parallelism [REF:2]."
        result = enforcer.verify_only(content, self.sources)

        if result.overall_score >= 0.5:
            assert result.compliance_level == ComplianceLevel.COMPLIANT

    def test_non_compliant_level(self):
        """Test NON_COMPLIANT level for low scores"""
        config = EnforcementConfig(min_compliance_score=0.99)
        enforcer = RAGEnforcer(config)

        # Use long sentences clearly unrelated to sources
        content = "Python is a programming language that was created by Guido van Rossum. JavaScript runs in web browsers and enables interactive web functionality."
        result = enforcer.verify_only(content, self.sources)

        assert result.compliance_level != ComplianceLevel.COMPLIANT


class TestEnforcementResult:
    """Test EnforcementResult data structure"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = EnforcementResult(
            content="Test content",
            is_compliant=True,
            compliance_level=ComplianceLevel.COMPLIANT,
            overall_score=0.95,
            citation_score=0.90,
            grounding_score=0.97,
            attempt_number=1,
            total_attempts=3,
            processing_time_ms=150.5
        )

        d = result.to_dict()

        assert d["is_compliant"] is True
        assert d["compliance_level"] == "compliant"
        assert d["overall_score"] == 0.95
        assert d["attempt_number"] == 1


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_enforcer(self):
        """Test create_enforcer function"""
        enforcer = create_enforcer(
            min_compliance=0.85,
            max_attempts=5,
            require_citations=False
        )

        assert enforcer.config.min_compliance_score == 0.85
        assert enforcer.config.max_attempts == 5
        assert enforcer.config.require_citations is False

    def test_verify_content_function(self):
        """Test verify_content convenience function"""
        sources = ["Kafka is streaming.", "Partitions enable parallelism."]
        content = "Kafka is a streaming platform."

        result = verify_content(content, sources, min_compliance=0.5)

        assert isinstance(result, EnforcementResult)
        assert result.overall_score >= 0


class TestWithoutCitations:
    """Test enforcement without requiring citations"""

    def test_no_citation_requirement(self):
        """Test verification when citations are not required"""
        config = EnforcementConfig(require_citations=False)
        enforcer = RAGEnforcer(config)

        sources = ["Kafka is a streaming platform."]
        content = "Kafka is a streaming platform."  # No citations

        result = enforcer.verify_only(content, sources)

        # Citation score should be 1.0 when not required
        assert result.citation_score == 1.0


class TestAsyncRAGEnforcer:
    """Test async version of RAG enforcer"""

    @pytest.mark.asyncio
    async def test_async_enforce(self):
        """Test async enforcement"""
        config = EnforcementConfig(min_compliance_score=0.5)
        enforcer = AsyncRAGEnforcer(config)

        sources = ["Kafka is streaming.", "Partitions are parallel."]

        async def generator(topic, sources, strictness, citation_prompt, **kwargs):
            return "Kafka is streaming [REF:1]. Partitions are parallel [REF:2]."

        result = await enforcer.enforce(
            generator_func=generator,
            sources=sources,
            topic="Kafka"
        )

        assert result is not None
        assert result.overall_score >= 0


class TestEdgeCases:
    """Test edge cases"""

    def setup_method(self):
        self.enforcer = RAGEnforcer()

    def test_empty_content(self):
        """Test with empty content"""
        sources = ["Some source."]
        result = self.enforcer.verify_only("", sources)

        assert result.total_attempts == 1

    def test_empty_sources(self):
        """Test with empty sources"""
        content = "Some content."
        result = self.enforcer.verify_only(content, [])

        assert result is not None

    def test_single_word_content(self):
        """Test with very short content"""
        sources = ["Kafka is a platform."]
        result = self.enforcer.verify_only("Kafka", sources)

        assert result is not None

    def test_very_long_content(self):
        """Test with very long content"""
        sources = ["Kafka is streaming."]
        content = "Kafka is streaming. " * 100

        result = self.enforcer.verify_only(content, sources)

        assert result is not None
        assert result.sentence_report.total_sentences > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
