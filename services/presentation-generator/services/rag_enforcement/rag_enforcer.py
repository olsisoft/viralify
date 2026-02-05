"""
RAG Enforcer

Main orchestrator for RAG compliance enforcement.
Combines citation validation, sentence verification, and retry logic.
"""

import time
import asyncio
from typing import List, Optional, Callable, Tuple, Any

# Support both package and standalone imports
try:
    from .models import (
        EnforcementConfig, EnforcementResult, ComplianceLevel,
        CitationReport, SentenceReport, FactStatus
    )
    from .citation_validator import CitationValidator
    from .sentence_verifier import SentenceVerifier, AsyncSentenceVerifier
except ImportError:
    from models import (
        EnforcementConfig, EnforcementResult, ComplianceLevel,
        CitationReport, SentenceReport, FactStatus
    )
    from citation_validator import CitationValidator
    from sentence_verifier import SentenceVerifier, AsyncSentenceVerifier


class RAGComplianceError(Exception):
    """Raised when content cannot meet RAG compliance requirements"""
    def __init__(self, message: str, result: Optional[EnforcementResult] = None):
        super().__init__(message)
        self.result = result


class RAGEnforcer:
    """
    Enforces RAG compliance through citation validation,
    sentence verification, and retry logic.

    Pipeline:
    1. Generate content with citation requirements
    2. Validate citations (must reference real sources)
    3. Verify sentences (each must be grounded in sources)
    4. If not compliant, regenerate with stricter prompt
    5. After max_attempts, raise RAGComplianceError
    """

    def __init__(
        self,
        config: Optional[EnforcementConfig] = None,
        embedding_func: Optional[Callable] = None
    ):
        self.config = config or EnforcementConfig()
        self._embed = embedding_func

        # Initialize sub-components
        self.citation_validator = CitationValidator(config, embedding_func)
        self.sentence_verifier = SentenceVerifier(config, embedding_func)

    def set_embedding_function(self, func: Callable):
        """Set embedding function for semantic similarity"""
        self._embed = func
        self.citation_validator._embed = func
        self.sentence_verifier.set_embedding_function(func)

    async def enforce(
        self,
        generator_func: Callable,
        sources: List[str],
        topic: str,
        **generator_kwargs
    ) -> EnforcementResult:
        """
        Generate content and enforce RAG compliance.

        Args:
            generator_func: Async function that generates content
                           Signature: (topic, sources, strictness, **kwargs) -> str
            sources: List of source chunks
            topic: Topic to generate content about
            **generator_kwargs: Additional kwargs for generator

        Returns:
            EnforcementResult with compliant content

        Raises:
            RAGComplianceError: If compliance cannot be achieved
        """
        start_time = time.time()
        best_result = None
        best_score = 0.0

        strictness_levels = ["standard", "strict", "ultra_strict"]

        for attempt in range(1, self.config.max_attempts + 1):
            strictness = strictness_levels[min(attempt - 1, len(strictness_levels) - 1)]

            print(f"[RAG_ENFORCER] Attempt {attempt}/{self.config.max_attempts} "
                  f"(strictness: {strictness})", flush=True)

            try:
                # Generate content
                content = await generator_func(
                    topic=topic,
                    sources=sources,
                    strictness=strictness,
                    citation_prompt=self.citation_validator.generate_citation_prompt(sources),
                    **generator_kwargs
                )

                # Verify content
                result = await self._verify_content(content, sources, attempt)
                result.processing_time_ms = (time.time() - start_time) * 1000

                # Track best result
                if result.overall_score > best_score:
                    best_score = result.overall_score
                    best_result = result

                # Check if compliant
                if result.is_compliant:
                    print(f"[RAG_ENFORCER] ✅ Compliant on attempt {attempt}: "
                          f"{result.overall_score:.0%}", flush=True)
                    return result

                print(f"[RAG_ENFORCER] ❌ Not compliant: {result.overall_score:.0%} "
                      f"(need {self.config.min_compliance_score:.0%})", flush=True)

                # Log specific issues
                if result.citation_report:
                    print(f"[RAG_ENFORCER]   - Citation rate: {result.citation_report.citation_rate:.0%}", flush=True)
                if result.sentence_report:
                    print(f"[RAG_ENFORCER]   - Grounding rate: {result.sentence_report.grounding_rate:.0%}", flush=True)
                if result.hallucinations:
                    print(f"[RAG_ENFORCER]   - Hallucinations: {len(result.hallucinations)}", flush=True)

            except Exception as e:
                print(f"[RAG_ENFORCER] Generation error on attempt {attempt}: {e}", flush=True)
                continue

        # Failed to achieve compliance
        if best_result:
            best_result.total_attempts = self.config.max_attempts
            raise RAGComplianceError(
                f"Cannot achieve {self.config.min_compliance_score:.0%} compliance "
                f"after {self.config.max_attempts} attempts. "
                f"Best score: {best_score:.0%}",
                result=best_result
            )
        else:
            raise RAGComplianceError(
                f"All {self.config.max_attempts} generation attempts failed"
            )

    async def _verify_content(
        self,
        content: str,
        sources: List[str],
        attempt: int
    ) -> EnforcementResult:
        """Verify content against sources"""
        result = EnforcementResult(
            content=content,
            attempt_number=attempt,
            total_attempts=self.config.max_attempts
        )

        # 1. Validate citations
        if self.config.require_citations:
            citation_report = self.citation_validator.validate_citations(content, sources)
            result.citation_report = citation_report
            result.citation_score = self._compute_citation_score(citation_report)
        else:
            result.citation_score = 1.0  # Perfect if not required

        # 2. Verify sentences
        sentence_report = self.sentence_verifier.verify_sentences(content, sources)
        result.sentence_report = sentence_report
        result.grounding_score = sentence_report.grounding_rate

        # 3. Extract hallucinations
        for score in sentence_report.sentence_scores:
            if score.fact_status == FactStatus.HALLUCINATION:
                result.hallucinations.append(score.sentence[:100])
            elif score.fact_status == FactStatus.UNSUPPORTED:
                result.ungrounded_facts.append(score.sentence[:100])

        # 4. Compute overall score
        result.overall_score = (
            self.config.citation_weight * result.citation_score +
            self.config.grounding_weight * result.grounding_score
        )

        # 5. Determine compliance level
        if result.overall_score >= self.config.min_compliance_score:
            result.is_compliant = True
            result.compliance_level = ComplianceLevel.COMPLIANT
        elif result.overall_score >= 0.70:
            result.compliance_level = ComplianceLevel.PARTIAL
        elif len(result.hallucinations) > 5:
            result.compliance_level = ComplianceLevel.REJECTED
        else:
            result.compliance_level = ComplianceLevel.NON_COMPLIANT

        return result

    def _compute_citation_score(self, report: CitationReport) -> float:
        """Compute citation-based score"""
        if report.total_sentences == 0:
            return 1.0

        # Factors:
        # 1. Citation rate (% of sentences with citations)
        # 2. Citation validity (% of valid citations)
        citation_rate = report.citation_rate
        validity_rate = report.validity_rate

        # Penalty for uncited sentences
        uncited_penalty = report.uncited_sentences / report.total_sentences

        score = (0.6 * citation_rate + 0.4 * validity_rate) - (0.2 * uncited_penalty)
        return max(0.0, min(1.0, score))

    def verify_only(
        self,
        content: str,
        sources: List[str]
    ) -> EnforcementResult:
        """
        Verify content without generation (synchronous).
        Useful for post-hoc verification.
        """
        result = EnforcementResult(content=content, attempt_number=1)

        # Citation validation
        if self.config.require_citations:
            citation_report = self.citation_validator.validate_citations(content, sources)
            result.citation_report = citation_report
            result.citation_score = self._compute_citation_score(citation_report)
        else:
            result.citation_score = 1.0

        # Sentence verification
        sentence_report = self.sentence_verifier.verify_sentences(content, sources)
        result.sentence_report = sentence_report
        result.grounding_score = sentence_report.grounding_rate

        # Hallucinations
        for score in sentence_report.sentence_scores:
            if score.fact_status == FactStatus.HALLUCINATION:
                result.hallucinations.append(score.sentence[:100])
            elif score.fact_status == FactStatus.UNSUPPORTED:
                result.ungrounded_facts.append(score.sentence[:100])

        # Overall score
        result.overall_score = (
            self.config.citation_weight * result.citation_score +
            self.config.grounding_weight * result.grounding_score
        )

        # Compliance
        result.is_compliant = result.overall_score >= self.config.min_compliance_score
        if result.is_compliant:
            result.compliance_level = ComplianceLevel.COMPLIANT
        elif result.overall_score >= 0.70:
            result.compliance_level = ComplianceLevel.PARTIAL
        else:
            result.compliance_level = ComplianceLevel.NON_COMPLIANT

        return result

    def generate_strictness_prompt(self, level: str) -> str:
        """Get strictness prompt for a given level"""
        return self.config.strictness_prompts.get(level, self.config.strictness_prompts["standard"])


class AsyncRAGEnforcer(RAGEnforcer):
    """Async version with async sentence verification"""

    def __init__(
        self,
        config: Optional[EnforcementConfig] = None,
        async_embedding_func: Optional[Callable] = None
    ):
        super().__init__(config)
        self._async_embed = async_embedding_func
        self.async_sentence_verifier = AsyncSentenceVerifier(config, async_embedding_func)

    async def _verify_content(
        self,
        content: str,
        sources: List[str],
        attempt: int
    ) -> EnforcementResult:
        """Async content verification"""
        result = EnforcementResult(
            content=content,
            attempt_number=attempt,
            total_attempts=self.config.max_attempts
        )

        # Citation validation (sync is fine, it's fast)
        if self.config.require_citations:
            citation_report = self.citation_validator.validate_citations(content, sources)
            result.citation_report = citation_report
            result.citation_score = self._compute_citation_score(citation_report)
        else:
            result.citation_score = 1.0

        # Async sentence verification
        sentence_report = await self.async_sentence_verifier.verify_sentences_async(
            content, sources
        )
        result.sentence_report = sentence_report
        result.grounding_score = sentence_report.grounding_rate

        # Hallucinations
        for score in sentence_report.sentence_scores:
            if score.fact_status == FactStatus.HALLUCINATION:
                result.hallucinations.append(score.sentence[:100])
            elif score.fact_status == FactStatus.UNSUPPORTED:
                result.ungrounded_facts.append(score.sentence[:100])

        # Overall score
        result.overall_score = (
            self.config.citation_weight * result.citation_score +
            self.config.grounding_weight * result.grounding_score
        )

        # Compliance
        result.is_compliant = result.overall_score >= self.config.min_compliance_score
        if result.is_compliant:
            result.compliance_level = ComplianceLevel.COMPLIANT
        elif result.overall_score >= 0.70:
            result.compliance_level = ComplianceLevel.PARTIAL
        else:
            result.compliance_level = ComplianceLevel.NON_COMPLIANT

        return result


# Convenience functions
def create_enforcer(
    min_compliance: float = 0.90,
    max_attempts: int = 3,
    require_citations: bool = True,
    embedding_func: Optional[Callable] = None
) -> RAGEnforcer:
    """Create a configured RAGEnforcer"""
    config = EnforcementConfig(
        min_compliance_score=min_compliance,
        max_attempts=max_attempts,
        require_citations=require_citations
    )
    return RAGEnforcer(config, embedding_func)


def verify_content(
    content: str,
    sources: List[str],
    min_compliance: float = 0.90
) -> EnforcementResult:
    """Quick verification of content against sources"""
    config = EnforcementConfig(min_compliance_score=min_compliance)
    enforcer = RAGEnforcer(config)
    return enforcer.verify_only(content, sources)
