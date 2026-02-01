"""
RAG Threshold Validator for Presentation Generator (v2 REINFORCED)

Validates that RAG context meets minimum requirements before generation.
Prevents hallucinations by blocking or warning when source documents
are insufficient.

v2 Reinforced improvements:
- Stricter token thresholds (750/3000 vs 500/2000)
- Topic relevance validation (content must cover requested topic)
- Document diversity check (multiple sources preferred)
- Content quality scoring

Thresholds:
- MINIMUM_TOKENS (750): Below this, generation is BLOCKED
- QUALITY_TOKENS (3000): Below this, generation continues with WARNING
- OPTIMAL_TOKENS (5000): Ideal minimum for comprehensive coverage
- Above QUALITY_TOKENS: Full RAG mode, optimal coverage expected

Environment Variables:
- RAG_MINIMUM_TOKENS: Hard block threshold (default: 750)
- RAG_QUALITY_TOKENS: Warning threshold (default: 3000)
- RAG_STRICT_THRESHOLD: Enable strict mode (default: false)
"""
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple

import tiktoken


class RAGMode(str, Enum):
    """RAG operation mode based on available context"""
    FULL = "full"  # Sufficient context (>3000 tokens)
    PARTIAL = "partial"  # Limited context (750-3000 tokens), warning issued
    BLOCKED = "blocked"  # Insufficient context (<750 tokens), generation refused
    NONE = "none"  # No RAG context provided (standard generation)


@dataclass
class RAGThresholdResult:
    """Result of RAG threshold validation"""
    mode: RAGMode
    token_count: int
    is_sufficient: bool  # True if generation should proceed
    warning_message: Optional[str] = None
    error_message: Optional[str] = None

    # v2 Reinforced fields
    topic_relevance_score: float = 1.0  # 0-1, how relevant content is to topic
    topic_coverage_issues: List[str] = field(default_factory=list)
    content_quality_score: float = 1.0  # 0-1, overall content quality
    unique_sources_count: int = 0  # Number of distinct document sources
    density_score: float = 1.0  # Information density (unique terms / tokens)

    @property
    def should_block(self) -> bool:
        return self.mode == RAGMode.BLOCKED

    @property
    def has_warning(self) -> bool:
        return self.warning_message is not None

    @property
    def quality_grade(self) -> str:
        """Return a quality grade A-F based on combined metrics."""
        combined = (
            (self.topic_relevance_score * 0.4) +
            (self.content_quality_score * 0.3) +
            (self.density_score * 0.3)
        )
        if combined >= 0.9:
            return "A"
        elif combined >= 0.75:
            return "B"
        elif combined >= 0.6:
            return "C"
        elif combined >= 0.4:
            return "D"
        return "F"


class RAGThresholdValidator:
    """
    Validates RAG context meets minimum thresholds (v2 REINFORCED).

    Prevents the LLM from compensating with its own knowledge
    when source documents are insufficient.

    v2 improvements:
    - Stricter token thresholds
    - Topic relevance validation
    - Content quality scoring
    - Document diversity check
    """

    # v2 REINFORCED: Stricter default thresholds
    DEFAULT_MINIMUM_TOKENS = 750   # Hard block below this (was 500)
    DEFAULT_QUALITY_TOKENS = 3000  # Warning below this (was 2000)
    DEFAULT_OPTIMAL_TOKENS = 5000  # Ideal minimum for comprehensive coverage (was 4000)

    # Strict mode thresholds (even higher requirements)
    STRICT_MINIMUM_TOKENS = 1000
    STRICT_QUALITY_TOKENS = 4000

    # Topic relevance thresholds
    MIN_TOPIC_RELEVANCE = 0.30  # Minimum relevance score to proceed
    WARN_TOPIC_RELEVANCE = 0.50  # Warn if below this

    def __init__(
        self,
        minimum_tokens: Optional[int] = None,
        quality_tokens: Optional[int] = None,
        strict_mode: bool = False,
    ):
        """
        Initialize the validator.

        Args:
            minimum_tokens: Hard minimum (block below this)
            quality_tokens: Quality threshold (warn below this)
            strict_mode: Use stricter thresholds
        """
        self.strict_mode = strict_mode or os.getenv("RAG_STRICT_THRESHOLD", "false").lower() == "true"

        if self.strict_mode:
            default_min = self.STRICT_MINIMUM_TOKENS
            default_quality = self.STRICT_QUALITY_TOKENS
        else:
            default_min = self.DEFAULT_MINIMUM_TOKENS
            default_quality = self.DEFAULT_QUALITY_TOKENS

        self.minimum_tokens = minimum_tokens or int(
            os.getenv("RAG_MINIMUM_TOKENS", default_min)
        )
        self.quality_tokens = quality_tokens or int(
            os.getenv("RAG_QUALITY_TOKENS", default_quality)
        )

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def _extract_unique_terms(self, text: str) -> set:
        """Extract unique significant terms from text."""
        # Remove common stopwords and short words
        stopwords = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has',
            'will', 'can', 'are', 'was', 'were', 'been', 'being', 'would', 'could',
            'vous', 'nous', 'elle', 'sont', 'avec', 'pour', 'dans', 'cette',
        }
        words = re.findall(r'\b[a-zA-Z\u00C0-\u017F]{4,}\b', text.lower())
        return {w for w in words if w not in stopwords}

    def _calculate_topic_relevance(
        self,
        rag_context: str,
        topic: Optional[str]
    ) -> Tuple[float, List[str]]:
        """
        Calculate how relevant the RAG context is to the requested topic.

        Returns:
            Tuple of (relevance_score, coverage_issues)
        """
        if not topic:
            return 1.0, []

        topic_lower = topic.lower()
        context_lower = rag_context.lower()
        issues = []

        # Extract topic keywords (split on spaces, remove common words)
        topic_words = set(topic_lower.split())
        topic_words = {w for w in topic_words if len(w) > 3}

        if not topic_words:
            return 1.0, []

        # Check presence of topic keywords in context
        found_keywords = sum(1 for w in topic_words if w in context_lower)
        keyword_coverage = found_keywords / len(topic_words) if topic_words else 1.0

        if keyword_coverage < 0.3:
            issues.append(f"Topic keywords poorly covered ({keyword_coverage:.0%})")

        # Check if topic phrase appears in context
        topic_phrase_found = topic_lower in context_lower
        if not topic_phrase_found and keyword_coverage < 0.5:
            issues.append("Topic phrase not found in source documents")

        # Calculate relevance score
        relevance = (keyword_coverage * 0.7) + (0.3 if topic_phrase_found else 0.0)

        return relevance, issues

    def _calculate_content_quality(self, rag_context: str) -> Tuple[float, float]:
        """
        Calculate content quality metrics.

        Returns:
            Tuple of (quality_score, density_score)
        """
        tokens = self.count_tokens(rag_context)
        unique_terms = self._extract_unique_terms(rag_context)

        # Density: unique terms / tokens (capped at reasonable range)
        density = len(unique_terms) / max(tokens, 1)
        density_normalized = min(1.0, density * 10)  # Normalize to 0-1

        # Quality factors
        # 1. Has structure (headers, lists)
        has_structure = bool(re.search(r'(^|\n)(#+\s|[-*]\s|\d+\.\s)', rag_context))

        # 2. Has technical terms (CamelCase, acronyms)
        has_technical = bool(re.search(r'\b[A-Z][a-z]+[A-Z]|[A-Z]{2,5}\b', rag_context))

        # 3. Has code or examples
        has_examples = bool(re.search(r'```|`[^`]+`|example|exemple', rag_context, re.IGNORECASE))

        # 4. Reasonable paragraph length (not all one blob)
        paragraphs = rag_context.split('\n\n')
        has_paragraphs = len(paragraphs) >= 3

        quality_factors = [has_structure, has_technical, has_examples, has_paragraphs]
        quality_score = sum(quality_factors) / len(quality_factors)

        return quality_score, density_normalized

    def _count_unique_sources(self, rag_context: str) -> int:
        """
        Estimate the number of unique document sources in context.

        Looks for document markers, section separators, or distinct sections.
        """
        # Look for common document separator patterns
        separators = [
            r'---+',  # Horizontal rules
            r'Document \d+',  # Document markers
            r'\[Source:',  # Source citations
            r'From:.*\.pdf',  # PDF references
            r'#{2,}\s+',  # Multiple headers (H2+)
        ]

        source_count = 1  # At least one source
        for pattern in separators:
            matches = len(re.findall(pattern, rag_context, re.IGNORECASE))
            source_count = max(source_count, matches + 1)

        return min(source_count, 10)  # Cap at reasonable number

    def validate(
        self,
        rag_context: Optional[str],
        has_documents: bool = False,
        strict_mode: bool = False,
        topic: Optional[str] = None,
    ) -> RAGThresholdResult:
        """
        Validate RAG context meets thresholds (v2 REINFORCED).

        Args:
            rag_context: The RAG context string
            has_documents: Whether documents were provided
            strict_mode: If True, treat PARTIAL as BLOCKED
            topic: Optional topic to check relevance against

        Returns:
            RAGThresholdResult with mode, messages, and quality metrics
        """
        # Use instance strict_mode if not overridden
        strict_mode = strict_mode or self.strict_mode

        # No RAG context at all
        if not rag_context or not rag_context.strip():
            if has_documents:
                # Documents were provided but no content retrieved
                return RAGThresholdResult(
                    mode=RAGMode.BLOCKED,
                    token_count=0,
                    is_sufficient=False,
                    error_message=(
                        "No content could be extracted from the provided documents. "
                        "Please check that the documents contain readable text content "
                        "related to the topic."
                    ),
                    topic_relevance_score=0.0,
                    content_quality_score=0.0,
                )
            else:
                # No documents provided - standard generation mode
                return RAGThresholdResult(
                    mode=RAGMode.NONE,
                    token_count=0,
                    is_sufficient=True,  # Allow standard generation
                    warning_message=(
                        "No source documents provided. Content will be generated "
                        "using AI knowledge only. For more accurate content, "
                        "consider uploading reference documents."
                    ),
                )

        # Count tokens
        token_count = self.count_tokens(rag_context)

        # v2: Calculate additional quality metrics
        topic_relevance, topic_issues = self._calculate_topic_relevance(rag_context, topic)
        quality_score, density_score = self._calculate_content_quality(rag_context)
        unique_sources = self._count_unique_sources(rag_context)

        # v2: Check topic relevance (can block even with sufficient tokens)
        if topic and topic_relevance < self.MIN_TOPIC_RELEVANCE:
            return RAGThresholdResult(
                mode=RAGMode.BLOCKED,
                token_count=token_count,
                is_sufficient=False,
                error_message=(
                    f"Source content is not sufficiently relevant to the topic '{topic}'. "
                    f"Relevance score: {topic_relevance:.0%} (minimum: {self.MIN_TOPIC_RELEVANCE:.0%}). "
                    "Please provide documents that directly cover the requested topic."
                ),
                topic_relevance_score=topic_relevance,
                topic_coverage_issues=topic_issues,
                content_quality_score=quality_score,
                density_score=density_score,
                unique_sources_count=unique_sources,
            )

        # Below minimum threshold - BLOCK
        if token_count < self.minimum_tokens:
            return RAGThresholdResult(
                mode=RAGMode.BLOCKED,
                token_count=token_count,
                is_sufficient=False,
                error_message=(
                    f"Insufficient source content: {token_count} tokens retrieved "
                    f"(minimum required: {self.minimum_tokens}). "
                    "Please provide more comprehensive documents or check that "
                    "the documents cover the requested topic."
                ),
                topic_relevance_score=topic_relevance,
                topic_coverage_issues=topic_issues,
                content_quality_score=quality_score,
                density_score=density_score,
                unique_sources_count=unique_sources,
            )

        # Below quality threshold - WARNING or BLOCK in strict mode
        if token_count < self.quality_tokens:
            # Build warning message with quality insights
            warnings = []
            warnings.append(
                f"Limited source content: {token_count} tokens "
                f"(recommended: {self.quality_tokens}+)"
            )

            if topic_relevance < self.WARN_TOPIC_RELEVANCE:
                warnings.append(f"Low topic relevance: {topic_relevance:.0%}")
            if topic_issues:
                warnings.extend(topic_issues)
            if unique_sources < 2:
                warnings.append("Single source document (multiple sources recommended)")
            if quality_score < 0.5:
                warnings.append(f"Content quality below optimal: {quality_score:.0%}")

            warning_msg = " | ".join(warnings)

            if strict_mode:
                return RAGThresholdResult(
                    mode=RAGMode.BLOCKED,
                    token_count=token_count,
                    is_sufficient=False,
                    error_message=(
                        f"Source content below quality threshold in strict mode. {warning_msg}. "
                        f"Required: {self.quality_tokens}+ tokens with high relevance."
                    ),
                    topic_relevance_score=topic_relevance,
                    topic_coverage_issues=topic_issues,
                    content_quality_score=quality_score,
                    density_score=density_score,
                    unique_sources_count=unique_sources,
                )

            return RAGThresholdResult(
                mode=RAGMode.PARTIAL,
                token_count=token_count,
                is_sufficient=True,  # Allow with warning
                warning_message=(
                    f"{warning_msg}. "
                    "Some content may be supplemented with AI knowledge. "
                    "For 90%+ source-based content, provide more comprehensive documents."
                ),
                topic_relevance_score=topic_relevance,
                topic_coverage_issues=topic_issues,
                content_quality_score=quality_score,
                density_score=density_score,
                unique_sources_count=unique_sources,
            )

        # Above quality threshold - FULL mode
        # Still add warnings if quality metrics are suboptimal
        warning_msg = None
        if topic_relevance < self.WARN_TOPIC_RELEVANCE or quality_score < 0.6:
            warnings = []
            if topic_relevance < self.WARN_TOPIC_RELEVANCE:
                warnings.append(f"Topic relevance could be improved: {topic_relevance:.0%}")
            if quality_score < 0.6:
                warnings.append(f"Content structure could be improved: {quality_score:.0%}")
            warning_msg = " | ".join(warnings)

        return RAGThresholdResult(
            mode=RAGMode.FULL,
            token_count=token_count,
            is_sufficient=True,
            warning_message=warning_msg,
            topic_relevance_score=topic_relevance,
            topic_coverage_issues=topic_issues,
            content_quality_score=quality_score,
            density_score=density_score,
            unique_sources_count=unique_sources,
        )


# Singleton instance
_validator_instance: Optional[RAGThresholdValidator] = None


def get_rag_threshold_validator() -> RAGThresholdValidator:
    """Get the singleton validator instance"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = RAGThresholdValidator()
    return _validator_instance


def validate_rag_threshold(
    rag_context: Optional[str],
    has_documents: bool = False,
    topic: Optional[str] = None,
    strict_mode: bool = False,
) -> RAGThresholdResult:
    """
    Convenience function to validate RAG threshold (v2 REINFORCED).

    Args:
        rag_context: The RAG context string
        has_documents: Whether documents were provided
        topic: Optional topic to validate relevance against
        strict_mode: Use stricter validation thresholds

    Returns:
        RAGThresholdResult with mode, messages, and quality metrics
    """
    validator = get_rag_threshold_validator()
    return validator.validate(rag_context, has_documents, strict_mode=strict_mode, topic=topic)


def validate_rag_for_generation(
    rag_context: Optional[str],
    topic: str,
    document_ids: Optional[List[str]] = None,
    strict_mode: bool = False,
) -> RAGThresholdResult:
    """
    Validate RAG context specifically for presentation generation.

    This is the recommended entry point for pre-generation validation.

    Args:
        rag_context: The RAG context string
        topic: The presentation topic (required for relevance check)
        document_ids: List of document IDs (to determine if documents were provided)
        strict_mode: Use stricter validation thresholds

    Returns:
        RAGThresholdResult with comprehensive validation
    """
    has_documents = bool(document_ids and len(document_ids) > 0)
    validator = get_rag_threshold_validator()
    return validator.validate(
        rag_context,
        has_documents=has_documents,
        strict_mode=strict_mode,
        topic=topic
    )
