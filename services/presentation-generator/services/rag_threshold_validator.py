"""
RAG Threshold Validator for Presentation Generator

Validates that RAG context meets minimum requirements before generation.
Prevents hallucinations by blocking or warning when source documents
are insufficient.

Thresholds:
- MINIMUM_TOKENS (500): Below this, generation is BLOCKED
- QUALITY_TOKENS (2000): Below this, generation continues with WARNING
- Above QUALITY_TOKENS: Full RAG mode, optimal coverage expected
"""
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import tiktoken


class RAGMode(str, Enum):
    """RAG operation mode based on available context"""
    FULL = "full"  # Sufficient context (>2000 tokens)
    PARTIAL = "partial"  # Limited context (500-2000 tokens), warning issued
    BLOCKED = "blocked"  # Insufficient context (<500 tokens), generation refused
    NONE = "none"  # No RAG context provided (standard generation)


@dataclass
class RAGThresholdResult:
    """Result of RAG threshold validation"""
    mode: RAGMode
    token_count: int
    is_sufficient: bool  # True if generation should proceed
    warning_message: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def should_block(self) -> bool:
        return self.mode == RAGMode.BLOCKED

    @property
    def has_warning(self) -> bool:
        return self.warning_message is not None


class RAGThresholdValidator:
    """
    Validates RAG context meets minimum thresholds.

    Prevents the LLM from compensating with its own knowledge
    when source documents are insufficient.
    """

    # Default thresholds (can be overridden via env vars)
    DEFAULT_MINIMUM_TOKENS = 500  # Hard block below this
    DEFAULT_QUALITY_TOKENS = 2000  # Warning below this
    DEFAULT_OPTIMAL_TOKENS = 4000  # Ideal minimum for comprehensive coverage

    def __init__(
        self,
        minimum_tokens: Optional[int] = None,
        quality_tokens: Optional[int] = None,
    ):
        """
        Initialize the validator.

        Args:
            minimum_tokens: Hard minimum (block below this)
            quality_tokens: Quality threshold (warn below this)
        """
        self.minimum_tokens = minimum_tokens or int(
            os.getenv("RAG_MINIMUM_TOKENS", self.DEFAULT_MINIMUM_TOKENS)
        )
        self.quality_tokens = quality_tokens or int(
            os.getenv("RAG_QUALITY_TOKENS", self.DEFAULT_QUALITY_TOKENS)
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

    def validate(
        self,
        rag_context: Optional[str],
        has_documents: bool = False,
        strict_mode: bool = False,
    ) -> RAGThresholdResult:
        """
        Validate RAG context meets thresholds.

        Args:
            rag_context: The RAG context string
            has_documents: Whether documents were provided
            strict_mode: If True, treat PARTIAL as BLOCKED

        Returns:
            RAGThresholdResult with mode and messages
        """
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
            )

        # Below quality threshold - WARNING
        if token_count < self.quality_tokens:
            if strict_mode:
                return RAGThresholdResult(
                    mode=RAGMode.BLOCKED,
                    token_count=token_count,
                    is_sufficient=False,
                    error_message=(
                        f"Source content below quality threshold: {token_count} tokens "
                        f"(required in strict mode: {self.quality_tokens}). "
                        "Please provide more comprehensive documents."
                    ),
                )

            return RAGThresholdResult(
                mode=RAGMode.PARTIAL,
                token_count=token_count,
                is_sufficient=True,  # Allow with warning
                warning_message=(
                    f"Limited source content: {token_count} tokens retrieved "
                    f"(recommended: {self.quality_tokens}+). "
                    "Some content may be supplemented with AI knowledge. "
                    "For 90%+ source-based content, provide more comprehensive documents."
                ),
            )

        # Above quality threshold - FULL mode
        return RAGThresholdResult(
            mode=RAGMode.FULL,
            token_count=token_count,
            is_sufficient=True,
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
) -> RAGThresholdResult:
    """
    Convenience function to validate RAG threshold.

    Args:
        rag_context: The RAG context string
        has_documents: Whether documents were provided

    Returns:
        RAGThresholdResult
    """
    validator = get_rag_threshold_validator()
    return validator.validate(rag_context, has_documents)
