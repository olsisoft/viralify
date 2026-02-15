"""
Context Builder for RAG

Builds combined context from search results with token-aware truncation
and smart prioritization.
"""

from typing import List, Optional
from dataclasses import dataclass

from .chunk_prioritizer import ChunkPrioritizer, get_chunk_prioritizer


@dataclass
class ContextBuildResult:
    """Result of context building."""
    context: str
    total_tokens: int
    chunks_included: int
    chunks_truncated: int
    chunks_excluded: int


class ContextBuilder:
    """
    Build combined context from RAG search results.

    Features:
    - Token-aware truncation
    - Chunk prioritization (definitions, examples boost)
    - Graceful truncation at sentence boundaries
    - Clear source attribution

    Usage:
        builder = ContextBuilder(tokenizer)
        result = builder.build(search_results, max_tokens=8000)
    """

    # Separator between chunks
    CHUNK_SEPARATOR = "\n\n---\n\n"

    def __init__(
        self,
        tokenizer=None,
        prioritizer: ChunkPrioritizer = None,
    ):
        """
        Initialize the context builder.

        Args:
            tokenizer: Tiktoken tokenizer for token counting
            prioritizer: Chunk prioritizer (uses default if not provided)
        """
        self.tokenizer = tokenizer or self._get_default_tokenizer()
        self.prioritizer = prioritizer or get_chunk_prioritizer()

    def _get_default_tokenizer(self):
        """Get default tokenizer."""
        try:
            import tiktoken
            return tiktoken.encoding_for_model("gpt-4")
        except (ImportError, KeyError):
            return None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # Rough estimate

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.

        Attempts to end at a sentence boundary for readability.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text
        """
        if not text:
            return ""

        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text

            # Truncate tokens and decode
            truncated_tokens = tokens[:max_tokens - 20]  # Reserve for message
            truncated_text = self.tokenizer.decode(truncated_tokens)
        else:
            # Estimate: 4 chars per token
            char_limit = max_tokens * 4
            if len(text) <= char_limit:
                return text
            truncated_text = text[:char_limit]

        # Try to end at sentence boundary
        last_period = truncated_text.rfind('.')
        if last_period > len(truncated_text) * 0.7:
            truncated_text = truncated_text[:last_period + 1]

        return truncated_text + "\n\n[... content truncated due to length ...]"

    def build(
        self,
        results: List,  # List[RAGChunkResult]
        max_tokens: int,
        include_source_attribution: bool = True,
    ) -> ContextBuildResult:
        """
        Build combined context from search results.

        Args:
            results: List of RAGChunkResult from vector search
            max_tokens: Maximum tokens for combined context
            include_source_attribution: Include source info in output

        Returns:
            ContextBuildResult with combined context and statistics
        """
        if not results:
            return ContextBuildResult(
                context="",
                total_tokens=0,
                chunks_included=0,
                chunks_truncated=0,
                chunks_excluded=0,
            )

        # Prioritize chunks (boost definitions, examples, etc.)
        sorted_results = self.prioritizer.prioritize(results)

        context_parts = []
        current_tokens = 0
        chunks_truncated = 0
        separator_tokens = self.count_tokens(self.CHUNK_SEPARATOR)

        for i, result in enumerate(sorted_results):
            # Get content from result
            content = getattr(result, 'content', str(result))
            if not content:
                continue

            # Add source attribution if requested
            if include_source_attribution:
                source = getattr(result, 'document_filename', None)
                if source:
                    content = f"[Source: {source}]\n{content}"

            chunk_tokens = self.count_tokens(content)

            # Check if chunk fits
            remaining_tokens = max_tokens - current_tokens

            if i > 0:
                remaining_tokens -= separator_tokens

            if chunk_tokens <= remaining_tokens:
                # Chunk fits completely
                context_parts.append(content)
                current_tokens += chunk_tokens
                if i > 0:
                    current_tokens += separator_tokens
            elif remaining_tokens > 100:
                # Partial fit - truncate
                truncated = self.truncate_to_tokens(content, remaining_tokens)
                context_parts.append(truncated)
                current_tokens += self.count_tokens(truncated)
                chunks_truncated += 1
                break
            else:
                # No more room
                break

        combined = self.CHUNK_SEPARATOR.join(context_parts)

        return ContextBuildResult(
            context=combined,
            total_tokens=current_tokens,
            chunks_included=len(context_parts),
            chunks_truncated=chunks_truncated,
            chunks_excluded=len(sorted_results) - len(context_parts),
        )

    def build_with_structure(
        self,
        results: List,  # List[RAGChunkResult]
        document_structure: str,
        max_tokens: int,
    ) -> ContextBuildResult:
        """
        Build context with document structure prepended.

        The document structure is prioritized over chunk content.

        Args:
            results: Search results
            document_structure: Extracted document structure
            max_tokens: Total token budget

        Returns:
            ContextBuildResult with structure + content
        """
        if not document_structure:
            return self.build(results, max_tokens)

        # Reserve tokens for structure
        structure_tokens = self.count_tokens(document_structure)
        content_tokens = max_tokens - structure_tokens - 100  # Buffer

        if content_tokens < 200:
            # Structure takes most of the budget
            return ContextBuildResult(
                context=document_structure,
                total_tokens=structure_tokens,
                chunks_included=0,
                chunks_truncated=0,
                chunks_excluded=len(results),
            )

        # Build content with remaining budget
        content_result = self.build(results, content_tokens)

        # Combine
        combined = f"{document_structure}\n\n{content_result.context}"

        return ContextBuildResult(
            context=combined,
            total_tokens=structure_tokens + content_result.total_tokens,
            chunks_included=content_result.chunks_included,
            chunks_truncated=content_result.chunks_truncated,
            chunks_excluded=content_result.chunks_excluded,
        )


# Module-level instance
_default_builder = None


def get_context_builder(tokenizer=None) -> ContextBuilder:
    """Get or create a context builder instance."""
    global _default_builder
    if _default_builder is None or tokenizer is not None:
        _default_builder = ContextBuilder(tokenizer)
    return _default_builder


def build_context(
    results: List,
    max_tokens: int,
) -> str:
    """
    Convenience function to build context.

    Args:
        results: Search results
        max_tokens: Token limit

    Returns:
        Combined context string
    """
    return get_context_builder().build(results, max_tokens).context
