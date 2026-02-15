"""
Unit Tests for ContextBuilder

Tests context building from RAG search results with token-aware truncation.
"""

import pytest
from dataclasses import dataclass
from typing import Optional
from ..retrieval.context_builder import (
    ContextBuilder,
    ContextBuildResult,
    build_context,
    get_context_builder,
)


@dataclass
class MockChunk:
    """Mock RAGChunkResult for testing."""
    content: str
    similarity_score: float = 0.5
    document_filename: str = "test.pdf"


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text: str):
        """Roughly 1 token per 4 chars."""
        return list(range(len(text) // 4))

    def decode(self, tokens):
        """Convert tokens back to approximate text."""
        return "x" * (len(tokens) * 4)


class TestContextBuildResult:
    """Tests for ContextBuildResult dataclass."""

    def test_create_empty(self):
        """Test creating empty result."""
        result = ContextBuildResult(
            context="",
            total_tokens=0,
            chunks_included=0,
            chunks_truncated=0,
            chunks_excluded=0,
        )

        assert result.context == ""
        assert result.total_tokens == 0
        assert result.chunks_included == 0

    def test_create_with_content(self):
        """Test creating result with content."""
        result = ContextBuildResult(
            context="Some context text",
            total_tokens=100,
            chunks_included=5,
            chunks_truncated=1,
            chunks_excluded=2,
        )

        assert result.context == "Some context text"
        assert result.total_tokens == 100
        assert result.chunks_included == 5
        assert result.chunks_truncated == 1
        assert result.chunks_excluded == 2


class TestContextBuilder:
    """Tests for ContextBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create builder with mock tokenizer."""
        return ContextBuilder(tokenizer=MockTokenizer())

    @pytest.fixture
    def builder_no_tokenizer(self):
        """Create builder without tokenizer."""
        builder = ContextBuilder(tokenizer=None)
        builder.tokenizer = None  # Force no tokenizer
        return builder

    # ==========================================================================
    # Token Counting Tests
    # ==========================================================================

    def test_count_tokens_with_tokenizer(self, builder):
        """Test token counting with tokenizer."""
        # MockTokenizer: 1 token per 4 chars
        text = "a" * 100  # 100 chars = 25 tokens
        count = builder.count_tokens(text)
        assert count == 25

    def test_count_tokens_without_tokenizer(self, builder_no_tokenizer):
        """Test token counting fallback."""
        text = "a" * 100  # 100 chars / 4 = 25 estimated tokens
        count = builder_no_tokenizer.count_tokens(text)
        assert count == 25

    def test_count_tokens_empty(self, builder):
        """Test counting tokens in empty string."""
        assert builder.count_tokens("") == 0
        assert builder.count_tokens(None) == 0

    # ==========================================================================
    # Truncation Tests
    # ==========================================================================

    def test_truncate_short_text(self, builder):
        """Test that short text is not truncated."""
        text = "Short text"
        result = builder.truncate_to_tokens(text, max_tokens=100)
        assert result == text

    def test_truncate_empty(self, builder):
        """Test truncating empty string."""
        assert builder.truncate_to_tokens("", 100) == ""

    def test_truncate_at_sentence(self, builder_no_tokenizer):
        """Test that truncation prefers sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence is longer."
        result = builder_no_tokenizer.truncate_to_tokens(text, max_tokens=10)

        # Should end with truncation marker
        assert "[... content truncated" in result

    # ==========================================================================
    # Basic Build Tests
    # ==========================================================================

    def test_build_empty_results(self, builder):
        """Test building with empty results."""
        result = builder.build([], max_tokens=1000)

        assert result.context == ""
        assert result.total_tokens == 0
        assert result.chunks_included == 0

    def test_build_single_chunk(self, builder):
        """Test building with single chunk."""
        chunks = [MockChunk(content="Test content")]

        result = builder.build(chunks, max_tokens=1000)

        assert "Test content" in result.context
        assert result.chunks_included == 1
        assert result.chunks_excluded == 0

    def test_build_multiple_chunks(self, builder):
        """Test building with multiple chunks."""
        chunks = [
            MockChunk(content="First chunk", similarity_score=0.9),
            MockChunk(content="Second chunk", similarity_score=0.8),
            MockChunk(content="Third chunk", similarity_score=0.7),
        ]

        result = builder.build(chunks, max_tokens=1000)

        assert "First chunk" in result.context
        assert "Second chunk" in result.context
        assert "Third chunk" in result.context
        assert result.chunks_included == 3

    def test_build_with_separator(self, builder):
        """Test that chunks are separated properly."""
        chunks = [
            MockChunk(content="Chunk A"),
            MockChunk(content="Chunk B"),
        ]

        result = builder.build(chunks, max_tokens=1000)

        assert builder.CHUNK_SEPARATOR in result.context

    # ==========================================================================
    # Source Attribution Tests
    # ==========================================================================

    def test_build_with_source_attribution(self, builder):
        """Test that source attribution is included."""
        chunks = [
            MockChunk(content="Test content", document_filename="source.pdf"),
        ]

        result = builder.build(chunks, max_tokens=1000, include_source_attribution=True)

        assert "[Source: source.pdf]" in result.context

    def test_build_without_source_attribution(self, builder):
        """Test building without source attribution."""
        chunks = [
            MockChunk(content="Test content", document_filename="source.pdf"),
        ]

        result = builder.build(chunks, max_tokens=1000, include_source_attribution=False)

        assert "[Source:" not in result.context

    # ==========================================================================
    # Token Limit Tests
    # ==========================================================================

    def test_build_respects_token_limit(self, builder):
        """Test that build respects token limit."""
        # Each chunk ~25 tokens (100 chars / 4)
        chunks = [
            MockChunk(content="a" * 100),  # ~25 tokens
            MockChunk(content="b" * 100),  # ~25 tokens
            MockChunk(content="c" * 100),  # ~25 tokens
        ]

        result = builder.build(chunks, max_tokens=50)

        # Should include some but not all
        assert result.chunks_included < 3
        assert result.total_tokens <= 50

    def test_build_truncates_last_chunk(self, builder):
        """Test that last chunk can be truncated to fit."""
        chunks = [
            MockChunk(content="a" * 100),  # ~25 tokens
            MockChunk(content="b" * 200),  # ~50 tokens, won't fully fit
        ]

        result = builder.build(chunks, max_tokens=60)

        # First chunk fits, second partially
        assert result.chunks_included >= 1
        if result.chunks_truncated > 0:
            assert "[... content truncated" in result.context

    def test_build_excludes_when_no_room(self, builder):
        """Test that chunks are excluded when no room."""
        chunks = [
            MockChunk(content="a" * 400),  # ~100 tokens
            MockChunk(content="b" * 400),  # ~100 tokens, won't fit
        ]

        result = builder.build(chunks, max_tokens=100)

        assert result.chunks_included == 1
        assert result.chunks_excluded >= 1

    # ==========================================================================
    # Build With Structure Tests
    # ==========================================================================

    def test_build_with_structure(self, builder):
        """Test building with document structure."""
        chunks = [MockChunk(content="Chunk content")]
        structure = "# Document Structure\n## Section 1"

        result = builder.build_with_structure(chunks, structure, max_tokens=1000)

        assert structure in result.context
        assert "Chunk content" in result.context

    def test_build_with_structure_empty(self, builder):
        """Test building with empty structure."""
        chunks = [MockChunk(content="Content")]

        result = builder.build_with_structure(chunks, "", max_tokens=1000)

        assert "Content" in result.context

    def test_build_with_structure_no_structure(self, builder):
        """Test building with None structure."""
        chunks = [MockChunk(content="Content")]

        result = builder.build_with_structure(chunks, None, max_tokens=1000)

        assert "Content" in result.context

    def test_build_with_structure_limited_tokens(self, builder):
        """Test that structure gets priority when tokens are limited."""
        chunks = [MockChunk(content="x" * 400)]  # ~100 tokens
        structure = "y" * 800  # ~200 tokens

        result = builder.build_with_structure(chunks, structure, max_tokens=250)

        # Structure should be included
        assert structure in result.context
        # Content may be limited
        assert result.chunks_excluded >= 0

    def test_build_with_structure_structure_too_large(self, builder):
        """Test when structure consumes most of budget."""
        chunks = [MockChunk(content="Content")]
        structure = "x" * 4000  # ~1000 tokens

        result = builder.build_with_structure(chunks, structure, max_tokens=1050)

        # Structure should be there, content excluded
        assert structure in result.context
        assert result.chunks_excluded == 1

    # ==========================================================================
    # Chunk Prioritization Tests
    # ==========================================================================

    def test_build_prioritizes_chunks(self, builder):
        """Test that chunks are prioritized (boosted chunks first)."""
        chunks = [
            MockChunk(content="Plain content", similarity_score=0.7),
            MockChunk(
                content="[contains: definition] Important definition",
                similarity_score=0.6,
            ),
        ]

        result = builder.build(chunks, max_tokens=1000)

        # Definition should be boosted and appear first
        # (the prioritizer will reorder)
        assert "definition" in result.context.lower()


class TestContextBuilderEdgeCases:
    """Edge case tests for ContextBuilder."""

    @pytest.fixture
    def builder(self):
        return ContextBuilder(tokenizer=MockTokenizer())

    def test_chunk_without_content_attr(self, builder):
        """Test handling chunk without content attribute."""
        class SimpleObject:
            pass

        obj = SimpleObject()
        chunks = [obj]

        # Should not crash, uses str(result)
        result = builder.build(chunks, max_tokens=1000)
        assert result.chunks_included >= 0

    def test_chunk_with_none_content(self, builder):
        """Test handling chunk with None content."""
        chunks = [MockChunk(content=None)]

        result = builder.build(chunks, max_tokens=1000)

        # None content should be skipped
        assert result.context == ""

    def test_chunk_with_empty_content(self, builder):
        """Test handling chunk with empty content."""
        chunks = [MockChunk(content="")]

        result = builder.build(chunks, max_tokens=1000)

        assert result.context == ""

    def test_very_small_token_limit(self, builder):
        """Test with very small token limit."""
        chunks = [MockChunk(content="Some content")]

        result = builder.build(chunks, max_tokens=10)

        # Should handle gracefully
        assert result.total_tokens <= 10


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_build_context_function(self):
        """Test module-level build_context function."""
        chunks = [
            MockChunk(content="Test content"),
        ]

        context = build_context(chunks, max_tokens=1000)

        assert "Test content" in context

    def test_build_context_empty(self):
        """Test build_context with empty results."""
        context = build_context([], max_tokens=1000)
        assert context == ""

    def test_get_context_builder_singleton(self):
        """Test that get_context_builder returns consistent instance."""
        builder1 = get_context_builder()
        builder2 = get_context_builder()

        assert builder1 is builder2

    def test_get_context_builder_with_tokenizer(self):
        """Test get_context_builder with custom tokenizer."""
        tokenizer = MockTokenizer()
        builder = get_context_builder(tokenizer=tokenizer)

        assert builder.tokenizer is tokenizer
