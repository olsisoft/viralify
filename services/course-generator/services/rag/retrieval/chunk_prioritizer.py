"""
Chunk Prioritizer for RAG

Prioritizes search result chunks based on content importance markers.
Boosts definitions, examples, code, and visual content.
"""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PrioritizedChunk:
    """Chunk with priority score."""
    chunk: object  # RAGChunkResult
    priority_score: float
    boost_reasons: List[str]


class ChunkPrioritizer:
    """
    Prioritize RAG chunks based on content importance.

    Boosts scores for:
    - Definitions and key concepts (+0.15)
    - Examples (+0.10)
    - Code snippets (+0.05)
    - Content with images (+0.05)

    Usage:
        prioritizer = ChunkPrioritizer()
        sorted_chunks = prioritizer.prioritize(chunks)
    """

    # Boost values for different content types
    BOOST_DEFINITION = 0.15
    BOOST_EXAMPLE = 0.10
    BOOST_CODE = 0.05
    BOOST_VISUAL = 0.05
    BOOST_KEY_CONCEPT = 0.10

    # Content markers (from SemanticChunker enriched format)
    DEFINITION_MARKERS = [
        '[contains: definition',
        'key concept',
        'définition:',
        'definition:',
    ]

    EXAMPLE_MARKERS = [
        '[contains: example',
        'contains: example',
        'exemple:',
        'example:',
        'for example',
        'par exemple',
    ]

    CODE_MARKERS = [
        '[content type: code',
        'contains: code',
        '```',
        'def ',
        'function ',
        'class ',
    ]

    VISUAL_MARKERS = [
        '[associated visuals:',
        '[image:',
        '[diagram:',
        '[figure:',
    ]

    def prioritize(
        self,
        chunks: List,  # List[RAGChunkResult]
    ) -> List:
        """
        Sort chunks by priority score.

        Args:
            chunks: List of RAGChunkResult objects

        Returns:
            Sorted list (highest priority first)
        """
        if not chunks:
            return []

        scored_chunks = []

        for chunk in chunks:
            result = self._score_chunk(chunk)
            scored_chunks.append((chunk, result.priority_score))

        # Sort by priority (highest first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [c[0] for c in scored_chunks]

    def prioritize_with_scores(
        self,
        chunks: List,
    ) -> List[PrioritizedChunk]:
        """
        Prioritize chunks and return with scores and boost reasons.

        Args:
            chunks: List of RAGChunkResult objects

        Returns:
            List of PrioritizedChunk with scores
        """
        if not chunks:
            return []

        results = []
        for chunk in chunks:
            result = self._score_chunk(chunk)
            results.append(result)

        # Sort by priority
        results.sort(key=lambda x: x.priority_score, reverse=True)

        return results

    def _score_chunk(self, chunk) -> PrioritizedChunk:
        """
        Calculate priority score for a chunk.

        Args:
            chunk: RAGChunkResult object

        Returns:
            PrioritizedChunk with score
        """
        # Start with similarity score
        base_score = getattr(chunk, 'similarity_score', 0.5)
        boost_reasons = []

        # Get content for marker detection
        content = getattr(chunk, 'content', str(chunk))
        content_lower = content.lower() if content else ''

        # Check for definition markers
        if self._has_markers(content_lower, self.DEFINITION_MARKERS):
            base_score += self.BOOST_DEFINITION
            boost_reasons.append('definition')

        # Check for example markers
        if self._has_markers(content_lower, self.EXAMPLE_MARKERS):
            base_score += self.BOOST_EXAMPLE
            boost_reasons.append('example')

        # Check for code markers
        if self._has_markers(content_lower, self.CODE_MARKERS):
            base_score += self.BOOST_CODE
            boost_reasons.append('code')

        # Check for visual markers
        if self._has_markers(content_lower, self.VISUAL_MARKERS):
            base_score += self.BOOST_VISUAL
            boost_reasons.append('visual')

        return PrioritizedChunk(
            chunk=chunk,
            priority_score=base_score,
            boost_reasons=boost_reasons,
        )

    def _has_markers(self, content: str, markers: List[str]) -> bool:
        """Check if content contains any of the markers."""
        return any(marker in content for marker in markers)

    def get_boost_summary(
        self,
        chunks: List[PrioritizedChunk],
    ) -> dict:
        """
        Get summary of boosts applied.

        Args:
            chunks: List of PrioritizedChunk

        Returns:
            Summary dict with counts
        """
        summary = {
            'total_chunks': len(chunks),
            'boosted_chunks': 0,
            'boost_counts': {
                'definition': 0,
                'example': 0,
                'code': 0,
                'visual': 0,
            },
        }

        for chunk in chunks:
            if chunk.boost_reasons:
                summary['boosted_chunks'] += 1
                for reason in chunk.boost_reasons:
                    if reason in summary['boost_counts']:
                        summary['boost_counts'][reason] += 1

        return summary


# Module-level instance
_default_prioritizer = None


def get_chunk_prioritizer() -> ChunkPrioritizer:
    """Get the default chunk prioritizer instance."""
    global _default_prioritizer
    if _default_prioritizer is None:
        _default_prioritizer = ChunkPrioritizer()
    return _default_prioritizer


def prioritize_chunks(chunks: List) -> List:
    """
    Convenience function to prioritize chunks.

    Args:
        chunks: List of RAGChunkResult

    Returns:
        Sorted list by priority
    """
    return get_chunk_prioritizer().prioritize(chunks)
