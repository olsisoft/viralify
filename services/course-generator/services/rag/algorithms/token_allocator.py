"""
Token Allocation for Weighted RAG

Allocates token budget to documents based on their relevance scores.
Ensures proportional representation while guaranteeing minimum allocations.
"""

from typing import List, Tuple

from ..models.scoring import DocumentRelevanceScore


class TokenAllocator:
    """
    Allocate token budget to documents based on relevance scores.

    Implements:
    - Proportional allocation based on normalized scores
    - Minimum guaranteed allocation for included documents
    - Threshold filtering for irrelevant documents
    - Budget scaling to prevent overflow

    Usage:
        allocator = TokenAllocator(
            min_relevance_threshold=0.25,
            min_allocation_percent=0.10
        )
        scores = allocator.allocate(scores, total_budget=8000)
    """

    def __init__(
        self,
        min_relevance_threshold: float = 0.25,
        min_allocation_percent: float = 0.10,
    ):
        """
        Initialize the token allocator.

        Args:
            min_relevance_threshold: Documents below this score are excluded
            min_allocation_percent: Minimum % of budget for included documents
        """
        self.min_relevance_threshold = min_relevance_threshold
        self.min_allocation_percent = min_allocation_percent

    def allocate(
        self,
        scores: List[DocumentRelevanceScore],
        total_budget: int,
    ) -> List[DocumentRelevanceScore]:
        """
        Allocate token budget to documents based on relevance scores.

        Documents below min_relevance_threshold are excluded (allocated_tokens=0).
        Included documents get proportional allocation with minimum guarantee.

        Args:
            scores: List of DocumentRelevanceScore (will be mutated)
            total_budget: Total tokens available

        Returns:
            Same list with allocated_tokens and contribution_percentage set
        """
        # Partition into relevant and excluded
        relevant, excluded = self._partition_by_relevance(scores)

        if not relevant:
            return scores

        # Calculate total score for normalization
        total_score = sum(s.final_score for s in relevant)

        # Allocate tokens proportionally
        for score in relevant:
            proportion = score.final_score / total_score

            # Ensure minimum allocation
            proportion = max(proportion, self.min_allocation_percent)

            score.allocated_tokens = int(total_budget * proportion)
            score.contribution_percentage = proportion * 100

        # Scale down if we exceeded budget
        total_allocated = sum(s.allocated_tokens for s in relevant)
        if total_allocated > total_budget:
            scale = total_budget / total_allocated
            for score in relevant:
                score.allocated_tokens = int(score.allocated_tokens * scale)
                score.contribution_percentage *= scale

        return scores

    def _partition_by_relevance(
        self,
        scores: List[DocumentRelevanceScore],
    ) -> Tuple[List[DocumentRelevanceScore], List[DocumentRelevanceScore]]:
        """
        Partition scores into relevant and excluded lists.

        Args:
            scores: All document scores

        Returns:
            Tuple of (relevant_scores, excluded_scores)
        """
        relevant = []
        excluded = []

        for score in scores:
            if score.final_score >= self.min_relevance_threshold:
                relevant.append(score)
            else:
                excluded.append(score)
                score.allocated_tokens = 0
                score.contribution_percentage = 0.0

        return relevant, excluded

    def get_allocation_summary(
        self,
        scores: List[DocumentRelevanceScore],
    ) -> dict:
        """
        Get a summary of the allocation.

        Args:
            scores: List of allocated DocumentRelevanceScore

        Returns:
            Summary dict with counts and totals
        """
        included = [s for s in scores if s.allocated_tokens > 0]
        excluded = [s for s in scores if s.allocated_tokens == 0]

        return {
            "total_documents": len(scores),
            "documents_included": len(included),
            "documents_excluded": len(excluded),
            "total_tokens_allocated": sum(s.allocated_tokens for s in included),
            "allocations": [
                {
                    "filename": s.filename,
                    "score": s.final_score,
                    "tokens": s.allocated_tokens,
                    "percentage": s.contribution_percentage,
                }
                for s in included
            ],
        }


# Module-level instance for convenience
_default_allocator = None


def get_token_allocator() -> TokenAllocator:
    """Get the default token allocator instance."""
    global _default_allocator
    if _default_allocator is None:
        _default_allocator = TokenAllocator()
    return _default_allocator


def allocate_tokens(
    scores: List[DocumentRelevanceScore],
    total_budget: int,
) -> List[DocumentRelevanceScore]:
    """
    Convenience function to allocate tokens using default allocator.

    Args:
        scores: Document scores
        total_budget: Token budget

    Returns:
        Scores with allocated_tokens set
    """
    return get_token_allocator().allocate(scores, total_budget)
