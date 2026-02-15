"""
RAG Scoring Models

Data models for document relevance scoring and weighted retrieval results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class DocumentRelevanceScore:
    """
    Relevance score for a document relative to a topic.

    Used by WeightedMultiSourceRAG to rank documents and allocate
    token budgets proportionally.

    Scoring Weights (default):
    - semantic_similarity: 40% (embedding-based)
    - keyword_coverage: 30% (exact keyword match)
    - freshness_score: 15% (recency)
    - document_type_score: 15% (PDF > DOCX > TXT)
    """

    # Document identification
    document_id: str
    filename: str

    # Individual scores (0.0 - 1.0)
    semantic_similarity: float = 0.0
    """Embedding-based similarity to topic query (cosine similarity)."""

    keyword_coverage: float = 0.0
    """Percentage of topic keywords found in document."""

    freshness_score: float = 1.0
    """Based on document date (newer = better, 2-year decay)."""

    document_type_score: float = 1.0
    """Based on document type (PDF=1.0, DOCX=0.9, TXT=0.7, etc.)."""

    # Weighted final score
    final_score: float = 0.0
    """Composite score computed from weighted individual scores."""

    # Metadata for traceability
    matched_keywords: List[str] = field(default_factory=list)
    """Keywords from query that were found in this document."""

    document_type: str = "unknown"
    """Type of document (pdf, docx, youtube, url, etc.)."""

    created_at: Optional[datetime] = None
    """Document creation/upload timestamp."""

    # Token allocation (set by allocate_tokens())
    allocated_tokens: int = 0
    """Number of tokens allocated to this document based on relevance."""

    contribution_percentage: float = 0.0
    """Percentage of total context this document contributes."""

    def __repr__(self) -> str:
        return (
            f"DocumentRelevanceScore("
            f"filename={self.filename!r}, "
            f"final_score={self.final_score:.2f}, "
            f"allocated_tokens={self.allocated_tokens})"
        )


@dataclass
class WeightedRAGResult:
    """
    Result of weighted multi-source RAG retrieval.

    Contains the combined context from all relevant sources,
    plus detailed traceability information for each document.
    """

    # Combined context from all relevant sources
    combined_context: str
    """Concatenated content from all included documents."""

    # Per-document breakdown for traceability
    document_scores: List[DocumentRelevanceScore]
    """Detailed scores for each document (sorted by relevance)."""

    # Statistics
    total_documents_provided: int = 0
    """Total number of documents provided for consideration."""

    documents_included: int = 0
    """Number of documents included in the context (above threshold)."""

    documents_excluded: int = 0
    """Number of documents excluded (below relevance threshold)."""

    total_tokens_used: int = 0
    """Total tokens in the combined context."""

    # Source contribution map (for traceability)
    source_contributions: Dict[str, float] = field(default_factory=dict)
    """Mapping of filename -> contribution percentage."""

    def get_top_sources(self, n: int = 3) -> List[str]:
        """Get the top N contributing source filenames."""
        sorted_sources = sorted(
            self.source_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [s[0] for s in sorted_sources[:n]]

    def __repr__(self) -> str:
        return (
            f"WeightedRAGResult("
            f"included={self.documents_included}/{self.total_documents_provided}, "
            f"tokens={self.total_tokens_used})"
        )
