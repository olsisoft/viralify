"""
Weighted Multi-Source RAG Algorithm

Core algorithm for scoring, allocating, and retrieving content from
multiple documents with relevance-based weighting.

Scoring weights (default):
- Semantic similarity: 40%
- Keyword coverage: 30%
- Document freshness: 15%
- Document type: 15%
"""

from datetime import datetime
from typing import Dict, List, Optional

from ..models.scoring import DocumentRelevanceScore, WeightedRAGResult
from .keyword_extractor import KeywordExtractor
from .token_allocator import TokenAllocator

# Avoid circular imports - these are injected
try:
    from models.document_models import Document, DocumentChunk
except ImportError:
    Document = None
    DocumentChunk = None


class WeightedMultiSourceRAG:
    """
    Weighted Multi-Source RAG Algorithm.

    Ensures balanced, relevance-weighted content from multiple sources:
    1. Score each document for relevance to the topic
    2. Filter out irrelevant documents (below threshold)
    3. Allocate tokens proportionally based on scores
    4. Retrieve content respecting each document's budget
    5. Track contributions for traceability

    Usage:
        rag = WeightedMultiSourceRAG(embedding_service=embed_svc)
        scores = await rag.score_documents(documents, topic="Apache Kafka")
        scores = rag.allocate_tokens(scores, total_budget=8000)
        result = await rag.retrieve_weighted_context(documents, scores, topic)
    """

    # Scoring weights (must sum to 1.0)
    WEIGHT_SEMANTIC = 0.40
    WEIGHT_KEYWORDS = 0.30
    WEIGHT_FRESHNESS = 0.15
    WEIGHT_DOC_TYPE = 0.15

    # Minimum relevance threshold (documents below this are excluded)
    MIN_RELEVANCE_THRESHOLD = 0.25

    # Document type scores (official docs score higher)
    DOC_TYPE_SCORES = {
        "pdf": 1.0,        # Official documentation
        "docx": 0.9,       # Formal documents
        "pptx": 0.85,      # Presentations
        "xlsx": 0.8,       # Structured data
        "md": 0.75,        # Technical docs
        "txt": 0.7,        # Plain text
        "url": 0.65,       # Web content
        "youtube": 0.6,    # Video transcripts
        "unknown": 0.5,
    }

    def __init__(
        self,
        embedding_service=None,
        tokenizer=None,
        keyword_extractor: KeywordExtractor = None,
        token_allocator: TokenAllocator = None,
    ):
        """
        Initialize the weighted RAG algorithm.

        Args:
            embedding_service: Service for computing embeddings (optional)
            tokenizer: Tokenizer for counting tokens (optional)
            keyword_extractor: Custom keyword extractor (optional)
            token_allocator: Custom token allocator (optional)
        """
        self.embedding_service = embedding_service
        self.tokenizer = tokenizer or self._get_default_tokenizer()
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self.token_allocator = token_allocator or TokenAllocator(
            min_relevance_threshold=self.MIN_RELEVANCE_THRESHOLD,
        )

    def _get_default_tokenizer(self):
        """Get default tokenizer for token counting."""
        try:
            import tiktoken
            return tiktoken.encoding_for_model("gpt-4")
        except (ImportError, KeyError):
            return None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count (exact if tiktoken available, estimate otherwise)
        """
        if not text:
            return 0
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # Rough estimate: ~4 chars per token

    async def score_documents(
        self,
        documents: List,  # List[Document]
        topic: str,
        description: Optional[str] = None,
    ) -> List[DocumentRelevanceScore]:
        """
        Score each document for relevance to the topic.

        Combines multiple signals:
        - Semantic similarity (embedding-based)
        - Keyword coverage (exact match)
        - Document freshness (recency)
        - Document type (PDF > TXT)

        Args:
            documents: List of Document objects to score
            topic: Course/content topic
            description: Optional detailed description

        Returns:
            List of DocumentRelevanceScore sorted by relevance (highest first)
        """
        query = f"{topic} {description or ''}".strip()
        query_keywords = self.keyword_extractor.extract(query)

        scores = []

        for doc in documents:
            # Skip documents without content
            raw_content = getattr(doc, 'raw_content', None)
            if not raw_content:
                continue

            doc_id = getattr(doc, 'id', str(id(doc)))
            filename = getattr(doc, 'filename', 'unknown')
            doc_type = getattr(doc, 'document_type', None)
            doc_type_str = doc_type.value if hasattr(doc_type, 'value') else str(doc_type or 'unknown')
            created_at = getattr(doc, 'created_at', None)

            score = DocumentRelevanceScore(
                document_id=doc_id,
                filename=filename,
                document_type=doc_type_str,
                created_at=created_at,
            )

            # 1. Semantic similarity (using embeddings if available)
            if self.embedding_service and raw_content:
                try:
                    score.semantic_similarity = await self._compute_semantic_similarity(
                        query, raw_content[:5000]  # First 5000 chars for efficiency
                    )
                except Exception as e:
                    print(f"[WEIGHTED_RAG] Semantic scoring failed for {filename}: {e}", flush=True)
                    score.semantic_similarity = 0.5  # Default
            else:
                # Fallback to keyword-based pseudo-similarity
                score.semantic_similarity = self.keyword_extractor.compute_similarity(
                    query, raw_content
                )

            # 2. Keyword coverage
            score.keyword_coverage = self.keyword_extractor.compute_coverage(query, raw_content)
            score.matched_keywords = self._find_matched_keywords(query_keywords, raw_content)

            # 3. Freshness score (documents from last year score 1.0, older decay)
            score.freshness_score = self._compute_freshness_score(created_at)

            # 4. Document type score
            score.document_type_score = self.DOC_TYPE_SCORES.get(doc_type_str, 0.5)

            # Calculate weighted final score
            score.final_score = (
                self.WEIGHT_SEMANTIC * score.semantic_similarity +
                self.WEIGHT_KEYWORDS * score.keyword_coverage +
                self.WEIGHT_FRESHNESS * score.freshness_score +
                self.WEIGHT_DOC_TYPE * score.document_type_score
            )

            scores.append(score)

            print(
                f"[WEIGHTED_RAG] {filename}: "
                f"semantic={score.semantic_similarity:.2f}, "
                f"keywords={score.keyword_coverage:.2f}, "
                f"fresh={score.freshness_score:.2f}, "
                f"type={score.document_type_score:.2f} "
                f"→ FINAL={score.final_score:.2f}",
                flush=True
            )

        # Sort by final score (highest first)
        scores.sort(key=lambda x: x.final_score, reverse=True)

        return scores

    def _find_matched_keywords(
        self,
        query_keywords: List[str],
        document: str,
    ) -> List[str]:
        """Find which query keywords appear in the document."""
        doc_lower = document.lower()
        return [kw for kw in query_keywords if kw.lower() in doc_lower]

    def _compute_freshness_score(
        self,
        created_at: Optional[datetime],
    ) -> float:
        """
        Compute freshness score based on document age.

        Documents from the last year score 1.0, with 2-year decay to 0.5 minimum.

        Args:
            created_at: Document creation timestamp

        Returns:
            Freshness score between 0.5 and 1.0
        """
        if not created_at:
            return 0.7  # Unknown date gets neutral score

        try:
            days_old = (datetime.utcnow() - created_at).days
            return max(0.5, 1.0 - (days_old / 730))  # 2-year decay
        except Exception:
            return 0.7

    async def _compute_semantic_similarity(
        self,
        query: str,
        document: str,
    ) -> float:
        """
        Compute semantic similarity using embeddings.

        Args:
            query: Query text
            document: Document text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not self.embedding_service:
            return 0.5

        try:
            query_emb = await self.embedding_service.embed(query)
            doc_emb = await self.embedding_service.embed(document)

            # Cosine similarity
            import numpy as np
            similarity = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            return max(0.0, min(1.0, float(similarity)))
        except Exception:
            return 0.5

    def allocate_tokens(
        self,
        scores: List[DocumentRelevanceScore],
        total_budget: int,
    ) -> List[DocumentRelevanceScore]:
        """
        Allocate token budget to documents based on relevance scores.

        Delegates to TokenAllocator for the actual allocation logic.

        Args:
            scores: Scored documents (sorted by relevance)
            total_budget: Total tokens available

        Returns:
            Updated scores with token allocations
        """
        return self.token_allocator.allocate(scores, total_budget)

    async def retrieve_weighted_context(
        self,
        documents: List,  # List[Document]
        scores: List[DocumentRelevanceScore],
        topic: str,
        chunks_by_doc: Dict[str, List] = None,  # Dict[str, List[DocumentChunk]]
    ) -> WeightedRAGResult:
        """
        Retrieve context from each document respecting its token budget.

        Args:
            documents: Source documents
            scores: Scored and allocated documents
            topic: Topic for chunk relevance
            chunks_by_doc: Pre-computed chunks per document (optional)

        Returns:
            WeightedRAGResult with combined context and traceability
        """
        doc_map = {getattr(d, 'id', str(id(d))): d for d in documents}
        context_parts = []
        source_contributions = {}

        included = [s for s in scores if s.allocated_tokens > 0]
        excluded = [s for s in scores if s.allocated_tokens == 0]

        for score in included:
            doc = doc_map.get(score.document_id)
            raw_content = getattr(doc, 'raw_content', None) if doc else None

            if not raw_content:
                continue

            filename = getattr(doc, 'filename', 'unknown')

            # Build document context within budget
            doc_content = self._extract_within_budget(
                raw_content,
                score.allocated_tokens,
                filename,
            )

            if doc_content:
                header = f"\n=== SOURCE: {filename} (Relevance: {score.final_score:.0%}) ===\n"
                context_parts.append(header + doc_content)
                source_contributions[filename] = score.contribution_percentage

        combined = "\n".join(context_parts)

        return WeightedRAGResult(
            combined_context=combined,
            document_scores=scores,
            total_documents_provided=len(documents),
            documents_included=len(included),
            documents_excluded=len(excluded),
            total_tokens_used=self.count_tokens(combined),
            source_contributions=source_contributions,
        )

    def _extract_within_budget(
        self,
        content: str,
        token_budget: int,
        filename: str,
    ) -> str:
        """
        Extract content within token budget, prioritizing beginning.

        Attempts to end at a sentence boundary for better readability.

        Args:
            content: Full document content
            token_budget: Maximum tokens allowed
            filename: Filename for logging

        Returns:
            Truncated content within budget
        """
        if not content:
            return ""

        current_tokens = self.count_tokens(content)

        if current_tokens <= token_budget:
            return content

        # Truncate to budget (rough estimate: 4 chars per token)
        char_budget = token_budget * 4
        truncated = content[:char_budget]

        # Try to end at sentence boundary
        last_period = truncated.rfind('.')
        if last_period > char_budget * 0.7:
            truncated = truncated[:last_period + 1]

        return truncated + f"\n[...truncated to {token_budget} tokens...]"


# Module-level instance for convenience
_default_rag = None


def get_weighted_rag(embedding_service=None) -> WeightedMultiSourceRAG:
    """
    Get or create a WeightedMultiSourceRAG instance.

    Args:
        embedding_service: Optional embedding service to use

    Returns:
        WeightedMultiSourceRAG instance
    """
    global _default_rag
    if _default_rag is None or embedding_service is not None:
        _default_rag = WeightedMultiSourceRAG(embedding_service=embedding_service)
    return _default_rag
