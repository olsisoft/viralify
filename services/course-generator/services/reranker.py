"""
Cross-Encoder Re-ranking Service

Implements semantic re-ranking of RAG results using cross-encoder models.
Cross-encoders are more accurate than bi-encoders because they see
query and document together, enabling deeper semantic understanding.

Models supported:
- ms-marco-MiniLM-L-6-v2: Fast, good quality (default)
- ms-marco-MiniLM-L-12-v2: Slower, better quality
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod


@dataclass
class RerankResult:
    """Result of re-ranking operation"""
    original_index: int
    original_score: float
    rerank_score: float
    content: str
    document_id: str


class RerankerBase(ABC):
    """Abstract base class for re-rankers"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents based on query relevance.

        Args:
            query: The search query
            documents: List of document texts to re-rank
            top_k: Return only top K results (None = all)

        Returns:
            List of (original_index, rerank_score) tuples, sorted by score desc
        """
        pass


class CrossEncoderReranker(RerankerBase):
    """
    Cross-Encoder based re-ranker using sentence-transformers.

    Cross-encoders process query and document together, providing
    more accurate relevance scoring than bi-encoder similarity.
    """

    MODEL_CONFIGS = {
        "fast": {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "max_length": 512,
        },
        "accurate": {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "max_length": 512,
        },
    }

    def __init__(self, model_type: str = "fast"):
        """
        Initialize CrossEncoder reranker.

        Args:
            model_type: "fast" or "accurate"
        """
        self.model_type = model_type
        self.model = None
        self.config = self.MODEL_CONFIGS.get(model_type, self.MODEL_CONFIGS["fast"])
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of the model"""
        if self._initialized:
            return

        try:
            from sentence_transformers import CrossEncoder

            print(f"[RERANKER] Loading CrossEncoder: {self.config['model_name']}", flush=True)
            self.model = CrossEncoder(
                self.config["model_name"],
                max_length=self.config["max_length"],
            )
            self._initialized = True
            print(f"[RERANKER] CrossEncoder loaded successfully", flush=True)

        except ImportError as e:
            print(f"[RERANKER] sentence-transformers not available: {e}", flush=True)
            raise ImportError(
                "CrossEncoder requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            print(f"[RERANKER] Failed to load CrossEncoder: {e}", flush=True)
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents using CrossEncoder.

        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Return only top K (None = all)

        Returns:
            List of (original_index, score) sorted by relevance
        """
        if not documents:
            return []

        self._ensure_initialized()

        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Get relevance scores from CrossEncoder
        scores = self.model.predict(pairs)

        # Create indexed results
        indexed_scores = [(i, float(scores[i])) for i in range(len(scores))]

        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k limit
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores


class TFIDFReranker(RerankerBase):
    """
    TF-IDF based re-ranker as fallback.

    Uses keyword overlap for re-ranking when CrossEncoder is not available.
    Less accurate but requires no heavy dependencies.
    """

    def __init__(self):
        self.vectorizer = None
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
        )
        self._cosine_similarity = cosine_similarity
        self._initialized = True
        print("[RERANKER] TF-IDF reranker initialized", flush=True)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Re-rank using TF-IDF similarity"""
        if not documents:
            return []

        self._ensure_initialized()

        # Fit on all texts
        all_texts = [query] + documents
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)

        # Query is first row, documents are rest
        query_vec = tfidf_matrix[0:1]
        doc_vecs = tfidf_matrix[1:]

        # Calculate similarities
        similarities = self._cosine_similarity(query_vec, doc_vecs)[0]

        # Create indexed results
        indexed_scores = [(i, float(similarities[i])) for i in range(len(similarities))]

        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores


class RerankerFactory:
    """Factory for creating reranker instances"""

    _instance: Optional[RerankerBase] = None
    _backend: str = "auto"

    @classmethod
    def create(cls, backend: str = "auto") -> RerankerBase:
        """
        Create a reranker instance.

        Args:
            backend: "auto", "cross-encoder", "cross-encoder-accurate", or "tfidf"

        Returns:
            Reranker instance
        """
        # Cache instance if same backend
        if cls._instance is not None and cls._backend == backend:
            return cls._instance

        cls._backend = backend

        if backend == "tfidf":
            cls._instance = TFIDFReranker()
            return cls._instance

        if backend == "cross-encoder":
            cls._instance = CrossEncoderReranker(model_type="fast")
            return cls._instance

        if backend == "cross-encoder-accurate":
            cls._instance = CrossEncoderReranker(model_type="accurate")
            return cls._instance

        # Auto mode: try CrossEncoder, fallback to TF-IDF
        if backend == "auto":
            try:
                cls._instance = CrossEncoderReranker(model_type="fast")
                # Test initialization
                cls._instance._ensure_initialized()
                return cls._instance
            except Exception as e:
                print(f"[RERANKER] CrossEncoder not available, using TF-IDF fallback: {e}", flush=True)
                cls._instance = TFIDFReranker()
                return cls._instance

        raise ValueError(f"Unknown reranker backend: {backend}")

    @classmethod
    def get_instance(cls) -> Optional[RerankerBase]:
        """Get the cached instance"""
        return cls._instance


def get_reranker(backend: str = None) -> RerankerBase:
    """
    Get a reranker instance.

    Args:
        backend: Reranker backend to use. If None, uses RERANKER_BACKEND env var.

    Returns:
        Reranker instance
    """
    if backend is None:
        backend = os.getenv("RERANKER_BACKEND", "auto")
    return RerankerFactory.create(backend)
