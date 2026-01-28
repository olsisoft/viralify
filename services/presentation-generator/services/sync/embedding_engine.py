"""
Embedding Engine for SSVS Synchronization

Provides multiple embedding backends for semantic similarity:
- MiniLM (default): Fast, good quality, 384 dimensions
- BGE-M3: Better multilingual, slower, 1024 dimensions
- TF-IDF (fallback): No dependencies, vocabulary-based

Usage:
    engine = EmbeddingEngineFactory.create("auto")  # MiniLM with TF-IDF fallback
    engine.build_vocabulary(documents)  # Required for TF-IDF, optional for others
    embedding = engine.embed(text)
    similarity = engine.similarity(emb1, emb2)
"""

import os
import logging

# ============================================================================
# SUPPRESS PROGRESS BARS BEFORE IMPORTING ML LIBRARIES
# Must be set BEFORE importing transformers/sentence_transformers
# ============================================================================
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Suppress verbose logging from ML libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("safetensors").setLevel(logging.ERROR)

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from collections import Counter
from enum import Enum
import re


class EmbeddingBackend(str, Enum):
    """Available embedding backends"""
    AUTO = "auto"        # MiniLM with TF-IDF fallback
    MINILM = "minilm"    # all-MiniLM-L6-v2 (384 dims)
    BGE_M3 = "bge-m3"    # BAAI/bge-m3 (1024 dims)
    E5_LARGE = "e5-large"  # intfloat/multilingual-e5-large (1024 dims) - BEST MULTILINGUAL
    TFIDF = "tfidf"      # TF-IDF fallback (no dependencies)


class EmbeddingEngineBase(ABC):
    """Abstract base class for embedding engines"""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensions"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return engine name for logging"""
        pass

    @abstractmethod
    def build_vocabulary(self, documents: List[str]) -> None:
        """
        Build vocabulary from documents.
        Required for TF-IDF, optional for transformer models.
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        pass

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))


class TFIDFEmbeddingEngine(EmbeddingEngineBase):
    """
    TF-IDF based embedding engine.

    No external dependencies, works as fallback.
    Quality is lower than transformer models but very fast.
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.vocab_size: int = 0
        self._dimensions: int = 0

        # French + English stop words
        self.stop_words = {
            # French
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'en', 'est',
            'que', 'qui', 'dans', 'ce', 'il', 'ne', 'sur', 'se', 'pas', 'plus',
            'par', 'pour', 'au', 'avec', 'son', 'sa', 'ses', 'ou', 'comme', 'mais',
            'nous', 'vous', 'leur', 'on', 'cette', 'ces', 'tout', 'elle', 'sont',
            'a', 'à', 'être', 'avoir', 'fait', 'faire', 'peut', 'aussi', 'bien',
            # English
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'and', 'or', 'but', 'if', 'while', 'although', 'because', 'until',
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
            'this', 'that', 'these', 'those', 'it', 'its', 'he', 'she', 'they',
            'we', 'you', 'i', 'me', 'my', 'your', 'our', 'their', 'who', 'which',
        }

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return "TF-IDF"

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text"""
        text = text.lower()
        text = re.sub(r"[^\w\s']", ' ', text)
        words = text.split()
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        return words

    def build_vocabulary(self, documents: List[str]) -> None:
        """Build vocabulary and compute IDF from corpus"""
        doc_count = len(documents)
        word_doc_count: Counter = Counter()

        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                word_doc_count[word] += 1

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(word_doc_count.keys()))}
        self.vocab_size = len(self.vocabulary)
        self._dimensions = self.vocab_size

        self.idf = {}
        for word, df in word_doc_count.items():
            self.idf[word] = math.log(doc_count / (1 + df))

        print(f"[EMBEDDING] TF-IDF vocabulary built: {self.vocab_size} terms", flush=True)

    def embed(self, text: str) -> np.ndarray:
        """Create TF-IDF embedding for text"""
        if self.vocab_size == 0:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")

        words = self._tokenize(text)
        tf: Counter = Counter(words)

        vector = np.zeros(self.vocab_size)

        for word, count in tf.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                vector[idx] = count * self.idf.get(word, 0)

        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        return [self.embed(text) for text in texts]


class SentenceTransformerEngine(EmbeddingEngineBase):
    """
    Sentence Transformer based embedding engine.

    Supports:
    - all-MiniLM-L6-v2: Fast, 384 dimensions
    - BAAI/bge-m3: Better multilingual, 1024 dimensions

    Requires: sentence-transformers, torch
    """

    # Model configurations
    MODEL_CONFIGS = {
        "minilm": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "display_name": "MiniLM-L6-v2",
            "multilingual": False,
        },
        "bge-m3": {
            "model_name": "BAAI/bge-m3",
            "dimensions": 1024,
            "display_name": "BGE-M3",
            "multilingual": True,
        },
        "e5-large": {
            "model_name": "intfloat/multilingual-e5-large",
            "dimensions": 1024,
            "display_name": "E5-Large-Multilingual",
            "multilingual": True,
            "instruction_prefix": "query: ",  # E5 recommends prefixing queries
        },
    }

    def __init__(self, model_key: str = "minilm"):
        """
        Initialize with specified model.

        Args:
            model_key: "minilm" or "bge-m3"
        """
        if model_key not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}. Choose from: {list(self.MODEL_CONFIGS.keys())}")

        self.model_key = model_key
        self.config = self.MODEL_CONFIGS[model_key]
        self._model = None
        self._load_model()

    def _load_model(self):
        """Lazy load the model (only if not already loaded)"""
        # Skip if already loaded (defensive check)
        if self._model is not None:
            print(f"[EMBEDDING] {self.config['display_name']} already loaded, skipping", flush=True)
            return

        try:
            import warnings
            from sentence_transformers import SentenceTransformer

            model_name = self.config["model_name"]
            print(f"[EMBEDDING] Loading {self.config['display_name']}...", flush=True)

            # Load model with progress bars suppressed (via env vars set at module level)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = SentenceTransformer(model_name)

            # Force CPU if no GPU available
            import torch
            if not torch.cuda.is_available():
                self._model = self._model.to('cpu')
                print(f"[EMBEDDING] {self.config['display_name']} loaded on CPU", flush=True)
            else:
                print(f"[EMBEDDING] {self.config['display_name']} loaded on GPU", flush=True)

        except ImportError as e:
            raise ImportError(
                f"sentence-transformers is required for {self.config['display_name']}. "
                "Install with: pip install sentence-transformers"
            ) from e

    @property
    def dimensions(self) -> int:
        return self.config["dimensions"]

    @property
    def name(self) -> str:
        return self.config["display_name"]

    def build_vocabulary(self, documents: List[str]) -> None:
        """
        Not required for transformer models.
        Kept for interface compatibility.
        """
        # Optionally pre-encode documents to warm up the model
        if documents and len(documents) <= 10:
            # Warm up with a few documents
            _ = self._model.encode(documents[:3], show_progress_bar=False)
        print(f"[EMBEDDING] {self.name} ready (no vocabulary needed)", flush=True)

    @property
    def is_multilingual(self) -> bool:
        """Check if model supports multilingual embeddings"""
        return self.config.get("multilingual", False)

    def _prepare_text(self, text: str) -> str:
        """Prepare text with model-specific prefixes if needed"""
        prefix = self.config.get("instruction_prefix", "")
        if prefix:
            return f"{prefix}{text}"
        return text

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        prepared_text = self._prepare_text(text)
        embedding = self._model.encode(prepared_text, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embedding)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts (batched for efficiency)"""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        prepared_texts = [self._prepare_text(t) for t in texts]
        embeddings = self._model.encode(prepared_texts, show_progress_bar=False, batch_size=32, normalize_embeddings=True)
        return [np.array(emb) for emb in embeddings]


class EmbeddingEngineFactory:
    """
    Factory for creating embedding engines.

    Supports automatic fallback from transformer models to TF-IDF
    if dependencies are not available.

    Uses singleton pattern to avoid loading the same model multiple times.
    """

    # Singleton cache for loaded engines
    _instances: Dict[str, EmbeddingEngineBase] = {}

    @classmethod
    def create(cls, backend: str = "auto") -> EmbeddingEngineBase:
        """
        Create or retrieve cached embedding engine instance.

        Uses singleton pattern to avoid loading the same model multiple times.
        E5-Large is ~2GB and takes several seconds to load.

        Args:
            backend: "auto", "minilm", "bge-m3", "e5-large", or "tfidf"

        Returns:
            EmbeddingEngineBase instance (cached if already loaded)
        """
        backend = backend.lower().strip()

        # Get from environment if not specified
        # Check unified env var first, then legacy vars
        if backend == "auto":
            backend = (
                os.getenv("EMBEDDING_BACKEND") or
                os.getenv("SSVS_EMBEDDING_BACKEND") or
                os.getenv("RAG_EMBEDDING_BACKEND") or
                "auto"
            ).lower()

        # Check cache first (Singleton pattern)
        if backend in cls._instances:
            cached_engine = cls._instances[backend]
            print(f"[EMBEDDING] ✓ Cache HIT: reusing {cached_engine.name} (id={id(cached_engine)})", flush=True)
            return cached_engine

        print(f"[EMBEDDING] ✗ Cache MISS: creating new engine for '{backend}' (cache has: {list(cls._instances.keys())})", flush=True)

        if backend == "tfidf":
            engine = TFIDFEmbeddingEngine()
            cls._instances[backend] = engine
            return engine

        if backend == "minilm":
            engine = cls._create_transformer("minilm")
            cls._instances[backend] = engine
            cls._instances["minilm"] = engine
            return engine

        if backend == "bge-m3":
            engine = cls._create_transformer("bge-m3")
            cls._instances[backend] = engine
            cls._instances["bge-m3"] = engine
            return engine

        if backend in ("e5-large", "e5_large", "multilingual"):
            engine = cls._create_transformer("e5-large")
            # Cache under all known aliases
            cls._instances["e5-large"] = engine
            cls._instances["e5_large"] = engine
            cls._instances["multilingual"] = engine
            return engine

        if backend == "auto":
            engine = cls._create_with_fallback()
            cls._instances["auto"] = engine
            # Also cache under the actual model name to avoid loading twice
            if hasattr(engine, 'model_key'):
                cls._instances[engine.model_key] = engine
            return engine

        raise ValueError(f"Unknown embedding backend: {backend}")

    @staticmethod
    def _create_transformer(model_key: str) -> EmbeddingEngineBase:
        """Create transformer engine, raise if not available"""
        try:
            return SentenceTransformerEngine(model_key)
        except ImportError as e:
            raise ImportError(
                f"Cannot create {model_key} engine: {e}\n"
                "Install with: pip install sentence-transformers torch"
            ) from e

    @staticmethod
    def _create_with_fallback() -> EmbeddingEngineBase:
        """
        Create best available engine with automatic fallback.

        Priority: MiniLM > TF-IDF
        """
        # Try MiniLM first (best balance of speed/quality)
        try:
            engine = SentenceTransformerEngine("minilm")
            print("[EMBEDDING] Using MiniLM (primary)", flush=True)
            return engine
        except ImportError:
            print("[EMBEDDING] MiniLM not available, falling back to TF-IDF", flush=True)

        # Fallback to TF-IDF
        print("[EMBEDDING] Using TF-IDF (fallback)", flush=True)
        return TFIDFEmbeddingEngine()

    @staticmethod
    def is_transformer_available() -> bool:
        """Check if sentence-transformers is available"""
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False

    @staticmethod
    def list_available_backends() -> List[str]:
        """List all available backends"""
        available = ["tfidf", "auto"]

        if EmbeddingEngineFactory.is_transformer_available():
            available.extend(["minilm", "bge-m3"])

        return available


# Convenience function for quick access
def get_embedding_engine(backend: str = "auto") -> EmbeddingEngineBase:
    """Get embedding engine instance"""
    return EmbeddingEngineFactory.create(backend)
