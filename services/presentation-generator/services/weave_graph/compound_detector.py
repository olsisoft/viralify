"""
ML-based Compound Term Detector for WeaveGraph

Hybrid approach combining:
1. PMI (Pointwise Mutual Information) - Statistical collocation detection
2. Embeddings - Semantic filtering for technical relevance

This replaces the hardcoded KNOWN_COMPOUND_TERMS with a learned approach.
"""

import re
import math
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompoundTermResult:
    """Result of compound term detection"""
    term: str
    original_form: str  # Preserves original case
    pmi_score: float
    semantic_score: float
    combined_score: float
    frequency: int
    is_technical: bool


@dataclass
class PMIConfig:
    """Configuration for PMI calculation"""
    min_frequency: int = 2  # Minimum occurrences to consider
    min_pmi: float = 2.0  # Minimum PMI score to keep
    max_ngram_size: int = 3  # Up to trigrams
    smoothing: float = 1.0  # Laplace smoothing


@dataclass
class CompoundDetectorConfig:
    """Configuration for the compound detector"""
    pmi_config: PMIConfig = field(default_factory=PMIConfig)
    semantic_threshold: float = 0.3  # Minimum semantic similarity to tech terms
    min_combined_score: float = 0.5  # Minimum combined score to keep
    use_embeddings: bool = True  # Whether to use embedding filtering
    cache_embeddings: bool = True  # Cache embeddings for performance


class PMICalculator:
    """
    Calculates Pointwise Mutual Information for n-grams.

    PMI(x, y) = log( P(x, y) / (P(x) * P(y)) )

    High PMI = words appear together more than expected by chance.
    """

    # Stop words to filter out
    STOP_WORDS = {
        # English
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'and', 'or', 'but', 'if',
        'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'on',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'we', 'you',
        'also', 'each', 'which', 'their', 'there', 'when', 'where', 'how',
        'very', 'just', 'only', 'more', 'most', 'other', 'some', 'such',
        # French
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'en', 'est',
        'que', 'qui', 'dans', 'ce', 'il', 'ne', 'sur', 'se', 'pas', 'plus',
        'par', 'pour', 'au', 'avec', 'son', 'sa', 'ses', 'ou', 'comme', 'mais',
    }

    def __init__(self, config: Optional[PMIConfig] = None):
        self.config = config or PMIConfig()
        self._unigram_counts: Counter = Counter()
        self._bigram_counts: Counter = Counter()
        self._trigram_counts: Counter = Counter()
        self._total_unigrams: int = 0
        self._total_bigrams: int = 0
        self._total_trigrams: int = 0
        self._is_trained: bool = False

    def train(self, texts: List[str]) -> None:
        """
        Train the PMI calculator on a corpus of texts.

        Args:
            texts: List of text documents
        """
        self._unigram_counts.clear()
        self._bigram_counts.clear()
        self._trigram_counts.clear()

        for text in texts:
            words = self._tokenize(text)

            # Count unigrams
            for word in words:
                if word.lower() not in self.STOP_WORDS:
                    self._unigram_counts[word.lower()] += 1

            # Count bigrams
            for i in range(len(words) - 1):
                if self._is_valid_ngram(words[i:i+2]):
                    bigram = ' '.join(w.lower() for w in words[i:i+2])
                    self._bigram_counts[bigram] += 1

            # Count trigrams
            if self.config.max_ngram_size >= 3:
                for i in range(len(words) - 2):
                    if self._is_valid_ngram(words[i:i+3]):
                        trigram = ' '.join(w.lower() for w in words[i:i+3])
                        self._trigram_counts[trigram] += 1

        self._total_unigrams = sum(self._unigram_counts.values())
        self._total_bigrams = sum(self._bigram_counts.values())
        self._total_trigrams = sum(self._trigram_counts.values())
        self._is_trained = True

        logger.debug(f"PMI trained: {len(self._unigram_counts)} unigrams, "
                    f"{len(self._bigram_counts)} bigrams, "
                    f"{len(self._trigram_counts)} trigrams")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text preserving case for original form.

        Supports:
        - ASCII letters
        - Unicode letters (accented chars like é, ñ, ü)
        - Numbers after first letter
        """
        # Use unicode letter pattern to support French, Spanish, etc.
        return re.findall(r'\b[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9]*\b', text)

    def _is_valid_ngram(self, words: List[str]) -> bool:
        """Check if n-gram is valid (no stop words)"""
        return not any(w.lower() in self.STOP_WORDS for w in words)

    def calculate_pmi(self, ngram: str) -> float:
        """
        Calculate PMI for an n-gram.

        PMI(w1, w2) = log2( P(w1, w2) / (P(w1) * P(w2)) )

        Returns:
            PMI score (higher = stronger collocation)
        """
        if not self._is_trained:
            raise ValueError("PMI calculator not trained. Call train() first.")

        words = ngram.lower().split()
        n = len(words)

        if n == 2:
            return self._calculate_bigram_pmi(words[0], words[1])
        elif n == 3:
            return self._calculate_trigram_pmi(words[0], words[1], words[2])
        else:
            return 0.0

    def _calculate_bigram_pmi(self, w1: str, w2: str) -> float:
        """Calculate PMI for a bigram"""
        if self._total_unigrams == 0 or self._total_bigrams == 0:
            return 0.0

        # Get counts with smoothing
        smooth = self.config.smoothing
        count_w1 = self._unigram_counts.get(w1, 0) + smooth
        count_w2 = self._unigram_counts.get(w2, 0) + smooth
        count_bigram = self._bigram_counts.get(f"{w1} {w2}", 0) + smooth

        # Calculate probabilities
        p_w1 = count_w1 / (self._total_unigrams + smooth * len(self._unigram_counts))
        p_w2 = count_w2 / (self._total_unigrams + smooth * len(self._unigram_counts))
        p_bigram = count_bigram / (self._total_bigrams + smooth * len(self._bigram_counts))

        # PMI = log2(P(w1,w2) / (P(w1) * P(w2)))
        if p_w1 * p_w2 > 0:
            pmi = math.log2(p_bigram / (p_w1 * p_w2))
            return max(0, pmi)  # Clip negative PMI
        return 0.0

    def _calculate_trigram_pmi(self, w1: str, w2: str, w3: str) -> float:
        """Calculate PMI for a trigram using chain rule approximation"""
        if self._total_unigrams == 0 or self._total_trigrams == 0:
            return 0.0

        smooth = self.config.smoothing

        # Get counts
        count_w1 = self._unigram_counts.get(w1, 0) + smooth
        count_w2 = self._unigram_counts.get(w2, 0) + smooth
        count_w3 = self._unigram_counts.get(w3, 0) + smooth
        count_trigram = self._trigram_counts.get(f"{w1} {w2} {w3}", 0) + smooth

        # Calculate probabilities
        vocab_size = len(self._unigram_counts) + 1
        p_w1 = count_w1 / (self._total_unigrams + smooth * vocab_size)
        p_w2 = count_w2 / (self._total_unigrams + smooth * vocab_size)
        p_w3 = count_w3 / (self._total_unigrams + smooth * vocab_size)
        p_trigram = count_trigram / (self._total_trigrams + smooth * len(self._trigram_counts) + 1)

        # PMI approximation for trigrams
        if p_w1 * p_w2 * p_w3 > 0:
            pmi = math.log2(p_trigram / (p_w1 * p_w2 * p_w3))
            return max(0, pmi)
        return 0.0

    def get_top_collocations(
        self,
        n: int = 2,
        top_k: int = 100,
        min_frequency: Optional[int] = None
    ) -> List[Tuple[str, float, int]]:
        """
        Get top collocations by PMI score.

        Args:
            n: N-gram size (2 or 3)
            top_k: Number of top results
            min_frequency: Minimum frequency filter

        Returns:
            List of (ngram, pmi_score, frequency) tuples
        """
        if not self._is_trained:
            raise ValueError("PMI calculator not trained. Call train() first.")

        min_freq = min_frequency or self.config.min_frequency

        if n == 2:
            counts = self._bigram_counts
        elif n == 3:
            counts = self._trigram_counts
        else:
            return []

        results = []
        for ngram, freq in counts.items():
            if freq >= min_freq:
                pmi = self.calculate_pmi(ngram)
                if pmi >= self.config.min_pmi:
                    results.append((ngram, pmi, freq))

        # Sort by PMI descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_frequency(self, ngram: str) -> int:
        """Get frequency of an n-gram"""
        words = ngram.lower().split()
        if len(words) == 1:
            return self._unigram_counts.get(ngram.lower(), 0)
        elif len(words) == 2:
            return self._bigram_counts.get(ngram.lower(), 0)
        elif len(words) == 3:
            return self._trigram_counts.get(ngram.lower(), 0)
        return 0


class SemanticFilter:
    """
    Filters compound terms based on semantic similarity to technical concepts.
    Uses embeddings to determine if a term is technical.
    """

    # Seed technical terms for similarity comparison
    TECH_SEED_TERMS = [
        # ML/AI
        "machine learning", "neural network", "deep learning", "artificial intelligence",
        # Data
        "data pipeline", "data warehouse", "database", "data processing",
        # Cloud/Infra
        "cloud computing", "microservices", "container", "kubernetes",
        # Programming
        "programming language", "software development", "api", "framework",
        # DevOps
        "continuous integration", "deployment", "infrastructure",
    ]

    def __init__(self, embedding_engine=None, threshold: float = 0.3):
        """
        Args:
            embedding_engine: Optional embedding engine (uses TF-IDF fallback if None)
            threshold: Minimum similarity to be considered technical
        """
        self.embedding_engine = embedding_engine
        self.threshold = threshold
        self._seed_embeddings = None
        self._use_tfidf_fallback = embedding_engine is None

    def _get_embedding_engine(self):
        """Lazy load embedding engine"""
        if self.embedding_engine is not None:
            return self.embedding_engine

        # Try to import and create embedding engine
        try:
            from .models import ConceptNode  # Check if we're in the right context
            from ..sync.embedding_engine import EmbeddingEngineFactory
            self.embedding_engine = EmbeddingEngineFactory.create("auto")
            self._use_tfidf_fallback = False
            return self.embedding_engine
        except ImportError:
            self._use_tfidf_fallback = True
            return None

    def _compute_seed_embeddings(self):
        """Compute embeddings for seed technical terms"""
        if self._seed_embeddings is not None:
            return

        engine = self._get_embedding_engine()
        if engine is not None and not self._use_tfidf_fallback:
            self._seed_embeddings = engine.encode(self.TECH_SEED_TERMS)
        else:
            # TF-IDF fallback: just use keyword matching
            self._seed_embeddings = None

    def is_technical(self, term: str) -> Tuple[bool, float]:
        """
        Check if a term is technical based on semantic similarity.

        Returns:
            Tuple of (is_technical, similarity_score)
        """
        self._compute_seed_embeddings()

        if self._use_tfidf_fallback or self._seed_embeddings is None:
            return self._is_technical_tfidf(term)

        try:
            engine = self._get_embedding_engine()
            term_embedding = engine.encode([term])[0]

            # Calculate max similarity to any seed term
            max_sim = 0.0
            for seed_emb in self._seed_embeddings:
                sim = self._cosine_similarity(term_embedding, seed_emb)
                max_sim = max(max_sim, sim)

            return (max_sim >= self.threshold, max_sim)
        except Exception as e:
            logger.warning(f"Embedding error, falling back to TF-IDF: {e}")
            return self._is_technical_tfidf(term)

    def _is_technical_tfidf(self, term: str) -> Tuple[bool, float]:
        """Fallback: check if term contains technical keywords"""
        term_lower = term.lower()
        term_words = set(term_lower.split())

        # Technical keyword indicators
        tech_keywords = {
            'data', 'api', 'cloud', 'machine', 'learning', 'neural', 'network',
            'database', 'pipeline', 'stream', 'batch', 'model', 'algorithm',
            'service', 'container', 'kubernetes', 'docker', 'server', 'client',
            'processing', 'computing', 'analytics', 'integration', 'deployment',
            'architecture', 'system', 'framework', 'library', 'engine', 'platform',
            'queue', 'broker', 'message', 'event', 'driven', 'distributed',
        }

        # Count matching keywords
        matches = term_words & tech_keywords

        if len(matches) > 0:
            score = len(matches) / len(term_words)
            return (score >= 0.3, score)

        # Check for technical patterns
        if re.search(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', term):  # CamelCase
            return (True, 0.5)
        if '_' in term:  # snake_case
            return (True, 0.5)

        return (False, 0.0)

    def _cosine_similarity(self, a, b) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def filter_terms(
        self,
        terms: List[Tuple[str, float, int]]
    ) -> List[Tuple[str, float, int, float]]:
        """
        Filter terms by technical relevance.

        Args:
            terms: List of (term, pmi_score, frequency) tuples

        Returns:
            List of (term, pmi_score, frequency, semantic_score) tuples
        """
        results = []
        for term, pmi, freq in terms:
            is_tech, sem_score = self.is_technical(term)
            if is_tech:
                results.append((term, pmi, freq, sem_score))
        return results


class CompoundTermDetector:
    """
    Hybrid compound term detector combining PMI and semantic filtering.

    Pipeline:
    1. Train PMI on corpus
    2. Extract high-PMI n-grams
    3. Filter by semantic similarity to technical concepts
    4. Return ranked compound terms
    """

    def __init__(self, config: Optional[CompoundDetectorConfig] = None):
        self.config = config or CompoundDetectorConfig()
        self.pmi_calculator = PMICalculator(self.config.pmi_config)
        self.semantic_filter = SemanticFilter(
            threshold=self.config.semantic_threshold
        ) if self.config.use_embeddings else None
        self._original_forms: Dict[str, str] = {}  # lowercase -> original

    def train(self, texts: List[str]) -> None:
        """
        Train the detector on a corpus.

        Args:
            texts: List of text documents
        """
        # Store original forms for case preservation
        # Use unicode-aware regex
        for text in texts:
            # Match 2-3 word phrases
            words = re.findall(
                r'\b[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9]*(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9]*){1,2}\b',
                text
            )
            for phrase in words:
                lower = phrase.lower()
                # Prefer title case versions (both words start with uppercase)
                is_title_case = all(w[0].isupper() for w in phrase.split() if w)
                if lower not in self._original_forms or is_title_case:
                    self._original_forms[lower] = phrase

        self.pmi_calculator.train(texts)

    def detect(
        self,
        text: Optional[str] = None,
        top_k: int = 100
    ) -> List[CompoundTermResult]:
        """
        Detect compound terms.

        Args:
            text: Optional new text to analyze (uses training corpus if None)
            top_k: Maximum number of results

        Returns:
            List of CompoundTermResult objects
        """
        results = []

        # Get high-PMI bigrams
        bigrams = self.pmi_calculator.get_top_collocations(n=2, top_k=top_k * 2)

        # Get high-PMI trigrams
        trigrams = self.pmi_calculator.get_top_collocations(n=3, top_k=top_k)

        all_candidates = bigrams + trigrams

        # Apply semantic filter if enabled
        if self.semantic_filter is not None and self.config.use_embeddings:
            filtered = self.semantic_filter.filter_terms(all_candidates)
        else:
            # Without semantic filter, use all high-PMI terms
            filtered = [(t, p, f, 0.5) for t, p, f in all_candidates]

        # Create results with combined scoring
        for term, pmi, freq, sem_score in filtered:
            # Normalize PMI to 0-1 range (typical PMI is 0-10)
            norm_pmi = min(1.0, pmi / 10.0)

            # Combined score: weighted average
            combined = 0.6 * norm_pmi + 0.4 * sem_score

            if combined >= self.config.min_combined_score:
                original = self._original_forms.get(term, term)
                results.append(CompoundTermResult(
                    term=term,
                    original_form=original,
                    pmi_score=pmi,
                    semantic_score=sem_score,
                    combined_score=combined,
                    frequency=freq,
                    is_technical=True
                ))

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]

    def is_compound_term(self, ngram: str) -> Tuple[bool, float]:
        """
        Check if a specific n-gram is a compound term.

        Returns:
            Tuple of (is_compound, score)
        """
        pmi = self.pmi_calculator.calculate_pmi(ngram)
        freq = self.pmi_calculator.get_frequency(ngram)

        if freq < self.config.pmi_config.min_frequency:
            return (False, 0.0)

        if pmi < self.config.pmi_config.min_pmi:
            return (False, 0.0)

        # Check semantic relevance
        if self.semantic_filter is not None:
            is_tech, sem_score = self.semantic_filter.is_technical(ngram)
            if not is_tech:
                return (False, 0.0)

            norm_pmi = min(1.0, pmi / 10.0)
            combined = 0.6 * norm_pmi + 0.4 * sem_score
            return (combined >= self.config.min_combined_score, combined)

        norm_pmi = min(1.0, pmi / 10.0)
        return (norm_pmi >= self.config.min_combined_score, norm_pmi)

    def extract_from_text(self, text: str) -> List[CompoundTermResult]:
        """
        Extract compound terms from a single text.

        This is a convenience method that trains on the text and extracts.
        For better results, train on a larger corpus first.
        """
        self.train([text])
        return self.detect(text)

    def get_known_terms(self) -> Set[str]:
        """
        Get all detected compound terms as a set.

        Useful for replacing the hardcoded KNOWN_COMPOUND_TERMS.
        """
        results = self.detect()
        return {r.term for r in results}


# Convenience function for one-off extraction
def detect_compound_terms(
    texts: List[str],
    min_pmi: float = 2.0,
    min_frequency: int = 2,
    use_embeddings: bool = True,
    top_k: int = 100
) -> List[CompoundTermResult]:
    """
    Convenience function to detect compound terms from texts.

    Args:
        texts: List of text documents
        min_pmi: Minimum PMI score
        min_frequency: Minimum occurrence count
        use_embeddings: Whether to use semantic filtering
        top_k: Maximum results

    Returns:
        List of CompoundTermResult objects
    """
    config = CompoundDetectorConfig(
        pmi_config=PMIConfig(min_pmi=min_pmi, min_frequency=min_frequency),
        use_embeddings=use_embeddings
    )
    detector = CompoundTermDetector(config)
    detector.train(texts)
    return detector.detect(top_k=top_k)
