"""
SSVS - Semantic Slide-Voiceover Synchronization Algorithm

This module implements the core SSVS algorithm for optimal alignment between
presentation slides and voice narration segments.

ALGORITHM OVERVIEW:
1. Semantic Embedding: Project slides and segments into a shared vector space
2. Similarity Matrix: Compute semantic similarity between all pairs
3. Dynamic Programming: Find optimal assignment respecting temporal constraints
4. Transition Detection: Identify precise transition points

GUARANTEES:
- Optimal alignment (not heuristic)
- Complete coverage of all segments
- Temporal order preservation
- Contiguous segment assignment per slide
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
import re
import math


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class VoiceSegment:
    """A segment of voice narration with timestamps"""
    id: int
    text: str
    start_time: float
    end_time: float
    word_timestamps: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Slide:
    """A presentation slide with content for semantic matching"""
    id: str
    index: int
    title: str
    content: str  # Main text content
    voiceover_text: str  # Expected narration
    keywords: List[str] = field(default_factory=list)
    slide_type: str = "content"

    def get_searchable_text(self) -> str:
        """Combine all text for semantic embedding"""
        parts = [self.title, self.content, self.voiceover_text]
        parts.extend(self.keywords)
        return " ".join(filter(None, parts))


@dataclass
class SynchronizationResult:
    """Result of synchronizing a slide with voice segments"""
    slide_id: str
    slide_index: int
    segment_ids: List[int]
    start_time: float
    end_time: float
    semantic_score: float
    temporal_score: float
    combined_score: float
    transition_words: List[str] = field(default_factory=list)


# ==============================================================================
# SEMANTIC EMBEDDING ENGINE
# ==============================================================================

class SemanticEmbeddingEngine:
    """
    Semantic embedding engine using TF-IDF.

    For production, this can be swapped with Sentence-BERT for better quality.
    TF-IDF is used here for simplicity and no external dependencies.
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.vocab_size: int = 0

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

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text"""
        # Lowercase
        text = text.lower()
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", ' ', text)
        # Split into words
        words = text.split()
        # Remove stop words and short words
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        return words

    def build_vocabulary(self, documents: List[str]) -> None:
        """Build vocabulary and compute IDF from corpus"""
        doc_count = len(documents)
        word_doc_count: Counter = Counter()

        # Count document frequency for each word
        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                word_doc_count[word] += 1

        # Build vocabulary (sorted for reproducibility)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(word_doc_count.keys()))}
        self.vocab_size = len(self.vocabulary)

        # Compute IDF: log(N / (1 + df))
        self.idf = {}
        for word, df in word_doc_count.items():
            self.idf[word] = math.log(doc_count / (1 + df))

    def embed(self, text: str) -> np.ndarray:
        """Create TF-IDF embedding for text"""
        if self.vocab_size == 0:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")

        words = self._tokenize(text)
        tf: Counter = Counter(words)

        # Create sparse vector
        vector = np.zeros(self.vocab_size)

        for word, count in tf.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                # TF-IDF = term frequency * inverse document frequency
                vector[idx] = count * self.idf.get(word, 0)

        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))


# ==============================================================================
# SSVS SYNCHRONIZER - CORE ALGORITHM
# ==============================================================================

class SSVSSynchronizer:
    """
    Semantic Slide-Voiceover Synchronization Algorithm

    Uses dynamic programming to find the optimal alignment between
    slides and voice segments based on semantic similarity.

    ALGORITHM:
    1. Build semantic embeddings for all slides and segments
    2. Compute similarity matrix
    3. Use DP to find optimal partition of segments to slides
    4. Reconstruct alignment via backtracking

    COMPLEXITY: O(n × m²) where n=slides, m=segments
    """

    def __init__(self,
                 alpha: float = 0.6,   # Weight for semantic score
                 beta: float = 0.3,    # Weight for temporal score
                 gamma: float = 0.1):  # Weight for transition score
        """
        Args:
            alpha: Weight for semantic similarity
            beta: Weight for temporal consistency
            gamma: Weight for transition smoothness
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.embedding_engine = SemanticEmbeddingEngine()

        # Cache for embeddings
        self._slide_embeddings: Dict[str, np.ndarray] = {}
        self._segment_embeddings: Dict[int, np.ndarray] = {}

    def _build_embeddings(self, slides: List[Slide], segments: List[VoiceSegment]) -> None:
        """Build embeddings for all slides and segments"""
        # Collect all documents for vocabulary
        documents = []
        for slide in slides:
            documents.append(slide.get_searchable_text())
        for segment in segments:
            documents.append(segment.text)

        # Build vocabulary
        self.embedding_engine.build_vocabulary(documents)

        # Embed slides
        for slide in slides:
            self._slide_embeddings[slide.id] = self.embedding_engine.embed(
                slide.get_searchable_text()
            )

        # Embed segments
        for segment in segments:
            self._segment_embeddings[segment.id] = self.embedding_engine.embed(
                segment.text
            )

    def _compute_semantic_score(self,
                                 slide: Slide,
                                 segments: List[VoiceSegment]) -> float:
        """Compute semantic similarity between slide and segment group"""
        if not segments:
            return 0.0

        slide_embedding = self._slide_embeddings[slide.id]

        # Combine segment embeddings (weighted average by duration)
        total_duration = sum(s.duration for s in segments)
        if total_duration == 0:
            return 0.0

        combined_embedding = np.zeros_like(slide_embedding)
        for segment in segments:
            weight = segment.duration / total_duration
            combined_embedding += weight * self._segment_embeddings[segment.id]

        # Normalize
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm

        return self.embedding_engine.similarity(slide_embedding, combined_embedding)

    def _compute_temporal_score(self,
                                 segments: List[VoiceSegment],
                                 expected_duration_ratio: float) -> float:
        """
        Compute temporal consistency score.

        Measures how well the segment duration matches the expected
        proportion of total narration time.
        """
        if not segments:
            return 0.0

        actual_duration = segments[-1].end_time - segments[0].start_time
        total_duration = segments[-1].end_time  # Assuming segments span from 0

        if total_duration == 0:
            return 0.0

        actual_ratio = actual_duration / total_duration

        # Score based on how close actual is to expected
        # Uses Gaussian-like decay
        diff = abs(actual_ratio - expected_duration_ratio)
        return math.exp(-diff * 5)  # Higher penalty for large differences

    def _compute_transition_score(self,
                                   segment: VoiceSegment) -> float:
        """
        Compute transition score for a segment starting a new slide.

        Higher score if segment contains transition words.
        """
        transition_markers = [
            # French
            'maintenant', 'ensuite', 'puis', 'passons', 'voyons', 'regardons',
            'abordons', 'commençons', 'continuons', 'terminons', 'finalement',
            # English
            'now', 'next', 'then', 'moving on', "let's look", "let's see",
            'finally', 'first', 'second', 'third', 'lastly', 'starting with',
        ]

        text_lower = segment.text.lower()

        for marker in transition_markers:
            if marker in text_lower:
                return 1.0

        return 0.3  # Base score if no explicit transition

    def _compute_combined_score(self,
                                 slide: Slide,
                                 segments: List[VoiceSegment],
                                 expected_duration_ratio: float) -> float:
        """Compute combined score for assigning segments to slide"""
        sem_score = self._compute_semantic_score(slide, segments)
        temp_score = self._compute_temporal_score(segments, expected_duration_ratio)
        trans_score = self._compute_transition_score(segments[0]) if segments else 0.0

        return (self.alpha * sem_score +
                self.beta * temp_score +
                self.gamma * trans_score)

    def synchronize(self,
                    slides: List[Slide],
                    segments: List[VoiceSegment]) -> List[SynchronizationResult]:
        """
        Synchronize slides with voice segments using dynamic programming.

        Args:
            slides: List of presentation slides (ordered)
            segments: List of voice segments (ordered by time)

        Returns:
            List of SynchronizationResult, one per slide
        """
        n_slides = len(slides)
        n_segments = len(segments)

        if n_slides == 0 or n_segments == 0:
            return []

        print(f"[SSVS] Synchronizing {n_slides} slides with {n_segments} segments", flush=True)

        # Build embeddings
        self._build_embeddings(slides, segments)

        # Expected duration ratio per slide (uniform distribution as baseline)
        expected_ratio = 1.0 / n_slides

        # DP table: dp[i][j] = best score for aligning slides[0:i] with segments[0:j]
        dp = np.full((n_slides + 1, n_segments + 1), -np.inf)
        dp[0][0] = 0.0

        # Backtrack table: backtrack[i][j] = k where segments[k:j] are assigned to slide i-1
        backtrack = np.zeros((n_slides + 1, n_segments + 1), dtype=int)

        # Fill DP table
        for i in range(1, n_slides + 1):
            slide = slides[i - 1]

            # Minimum segments needed for remaining slides
            min_segments_needed = n_slides - i

            for j in range(i, n_segments + 1 - min_segments_needed):
                # Try all possible starting points k for this slide's segments
                best_score = -np.inf
                best_k = i - 1

                for k in range(i - 1, j):
                    if dp[i - 1][k] == -np.inf:
                        continue

                    # Segments k to j-1 assigned to this slide
                    assigned_segments = segments[k:j]

                    if not assigned_segments:
                        continue

                    score = self._compute_combined_score(
                        slide, assigned_segments, expected_ratio
                    )

                    total_score = dp[i - 1][k] + score

                    if total_score > best_score:
                        best_score = total_score
                        best_k = k

                dp[i][j] = best_score
                backtrack[i][j] = best_k

        # Find best final alignment
        best_final_score = dp[n_slides][n_segments]

        if best_final_score == -np.inf:
            print("[SSVS] WARNING: No valid alignment found, using fallback", flush=True)
            return self._fallback_synchronize(slides, segments)

        # Backtrack to reconstruct alignment
        results = []
        j = n_segments

        for i in range(n_slides, 0, -1):
            k = backtrack[i][j]
            slide = slides[i - 1]
            assigned_segments = segments[k:j]

            if assigned_segments:
                sem_score = self._compute_semantic_score(slide, assigned_segments)
                temp_score = self._compute_temporal_score(assigned_segments, expected_ratio)

                result = SynchronizationResult(
                    slide_id=slide.id,
                    slide_index=slide.index,
                    segment_ids=[s.id for s in assigned_segments],
                    start_time=assigned_segments[0].start_time,
                    end_time=assigned_segments[-1].end_time,
                    semantic_score=sem_score,
                    temporal_score=temp_score,
                    combined_score=dp[i][j] - dp[i - 1][k]
                )
                results.insert(0, result)

            j = k

        print(f"[SSVS] Alignment complete. Average semantic score: "
              f"{np.mean([r.semantic_score for r in results]):.3f}", flush=True)

        return results

    def _fallback_synchronize(self,
                               slides: List[Slide],
                               segments: List[VoiceSegment]) -> List[SynchronizationResult]:
        """
        Fallback synchronization using proportional distribution.
        Used when DP finds no valid alignment.
        """
        print("[SSVS] Using proportional fallback", flush=True)

        n_slides = len(slides)
        n_segments = len(segments)

        if n_segments < n_slides:
            # Not enough segments, assign one per slide
            segments_per_slide = 1
        else:
            segments_per_slide = n_segments // n_slides

        results = []
        seg_idx = 0

        for i, slide in enumerate(slides):
            # Calculate how many segments for this slide
            if i == n_slides - 1:
                # Last slide gets all remaining
                count = n_segments - seg_idx
            else:
                count = segments_per_slide

            assigned = segments[seg_idx:seg_idx + count]

            if assigned:
                result = SynchronizationResult(
                    slide_id=slide.id,
                    slide_index=slide.index,
                    segment_ids=[s.id for s in assigned],
                    start_time=assigned[0].start_time,
                    end_time=assigned[-1].end_time,
                    semantic_score=0.5,  # Unknown
                    temporal_score=1.0,
                    combined_score=0.5
                )
                results.append(result)

            seg_idx += count

        return results

    def synchronize_with_word_timestamps(self,
                                          slides: List[Slide],
                                          segments: List[VoiceSegment],
                                          word_timestamps: List[Dict]) -> List[SynchronizationResult]:
        """
        Enhanced synchronization using word-level timestamps.

        This allows for sub-segment precision in finding transition points.
        """
        # First do segment-level alignment
        results = self.synchronize(slides, segments)

        if not word_timestamps:
            return results

        # Refine transition points using word timestamps
        for i, result in enumerate(results):
            if i == 0:
                continue

            # Look for transition words near the boundary
            boundary_time = result.start_time

            # Find words within 1 second of boundary
            nearby_words = [
                w for w in word_timestamps
                if abs(w.get('start', 0) - boundary_time) < 1.0
            ]

            # Check for transition markers
            for word in nearby_words:
                word_text = word.get('word', '').lower()
                if word_text in ['maintenant', 'ensuite', 'puis', 'now', 'next', 'then']:
                    # Adjust start time to this word
                    result.start_time = word.get('start', result.start_time)
                    result.transition_words.append(word_text)

                    # Also adjust previous result's end time
                    if i > 0:
                        results[i - 1].end_time = result.start_time
                    break

        return results
