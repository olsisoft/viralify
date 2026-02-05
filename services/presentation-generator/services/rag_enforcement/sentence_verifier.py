"""
Sentence-Level Verifier

Verifies each sentence in generated content against source documents.
Identifies ungrounded sentences that may be hallucinations.
"""

import re
from typing import List, Tuple, Optional, Callable
import numpy as np

# Support both package and standalone imports
try:
    from .models import SentenceScore, SentenceReport, FactStatus, EnforcementConfig
except ImportError:
    from models import SentenceScore, SentenceReport, FactStatus, EnforcementConfig


class SentenceVerifier:
    """
    Verifies sentences against source documents.

    For each sentence:
    1. Finds the most similar source chunk
    2. Determines if similarity is above threshold (grounded)
    3. Flags ungrounded sentences as potential hallucinations
    """

    # Sentence splitting pattern
    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')

    # Citation pattern (to remove before processing)
    CITATION_PATTERN = re.compile(r'\[REF:\d+\]')

    def __init__(
        self,
        config: Optional[EnforcementConfig] = None,
        embedding_func: Optional[Callable] = None
    ):
        self.config = config or EnforcementConfig()
        self._embed = embedding_func
        self._source_embeddings: Optional[List[np.ndarray]] = None

    def set_embedding_function(self, func: Callable):
        """Set the embedding function"""
        self._embed = func

    def precompute_source_embeddings(self, sources: List[str]) -> None:
        """Precompute embeddings for sources to avoid redundant computation"""
        if self._embed:
            self._source_embeddings = [self._embed(s) for s in sources]

    def verify_sentences(
        self,
        content: str,
        sources: List[str],
        precomputed_embeddings: bool = True
    ) -> SentenceReport:
        """
        Verify all sentences in content against sources.

        Args:
            content: Generated content
            sources: List of source chunks
            precomputed_embeddings: Whether to precompute source embeddings

        Returns:
            SentenceReport with per-sentence verification
        """
        # Precompute source embeddings if not already done
        if precomputed_embeddings and self._embed and self._source_embeddings is None:
            self.precompute_source_embeddings(sources)

        # Split into sentences
        sentences = self._split_sentences(content)

        report = SentenceReport(total_sentences=len(sentences))
        total_similarity = 0.0

        for sentence in sentences:
            score = self._verify_single_sentence(sentence, sources)
            report.sentence_scores.append(score)

            if score.is_grounded:
                report.grounded_sentences += 1
            else:
                report.ungrounded_sentences += 1

            total_similarity += score.similarity

        # Calculate average similarity
        if report.total_sentences > 0:
            report.average_similarity = total_similarity / report.total_sentences

        return report

    def _verify_single_sentence(
        self,
        sentence: str,
        sources: List[str]
    ) -> SentenceScore:
        """Verify a single sentence against sources"""
        # Clean sentence (remove citations)
        clean_sentence = self.CITATION_PATTERN.sub('', sentence).strip()

        # Skip very short sentences
        if len(clean_sentence.split()) < 5:
            return SentenceScore(
                sentence=sentence,
                similarity=1.0,  # Assume short sentences are OK
                is_grounded=True,
                fact_status=FactStatus.SUPPORTED
            )

        # Find best matching source
        best_similarity = 0.0
        best_source = None

        if self._embed and self._source_embeddings:
            # Use precomputed embeddings
            sentence_emb = self._embed(clean_sentence)
            for i, source_emb in enumerate(self._source_embeddings):
                sim = self._cosine_similarity(sentence_emb, source_emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_source = sources[i]
        else:
            # Fallback to keyword matching
            for source in sources:
                sim = self._keyword_similarity(clean_sentence, source)
                if sim > best_similarity:
                    best_similarity = sim
                    best_source = source

        # Determine if grounded
        is_grounded = best_similarity >= self.config.sentence_similarity_threshold

        # Determine fact status
        if best_similarity >= 0.8:
            fact_status = FactStatus.SUPPORTED
        elif best_similarity >= self.config.sentence_similarity_threshold:
            fact_status = FactStatus.SUPPORTED
        elif best_similarity >= 0.3:
            fact_status = FactStatus.UNSUPPORTED
        else:
            fact_status = FactStatus.HALLUCINATION

        return SentenceScore(
            sentence=sentence,
            similarity=best_similarity,
            matched_source=best_source[:200] if best_source else None,
            is_grounded=is_grounded,
            fact_status=fact_status
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Split by sentence boundaries
        sentences = self.SENTENCE_PATTERN.split(text)

        # Filter and clean
        cleaned = []
        for s in sentences:
            s = s.strip()
            if s and len(s) > 5:
                cleaned.append(s)

        return cleaned

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Keyword-based similarity (fallback)"""
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        # Jaccard similarity
        jaccard = len(intersection) / len(union)

        # Also consider how many query words are in source
        coverage = len(intersection) / len(words1) if words1 else 0

        # Weighted combination
        return 0.5 * jaccard + 0.5 * coverage

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text"""
        # Remove punctuation and lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()

        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'and', 'or', 'but', 'if',
            'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'on',
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'en', 'est',
            'que', 'qui', 'dans', 'ce', 'il', 'ne', 'sur', 'se', 'pas', 'plus',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'we', 'you',
        }

        return [w for w in words if w not in stop_words and len(w) > 2]

    def get_hallucination_candidates(
        self,
        report: SentenceReport,
        max_candidates: int = 10
    ) -> List[SentenceScore]:
        """
        Get sentences most likely to be hallucinations.

        Returns sentences sorted by likelihood of being hallucinated.
        """
        # Filter to ungrounded sentences
        candidates = [s for s in report.sentence_scores if not s.is_grounded]

        # Sort by similarity (lowest first = most likely hallucination)
        candidates.sort(key=lambda x: x.similarity)

        return candidates[:max_candidates]

    def generate_feedback(self, report: SentenceReport) -> str:
        """Generate human-readable feedback about verification results"""
        lines = []

        lines.append(f"Sentence Verification Report")
        lines.append(f"=" * 40)
        lines.append(f"Total sentences: {report.total_sentences}")
        lines.append(f"Grounded: {report.grounded_sentences} ({report.grounding_rate:.0%})")
        lines.append(f"Ungrounded: {report.ungrounded_sentences}")
        lines.append(f"Average similarity: {report.average_similarity:.2f}")
        lines.append("")

        if report.ungrounded_sentences > 0:
            lines.append("⚠️ UNGROUNDED SENTENCES:")
            for score in report.get_worst_sentences(5):
                status_icon = "❌" if score.fact_status == FactStatus.HALLUCINATION else "⚠️"
                lines.append(f"  {status_icon} [{score.similarity:.2f}] {score.sentence[:80]}...")

        return "\n".join(lines)


class AsyncSentenceVerifier(SentenceVerifier):
    """Async version of SentenceVerifier for use with async embedding functions"""

    def __init__(
        self,
        config: Optional[EnforcementConfig] = None,
        async_embedding_func: Optional[Callable] = None
    ):
        super().__init__(config)
        self._async_embed = async_embedding_func

    async def verify_sentences_async(
        self,
        content: str,
        sources: List[str]
    ) -> SentenceReport:
        """Async version of verify_sentences"""
        sentences = self._split_sentences(content)
        report = SentenceReport(total_sentences=len(sentences))

        # Precompute source embeddings
        source_embeddings = []
        if self._async_embed:
            for source in sources:
                emb = await self._async_embed(source)
                source_embeddings.append(emb)

        total_similarity = 0.0

        for sentence in sentences:
            score = await self._verify_single_sentence_async(
                sentence, sources, source_embeddings
            )
            report.sentence_scores.append(score)

            if score.is_grounded:
                report.grounded_sentences += 1
            else:
                report.ungrounded_sentences += 1

            total_similarity += score.similarity

        if report.total_sentences > 0:
            report.average_similarity = total_similarity / report.total_sentences

        return report

    async def _verify_single_sentence_async(
        self,
        sentence: str,
        sources: List[str],
        source_embeddings: List[np.ndarray]
    ) -> SentenceScore:
        """Async verification of a single sentence"""
        clean_sentence = self.CITATION_PATTERN.sub('', sentence).strip()

        if len(clean_sentence.split()) < 5:
            return SentenceScore(
                sentence=sentence,
                similarity=1.0,
                is_grounded=True,
                fact_status=FactStatus.SUPPORTED
            )

        best_similarity = 0.0
        best_source = None

        if self._async_embed and source_embeddings:
            sentence_emb = await self._async_embed(clean_sentence)
            for i, source_emb in enumerate(source_embeddings):
                sim = self._cosine_similarity(sentence_emb, source_emb)
                if sim > best_similarity:
                    best_similarity = sim
                    best_source = sources[i]
        else:
            for source in sources:
                sim = self._keyword_similarity(clean_sentence, source)
                if sim > best_similarity:
                    best_similarity = sim
                    best_source = source

        is_grounded = best_similarity >= self.config.sentence_similarity_threshold

        if best_similarity >= 0.8:
            fact_status = FactStatus.SUPPORTED
        elif best_similarity >= self.config.sentence_similarity_threshold:
            fact_status = FactStatus.SUPPORTED
        elif best_similarity >= 0.3:
            fact_status = FactStatus.UNSUPPORTED
        else:
            fact_status = FactStatus.HALLUCINATION

        return SentenceScore(
            sentence=sentence,
            similarity=best_similarity,
            matched_source=best_source[:200] if best_source else None,
            is_grounded=is_grounded,
            fact_status=fact_status
        )
