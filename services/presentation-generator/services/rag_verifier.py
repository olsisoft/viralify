"""
RAG Verification Service

Verifies that generated content actually uses the RAG source documents.
Uses MiniLM embeddings for semantic similarity comparison.
"""
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class RAGVerificationResult:
    """Result of RAG verification analysis."""
    # Overall coverage score (0-100%)
    overall_coverage: float = 0.0

    # Per-slide coverage scores
    slide_coverage: List[Dict[str, Any]] = field(default_factory=list)

    # Detected potential hallucinations (content not in source)
    potential_hallucinations: List[Dict[str, Any]] = field(default_factory=list)

    # Compliance status
    is_compliant: bool = False

    # Summary message
    summary: str = ""

    # Embedding backend used
    backend_used: str = ""


class RAGVerifier:
    """
    Verifies RAG content usage using semantic similarity.

    Uses MiniLM embeddings to compare generated content with source documents.
    This captures meaning rather than exact text matches.
    """

    def __init__(self, min_coverage_threshold: float = 0.50):
        """
        Initialize the verifier.

        Args:
            min_coverage_threshold: Minimum semantic similarity to be compliant (default 50%)
        """
        self.min_coverage_threshold = min_coverage_threshold
        self._embedding_engine = None
        self._engine_loaded = False

    def _get_engine(self):
        """Lazy load the embedding engine."""
        if not self._engine_loaded:
            try:
                from services.sync.embedding_engine import EmbeddingEngineFactory
                self._embedding_engine = EmbeddingEngineFactory.create("auto")
                print(f"[RAG_VERIFIER] Using embedding engine: {self._embedding_engine.name}", flush=True)
            except Exception as e:
                print(f"[RAG_VERIFIER] Could not load embedding engine: {e}", flush=True)
                self._embedding_engine = None
            self._engine_loaded = True
        return self._embedding_engine

    def verify(
        self,
        generated_content: Dict[str, Any],
        source_documents: str,
        verbose: bool = False
    ) -> RAGVerificationResult:
        """
        Verify that generated content uses source documents.

        Args:
            generated_content: The generated presentation script (dict with slides)
            source_documents: The RAG context string from source documents
            verbose: If True, print detailed analysis

        Returns:
            RAGVerificationResult with coverage metrics
        """
        result = RAGVerificationResult()

        if not source_documents:
            result.summary = "No source documents provided - cannot verify RAG usage"
            result.overall_coverage = 0.0
            return result

        slides = generated_content.get("slides", [])
        if not slides:
            result.summary = "No slides in generated content"
            result.overall_coverage = 0.0
            return result

        # Try to use embedding engine
        engine = self._get_engine()

        if engine:
            result.backend_used = engine.name
            return self._verify_with_embeddings(slides, source_documents, engine, verbose, result)
        else:
            result.backend_used = "keyword-fallback"
            return self._verify_with_keywords(slides, source_documents, verbose, result)

    def _verify_with_embeddings(
        self,
        slides: List[Dict],
        source_documents: str,
        engine,
        verbose: bool,
        result: RAGVerificationResult
    ) -> RAGVerificationResult:
        """Verify using semantic embeddings (MiniLM)."""

        # Chunk source documents for better matching
        source_chunks = self._chunk_text(source_documents, chunk_size=500, overlap=100)

        if verbose:
            print(f"[RAG_VERIFIER] Source: {len(source_documents)} chars -> {len(source_chunks)} chunks", flush=True)

        # Embed all source chunks at once (batched for efficiency)
        try:
            source_embeddings = engine.embed_batch(source_chunks)
        except Exception as e:
            print(f"[RAG_VERIFIER] Embedding error: {e}, falling back to keywords", flush=True)
            result.backend_used = "keyword-fallback"
            return self._verify_with_keywords(slides, source_documents, verbose, result)

        total_similarity = 0.0
        slide_results = []
        potential_hallucinations = []

        for i, slide in enumerate(slides):
            slide_text = self._extract_slide_text(slide)

            if not slide_text.strip():
                slide_results.append({
                    "slide_index": i,
                    "slide_type": slide.get("type", "unknown"),
                    "similarity": 1.0,
                    "reason": "Empty slide"
                })
                continue

            # Embed the slide content
            try:
                slide_embedding = engine.embed(slide_text)
            except Exception:
                slide_results.append({
                    "slide_index": i,
                    "slide_type": slide.get("type", "unknown"),
                    "similarity": 0.5,
                    "reason": "Embedding failed"
                })
                total_similarity += 0.5
                continue

            # Find max similarity with any source chunk
            max_similarity = 0.0
            for source_emb in source_embeddings:
                sim = engine.similarity(slide_embedding, source_emb)
                max_similarity = max(max_similarity, sim)

            # Normalize similarity to 0-1 range (cosine can be negative)
            max_similarity = max(0.0, min(1.0, (max_similarity + 1) / 2)) if max_similarity < 0 else max_similarity

            slide_result = {
                "slide_index": i,
                "slide_type": slide.get("type", "unknown"),
                "title": slide.get("title", "Untitled"),
                "similarity": round(max_similarity, 3),
            }
            slide_results.append(slide_result)

            # Only flag as hallucination if very low similarity AND substantial content
            if max_similarity < 0.3 and len(slide_text) > 200:
                potential_hallucinations.append({
                    "slide_index": i,
                    "slide_type": slide.get("type", "unknown"),
                    "title": slide.get("title", ""),
                    "similarity": round(max_similarity, 3),
                    "content_preview": slide_text[:150] + "..." if len(slide_text) > 150 else slide_text
                })

            total_similarity += max_similarity

            if verbose:
                print(f"[RAG_VERIFIER] Slide {i} ({slide.get('type', 'unknown')}): "
                      f"{max_similarity:.1%} similarity", flush=True)

        # Calculate overall similarity
        if slides:
            result.overall_coverage = round(total_similarity / len(slides), 3)
        else:
            result.overall_coverage = 0.0

        result.slide_coverage = slide_results
        result.potential_hallucinations = potential_hallucinations
        result.is_compliant = result.overall_coverage >= self.min_coverage_threshold

        # Generate summary
        if result.is_compliant:
            result.summary = f"✅ RAG COMPLIANT: {result.overall_coverage:.1%} semantic similarity ({engine.name})"
        else:
            result.summary = f"⚠️ RAG LOW SIMILARITY: {result.overall_coverage:.1%} (threshold: {self.min_coverage_threshold:.0%})"
            if potential_hallucinations:
                result.summary += f" - {len(potential_hallucinations)} slides may need review"

        if verbose:
            print(f"[RAG_VERIFIER] {result.summary}", flush=True)

        return result

    def _verify_with_keywords(
        self,
        slides: List[Dict],
        source_documents: str,
        verbose: bool,
        result: RAGVerificationResult
    ) -> RAGVerificationResult:
        """Fallback verification using keyword overlap."""

        if verbose:
            print(f"[RAG_VERIFIER] Using keyword fallback (no embeddings)", flush=True)

        # Extract significant words from source
        source_words = self._extract_significant_words(source_documents)

        if verbose:
            print(f"[RAG_VERIFIER] Source: {len(source_documents)} chars, {len(source_words)} keywords", flush=True)

        total_coverage = 0.0
        slide_results = []
        potential_hallucinations = []

        for i, slide in enumerate(slides):
            slide_text = self._extract_slide_text(slide)

            if not slide_text.strip():
                slide_results.append({
                    "slide_index": i,
                    "slide_type": slide.get("type", "unknown"),
                    "similarity": 1.0,
                    "reason": "Empty slide"
                })
                continue

            # Extract words from slide and calculate overlap
            slide_words = self._extract_significant_words(slide_text)

            if not slide_words:
                coverage = 0.5  # Default for empty content
            else:
                overlap = slide_words.intersection(source_words)
                coverage = len(overlap) / len(slide_words) if slide_words else 0

                # Boost if many keywords match
                if len(overlap) >= 5:
                    coverage = min(1.0, coverage * 1.3)

            slide_result = {
                "slide_index": i,
                "slide_type": slide.get("type", "unknown"),
                "title": slide.get("title", "Untitled"),
                "similarity": round(coverage, 3),
                "keywords_matched": len(slide_words.intersection(source_words)) if slide_words else 0,
            }
            slide_results.append(slide_result)

            if coverage < 0.2 and len(slide_text) > 200:
                potential_hallucinations.append({
                    "slide_index": i,
                    "slide_type": slide.get("type", "unknown"),
                    "title": slide.get("title", ""),
                    "similarity": round(coverage, 3),
                    "content_preview": slide_text[:150] + "..."
                })

            total_coverage += coverage

            if verbose:
                print(f"[RAG_VERIFIER] Slide {i} ({slide.get('type', 'unknown')}): "
                      f"{coverage:.1%} keyword coverage", flush=True)

        if slides:
            result.overall_coverage = round(total_coverage / len(slides), 3)
        else:
            result.overall_coverage = 0.0

        result.slide_coverage = slide_results
        result.potential_hallucinations = potential_hallucinations
        result.is_compliant = result.overall_coverage >= self.min_coverage_threshold

        if result.is_compliant:
            result.summary = f"✅ RAG COMPLIANT: {result.overall_coverage:.1%} keyword coverage"
        else:
            result.summary = f"⚠️ RAG LOW COVERAGE: {result.overall_coverage:.1%} (threshold: {self.min_coverage_threshold:.0%})"
            if potential_hallucinations:
                result.summary += f" - {len(potential_hallucinations)} slides may need review"

        if verbose:
            print(f"[RAG_VERIFIER] {result.summary}", flush=True)

        return result

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        if len(words) <= chunk_size:
            return [text]

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))

            if i + chunk_size >= len(words):
                break

        return chunks

    def _extract_slide_text(self, slide: Dict[str, Any]) -> str:
        """Extract all text content from a slide."""
        parts = []

        if slide.get("title"):
            parts.append(slide["title"])
        if slide.get("subtitle"):
            parts.append(slide["subtitle"])
        if slide.get("content"):
            parts.append(slide["content"])
        if slide.get("bullet_points"):
            parts.extend(slide["bullet_points"])
        if slide.get("voiceover_text"):
            parts.append(slide["voiceover_text"])
        if slide.get("diagram_description"):
            parts.append(slide["diagram_description"])

        return " ".join(parts)

    def _extract_significant_words(self, text: str) -> set:
        """Extract significant words (not stopwords, length > 3)."""
        stopwords = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has',
            'will', 'can', 'are', 'was', 'were', 'been', 'being', 'would', 'could',
            'should', 'may', 'might', 'must', 'need', 'also', 'just', 'like',
            'make', 'made', 'use', 'used', 'using', 'then', 'than', 'more', 'most',
            'some', 'such', 'only', 'other', 'into', 'over', 'which', 'where',
            'when', 'what', 'while', 'there', 'here', 'they', 'their', 'them',
            'vous', 'nous', 'elle', 'sont', 'avec', 'pour', 'dans', 'cette',
            'cela', 'mais', 'donc', 'ainsi', 'comme', 'tout', 'tous', 'plus',
            'moins', 'bien', 'fait', 'faire', 'peut', 'avoir', 'etre',
        }

        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        return {w for w in words if w not in stopwords}


# Singleton instance
_rag_verifier = None


def get_rag_verifier() -> RAGVerifier:
    """Get singleton RAG verifier instance."""
    global _rag_verifier
    if _rag_verifier is None:
        _rag_verifier = RAGVerifier(min_coverage_threshold=0.50)
    return _rag_verifier


def verify_rag_usage(
    generated_script: Dict[str, Any],
    rag_context: str,
    verbose: bool = True
) -> RAGVerificationResult:
    """
    Convenience function to verify RAG usage in generated content.

    Args:
        generated_script: The generated presentation script
        rag_context: The RAG context that was provided
        verbose: Print detailed analysis

    Returns:
        RAGVerificationResult with metrics
    """
    verifier = get_rag_verifier()
    return verifier.verify(generated_script, rag_context, verbose=verbose)
