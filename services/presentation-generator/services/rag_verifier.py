"""
RAG Verification Service

Verifies that generated content actually uses the RAG source documents.
Provides metrics on RAG coverage and detects potential hallucinations.
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class RAGVerificationResult:
    """Result of RAG verification analysis."""
    # Overall coverage score (0-100%)
    overall_coverage: float

    # Per-slide coverage scores
    slide_coverage: List[Dict[str, Any]] = field(default_factory=list)

    # Detected potential hallucinations (content not in source)
    potential_hallucinations: List[Dict[str, Any]] = field(default_factory=list)

    # Key terms from source that were used
    source_terms_used: List[str] = field(default_factory=list)

    # Key terms from source that were missed
    source_terms_missed: List[str] = field(default_factory=list)

    # Compliance status
    is_compliant: bool = False  # True if coverage >= 90%

    # Summary message
    summary: str = ""


class RAGVerifier:
    """
    Verifies RAG content usage in generated presentations.

    Uses multiple techniques:
    1. N-gram overlap analysis
    2. Key term extraction and matching
    3. Semantic similarity (if embeddings available)
    4. Code pattern matching
    """

    def __init__(self, min_coverage_threshold: float = 0.90):
        """
        Initialize the verifier.

        Args:
            min_coverage_threshold: Minimum coverage to be considered compliant (default 90%)
        """
        self.min_coverage_threshold = min_coverage_threshold

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

        # Extract text content from generated presentation
        slides = generated_content.get("slides", [])
        if not slides:
            result.summary = "No slides in generated content"
            result.overall_coverage = 0.0
            return result

        # Normalize source documents
        source_normalized = self._normalize_text(source_documents)
        source_terms = self._extract_key_terms(source_documents)
        source_ngrams = self._extract_ngrams(source_normalized, n=3)

        if verbose:
            print(f"[RAG_VERIFIER] Source: {len(source_normalized)} chars, {len(source_terms)} key terms, {len(source_ngrams)} trigrams", flush=True)

        total_coverage = 0.0
        slide_results = []
        all_used_terms = set()
        potential_hallucinations = []

        for i, slide in enumerate(slides):
            slide_text = self._extract_slide_text(slide)
            slide_normalized = self._normalize_text(slide_text)

            if not slide_normalized.strip():
                slide_results.append({
                    "slide_index": i,
                    "slide_type": slide.get("type", "unknown"),
                    "coverage": 1.0,  # Empty slides are compliant
                    "reason": "Empty slide"
                })
                continue

            # Calculate coverage using multiple methods
            ngram_coverage = self._calculate_ngram_coverage(slide_normalized, source_ngrams)
            term_coverage, used_terms = self._calculate_term_coverage(slide_text, source_terms)
            sequence_coverage = self._calculate_sequence_coverage(slide_normalized, source_normalized)

            # Weighted average (ngrams most important for exact matches)
            slide_coverage = (ngram_coverage * 0.4) + (term_coverage * 0.3) + (sequence_coverage * 0.3)

            all_used_terms.update(used_terms)

            slide_result = {
                "slide_index": i,
                "slide_type": slide.get("type", "unknown"),
                "title": slide.get("title", "Untitled"),
                "coverage": round(slide_coverage, 3),
                "ngram_coverage": round(ngram_coverage, 3),
                "term_coverage": round(term_coverage, 3),
                "sequence_coverage": round(sequence_coverage, 3),
            }
            slide_results.append(slide_result)

            # Detect potential hallucinations (low coverage slides with substantial content)
            if slide_coverage < 0.5 and len(slide_normalized) > 100:
                potential_hallucinations.append({
                    "slide_index": i,
                    "slide_type": slide.get("type", "unknown"),
                    "title": slide.get("title", ""),
                    "coverage": round(slide_coverage, 3),
                    "content_preview": slide_text[:200] + "..." if len(slide_text) > 200 else slide_text
                })

            total_coverage += slide_coverage

            if verbose:
                print(f"[RAG_VERIFIER] Slide {i} ({slide.get('type', 'unknown')}): {slide_coverage:.1%} coverage", flush=True)

        # Calculate overall coverage
        if slides:
            result.overall_coverage = round(total_coverage / len(slides), 3)
        else:
            result.overall_coverage = 0.0

        result.slide_coverage = slide_results
        result.potential_hallucinations = potential_hallucinations
        result.source_terms_used = list(all_used_terms)[:50]  # Top 50 terms
        result.source_terms_missed = [t for t in source_terms if t not in all_used_terms][:30]
        result.is_compliant = result.overall_coverage >= self.min_coverage_threshold

        # Generate summary
        if result.is_compliant:
            result.summary = f"✅ RAG COMPLIANT: {result.overall_coverage:.1%} coverage (threshold: {self.min_coverage_threshold:.0%})"
        else:
            result.summary = f"⚠️ RAG NON-COMPLIANT: {result.overall_coverage:.1%} coverage (required: {self.min_coverage_threshold:.0%})"
            if potential_hallucinations:
                result.summary += f" - {len(potential_hallucinations)} slides may contain hallucinations"

        if verbose:
            print(f"[RAG_VERIFIER] {result.summary}", flush=True)

        return result

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove special formatting, extra whitespace, etc.
        text = text.lower()
        text = re.sub(r'\[SYNC:[\w_]+\]', '', text)  # Remove sync markers
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()

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

        # Include code blocks (comments and variable names can indicate topic)
        for block in slide.get("code_blocks", []):
            if block.get("code"):
                parts.append(block["code"])

        return " ".join(parts)

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key technical terms from text."""
        # Normalize
        text_lower = text.lower()

        # Split into words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text_lower)

        # Filter to meaningful terms (length > 3, not common words)
        stopwords = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has',
            'will', 'can', 'are', 'was', 'were', 'been', 'being', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'need', 'also', 'just',
            'like', 'make', 'made', 'use', 'used', 'using', 'then', 'than',
            'more', 'most', 'some', 'such', 'only', 'other', 'into', 'over',
            'after', 'before', 'between', 'under', 'during', 'through', 'about',
            'against', 'each', 'every', 'both', 'these', 'those', 'your', 'their',
            'vous', 'nous', 'elle', 'elles', 'sont', 'avec', 'pour', 'dans',
            'cette', 'cela', 'mais', 'donc', 'ainsi', 'comme', 'tout', 'tous',
            'plus', 'moins', 'bien', 'tres', 'tres', 'fait', 'faire', 'peut',
        }

        key_terms = [w for w in words if len(w) > 3 and w not in stopwords]

        # Return unique terms
        return list(set(key_terms))

    def _extract_ngrams(self, text: str, n: int = 3) -> set:
        """Extract n-grams from text."""
        words = text.split()
        if len(words) < n:
            return set()
        return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))

    def _calculate_ngram_coverage(self, generated: str, source_ngrams: set) -> float:
        """Calculate what percentage of generated n-grams appear in source."""
        if not source_ngrams:
            return 0.0

        generated_ngrams = self._extract_ngrams(generated, n=3)
        if not generated_ngrams:
            return 1.0  # No content to verify

        matches = generated_ngrams.intersection(source_ngrams)
        return len(matches) / len(generated_ngrams)

    def _calculate_term_coverage(self, generated: str, source_terms: List[str]) -> Tuple[float, set]:
        """Calculate what percentage of source terms appear in generated content."""
        if not source_terms:
            return 0.0, set()

        generated_lower = generated.lower()
        used_terms = {term for term in source_terms if term in generated_lower}

        # Also extract terms from generated and see how many are in source
        generated_terms = set(self._extract_key_terms(generated))
        source_term_set = set(source_terms)
        generated_from_source = generated_terms.intersection(source_term_set)

        if not generated_terms:
            return 1.0, used_terms

        coverage = len(generated_from_source) / len(generated_terms)
        return coverage, used_terms

    def _calculate_sequence_coverage(self, generated: str, source: str) -> float:
        """Calculate sequence similarity between generated and source."""
        if not generated or not source:
            return 0.0

        # Use SequenceMatcher for fuzzy matching
        # Sample if texts are too long (performance)
        max_len = 10000
        if len(generated) > max_len:
            generated = generated[:max_len]
        if len(source) > max_len:
            # Sample from different parts of source
            source = source[:max_len//2] + source[-max_len//2:]

        matcher = SequenceMatcher(None, generated, source)
        return matcher.ratio()


# Singleton instance
_rag_verifier = None


def get_rag_verifier() -> RAGVerifier:
    """Get singleton RAG verifier instance."""
    global _rag_verifier
    if _rag_verifier is None:
        _rag_verifier = RAGVerifier(min_coverage_threshold=0.90)
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
