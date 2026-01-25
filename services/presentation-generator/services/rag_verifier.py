"""
RAG Verification Service v3

Verifies that generated content actually uses the RAG source documents.
Uses multiple validation methods:
1. Semantic similarity (MiniLM embeddings)
2. Source keyword validation (technical terms must exist in source)
3. Topic matching (extracted topics must match)
4. Hallucination detection (content not traceable to source)
"""
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class RAGVerificationResult:
    """Result of RAG verification analysis."""
    # Overall coverage score (0-100%)
    overall_coverage: float = 0.0

    # Per-slide coverage scores
    slide_coverage: List[Dict[str, Any]] = field(default_factory=list)

    # Detected potential hallucinations (content not in source)
    potential_hallucinations: List[Dict[str, Any]] = field(default_factory=list)

    # Source keyword analysis
    source_keywords_found: int = 0
    source_keywords_missing: List[str] = field(default_factory=list)
    keyword_coverage: float = 0.0

    # Topic analysis
    source_topics: List[str] = field(default_factory=list)
    generated_topics: List[str] = field(default_factory=list)
    topic_match_score: float = 0.0

    # Compliance status
    is_compliant: bool = False

    # Failure reasons (if not compliant)
    failure_reasons: List[str] = field(default_factory=list)

    # Summary message
    summary: str = ""

    # Embedding backend used
    backend_used: str = ""


class RAGVerifier:
    """
    Verifies RAG content usage using multiple validation methods (v3).

    Validation methods:
    1. Semantic similarity (MiniLM embeddings) - captures meaning
    2. Source keyword validation - technical terms must exist in source
    3. Topic matching - extracted topics from source must be present
    4. Hallucination detection - flags content not traceable to source

    Algorithm improvements (v3):
    - Combines semantic + keyword + topic validation
    - Stricter threshold (55% instead of 40%)
    - Keyword validation: 60% of technical terms must be in source
    - Topic matching: generated topics must match source topics
    - Multi-reason failure reporting
    """

    # Slide type weight adjustments (some slides naturally have lower RAG relevance)
    SLIDE_TYPE_WEIGHTS = {
        "title": 0.5,        # Title slides are generic, lower expectation
        "conclusion": 0.7,   # Conclusion summarizes, may differ from source
        "content": 1.0,      # Standard content should match well
        "code": 1.0,         # Code should match source examples
        "code_demo": 1.0,    # Code demos should match source
        "diagram": 0.8,      # Diagrams describe concepts, slight tolerance
    }

    # Minimum thresholds for each validation method
    MIN_SEMANTIC_THRESHOLD = 0.55      # 55% semantic similarity (increased from 40%)
    MIN_KEYWORD_THRESHOLD = 0.50       # 50% of technical keywords must be in source
    MIN_TOPIC_THRESHOLD = 0.40         # 40% of topics must match
    MAX_HALLUCINATION_RATIO = 0.30     # Max 30% of slides can be flagged as hallucinations

    def __init__(self, min_coverage_threshold: float = 0.55):
        """
        Initialize the verifier.

        Args:
            min_coverage_threshold: Minimum semantic similarity to be compliant (default 55%)
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
        """
        Verify using semantic embeddings (MiniLM).

        Algorithm v2 improvements:
        - Uses Top-3 average similarity instead of MAX single chunk
        - Excludes empty slides from average calculation
        - Applies slide type weight adjustments
        - Better normalization for cosine similarity
        """

        # Chunk source documents for better matching
        # Increased chunk size for better context, reduced overlap
        source_chunks = self._chunk_text(source_documents, chunk_size=800, overlap=200)

        if verbose:
            print(f"[RAG_VERIFIER] Source: {len(source_documents)} chars -> {len(source_chunks)} chunks", flush=True)

        # Embed all source chunks at once (batched for efficiency)
        try:
            source_embeddings = engine.embed_batch(source_chunks)
        except Exception as e:
            print(f"[RAG_VERIFIER] Embedding error: {e}, falling back to keywords", flush=True)
            result.backend_used = "keyword-fallback"
            return self._verify_with_keywords(slides, source_documents, verbose, result)

        weighted_similarity_sum = 0.0
        total_weight = 0.0
        slide_results = []
        potential_hallucinations = []
        non_empty_slides = 0

        for i, slide in enumerate(slides):
            slide_text = self._extract_slide_text(slide)
            slide_type = slide.get("type", "content")

            if not slide_text.strip():
                slide_results.append({
                    "slide_index": i,
                    "slide_type": slide_type,
                    "similarity": 1.0,
                    "reason": "Empty slide (excluded from average)"
                })
                # Don't count empty slides in the average
                continue

            non_empty_slides += 1

            # Embed the slide content
            try:
                slide_embedding = engine.embed(slide_text)
            except Exception:
                slide_results.append({
                    "slide_index": i,
                    "slide_type": slide_type,
                    "similarity": 0.5,
                    "reason": "Embedding failed"
                })
                # Count as 0.5 with weight 1.0
                weighted_similarity_sum += 0.5
                total_weight += 1.0
                continue

            # Calculate similarities with ALL source chunks
            similarities = []
            for source_emb in source_embeddings:
                sim = engine.similarity(slide_embedding, source_emb)
                # Cosine similarity is already in [-1, 1] range
                # For normalized vectors, it's typically in [0, 1]
                # Clamp to valid range
                sim = max(0.0, min(1.0, sim))
                similarities.append(sim)

            # Use Top-3 average instead of just MAX
            # This gives a more stable measure of relevance
            similarities.sort(reverse=True)
            top_n = min(3, len(similarities))
            if top_n > 0:
                avg_similarity = sum(similarities[:top_n]) / top_n
            else:
                avg_similarity = 0.0

            # Apply slide type weight adjustment
            slide_weight = self.SLIDE_TYPE_WEIGHTS.get(slide_type, 1.0)

            # Content length also affects weight (longer = more important)
            content_length_factor = min(1.0, len(slide_text) / 500)  # Cap at 500 chars
            effective_weight = slide_weight * (0.5 + 0.5 * content_length_factor)

            slide_result = {
                "slide_index": i,
                "slide_type": slide_type,
                "title": slide.get("title", "Untitled"),
                "similarity": round(avg_similarity, 3),
                "top_3_avg": round(avg_similarity, 3),
                "max_similarity": round(similarities[0] if similarities else 0, 3),
                "weight": round(effective_weight, 2),
            }
            slide_results.append(slide_result)

            # Accumulate weighted similarity
            weighted_similarity_sum += avg_similarity * effective_weight
            total_weight += effective_weight

            # Flag potential hallucination if very low similarity AND substantial content
            # Threshold lowered to 0.25 since we're using Top-3 average now
            if avg_similarity < 0.25 and len(slide_text) > 200:
                potential_hallucinations.append({
                    "slide_index": i,
                    "slide_type": slide_type,
                    "title": slide.get("title", ""),
                    "similarity": round(avg_similarity, 3),
                    "content_preview": slide_text[:150] + "..." if len(slide_text) > 150 else slide_text
                })

            if verbose:
                print(f"[RAG_VERIFIER] Slide {i} ({slide_type}): "
                      f"{avg_similarity:.1%} similarity (top-3 avg, weight={effective_weight:.2f})", flush=True)

        # Calculate weighted overall similarity
        if total_weight > 0:
            result.overall_coverage = round(weighted_similarity_sum / total_weight, 3)
        else:
            result.overall_coverage = 0.0

        result.slide_coverage = slide_results
        result.potential_hallucinations = potential_hallucinations
        result.is_compliant = result.overall_coverage >= self.min_coverage_threshold

        # Generate summary with more context
        if result.is_compliant:
            result.summary = f"✅ RAG COMPLIANT: {result.overall_coverage:.1%} weighted similarity ({engine.name}, {non_empty_slides} slides)"
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

    def _extract_technical_terms(self, text: str) -> Set[str]:
        """
        Extract technical/domain-specific terms from text.

        These are typically:
        - CamelCase or snake_case identifiers
        - Capitalized acronyms (API, REST, SQL, etc.)
        - Multi-word technical phrases
        - Domain terms (often with specific patterns)
        """
        terms = set()
        text_lower = text.lower()

        # 1. CamelCase identifiers (e.g., MessageBroker, ServiceBus)
        camel_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
        terms.update(m.lower() for m in re.findall(camel_pattern, text))

        # 2. Acronyms and uppercase terms (e.g., API, REST, ESB, SOA)
        acronym_pattern = r'\b[A-Z]{2,6}\b'
        terms.update(m.lower() for m in re.findall(acronym_pattern, text))

        # 3. Technical compound terms (e.g., message-broker, service-oriented)
        compound_pattern = r'\b[a-z]+[-_][a-z]+(?:[-_][a-z]+)*\b'
        terms.update(re.findall(compound_pattern, text_lower))

        # 4. Known technical term patterns
        tech_patterns = [
            r'\b(?:micro)?services?\b',
            r'\b(?:message|event)\s*(?:broker|bus|queue)\b',
            r'\bintegration\s*patterns?\b',
            r'\b(?:publish|subscribe|pub|sub)\b',
            r'\b(?:enterprise|service)\s*bus\b',
            r'\barchitecture\s*(?:oriented|patterns?|design)\b',
            r'\b(?:data|message)\s*(?:flow|pipeline|stream)\b',
            r'\b(?:api|rest|soap|grpc)\s*(?:gateway|endpoint)?\b',
            r'\b(?:kafka|rabbitmq|activemq|pulsar|redis)\b',
            r'\b(?:docker|kubernetes|k8s|helm)\b',
            r'\b(?:aws|azure|gcp|cloud)\b',
        ]
        for pattern in tech_patterns:
            terms.update(m.lower().replace(' ', '_') for m in re.findall(pattern, text_lower))

        # Filter out very common words that might match patterns
        common_words = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'are', 'was'}
        return terms - common_words

    def _extract_topics(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract main topics/concepts from text using frequency analysis.

        Returns the top N most frequent significant terms that are likely topics.
        """
        # Get all significant words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Common stopwords to exclude
        stopwords = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has',
            'will', 'can', 'are', 'was', 'were', 'been', 'being', 'would', 'could',
            'should', 'may', 'might', 'must', 'need', 'also', 'just', 'like',
            'make', 'made', 'use', 'used', 'using', 'then', 'than', 'more', 'most',
            'some', 'such', 'only', 'other', 'into', 'over', 'which', 'where',
            'when', 'what', 'while', 'there', 'here', 'they', 'their', 'them',
            'example', 'examples', 'section', 'chapter', 'part', 'page', 'figure',
            'able', 'about', 'above', 'according', 'across', 'after', 'again',
            'vous', 'nous', 'elle', 'sont', 'avec', 'pour', 'dans', 'cette',
            'cela', 'mais', 'donc', 'ainsi', 'comme', 'tout', 'tous', 'plus',
            'moins', 'bien', 'fait', 'faire', 'peut', 'avoir', 'etre', 'tres',
        }

        # Count word frequency
        word_counts = Counter(w for w in words if w not in stopwords)

        # Return top N most common
        return [word for word, _ in word_counts.most_common(top_n)]

    def _validate_keywords(
        self,
        generated_content: Dict[str, Any],
        source_documents: str,
        verbose: bool = False
    ) -> Tuple[float, List[str], int]:
        """
        Validate that technical keywords in generated content exist in source.

        Returns:
            Tuple of (coverage_ratio, missing_keywords, found_count)
        """
        # Extract all text from slides
        slides = generated_content.get("slides", [])
        all_generated_text = " ".join(self._extract_slide_text(s) for s in slides)

        # Extract technical terms from generated content
        generated_terms = self._extract_technical_terms(all_generated_text)

        # Extract terms from source
        source_terms = self._extract_technical_terms(source_documents)
        source_words = self._extract_significant_words(source_documents)
        source_combined = source_terms | source_words

        if not generated_terms:
            return 1.0, [], 0  # No technical terms = nothing to validate

        # Check which generated terms exist in source
        found_terms = generated_terms & source_combined
        missing_terms = generated_terms - source_combined

        # Filter out very short missing terms (likely false positives)
        missing_terms = {t for t in missing_terms if len(t) > 4}

        coverage = len(found_terms) / len(generated_terms) if generated_terms else 1.0

        if verbose and missing_terms:
            print(f"[RAG_VERIFIER] Keyword validation: {len(found_terms)}/{len(generated_terms)} "
                  f"found ({coverage:.1%})", flush=True)
            if len(missing_terms) <= 10:
                print(f"[RAG_VERIFIER] Missing keywords: {list(missing_terms)[:10]}", flush=True)

        return coverage, list(missing_terms)[:20], len(found_terms)

    def _validate_topics(
        self,
        generated_content: Dict[str, Any],
        source_documents: str,
        verbose: bool = False
    ) -> Tuple[float, List[str], List[str]]:
        """
        Validate that generated topics match source document topics.

        Returns:
            Tuple of (match_score, source_topics, generated_topics)
        """
        # Extract all text from slides
        slides = generated_content.get("slides", [])
        all_generated_text = " ".join(self._extract_slide_text(s) for s in slides)

        # Get topics from both
        source_topics = self._extract_topics(source_documents, top_n=30)
        generated_topics = self._extract_topics(all_generated_text, top_n=20)

        if not generated_topics or not source_topics:
            return 1.0, source_topics, generated_topics

        # Calculate overlap
        source_set = set(source_topics)
        generated_set = set(generated_topics)
        overlap = source_set & generated_set

        # Match score: how many of the generated topics appear in source
        match_score = len(overlap) / len(generated_set) if generated_set else 1.0

        if verbose:
            print(f"[RAG_VERIFIER] Topic validation: {len(overlap)}/{len(generated_topics)} "
                  f"match ({match_score:.1%})", flush=True)
            print(f"[RAG_VERIFIER] Source top topics: {source_topics[:10]}", flush=True)
            print(f"[RAG_VERIFIER] Generated topics: {generated_topics[:10]}", flush=True)

        return match_score, source_topics[:20], generated_topics[:15]

    def verify_comprehensive(
        self,
        generated_content: Dict[str, Any],
        source_documents: str,
        verbose: bool = False
    ) -> RAGVerificationResult:
        """
        Comprehensive verification using all validation methods.

        Combines:
        1. Semantic similarity (embeddings)
        2. Keyword validation
        3. Topic matching
        4. Hallucination detection
        """
        result = RAGVerificationResult()
        failure_reasons = []

        if not source_documents:
            result.summary = "No source documents provided - cannot verify RAG usage"
            result.overall_coverage = 0.0
            result.failure_reasons = ["no_source_documents"]
            return result

        slides = generated_content.get("slides", [])
        if not slides:
            result.summary = "No slides in generated content"
            result.overall_coverage = 0.0
            result.failure_reasons = ["no_slides"]
            return result

        # 1. Semantic similarity (main check)
        engine = self._get_engine()
        if engine:
            result.backend_used = engine.name
            semantic_result = self._verify_with_embeddings(
                slides, source_documents, engine, verbose, RAGVerificationResult()
            )
            result.slide_coverage = semantic_result.slide_coverage
            result.potential_hallucinations = semantic_result.potential_hallucinations
            result.overall_coverage = semantic_result.overall_coverage
        else:
            result.backend_used = "keyword-fallback"
            keyword_result = self._verify_with_keywords(
                slides, source_documents, verbose, RAGVerificationResult()
            )
            result.slide_coverage = keyword_result.slide_coverage
            result.potential_hallucinations = keyword_result.potential_hallucinations
            result.overall_coverage = keyword_result.overall_coverage

        # 2. Keyword validation
        keyword_coverage, missing_keywords, found_count = self._validate_keywords(
            generated_content, source_documents, verbose
        )
        result.keyword_coverage = keyword_coverage
        result.source_keywords_found = found_count
        result.source_keywords_missing = missing_keywords

        # 3. Topic matching
        topic_score, source_topics, generated_topics = self._validate_topics(
            generated_content, source_documents, verbose
        )
        result.topic_match_score = topic_score
        result.source_topics = source_topics
        result.generated_topics = generated_topics

        # 4. Hallucination ratio check
        non_empty_slides = sum(1 for s in slides if self._extract_slide_text(s).strip())
        hallucination_ratio = (
            len(result.potential_hallucinations) / non_empty_slides
            if non_empty_slides > 0 else 0
        )

        # Determine compliance
        is_semantic_ok = result.overall_coverage >= self.MIN_SEMANTIC_THRESHOLD
        is_keyword_ok = keyword_coverage >= self.MIN_KEYWORD_THRESHOLD
        is_topic_ok = topic_score >= self.MIN_TOPIC_THRESHOLD
        is_hallucination_ok = hallucination_ratio <= self.MAX_HALLUCINATION_RATIO

        # Build failure reasons
        if not is_semantic_ok:
            failure_reasons.append(
                f"semantic_similarity_low ({result.overall_coverage:.1%} < {self.MIN_SEMANTIC_THRESHOLD:.0%})"
            )
        if not is_keyword_ok:
            failure_reasons.append(
                f"keywords_not_from_source ({keyword_coverage:.1%} < {self.MIN_KEYWORD_THRESHOLD:.0%})"
            )
        if not is_topic_ok:
            failure_reasons.append(
                f"topics_mismatch ({topic_score:.1%} < {self.MIN_TOPIC_THRESHOLD:.0%})"
            )
        if not is_hallucination_ok:
            failure_reasons.append(
                f"too_many_hallucinations ({hallucination_ratio:.1%} > {self.MAX_HALLUCINATION_RATIO:.0%})"
            )

        result.failure_reasons = failure_reasons

        # Compliance requires at least semantic + (keyword OR topic) checks to pass
        result.is_compliant = is_semantic_ok and (is_keyword_ok or is_topic_ok) and is_hallucination_ok

        # Build summary
        if result.is_compliant:
            result.summary = (
                f"✅ RAG COMPLIANT: {result.overall_coverage:.1%} semantic, "
                f"{keyword_coverage:.1%} keywords, {topic_score:.1%} topics"
            )
        else:
            result.summary = (
                f"❌ RAG NON-COMPLIANT: {', '.join(failure_reasons)}"
            )
            if missing_keywords:
                result.summary += f" | Missing keywords: {missing_keywords[:5]}"

        if verbose:
            print(f"[RAG_VERIFIER] {result.summary}", flush=True)

        return result


# Singleton instance
_rag_verifier = None


def get_rag_verifier() -> RAGVerifier:
    """Get singleton RAG verifier instance."""
    global _rag_verifier
    if _rag_verifier is None:
        # v3: Threshold increased to 55% with multi-method validation
        _rag_verifier = RAGVerifier(min_coverage_threshold=0.55)
    return _rag_verifier


def verify_rag_usage(
    generated_script: Dict[str, Any],
    rag_context: str,
    verbose: bool = True,
    comprehensive: bool = True
) -> RAGVerificationResult:
    """
    Convenience function to verify RAG usage in generated content.

    Args:
        generated_script: The generated presentation script
        rag_context: The RAG context that was provided
        verbose: Print detailed analysis
        comprehensive: Use comprehensive multi-method verification (v3)

    Returns:
        RAGVerificationResult with metrics
    """
    verifier = get_rag_verifier()
    if comprehensive:
        return verifier.verify_comprehensive(generated_script, rag_context, verbose=verbose)
    else:
        return verifier.verify(generated_script, rag_context, verbose=verbose)
