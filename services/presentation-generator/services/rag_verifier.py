"""
RAG Verification Service v5

Verifies that generated content actually uses the RAG source documents.

v5 improvements:
- WeaveGraph integration for query expansion (discovers related concepts)
- Expanded matching using concept graph edges

v4 improvements:
- Multilingual support with E5-large embeddings (cross-language semantic matching)
- Automatic language detection for source/generated content
- Semantic-only mode for cross-language verification (disables keyword/topic)
- Configurable via RAG_VERIFIER_MODE environment variable

Modes:
- "auto": Detects if cross-language and uses appropriate mode
- "semantic_only": Only uses semantic similarity (best for multilingual)
- "comprehensive": Uses all methods (keyword, topic, semantic)

Environment Variables:
- RAG_VERIFIER_MODE: "auto" (default), "semantic_only", or "comprehensive"
- RAG_EMBEDDING_BACKEND: "e5-large" (default for multilingual), "minilm", "bge-m3"
- WEAVE_GRAPH_ENABLED: "true" (default) to enable WeaveGraph query expansion
"""
import os
import re
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter

# WeaveGraph imports (optional - graceful fallback if not available)
try:
    from services.weave_graph import (
        WeaveGraphBuilder,
        QueryExpansion,
        ResonanceMatcher,
        ResonanceConfig,
        ResonanceResult
    )
    HAS_WEAVE_GRAPH = True
except ImportError:
    HAS_WEAVE_GRAPH = False
    WeaveGraphBuilder = None
    QueryExpansion = None
    ResonanceMatcher = None
    ResonanceConfig = None
    ResonanceResult = None


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

    # WeaveGraph v5 fields
    weave_graph_enabled: bool = False
    expanded_terms: List[str] = field(default_factory=list)
    expansion_boost: float = 0.0  # How much WeaveGraph improved coverage

    # Resonance v6 fields
    resonance_enabled: bool = False
    resonance_boost: float = 0.0  # How much resonance improved coverage
    direct_matches: int = 0
    propagated_matches: int = 0
    max_resonance_depth: int = 0
    top_resonating_concepts: List[Dict] = field(default_factory=list)


class RAGVerifier:
    """
    Verifies RAG content usage using multiple validation methods (v4).

    v4 improvements:
    - Multilingual support with E5-large embeddings
    - Automatic language detection
    - Semantic-only mode for cross-language content
    - Configurable verification mode

    Validation methods:
    1. Semantic similarity (E5-large multilingual) - captures meaning across languages
    2. Source keyword validation - technical terms must exist in source (same-language only)
    3. Topic matching - extracted topics from source must be present (same-language only)
    4. Hallucination detection - flags content not traceable to source
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
    # Thresholds adjusted for multilingual E5-large embeddings
    MIN_SEMANTIC_THRESHOLD = 0.35      # 35% for E5-large cross-lingual (was 40%)
    MIN_SEMANTIC_THRESHOLD_SAME_LANG = 0.45  # 45% for same language
    MIN_KEYWORD_THRESHOLD = 0.30       # 30% of technical keywords
    MIN_TOPIC_THRESHOLD = 0.25         # 25% of topics must match
    MAX_HALLUCINATION_RATIO = 0.40     # Max 40% of slides can be flagged

    # Common French words for language detection
    FRENCH_INDICATORS = {
        'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'est', 'sont',
        'pour', 'dans', 'avec', 'sur', 'que', 'qui', 'ce', 'cette', 'ces',
        'nous', 'vous', 'ils', 'elle', 'être', 'avoir', 'fait', 'faire',
        'peut', 'peuvent', 'aussi', 'plus', 'très', 'bien', 'comme', 'mais',
    }

    # Common English words for language detection
    ENGLISH_INDICATORS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'and', 'or', 'but', 'if', 'for', 'with', 'this', 'that', 'these',
        'from', 'by', 'as', 'at', 'to', 'of', 'in', 'on', 'it', 'its',
    }

    def __init__(self, min_coverage_threshold: float = 0.55, mode: str = "auto"):
        """
        Initialize the verifier.

        Args:
            min_coverage_threshold: Minimum semantic similarity to be compliant (default 55%)
            mode: Verification mode - "auto", "semantic_only", or "comprehensive"
        """
        self.min_coverage_threshold = min_coverage_threshold
        self.mode = os.getenv("RAG_VERIFIER_MODE", mode).lower()
        self._embedding_engine = None
        self._engine_loaded = False

        # Preferred backend for multilingual
        # Use unified env var, fallback to legacy, then default to e5-large for multilingual
        self._preferred_backend = (
            os.getenv("EMBEDDING_BACKEND") or
            os.getenv("RAG_EMBEDDING_BACKEND") or
            "e5-large"
        )

        # WeaveGraph v5 settings
        self._weave_graph_enabled = os.getenv("WEAVE_GRAPH_ENABLED", "true").lower() == "true"
        self._weave_graph_builder: Optional[WeaveGraphBuilder] = None

        # Resonance v6 settings
        self._resonance_enabled = os.getenv("RESONANCE_ENABLED", "true").lower() == "true"
        self._resonance_matcher: Optional[ResonanceMatcher] = None
        self._resonance_decay = float(os.getenv("RESONANCE_DECAY", "0.7"))
        self._resonance_max_depth = int(os.getenv("RESONANCE_MAX_DEPTH", "3"))

        if self._weave_graph_enabled and HAS_WEAVE_GRAPH:
            try:
                self._weave_graph_builder = WeaveGraphBuilder()

                # Initialize resonance matcher if enabled
                if self._resonance_enabled and ResonanceMatcher:
                    config = ResonanceConfig(
                        decay_factor=self._resonance_decay,
                        max_depth=self._resonance_max_depth,
                        min_resonance=0.10,
                        boost_translation=1.2,
                        boost_synonym=1.1
                    )
                    self._resonance_matcher = ResonanceMatcher(config)
                    print(f"[RAG_VERIFIER] Initialized v6 - mode: {self.mode}, backend: {self._preferred_backend}, "
                          f"WeaveGraph: enabled, Resonance: enabled (decay={self._resonance_decay}, depth={self._resonance_max_depth})", flush=True)
                else:
                    print(f"[RAG_VERIFIER] Initialized v5 - mode: {self.mode}, backend: {self._preferred_backend}, WeaveGraph: enabled", flush=True)
            except Exception as e:
                print(f"[RAG_VERIFIER] WeaveGraph init failed: {e}, disabled", flush=True)
                self._weave_graph_enabled = False
        else:
            print(f"[RAG_VERIFIER] Initialized v6 - mode: {self.mode}, backend: {self._preferred_backend}, WeaveGraph: disabled", flush=True)

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on common word frequency.

        Returns: "en", "fr", or "unknown"
        """
        if not text:
            return "unknown"

        words = set(re.findall(r'\b[a-zA-Z]{2,}\b', text.lower()))

        french_count = len(words & self.FRENCH_INDICATORS)
        english_count = len(words & self.ENGLISH_INDICATORS)

        if french_count > english_count * 1.5:
            return "fr"
        elif english_count > french_count * 1.5:
            return "en"
        elif french_count > 5:
            return "fr"
        elif english_count > 5:
            return "en"
        return "unknown"

    def _is_cross_language(self, source_text: str, generated_text: str) -> bool:
        """Check if source and generated content are in different languages."""
        source_lang = self._detect_language(source_text[:2000])  # Sample first 2000 chars
        generated_lang = self._detect_language(generated_text[:2000])

        is_cross = source_lang != generated_lang and source_lang != "unknown" and generated_lang != "unknown"

        if is_cross:
            print(f"[RAG_VERIFIER] Cross-language detected: source={source_lang}, generated={generated_lang}", flush=True)

        return is_cross

    async def _expand_with_weave_graph(
        self,
        query_terms: List[str],
        user_id: str = "default"
    ) -> Tuple[List[str], float]:
        """
        Expand query terms using WeaveGraph.

        Returns:
            Tuple of (expanded_terms, expansion_boost_factor)
        """
        if not self._weave_graph_enabled or not self._weave_graph_builder:
            return query_terms, 0.0

        try:
            # Expand using the graph
            expansion = await self._weave_graph_builder.expand_query(
                " ".join(query_terms),
                user_id,
                max_expansions=15
            )

            if expansion and expansion.expanded_terms:
                # Calculate boost based on how many new terms were found
                original_count = len(query_terms)
                expanded_count = len(expansion.expanded_terms)
                new_terms = expanded_count - original_count

                # Boost factor: more expansions = better matching potential
                boost = min(0.15, new_terms * 0.02)  # Max 15% boost

                print(f"[RAG_VERIFIER] WeaveGraph expanded {original_count} -> {expanded_count} terms (+{boost:.1%} boost)", flush=True)
                print(f"[RAG_VERIFIER] Expanded terms: {expansion.expanded_terms[:10]}", flush=True)

                return expansion.expanded_terms, boost

        except Exception as e:
            print(f"[RAG_VERIFIER] WeaveGraph expansion error: {e}", flush=True)

        return query_terms, 0.0

    def _expand_with_weave_graph_sync(
        self,
        query_terms: List[str],
        user_id: str = "default"
    ) -> Tuple[List[str], float]:
        """Synchronous wrapper for WeaveGraph expansion."""
        import concurrent.futures

        print(f"[RAG_VERIFIER] WeaveGraph sync expansion starting for {len(query_terms)} terms...", flush=True)

        def run_in_new_loop():
            """Run async code in a fresh event loop (separate thread)."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self._expand_with_weave_graph(query_terms, user_id)
                )
            finally:
                new_loop.close()

        try:
            # Always use a separate thread with its own event loop
            # This avoids conflicts with the main FastAPI event loop
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(run_in_new_loop)
                return future.result(timeout=10.0)
        except Exception as e:
            print(f"[RAG_VERIFIER] Sync expansion failed: {e}", flush=True)
            return query_terms, 0.0

    async def _compute_resonance(
        self,
        generated_terms: List[str],
        source_terms: List[str],
        user_id: str = "default"
    ) -> Optional[Dict]:
        """
        Compute resonance scores using the concept graph.

        Returns dict with boost and match statistics.
        """
        if not self._resonance_matcher or not self._weave_graph_builder:
            return None

        try:
            # Get the graph from the store
            store = self._weave_graph_builder.store
            await store.initialize()

            # Find matching concepts in the graph
            matched_concept_ids = []
            source_set = set(t.lower() for t in source_terms)

            for term in generated_terms:
                concept = await store.get_concept_by_name(
                    term.lower().replace(' ', '_'), user_id
                )
                if concept and (
                    concept.canonical_name in source_set or
                    term.lower() in source_set
                ):
                    matched_concept_ids.append(concept.id)

            if not matched_concept_ids:
                return {'boost': 0.0, 'direct_matches': 0, 'propagated_matches': 0}

            # Build a local WeaveGraph for resonance propagation
            from services.weave_graph import WeaveGraph

            graph = WeaveGraph(user_id=user_id)

            # Get concepts and edges from store
            for cid in matched_concept_ids:
                neighbors = await store.get_concept_neighbors(cid, user_id, min_weight=0.5)
                for neighbor, weight, rel_type in neighbors:
                    graph.add_concept(neighbor)
                    from services.weave_graph import ConceptEdge, RelationType
                    edge = ConceptEdge(
                        source_id=cid,
                        target_id=neighbor.id,
                        relation_type=RelationType(rel_type),
                        weight=weight,
                        bidirectional=True
                    )
                    graph.add_edge(edge)

            # Propagate resonance
            result = self._resonance_matcher.propagate(matched_concept_ids, graph)

            # Calculate boost based on propagation
            if result.propagated_matches > 0:
                propagation_ratio = min(1.0, result.propagated_matches / 20)
                avg_resonance = result.total_resonance / (result.direct_matches + result.propagated_matches)
                boost = min(0.15, propagation_ratio * avg_resonance * 0.25)
            else:
                boost = 0.0

            return {
                'boost': boost,
                'direct_matches': result.direct_matches,
                'propagated_matches': result.propagated_matches,
                'max_depth': result.max_depth_reached,
                'top_concepts': [
                    {'id': cid, 'score': score}
                    for cid, score in result.get_top_concepts(5)
                ]
            }

        except Exception as e:
            print(f"[RAG_VERIFIER] Resonance computation error: {e}", flush=True)
            return None

    def _compute_resonance_sync(
        self,
        generated_terms: List[str],
        source_terms: List[str],
        user_id: str = "default"
    ) -> Optional[Dict]:
        """Synchronous wrapper for resonance computation."""
        import concurrent.futures

        def run_in_new_loop():
            """Run async code in a fresh event loop (separate thread)."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    self._compute_resonance(generated_terms, source_terms, user_id)
                )
            finally:
                new_loop.close()

        try:
            # Always use a separate thread with its own event loop
            # This avoids conflicts with the main FastAPI event loop
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(run_in_new_loop)
                return future.result(timeout=15.0)
        except Exception as e:
            print(f"[RAG_VERIFIER] Sync resonance failed: {e}", flush=True)
            return None

    def _get_engine(self, prefer_multilingual: bool = True):
        """
        Lazy load the embedding engine.

        Args:
            prefer_multilingual: If True, try to load E5-large for cross-language support
        """
        if not self._engine_loaded:
            try:
                from services.sync.embedding_engine import EmbeddingEngineFactory

                # Always use the preferred backend to avoid loading multiple models
                # The preferred backend is set via EMBEDDING_BACKEND env var
                backend = self._preferred_backend

                try:
                    self._embedding_engine = EmbeddingEngineFactory.create(backend)
                    print(f"[RAG_VERIFIER] Using embedding engine: {self._embedding_engine.name}", flush=True)

                    # Check if multilingual
                    if hasattr(self._embedding_engine, 'is_multilingual'):
                        print(f"[RAG_VERIFIER] Multilingual support: {self._embedding_engine.is_multilingual}", flush=True)

                except Exception as e:
                    # Fallback to auto if preferred backend fails
                    print(f"[RAG_VERIFIER] {backend} failed: {e}, falling back to auto", flush=True)
                    self._embedding_engine = EmbeddingEngineFactory.create("auto")
                    print(f"[RAG_VERIFIER] Using fallback engine: {self._embedding_engine.name}", flush=True)

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

        # Filter out common words and French non-technical terms
        non_technical = {
            # English common words
            'the', 'and', 'for', 'with', 'from', 'this', 'that', 'are', 'was',
            # French common patterns that aren't technical
            'est-ce', 'qu\'est', 'c\'est', 'n\'est', 'd\'un', 'd\'une',
            'qu\'il', 'qu\'on', 'qu\'une', 's\'il', 'jusqu\'',
            'lorsqu\'', 'puisqu\'', 'aujourd\'',
        }
        return terms - non_technical

    # Common translation mappings for technical terms (EN <-> FR)
    TRANSLATION_MAP = {
        # EN -> FR mappings
        'enterprise': 'entreprise', 'integration': 'intégration', 'pattern': 'patron',
        'patterns': 'patrons', 'messaging': 'messagerie', 'message': 'message',
        'messages': 'messages', 'system': 'système', 'systems': 'systèmes',
        'application': 'application', 'applications': 'applications',
        'service': 'service', 'services': 'services', 'architecture': 'architecture',
        'component': 'composant', 'components': 'composants', 'channel': 'canal',
        'channels': 'canaux', 'queue': 'file', 'queues': 'files',
        'broker': 'courtier', 'endpoint': 'endpoint', 'router': 'routeur',
        'filter': 'filtre', 'transformer': 'transformateur', 'adapter': 'adaptateur',
        'gateway': 'passerelle', 'pipeline': 'pipeline', 'workflow': 'workflow',
        'event': 'événement', 'events': 'événements', 'publish': 'publier',
        'subscribe': 'souscrire', 'consumer': 'consommateur', 'producer': 'producteur',
        'synchronous': 'synchrone', 'asynchronous': 'asynchrone',
        'request': 'requête', 'response': 'réponse', 'protocol': 'protocole',
        # FR -> EN reverse mappings
        'entreprise': 'enterprise', 'intégration': 'integration', 'patron': 'pattern',
        'patrons': 'patterns', 'messagerie': 'messaging', 'système': 'system',
        'systèmes': 'systems', 'composant': 'component', 'composants': 'components',
        'canal': 'channel', 'canaux': 'channels', 'file': 'queue', 'files': 'queues',
        'courtier': 'broker', 'routeur': 'router', 'filtre': 'filter',
        'transformateur': 'transformer', 'adaptateur': 'adapter', 'passerelle': 'gateway',
        'événement': 'event', 'événements': 'events', 'publier': 'publish',
        'souscrire': 'subscribe', 'consommateur': 'consumer', 'producteur': 'producer',
        'synchrone': 'synchronous', 'asynchrone': 'asynchronous',
        'requête': 'request', 'réponse': 'response', 'protocole': 'protocol',
    }

    def _normalize_topic(self, topic: str) -> str:
        """Normalize a topic by mapping translations to a canonical form."""
        topic_lower = topic.lower()
        # If it's in our translation map, return the English equivalent (canonical)
        if topic_lower in self.TRANSLATION_MAP:
            return self.TRANSLATION_MAP[topic_lower]
        return topic_lower

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity ratio between two strings (0-1)."""
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 < len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        # Quick reject: if length difference > 30%, unlikely to be cognates
        if (len1 - len2) / len1 > 0.3:
            return 0.0

        # Simple Levenshtein distance calculation
        prev_row = list(range(len2 + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        distance = prev_row[-1]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len)

    def _is_cognate(self, word1: str, word2: str, threshold: float = 0.75) -> bool:
        """
        Check if two words are cognates (similar spelling across languages).

        Examples: integration/intégration, architecture/architecture,
                  service/service, application/application
        """
        # Remove accents for comparison
        import unicodedata
        def remove_accents(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            )

        w1 = remove_accents(word1.lower())
        w2 = remove_accents(word2.lower())

        # Exact match after accent removal
        if w1 == w2:
            return True

        # Check Levenshtein similarity
        return self._levenshtein_ratio(w1, w2) >= threshold

    def _semantic_topic_match(
        self,
        source_topics: List[str],
        generated_topics: List[str],
        engine,
        threshold: float = 0.70
    ) -> Tuple[float, List[str]]:
        """
        Use embeddings to match topics semantically across languages.

        Returns:
            Tuple of (match_ratio, matched_topics)
        """
        if not source_topics or not generated_topics or not engine:
            return 0.0, []

        try:
            # Embed all topics
            source_embeddings = engine.embed_batch(source_topics)
            generated_embeddings = engine.embed_batch(generated_topics)

            matched = []
            for i, gen_emb in enumerate(generated_embeddings):
                # Find best match in source
                best_sim = 0.0
                for src_emb in source_embeddings:
                    sim = engine.similarity(gen_emb, src_emb)
                    best_sim = max(best_sim, sim)

                if best_sim >= threshold:
                    matched.append(generated_topics[i])

            match_ratio = len(matched) / len(generated_topics) if generated_topics else 0.0
            return match_ratio, matched

        except Exception as e:
            print(f"[RAG_VERIFIER] Semantic topic match error: {e}", flush=True)
            return 0.0, []

    def _extract_topics(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract main topics/concepts from text using frequency analysis.

        Returns the top N most frequent significant terms that are likely topics.
        """
        # Get all significant words
        words = re.findall(r'\b[a-zA-Z\u00C0-\u017F]{4,}\b', text.lower())  # Include accented chars

        # Extended stopwords (EN + FR)
        stopwords = {
            # English stopwords
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has',
            'will', 'can', 'are', 'was', 'were', 'been', 'being', 'would', 'could',
            'should', 'may', 'might', 'must', 'need', 'also', 'just', 'like',
            'make', 'made', 'use', 'used', 'using', 'then', 'than', 'more', 'most',
            'some', 'such', 'only', 'other', 'into', 'over', 'which', 'where',
            'when', 'what', 'while', 'there', 'here', 'they', 'their', 'them',
            'example', 'examples', 'section', 'chapter', 'part', 'page', 'figure',
            'able', 'about', 'above', 'according', 'across', 'after', 'again',
            'each', 'every', 'both', 'either', 'neither', 'first', 'second', 'third',
            'before', 'between', 'through', 'during', 'under', 'along', 'following',
            'however', 'therefore', 'although', 'because', 'since', 'unless',
            # French stopwords (extended)
            'vous', 'nous', 'elle', 'sont', 'avec', 'pour', 'dans', 'cette',
            'cela', 'mais', 'donc', 'ainsi', 'comme', 'tout', 'tous', 'plus',
            'moins', 'bien', 'fait', 'faire', 'peut', 'avoir', 'etre', 'tres',
            'comment', 'quand', 'pourquoi', 'quel', 'quelle', 'quels', 'quelles',
            'chaque', 'autre', 'autres', 'notre', 'votre', 'leur', 'leurs',
            'celui', 'celle', 'ceux', 'celles', 'meme', 'aussi', 'encore',
            'toujours', 'jamais', 'souvent', 'parfois', 'entre', 'vers', 'chez',
            'depuis', 'pendant', 'avant', 'apres', 'sous', 'sans', 'selon',
            'afin', 'alors', 'besoins', 'besoin', 'projet', 'projets',
            'choisir', 'choix', 'faut', 'falloir', 'doit', 'doivent',
            'permettre', 'permet', 'permettent', 'utiliser', 'utilise',
            # Question words that aren't topics
            'quoi', 'lequel', 'laquelle', 'lesquels', 'lesquelles',
        }

        # Count word frequency
        word_counts = Counter(w for w in words if w not in stopwords and len(w) > 3)

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

        Uses three methods for cross-language (EN/FR) matching:
        1. Exact match after normalization (translation map)
        2. Cognate detection (Levenshtein similarity)
        3. Semantic similarity for remaining terms

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

        # Normalize source for matching
        source_normalized = {self._normalize_topic(t) for t in source_combined}
        source_normalized.update(source_combined)

        found_terms = set()
        missing_terms = set()

        for term in generated_terms:
            found = False

            # Method 1: Exact match or normalized match
            normalized = self._normalize_topic(term)
            if term in source_combined or normalized in source_normalized:
                found = True

            # Method 2: Cognate detection
            if not found:
                for src_term in source_combined:
                    if self._is_cognate(term, src_term, threshold=0.80):
                        found = True
                        break

            if found:
                found_terms.add(term)
            else:
                missing_terms.add(term)

        # Filter out very short missing terms (likely false positives)
        missing_terms = {t for t in missing_terms if len(t) > 4}

        coverage = len(found_terms) / len(generated_terms) if generated_terms else 1.0

        if verbose:
            print(f"[RAG_VERIFIER] Keyword validation: {len(found_terms)}/{len(generated_terms)} "
                  f"found ({coverage:.1%})", flush=True)
            if missing_terms and len(missing_terms) <= 10:
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

        Uses three methods for cross-language matching:
        1. Exact match after normalization (translation map)
        2. Cognate detection (Levenshtein similarity after accent removal)
        3. Semantic similarity via embeddings (if engine available)

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

        matched_topics = set()

        # Method 1: Exact match after normalization
        source_normalized = {self._normalize_topic(t) for t in source_topics}
        for gen_topic in generated_topics:
            gen_normalized = self._normalize_topic(gen_topic)
            if gen_normalized in source_normalized or gen_topic in source_topics:
                matched_topics.add(gen_topic)

        # Method 2: Cognate detection (for words not matched yet)
        unmatched = [t for t in generated_topics if t not in matched_topics]
        for gen_topic in unmatched:
            for src_topic in source_topics:
                if self._is_cognate(gen_topic, src_topic, threshold=0.80):
                    matched_topics.add(gen_topic)
                    break

        # Method 3: Semantic matching with embeddings (for remaining unmatched)
        unmatched = [t for t in generated_topics if t not in matched_topics]
        if unmatched:
            engine = self._get_engine()
            if engine:
                sem_ratio, sem_matched = self._semantic_topic_match(
                    source_topics, unmatched, engine, threshold=0.70
                )
                matched_topics.update(sem_matched)

        # Calculate final match score
        match_score = len(matched_topics) / len(generated_topics) if generated_topics else 1.0

        if verbose:
            print(f"[RAG_VERIFIER] Topic validation: {len(matched_topics)}/{len(generated_topics)} "
                  f"match ({match_score:.1%})", flush=True)
            print(f"[RAG_VERIFIER] Source top topics: {source_topics[:10]}", flush=True)
            print(f"[RAG_VERIFIER] Generated topics: {generated_topics[:10]}", flush=True)
            if matched_topics:
                print(f"[RAG_VERIFIER] Matched topics: {list(matched_topics)[:10]}", flush=True)

        return match_score, source_topics[:20], generated_topics[:15]

    def verify_comprehensive(
        self,
        generated_content: Dict[str, Any],
        source_documents: str,
        verbose: bool = False,
        user_id: str = "default"
    ) -> RAGVerificationResult:
        """
        Comprehensive verification using all validation methods.

        v6: Resonance propagation through concept graph for better semantic matching.
        v5: WeaveGraph query expansion for better concept matching.
        v4: Automatically switches to semantic-only mode for cross-language content.

        Combines:
        1. WeaveGraph query expansion (v5)
        2. Resonance propagation (v6)
        3. Semantic similarity (E5-large multilingual embeddings)
        4. Keyword validation (same-language only)
        5. Topic matching (same-language only)
        6. Hallucination detection
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

        # Extract all generated text for language detection
        all_generated_text = " ".join(self._extract_slide_text(s) for s in slides)

        # v5: WeaveGraph query expansion
        expansion_boost = 0.0
        expanded_terms = []
        resonance_boost = 0.0

        if self._weave_graph_enabled and self._weave_graph_builder:
            result.weave_graph_enabled = True
            # Extract key terms from generated content for expansion
            generated_terms = list(self._extract_technical_terms(all_generated_text))[:20]
            expanded_terms, expansion_boost = self._expand_with_weave_graph_sync(
                generated_terms, user_id
            )
            result.expanded_terms = expanded_terms
            result.expansion_boost = expansion_boost

            # v6: Resonance propagation through concept graph
            if self._resonance_enabled and self._resonance_matcher:
                result.resonance_enabled = True
                source_terms = list(self._extract_technical_terms(source_documents))[:50]

                try:
                    # Build a local graph from the expanded terms for resonance
                    from services.weave_graph import WeaveGraph

                    # Get graph from builder's store for the user
                    resonance_result = self._compute_resonance_sync(
                        generated_terms, source_terms, user_id
                    )

                    if resonance_result:
                        resonance_boost = resonance_result.get('boost', 0.0)
                        result.resonance_boost = resonance_boost
                        result.direct_matches = resonance_result.get('direct_matches', 0)
                        result.propagated_matches = resonance_result.get('propagated_matches', 0)
                        result.max_resonance_depth = resonance_result.get('max_depth', 0)
                        result.top_resonating_concepts = resonance_result.get('top_concepts', [])

                        if verbose:
                            print(f"[RAG_VERIFIER] Resonance: {result.direct_matches} direct, "
                                  f"{result.propagated_matches} propagated, +{resonance_boost:.1%} boost", flush=True)

                except Exception as e:
                    print(f"[RAG_VERIFIER] Resonance computation failed: {e}", flush=True)

        # Combined boost from expansion and resonance
        total_boost = expansion_boost + resonance_boost

        # Detect if cross-language (e.g., EN source -> FR generated)
        is_cross_lang = self._is_cross_language(source_documents, all_generated_text)

        # Determine verification mode
        use_semantic_only = (
            self.mode == "semantic_only" or
            (self.mode == "auto" and is_cross_lang)
        )

        if use_semantic_only:
            print(f"[RAG_VERIFIER] Using SEMANTIC-ONLY mode (cross-language: {is_cross_lang})", flush=True)

        # Adjust threshold for cross-language
        semantic_threshold = (
            self.MIN_SEMANTIC_THRESHOLD if is_cross_lang
            else self.MIN_SEMANTIC_THRESHOLD_SAME_LANG
        )

        # 1. Semantic similarity (main check - works cross-language with E5-large)
        engine = self._get_engine(prefer_multilingual=is_cross_lang)
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

        # For semantic-only mode, skip keyword/topic validation
        if use_semantic_only:
            # In semantic-only mode, compliance is based only on semantic similarity
            result.keyword_coverage = 1.0  # Skip
            result.topic_match_score = 1.0  # Skip
            result.source_topics = []
            result.generated_topics = []

            # Hallucination check still applies
            non_empty_slides = sum(1 for s in slides if self._extract_slide_text(s).strip())
            hallucination_ratio = (
                len(result.potential_hallucinations) / non_empty_slides
                if non_empty_slides > 0 else 0
            )

            # Apply WeaveGraph boost to coverage
            boosted_coverage = min(1.0, result.overall_coverage + total_boost)

            is_semantic_ok = boosted_coverage >= semantic_threshold
            is_hallucination_ok = hallucination_ratio <= self.MAX_HALLUCINATION_RATIO

            if not is_semantic_ok:
                failure_reasons.append(
                    f"semantic_similarity_low ({boosted_coverage:.1%} < {semantic_threshold:.0%})"
                )
            if not is_hallucination_ok:
                failure_reasons.append(
                    f"too_many_hallucinations ({hallucination_ratio:.1%} > {self.MAX_HALLUCINATION_RATIO:.0%})"
                )

            result.failure_reasons = failure_reasons
            result.is_compliant = is_semantic_ok and is_hallucination_ok

            # Build summary
            mode_label = "SEMANTIC-ONLY" if is_cross_lang else "SEMANTIC"
            boost_labels = []
            if expansion_boost > 0:
                boost_labels.append("WeaveGraph")
            if resonance_boost > 0:
                boost_labels.append("Resonance")
            boost_label = f"+{'+'.join(boost_labels)}" if boost_labels else ""
            if result.is_compliant:
                result.summary = (
                    f"✅ RAG COMPLIANT ({mode_label}{boost_label}): {boosted_coverage:.1%} semantic similarity"
                )
            else:
                result.summary = f"❌ RAG NON-COMPLIANT: {', '.join(failure_reasons)}"

            if verbose:
                print(f"[RAG_VERIFIER] {result.summary}", flush=True)

            return result

        # 2. Keyword validation (same-language only)
        keyword_coverage, missing_keywords, found_count = self._validate_keywords(
            generated_content, source_documents, verbose
        )
        result.keyword_coverage = keyword_coverage
        result.source_keywords_found = found_count
        result.source_keywords_missing = missing_keywords

        # 3. Topic matching (same-language only)
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

        # Apply WeaveGraph boost to coverage
        boosted_coverage = min(1.0, result.overall_coverage + total_boost)

        # Determine compliance
        is_semantic_ok = boosted_coverage >= semantic_threshold
        is_keyword_ok = keyword_coverage >= self.MIN_KEYWORD_THRESHOLD
        is_topic_ok = topic_score >= self.MIN_TOPIC_THRESHOLD
        is_hallucination_ok = hallucination_ratio <= self.MAX_HALLUCINATION_RATIO

        # Build failure reasons
        if not is_semantic_ok:
            failure_reasons.append(
                f"semantic_similarity_low ({boosted_coverage:.1%} < {semantic_threshold:.0%})"
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
        boost_labels = []
        if expansion_boost > 0:
            boost_labels.append("WeaveGraph")
        if resonance_boost > 0:
            boost_labels.append("Resonance")
        boost_label = f" +{'+'.join(boost_labels)}" if boost_labels else ""
        if result.is_compliant:
            result.summary = (
                f"✅ RAG COMPLIANT{boost_label}: {boosted_coverage:.1%} semantic, "
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
        # v4: Multilingual support with E5-large, auto mode for cross-language detection
        mode = os.getenv("RAG_VERIFIER_MODE", "auto")
        _rag_verifier = RAGVerifier(min_coverage_threshold=0.55, mode=mode)
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
