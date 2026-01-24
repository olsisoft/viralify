"""
Traceability Service

Generates and manages source traceability for course content.
Links each piece of generated content back to its source documents.
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from models.traceability_models import (
    ContentReference,
    SlideTraceability,
    ConceptTraceability,
    LectureTraceability,
    CourseTraceability,
    SourceCitationConfig,
    CitationStyle,
)
from models.source_models import Source, PedagogicalRole


class TraceabilityService:
    """
    Service for generating and managing source traceability.

    Responsibilities:
    1. Analyze generated content and match to source chunks
    2. Create ContentReference objects linking content to sources
    3. Build slide-level and concept-level traceability
    4. Generate vocal citations for voiceover if enabled
    """

    def __init__(self):
        self._embedding_engine = None

    def _get_embedding_engine(self):
        """Lazy load embedding engine for semantic matching."""
        if self._embedding_engine is None:
            try:
                # Try to import from source_library's vectorization
                from services.vector_store import EmbeddingService
                self._embedding_engine = EmbeddingService()
            except Exception:
                self._embedding_engine = None
        return self._embedding_engine

    async def build_content_references(
        self,
        generated_text: str,
        sources: List[Source],
        source_chunks: List[Dict[str, Any]],
        content_type: str = "content",
    ) -> List[ContentReference]:
        """
        Build content references by matching generated text to source chunks.

        Args:
            generated_text: The generated content (voiceover, slide text, etc.)
            sources: List of Source objects used for generation
            source_chunks: List of chunks with embeddings and metadata
            content_type: Type of content being traced

        Returns:
            List of ContentReference objects
        """
        if not generated_text or not sources:
            return []

        references = []
        source_map = {s.id: s for s in sources}

        # Try semantic matching first
        engine = self._get_embedding_engine()

        if engine and source_chunks:
            # Semantic matching
            references = await self._semantic_match(
                generated_text, sources, source_chunks, engine
            )
        else:
            # Fallback to keyword matching
            references = self._keyword_match(generated_text, sources)

        return references

    async def _semantic_match(
        self,
        text: str,
        sources: List[Source],
        chunks: List[Dict[str, Any]],
        engine,
    ) -> List[ContentReference]:
        """Match text to source chunks using semantic similarity."""
        references = []
        source_map = {s.id: s for s in sources}

        try:
            # Embed the generated text
            text_embedding = await engine.embed(text)

            # Find top matching chunks
            matches = []
            for chunk in chunks:
                if "embedding" in chunk:
                    similarity = engine.cosine_similarity(
                        text_embedding, chunk["embedding"]
                    )
                    if similarity > 0.5:  # Threshold
                        matches.append((chunk, similarity))

            # Sort by similarity and take top 3
            matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = matches[:3]

            for chunk, similarity in top_matches:
                source_id = chunk.get("source_id") or chunk.get("document_id")
                source = source_map.get(source_id)

                if source:
                    ref = ContentReference(
                        source_id=source.id,
                        source_name=source.name,
                        source_type=source.source_type.value,
                        pedagogical_role=source.pedagogical_role.value,
                        location=self._extract_location(chunk),
                        page_number=chunk.get("page_number"),
                        timestamp_seconds=chunk.get("timestamp_seconds"),
                        section_title=chunk.get("section_title"),
                        quote_excerpt=self._extract_excerpt(chunk.get("content", ""), 150),
                        confidence=similarity,
                        relevance_score=similarity,
                    )
                    references.append(ref)

        except Exception as e:
            print(f"[TRACEABILITY] Semantic matching error: {e}", flush=True)
            # Fallback to keyword matching
            references = self._keyword_match(text, sources)

        return references

    def _keyword_match(
        self,
        text: str,
        sources: List[Source],
    ) -> List[ContentReference]:
        """Match text to sources using keyword overlap."""
        references = []
        text_lower = text.lower()
        text_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', text_lower))

        for source in sources:
            if not source.raw_content:
                continue

            source_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', source.raw_content.lower()))
            overlap = text_words.intersection(source_words)

            if len(overlap) >= 3:  # At least 3 common significant words
                relevance = len(overlap) / len(text_words) if text_words else 0

                ref = ContentReference(
                    source_id=source.id,
                    source_name=source.name,
                    source_type=source.source_type.value,
                    pedagogical_role=source.pedagogical_role.value,
                    location="",
                    quote_excerpt=self._find_matching_excerpt(text, source.raw_content),
                    confidence=min(relevance * 1.2, 1.0),
                    relevance_score=relevance,
                )
                references.append(ref)

        # Sort by relevance and return top 3
        references.sort(key=lambda r: r.relevance_score, reverse=True)
        return references[:3]

    def _extract_location(self, chunk: Dict[str, Any]) -> str:
        """Extract human-readable location from chunk metadata."""
        parts = []

        if chunk.get("page_number"):
            parts.append(f"page {chunk['page_number']}")
        if chunk.get("section_title"):
            parts.append(chunk["section_title"])
        if chunk.get("timestamp_seconds"):
            minutes = chunk["timestamp_seconds"] // 60
            seconds = chunk["timestamp_seconds"] % 60
            parts.append(f"timestamp {minutes}:{seconds:02d}")
        if chunk.get("chunk_index") is not None:
            parts.append(f"section {chunk['chunk_index'] + 1}")

        return ", ".join(parts) if parts else ""

    def _extract_excerpt(self, text: str, max_length: int = 150) -> str:
        """Extract a clean excerpt from text."""
        if not text:
            return ""

        # Clean and truncate
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= max_length:
            return text

        # Try to cut at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_comma = truncated.rfind(',')

        if last_period > max_length * 0.6:
            return truncated[:last_period + 1]
        elif last_comma > max_length * 0.6:
            return truncated[:last_comma] + "..."
        else:
            return truncated.rsplit(' ', 1)[0] + "..."

    def _find_matching_excerpt(self, generated: str, source: str, max_length: int = 150) -> str:
        """Find the most relevant excerpt from source that matches generated text."""
        if not source:
            return ""

        # Extract significant words from generated text
        gen_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', generated.lower()))

        # Split source into sentences
        sentences = re.split(r'[.!?]+', source)

        best_sentence = ""
        best_score = 0

        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue

            sent_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', sentence.lower()))
            overlap = gen_words.intersection(sent_words)
            score = len(overlap)

            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()

        return self._extract_excerpt(best_sentence, max_length)

    def build_slide_traceability(
        self,
        slide_index: int,
        slide_data: Dict[str, Any],
        content_refs: List[ContentReference],
        voiceover_refs: List[ContentReference],
        code_refs: List[ContentReference] = None,
        diagram_refs: List[ContentReference] = None,
    ) -> SlideTraceability:
        """Build traceability for a single slide."""

        # Find primary source (most referenced)
        all_refs = content_refs + voiceover_refs + (code_refs or []) + (diagram_refs or [])
        source_counts = {}
        for ref in all_refs:
            source_counts[ref.source_id] = source_counts.get(ref.source_id, 0) + 1

        primary_source = max(source_counts, key=source_counts.get) if source_counts else None

        # Calculate coverage
        total_refs = len(all_refs)
        source_coverage = min(total_refs / 3, 1.0) if total_refs > 0 else 0.0

        return SlideTraceability(
            slide_index=slide_index,
            slide_type=slide_data.get("type", "content"),
            slide_title=slide_data.get("title"),
            title_references=[],
            content_references=content_refs,
            voiceover_references=voiceover_refs,
            code_references=code_refs or [],
            diagram_references=diagram_refs or [],
            primary_source_id=primary_source,
            source_coverage=source_coverage,
        )

    def build_lecture_traceability(
        self,
        lecture_id: str,
        lecture_title: str,
        slides: List[SlideTraceability],
        section_id: str = None,
        section_title: str = None,
    ) -> LectureTraceability:
        """Build traceability for a complete lecture."""

        # Collect all sources
        all_sources = set()
        primary_sources = {}

        for slide in slides:
            all_refs = (
                slide.content_references +
                slide.voiceover_references +
                slide.code_references +
                slide.diagram_references
            )
            for ref in all_refs:
                all_sources.add(ref.source_id)
                primary_sources[ref.source_id] = primary_sources.get(ref.source_id, 0) + 1

        # Get top 3 most used sources as primary
        sorted_sources = sorted(primary_sources.items(), key=lambda x: x[1], reverse=True)
        top_primary = [s[0] for s in sorted_sources[:3]]

        # Calculate overall coverage
        coverages = [s.source_coverage for s in slides if s.source_coverage > 0]
        overall_coverage = sum(coverages) / len(coverages) if coverages else 0.0

        return LectureTraceability(
            lecture_id=lecture_id,
            lecture_title=lecture_title,
            section_id=section_id,
            section_title=section_title,
            slides=slides,
            concepts_covered=[],  # Will be filled by knowledge graph
            sources_used=list(all_sources),
            primary_sources=top_primary,
            overall_source_coverage=overall_coverage,
        )

    def build_course_traceability(
        self,
        course_id: str,
        course_title: str,
        lectures: List[LectureTraceability],
        citation_config: SourceCitationConfig,
    ) -> CourseTraceability:
        """Build complete course traceability."""

        # Collect all sources and stats
        all_sources = set()
        source_stats = {}

        for lecture in lectures:
            for source_id in lecture.sources_used:
                all_sources.add(source_id)

                if source_id not in source_stats:
                    source_stats[source_id] = {
                        "usage_count": 0,
                        "lectures": [],
                        "slides": 0,
                    }

                source_stats[source_id]["usage_count"] += 1
                source_stats[source_id]["lectures"].append(lecture.lecture_id)
                source_stats[source_id]["slides"] += len([
                    s for s in lecture.slides
                    if any(r.source_id == source_id for r in
                           s.content_references + s.voiceover_references)
                ])

        # Calculate overall coverage
        coverages = [l.overall_source_coverage for l in lectures]
        overall_coverage = sum(coverages) / len(coverages) if coverages else 0.0

        # Count total references
        total_refs = sum(
            len(s.content_references) + len(s.voiceover_references) +
            len(s.code_references) + len(s.diagram_references)
            for l in lectures for s in l.slides
        )

        return CourseTraceability(
            course_id=course_id,
            course_title=course_title,
            citation_config=citation_config,
            lectures=lectures,
            concepts=[],  # Will be filled by knowledge graph
            all_sources_used=list(all_sources),
            source_usage_stats=source_stats,
            overall_source_coverage=overall_coverage,
            total_references=total_refs,
        )

    def generate_vocal_citation(
        self,
        source: Source,
        config: SourceCitationConfig,
        context: str = "",
    ) -> str:
        """
        Generate a vocal citation for use in voiceover.

        Args:
            source: The source to cite
            config: Citation configuration
            context: Context of what's being cited

        Returns:
            Citation text to include in voiceover
        """
        if not config.enable_vocal_citations:
            return ""

        source_name = source.name

        # Shorten very long names
        if len(source_name) > 50:
            source_name = source_name[:47] + "..."

        if config.citation_style == CitationStyle.NATURAL:
            templates = [
                f"Comme expliqué dans {source_name}",
                f"Selon {source_name}",
                f"D'après {source_name}",
                f"Tel que mentionné dans {source_name}",
            ]
            # Rotate through templates based on context hash
            idx = hash(context) % len(templates)
            return templates[idx]

        elif config.citation_style == CitationStyle.ACADEMIC:
            # Try to extract author/year if available
            metadata = source.extracted_metadata or {}
            author = metadata.get("author", "")
            year = metadata.get("year", "")

            if author and year:
                return f"Selon {author} ({year})"
            elif author:
                return f"Selon {author}"
            else:
                return f"Selon {source_name}"

        elif config.citation_style == CitationStyle.MINIMAL:
            if source.source_type.value == "youtube":
                return "Comme démontré dans la vidéo"
            elif source.source_type.value == "file":
                return "Selon la documentation"
            elif source.source_type.value == "note":
                return "Comme vous l'avez noté"
            else:
                return "Selon les sources"

        return ""

    def enrich_voiceover_with_citations(
        self,
        voiceover_text: str,
        references: List[ContentReference],
        sources: List[Source],
        config: SourceCitationConfig,
    ) -> str:
        """
        Enrich voiceover text with vocal citations if enabled.

        Args:
            voiceover_text: Original voiceover text
            references: Content references for this text
            sources: Available sources
            config: Citation configuration

        Returns:
            Voiceover text with citations inserted
        """
        if not config.enable_vocal_citations or not references:
            return voiceover_text

        source_map = {s.id: s for s in sources}

        # Find the most relevant source
        if references:
            best_ref = max(references, key=lambda r: r.confidence)
            source = source_map.get(best_ref.source_id)

            if source:
                citation = self.generate_vocal_citation(
                    source, config, voiceover_text[:50]
                )

                if citation:
                    # Insert citation at beginning of first sentence
                    sentences = voiceover_text.split('. ', 1)
                    if len(sentences) > 1:
                        # Insert after first sentence
                        return f"{sentences[0]}. {citation}, {sentences[1]}"
                    else:
                        # Prepend to single sentence
                        return f"{citation}, {voiceover_text[0].lower()}{voiceover_text[1:]}"

        return voiceover_text


# Singleton instance
_traceability_service = None


def get_traceability_service() -> TraceabilityService:
    """Get singleton traceability service instance."""
    global _traceability_service
    if _traceability_service is None:
        _traceability_service = TraceabilityService()
    return _traceability_service
