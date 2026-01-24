"""
RAG Verification Service

Verifies that generated content actually uses the RAG source documents.
Uses semantic matching to handle paraphrasing and reformulation.
"""
import re
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

    # Key concepts from source that were used
    source_concepts_used: List[str] = field(default_factory=list)

    # Key concepts from source that were missed
    source_concepts_missed: List[str] = field(default_factory=list)

    # Compliance status
    is_compliant: bool = False

    # Summary message
    summary: str = ""


class RAGVerifier:
    """
    Verifies RAG content usage in generated presentations.

    Uses semantic concept matching rather than strict text matching:
    1. Extract key concepts/topics from source
    2. Check if concepts are addressed in generated content
    3. Allow for paraphrasing and reformulation
    """

    # Technical term patterns for different domains
    TECHNICAL_PATTERNS = [
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
        r'\b[A-Z]{2,}\b',  # Acronyms (API, REST, etc.)
        r'\b\w+(?:_\w+)+\b',  # snake_case
        r'\b(?:micro)?service[s]?\b',
        r'\b(?:data)?base[s]?\b',
        r'\b(?:end)?point[s]?\b',
        r'\bpattern[s]?\b',
        r'\barchitecture\b',
        r'\bintegration\b',
        r'\bmessag(?:e|ing)\b',
        r'\bqueue[s]?\b',
        r'\bpipeline[s]?\b',
        r'\bframework[s]?\b',
        r'\blibrar(?:y|ies)\b',
        r'\bprotocol[s]?\b',
    ]

    def __init__(self, min_coverage_threshold: float = 0.40):
        """
        Initialize the verifier.

        Args:
            min_coverage_threshold: Minimum coverage to be considered compliant (default 40%)
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

        # Extract key concepts from source (not just words)
        source_concepts = self._extract_concepts(source_documents)
        source_terms = self._extract_technical_terms(source_documents)

        # Combine concepts and technical terms
        all_source_items = source_concepts | source_terms

        if verbose:
            print(f"[RAG_VERIFIER] Source: {len(source_documents)} chars, "
                  f"{len(source_concepts)} concepts, {len(source_terms)} tech terms", flush=True)

        total_coverage = 0.0
        slide_results = []
        all_used_concepts = set()
        potential_hallucinations = []

        for i, slide in enumerate(slides):
            slide_text = self._extract_slide_text(slide)

            if not slide_text.strip():
                slide_results.append({
                    "slide_index": i,
                    "slide_type": slide.get("type", "unknown"),
                    "coverage": 1.0,
                    "reason": "Empty slide"
                })
                continue

            # Calculate coverage using semantic matching
            concept_coverage, used_concepts = self._calculate_concept_coverage(
                slide_text, all_source_items
            )

            # Calculate topic relevance (does the slide address source topics?)
            topic_relevance = self._calculate_topic_relevance(
                slide_text, source_documents
            )

            # Weighted: concept matching is primary, topic relevance as bonus
            slide_coverage = (concept_coverage * 0.7) + (topic_relevance * 0.3)

            all_used_concepts.update(used_concepts)

            slide_result = {
                "slide_index": i,
                "slide_type": slide.get("type", "unknown"),
                "title": slide.get("title", "Untitled"),
                "coverage": round(slide_coverage, 3),
                "concept_coverage": round(concept_coverage, 3),
                "topic_relevance": round(topic_relevance, 3),
                "concepts_used": len(used_concepts),
            }
            slide_results.append(slide_result)

            # Only flag as hallucination if very low coverage AND substantial content
            if slide_coverage < 0.15 and len(slide_text) > 200:
                potential_hallucinations.append({
                    "slide_index": i,
                    "slide_type": slide.get("type", "unknown"),
                    "title": slide.get("title", ""),
                    "coverage": round(slide_coverage, 3),
                    "content_preview": slide_text[:150] + "..." if len(slide_text) > 150 else slide_text
                })

            total_coverage += slide_coverage

            if verbose:
                print(f"[RAG_VERIFIER] Slide {i} ({slide.get('type', 'unknown')}): "
                      f"{slide_coverage:.1%} coverage ({len(used_concepts)} concepts)", flush=True)

        # Calculate overall coverage
        if slides:
            result.overall_coverage = round(total_coverage / len(slides), 3)
        else:
            result.overall_coverage = 0.0

        result.slide_coverage = slide_results
        result.potential_hallucinations = potential_hallucinations
        result.source_concepts_used = list(all_used_concepts)[:50]
        result.source_concepts_missed = [c for c in all_source_items if c not in all_used_concepts][:30]
        result.is_compliant = result.overall_coverage >= self.min_coverage_threshold

        # Generate summary
        if result.is_compliant:
            result.summary = f"✅ RAG COMPLIANT: {result.overall_coverage:.1%} coverage ({len(all_used_concepts)} concepts used)"
        else:
            result.summary = f"⚠️ RAG LOW COVERAGE: {result.overall_coverage:.1%} (threshold: {self.min_coverage_threshold:.0%})"
            if potential_hallucinations:
                result.summary += f" - {len(potential_hallucinations)} slides may need review"

        if verbose:
            print(f"[RAG_VERIFIER] {result.summary}", flush=True)

        return result

    def _extract_concepts(self, text: str) -> Set[str]:
        """
        Extract key concepts from text.
        Focuses on meaningful multi-word phrases and important terms.
        """
        concepts = set()
        text_lower = text.lower()

        # Extract capitalized phrases (often important concepts)
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        concepts.update(p.lower() for p in cap_phrases)

        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        concepts.update(q.lower() for q in quoted if len(q) > 3)

        # Extract terms after "called", "known as", "named"
        named = re.findall(r'(?:called|known as|named)\s+["\']?(\w+(?:\s+\w+)?)["\']?', text_lower)
        concepts.update(named)

        # Extract compound technical terms (word-word or word_word)
        compounds = re.findall(r'\b\w+[-_]\w+(?:[-_]\w+)*\b', text_lower)
        concepts.update(compounds)

        # Filter out very short or common concepts
        concepts = {c for c in concepts if len(c) > 3 and not self._is_common_word(c)}

        return concepts

    def _extract_technical_terms(self, text: str) -> Set[str]:
        """Extract technical terms using patterns."""
        terms = set()

        for pattern in self.TECHNICAL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.update(m.lower() for m in matches if len(m) > 2)

        # Also extract single important technical words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = Counter(words)

        # Terms that appear multiple times are likely important
        frequent_terms = {word for word, count in word_freq.items()
                        if count >= 2 and not self._is_common_word(word)}
        terms.update(frequent_terms)

        return terms

    def _is_common_word(self, word: str) -> bool:
        """Check if word is a common stopword."""
        stopwords = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has',
            'will', 'can', 'are', 'was', 'were', 'been', 'being', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'need', 'also', 'just',
            'like', 'make', 'made', 'use', 'used', 'using', 'then', 'than',
            'more', 'most', 'some', 'such', 'only', 'other', 'into', 'over',
            'after', 'before', 'between', 'under', 'during', 'through', 'about',
            'which', 'where', 'when', 'what', 'while', 'there', 'here', 'they',
            'their', 'them', 'these', 'those', 'your', 'yours', 'ours', 'been',
            # French
            'vous', 'nous', 'elle', 'elles', 'sont', 'avec', 'pour', 'dans',
            'cette', 'cela', 'mais', 'donc', 'ainsi', 'comme', 'tout', 'tous',
            'plus', 'moins', 'bien', 'tres', 'fait', 'faire', 'peut', 'avoir',
            'etre', 'etait', 'sont', 'seront', 'quand', 'comment', 'pourquoi',
        }
        return word.lower() in stopwords

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

        # Include code block comments (can indicate topic)
        for block in slide.get("code_blocks", []):
            if block.get("code"):
                # Extract comments from code
                comments = re.findall(r'#.*|//.*|/\*.*?\*/', block["code"])
                parts.extend(comments)

        return " ".join(parts)

    def _calculate_concept_coverage(
        self,
        generated: str,
        source_concepts: Set[str]
    ) -> Tuple[float, Set[str]]:
        """
        Calculate what percentage of source concepts appear in generated content.
        Uses fuzzy matching to handle variations.
        """
        if not source_concepts:
            return 1.0, set()

        generated_lower = generated.lower()
        used_concepts = set()

        for concept in source_concepts:
            # Direct match
            if concept in generated_lower:
                used_concepts.add(concept)
                continue

            # Check for partial match (concept words appear)
            concept_words = concept.split()
            if len(concept_words) > 1:
                # Multi-word: check if most words appear
                matches = sum(1 for w in concept_words if w in generated_lower)
                if matches >= len(concept_words) * 0.7:
                    used_concepts.add(concept)
                    continue

            # Check for variations (singular/plural, verb forms)
            variations = self._get_variations(concept)
            if any(v in generated_lower for v in variations):
                used_concepts.add(concept)

        # Coverage based on how many source concepts are used
        coverage = len(used_concepts) / len(source_concepts) if source_concepts else 0
        return coverage, used_concepts

    def _get_variations(self, term: str) -> List[str]:
        """Get common variations of a term."""
        variations = [term]

        # Singular/plural
        if term.endswith('s'):
            variations.append(term[:-1])
        else:
            variations.append(term + 's')

        # Common suffixes
        if term.endswith('ing'):
            variations.append(term[:-3])
            variations.append(term[:-3] + 'e')
        elif term.endswith('tion'):
            variations.append(term[:-4] + 'te')
        elif term.endswith('er'):
            variations.append(term[:-2])
            variations.append(term[:-2] + 'e')

        return variations

    def _calculate_topic_relevance(self, generated: str, source: str) -> float:
        """
        Calculate if the generated content is discussing the same topic as source.
        Uses keyword overlap as a proxy for topic relevance.
        """
        # Extract significant words from both
        gen_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', generated.lower()))
        src_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', source.lower()))

        # Remove common words
        gen_words = {w for w in gen_words if not self._is_common_word(w)}
        src_words = {w for w in src_words if not self._is_common_word(w)}

        if not gen_words:
            return 1.0  # Empty generated text

        # How many generated words are related to source topic?
        overlap = gen_words.intersection(src_words)
        relevance = len(overlap) / len(gen_words) if gen_words else 0

        # Boost score if there's meaningful overlap
        if len(overlap) >= 5:
            relevance = min(1.0, relevance * 1.5)

        return relevance


# Singleton instance
_rag_verifier = None


def get_rag_verifier() -> RAGVerifier:
    """Get singleton RAG verifier instance."""
    global _rag_verifier
    if _rag_verifier is None:
        _rag_verifier = RAGVerifier(min_coverage_threshold=0.40)
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
