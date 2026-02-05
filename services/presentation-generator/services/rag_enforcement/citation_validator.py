"""
Citation Validator

Validates inline citations in generated content against source documents.
Ensures every fact has a traceable source.
"""

import re
from typing import List, Tuple, Optional, Dict
import numpy as np

# Support both package and standalone imports
try:
    from .models import Citation, CitationReport, EnforcementConfig
except ImportError:
    from models import Citation, CitationReport, EnforcementConfig


class CitationValidator:
    """
    Validates inline citations in generated content.

    Expected citation format: [REF:1], [REF:2], etc.
    Each citation must map to a valid source chunk.
    """

    # Regex patterns for citations
    CITATION_PATTERN = re.compile(r'\[REF:(\d+)\]')
    SENTENCE_PATTERN = re.compile(r'[^.!?]*[.!?]')

    def __init__(
        self,
        config: Optional[EnforcementConfig] = None,
        embedding_func=None
    ):
        self.config = config or EnforcementConfig()
        self._embed = embedding_func  # Optional: for similarity matching

    def extract_citations(self, content: str) -> List[Citation]:
        """Extract all citations from content"""
        citations = []

        for match in self.CITATION_PATTERN.finditer(content):
            ref_id = match.group(1)

            # Get surrounding context (sentence containing the citation)
            start = max(0, match.start() - 200)
            end = min(len(content), match.end() + 50)
            context = content[start:end]

            # Find the sentence containing this citation
            sentence = self._extract_sentence_with_citation(content, match.start())

            citations.append(Citation(
                ref_id=ref_id,
                text=sentence,
                is_valid=False,  # Will be validated later
                similarity=0.0
            ))

        return citations

    def _extract_sentence_with_citation(self, content: str, citation_pos: int) -> str:
        """Extract the sentence containing a citation"""
        # Find sentence boundaries
        start = citation_pos
        while start > 0 and content[start-1] not in '.!?\n':
            start -= 1

        end = citation_pos
        while end < len(content) and content[end] not in '.!?\n':
            end += 1

        return content[start:end+1].strip()

    def validate_citations(
        self,
        content: str,
        sources: List[str],
        source_map: Optional[Dict[str, str]] = None
    ) -> CitationReport:
        """
        Validate all citations in content against sources.

        Args:
            content: Generated content with [REF:X] citations
            sources: List of source chunks (indexed 1-N)
            source_map: Optional mapping of ref_id -> source content

        Returns:
            CitationReport with validation results
        """
        citations = self.extract_citations(content)
        sentences = self._split_sentences(content)

        report = CitationReport(
            total_citations=len(citations),
            total_sentences=len(sentences)
        )

        # Validate each citation
        for citation in citations:
            ref_idx = int(citation.ref_id) - 1  # Convert to 0-indexed

            if source_map and citation.ref_id in source_map:
                citation.source_chunk = source_map[citation.ref_id]
                citation.is_valid = True
                citation.similarity = 1.0
            elif 0 <= ref_idx < len(sources):
                citation.source_chunk = sources[ref_idx]
                citation.is_valid = True

                # Calculate similarity if embedding function available
                if self._embed:
                    citation.similarity = self._calculate_similarity(
                        citation.text, citation.source_chunk
                    )
                else:
                    citation.similarity = self._keyword_similarity(
                        citation.text, citation.source_chunk
                    )
            else:
                citation.is_valid = False
                citation.similarity = 0.0

            report.citations.append(citation)

        # Count valid/invalid
        report.valid_citations = sum(1 for c in citations if c.is_valid)
        report.invalid_citations = sum(1 for c in citations if not c.is_valid)

        # Find uncited sentences
        cited_sentence_indices = set()
        for citation in citations:
            # Clean citation text for comparison (remove [REF:X] markers)
            clean_citation = self.CITATION_PATTERN.sub('', citation.text).strip()
            for i, sentence in enumerate(sentences):
                # Check if the clean citation text matches the sentence
                if clean_citation in sentence or sentence in clean_citation:
                    cited_sentence_indices.add(i)

        for i, sentence in enumerate(sentences):
            words = sentence.split()
            # Only count substantial sentences
            if len(words) >= self.config.min_words_for_citation:
                if i not in cited_sentence_indices:
                    report.uncited_sentences += 1
                    report.uncited_sentence_list.append(sentence[:100])

        return report

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Remove citations for cleaner splitting
        clean_text = self.CITATION_PATTERN.sub('', text)

        # Normalize whitespace
        clean_text = ' '.join(clean_text.split())

        # Split by sentence boundaries
        sentences = []
        for match in self.SENTENCE_PATTERN.finditer(clean_text):
            sentence = match.group().strip()
            if sentence and len(sentence) > 5:
                sentences.append(sentence)

        return sentences

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Simple keyword-based similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Remove stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'le', 'la', 'les', 'un', 'une', 'des', 'est', 'sont',
                      'and', 'or', 'but', 'if', 'et', 'ou', 'mais', 'si'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        if not self._embed:
            return self._keyword_similarity(text1, text2)

        try:
            emb1 = self._embed(text1)
            emb2 = self._embed(text2)

            # Cosine similarity
            dot = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot / (norm1 * norm2))
        except Exception:
            return self._keyword_similarity(text1, text2)

    def generate_citation_prompt(self, sources: List[str]) -> str:
        """Generate prompt section for citation requirements"""
        source_refs = []
        for i, source in enumerate(sources, 1):
            # Truncate long sources
            truncated = source[:500] + "..." if len(source) > 500 else source
            source_refs.append(f"[REF:{i}] {truncated}")

        return f"""
## CITATION REQUIREMENTS (MANDATORY)

Every factual statement MUST include a citation in the format [REF:X].

AVAILABLE SOURCES:
{chr(10).join(source_refs)}

RULES:
1. Each fact must have at least one [REF:X] citation
2. Only use the REF numbers listed above
3. Place [REF:X] at the end of the sentence containing the fact
4. If you cannot cite a source, DO NOT include that fact

EXAMPLE:
"Apache Kafka uses partitions for scalability [REF:1]. Each partition
can handle thousands of messages per second [REF:2]."

FORBIDDEN:
- Making up facts without citations
- Using [REF:X] numbers that don't exist
- Writing substantial claims without any citation
"""

    def check_citation_density(
        self,
        content: str,
        min_citations_per_paragraph: int = 2
    ) -> Tuple[bool, List[str]]:
        """
        Check if citation density is sufficient.

        Returns:
            Tuple of (is_sufficient, list_of_undercited_paragraphs)
        """
        paragraphs = content.split('\n\n')
        undercited = []

        for para in paragraphs:
            if len(para.split()) < 20:  # Skip short paragraphs
                continue

            citations = self.CITATION_PATTERN.findall(para)
            if len(citations) < min_citations_per_paragraph:
                undercited.append(para[:100] + "...")

        return len(undercited) == 0, undercited
