"""
Keyword Extraction for RAG

Extracts significant keywords from text for document relevance scoring.
Supports both English and French stopwords.
"""

import re
from typing import List, Set


class KeywordExtractor:
    """
    Extract significant keywords from text.

    Used for:
    - Computing keyword_coverage score in document relevance
    - Fallback similarity when embeddings are unavailable
    - Topic detection and matching

    Supports English and French stopwords.
    """

    # Combined English and French stopwords
    STOPWORDS: Set[str] = {
        # English stopwords
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
        'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'also', 'now', 'about', 'any',
        'both', 'but', 'down', 'even', 'if', 'it', 'its', 'like',
        'out', 'up', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'and', 'or', 'because', 'until',
        'while', 'yet', 'you', 'your', 'we', 'our', 'they', 'their',
        'he', 'she', 'him', 'her', 'his', 'hers', 'me', 'my',
        # French stopwords
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et',
        'en', 'est', 'que', 'qui', 'dans', 'ce', 'il', 'ne', 'sur',
        'se', 'pas', 'plus', 'par', 'pour', 'au', 'avec', 'son',
        'sa', 'ses', 'ou', 'mais', 'comme', 'on', 'tout', 'nous',
        'vous', 'elle', 'leur', 'leurs', 'bien', 'fait', 'cette',
        'ces', 'aux', 'été', 'être', 'avoir', 'faire', 'dit',
        'aussi', 'peut', 'même', 'donc', 'car', 'dont', 'si',
        'entre', 'vers', 'chez', 'sans', 'sous', 'autre', 'autres',
        'tous', 'toutes', 'peu', 'très', 'trop', 'ici', 'là',
        'quand', 'où', 'comment', 'pourquoi', 'parce', 'alors',
        'ainsi', 'encore', 'toujours', 'jamais', 'rien', 'quelque',
        'chaque', 'quel', 'quelle', 'quels', 'quelles', 'celui',
        'celle', 'ceux', 'celles', 'ci', 'là', 'fois', 'moins',
    }

    # Minimum keyword length
    MIN_KEYWORD_LENGTH = 3

    # Pattern for extracting words (supports accented characters)
    WORD_PATTERN = re.compile(r'\b[a-zA-Z\u00C0-\u017F]{3,}\b')

    def __init__(self, min_length: int = 3, custom_stopwords: Set[str] = None):
        """
        Initialize the keyword extractor.

        Args:
            min_length: Minimum keyword length (default: 3)
            custom_stopwords: Additional stopwords to include
        """
        self.min_length = min_length
        self.stopwords = self.STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

    def extract(self, text: str, max_keywords: int = None) -> List[str]:
        """
        Extract significant keywords from text.

        Args:
            text: Input text to extract keywords from
            max_keywords: Maximum number of keywords to return (None = all)

        Returns:
            List of unique keywords, preserving order of first occurrence
        """
        if not text:
            return []

        # Extract all words matching pattern
        words = self.WORD_PATTERN.findall(text.lower())

        # Filter stopwords and short words
        keywords = [
            w for w in words
            if w not in self.stopwords and len(w) >= self.min_length
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)

        if max_keywords:
            return unique_keywords[:max_keywords]

        return unique_keywords

    def compute_coverage(
        self,
        query: str,
        document: str,
        max_doc_chars: int = 10000,
    ) -> float:
        """
        Compute keyword coverage of query in document.

        Args:
            query: Query text
            document: Document text
            max_doc_chars: Maximum document characters to analyze

        Returns:
            Coverage score between 0.0 and 1.0
        """
        query_keywords = set(self.extract(query))

        if not query_keywords:
            return 0.0

        # Limit document analysis for performance
        doc_text = document[:max_doc_chars] if len(document) > max_doc_chars else document
        doc_keywords = set(self.extract(doc_text))

        # Calculate overlap
        overlap = query_keywords.intersection(doc_keywords)
        return len(overlap) / len(query_keywords)

    def compute_similarity(
        self,
        text1: str,
        text2: str,
        max_chars: int = 10000,
    ) -> float:
        """
        Compute keyword-based similarity between two texts.

        Used as fallback when embeddings are unavailable.

        Args:
            text1: First text
            text2: Second text
            max_chars: Maximum characters to analyze per text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        kw1 = set(self.extract(text1[:max_chars]))
        kw2 = set(self.extract(text2[:max_chars]))

        if not kw1:
            return 0.5  # Default when no keywords

        overlap = kw1.intersection(kw2)
        return len(overlap) / len(kw1)


# Module-level instance for convenience
_default_extractor = None


def get_keyword_extractor() -> KeywordExtractor:
    """Get the default keyword extractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = KeywordExtractor()
    return _default_extractor


def extract_keywords(text: str, max_keywords: int = None) -> List[str]:
    """
    Convenience function to extract keywords using default extractor.

    Args:
        text: Input text
        max_keywords: Maximum keywords to return

    Returns:
        List of keywords
    """
    return get_keyword_extractor().extract(text, max_keywords)
