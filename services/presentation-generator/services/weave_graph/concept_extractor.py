"""
Concept Extractor for WeaveGraph

Extracts concepts from text using NLP techniques:
- Technical term patterns (regex)
- Keyword extraction (TF-IDF)
- Named entity recognition (optional spaCy)
- Domain-specific patterns
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
import math

from .models import ConceptNode, ConceptSource


@dataclass
class ExtractionConfig:
    """Configuration for concept extraction"""
    min_term_length: int = 3
    max_term_length: int = 50
    min_frequency: int = 1
    max_concepts: int = 500
    include_bigrams: bool = True
    include_trigrams: bool = True
    extract_code_terms: bool = True
    extract_acronyms: bool = True
    language_detection: bool = True


class ConceptExtractor:
    """
    Extracts concepts from documents using NLP techniques.

    Uses a combination of:
    - Regex patterns for technical terms
    - TF-IDF for keyword importance
    - Pattern matching for code/API terms
    """

    # Technical term patterns
    PATTERNS = {
        # CamelCase: DataFrame, TensorFlow
        "camel_case": re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b'),

        # snake_case: data_frame, tensor_flow
        "snake_case": re.compile(r'\b([a-z]+(?:_[a-z]+)+)\b'),

        # Acronyms: API, REST, SQL, AWS
        "acronym": re.compile(r'\b([A-Z]{2,6})\b'),

        # Version numbers: v1.0, Python 3.11
        "version": re.compile(r'\b([A-Za-z]+\s*\d+(?:\.\d+)*)\b'),

        # Code elements: function(), Class.method
        "code_element": re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\(\)|\.[\w]+))\b'),

        # Technical compounds: machine learning, deep learning
        "compound_term": re.compile(r'\b([A-Z]?[a-z]+(?:\s+[a-z]+){1,3})\b'),

        # Hyphenated: real-time, open-source
        "hyphenated": re.compile(r'\b([a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)?)\b'),

        # Package/module: kafka.consumer, aws.s3
        "package": re.compile(r'\b([a-z]+(?:\.[a-z]+)+)\b'),
    }

    # Domain-specific keywords to always include
    TECH_DOMAINS = {
        "data": ["pipeline", "etl", "elt", "warehouse", "lake", "streaming", "batch", "ingestion"],
        "cloud": ["aws", "azure", "gcp", "kubernetes", "docker", "serverless", "lambda", "ec2"],
        "ml": ["model", "training", "inference", "feature", "embedding", "neural", "transformer"],
        "web": ["api", "rest", "graphql", "endpoint", "microservice", "gateway", "authentication"],
        "database": ["sql", "nosql", "mongodb", "postgresql", "redis", "cassandra", "index"],
        "messaging": ["kafka", "rabbitmq", "pubsub", "queue", "consumer", "producer", "topic"],
    }

    # Stop words for filtering
    STOP_WORDS = {
        # French
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'en', 'est',
        'que', 'qui', 'dans', 'ce', 'il', 'ne', 'sur', 'se', 'pas', 'plus',
        'par', 'pour', 'au', 'avec', 'son', 'sa', 'ses', 'ou', 'comme', 'mais',
        'nous', 'vous', 'leur', 'cette', 'ces', 'tout', 'elle', 'sont',
        # English
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'and', 'or', 'but', 'if',
        'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'on',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'we', 'you',
        'also', 'each', 'which', 'their', 'there', 'when', 'where', 'how',
        'very', 'just', 'only', 'more', 'most', 'other', 'some', 'such',
    }

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._idf_cache: Dict[str, float] = {}

    def extract_concepts(
        self,
        text: str,
        document_id: Optional[str] = None,
        existing_concepts: Optional[Dict[str, ConceptNode]] = None
    ) -> List[ConceptNode]:
        """
        Extract concepts from text.

        Args:
            text: The text to extract concepts from
            document_id: Optional document ID for tracking
            existing_concepts: Existing concepts to merge with

        Returns:
            List of extracted ConceptNode objects
        """
        concepts = {}
        existing = existing_concepts or {}

        # Detect language
        language = self._detect_language(text) if self.config.language_detection else "en"

        # Extract using different methods
        pattern_terms = self._extract_pattern_terms(text)
        keyword_terms = self._extract_keywords(text)
        domain_terms = self._extract_domain_terms(text)

        # Merge all terms
        all_terms = {}

        for term, source in pattern_terms:
            canonical = self._canonicalize(term)
            if self._is_valid_term(term, canonical):
                if canonical not in all_terms:
                    all_terms[canonical] = {"name": term, "source": source, "freq": 1}
                else:
                    all_terms[canonical]["freq"] += 1

        for term, score in keyword_terms:
            canonical = self._canonicalize(term)
            if self._is_valid_term(term, canonical):
                if canonical not in all_terms:
                    all_terms[canonical] = {"name": term, "source": ConceptSource.KEYWORD, "freq": 1, "score": score}
                else:
                    all_terms[canonical]["freq"] += 1
                    all_terms[canonical]["score"] = max(all_terms[canonical].get("score", 0), score)

        for term in domain_terms:
            canonical = self._canonicalize(term)
            if canonical not in all_terms:
                all_terms[canonical] = {"name": term, "source": ConceptSource.TECHNICAL_TERM, "freq": 1}
            else:
                all_terms[canonical]["freq"] += 1

        # Create ConceptNode objects
        for canonical, data in all_terms.items():
            # Check if exists in existing concepts
            if canonical in existing:
                node = existing[canonical]
                node.frequency += data["freq"]
                if document_id and document_id not in node.source_document_ids:
                    node.source_document_ids.append(document_id)
            else:
                node = ConceptNode(
                    name=data["name"],
                    canonical_name=canonical,
                    language=language,
                    frequency=data["freq"],
                    source_type=data["source"],
                    source_document_ids=[document_id] if document_id else []
                )
                concepts[canonical] = node

        # Sort by frequency and limit
        sorted_concepts = sorted(
            concepts.values(),
            key=lambda c: c.frequency,
            reverse=True
        )[:self.config.max_concepts]

        return sorted_concepts

    def _extract_pattern_terms(self, text: str) -> List[Tuple[str, ConceptSource]]:
        """Extract terms using regex patterns"""
        terms = []

        for pattern_name, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            source = ConceptSource.TECHNICAL_TERM if pattern_name in ["camel_case", "snake_case", "code_element", "package"] else ConceptSource.NLP_EXTRACTION

            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                terms.append((match, source))

        return terms

    def _extract_keywords(self, text: str, top_n: int = 100) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF scoring"""
        # Tokenize
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text.lower())
        words = [w for w in words if w not in self.STOP_WORDS and len(w) >= self.config.min_term_length]

        # Count term frequency
        tf = Counter(words)
        total_words = len(words)

        if total_words == 0:
            return []

        # Calculate TF-IDF-like score (simplified)
        # Higher score for less common words
        scored_terms = []
        for term, count in tf.items():
            tf_score = count / total_words
            # Boost technical-looking terms
            boost = 1.0
            if any(c.isupper() for c in term):
                boost = 1.5
            if '_' in term or '-' in term:
                boost = 1.5
            if len(term) > 8:
                boost *= 1.2

            score = tf_score * boost
            scored_terms.append((term, score))

        # Also extract bigrams if enabled
        if self.config.include_bigrams:
            bigrams = self._extract_ngrams(words, 2)
            for bigram, count in bigrams.items():
                score = (count / total_words) * 1.5  # Boost bigrams
                scored_terms.append((bigram, score))

        # Sort by score
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        return scored_terms[:top_n]

    def _extract_ngrams(self, words: List[str], n: int) -> Counter:
        """Extract n-grams from word list"""
        ngrams = Counter()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            # Filter out ngrams with stop words
            if not any(w in self.STOP_WORDS for w in words[i:i+n]):
                ngrams[ngram] += 1
        return ngrams

    def _extract_domain_terms(self, text: str) -> List[str]:
        """Extract known domain-specific terms"""
        text_lower = text.lower()
        found_terms = []

        for domain, terms in self.TECH_DOMAINS.items():
            for term in terms:
                if term in text_lower:
                    found_terms.append(term)

        return found_terms

    def _canonicalize(self, term: str) -> str:
        """Convert term to canonical form"""
        # Lowercase and replace separators with underscore
        canonical = term.lower().strip()
        canonical = re.sub(r'[\s\-\.]+', '_', canonical)
        canonical = re.sub(r'[^a-z0-9_]', '', canonical)
        return canonical

    def _is_valid_term(self, term: str, canonical: str) -> bool:
        """Check if term is valid for extraction"""
        if len(canonical) < self.config.min_term_length:
            return False
        if len(canonical) > self.config.max_term_length:
            return False
        if canonical in self.STOP_WORDS:
            return False
        if canonical.isdigit():
            return False
        return True

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words"""
        text_lower = text.lower()

        french_indicators = ['le', 'la', 'les', 'de', 'du', 'des', 'est', 'sont', 'avec', 'pour', 'dans', 'cette', 'ces']
        english_indicators = ['the', 'is', 'are', 'with', 'for', 'this', 'that', 'from', 'have', 'has']

        french_count = sum(1 for word in french_indicators if f' {word} ' in f' {text_lower} ')
        english_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_lower} ')

        return 'fr' if french_count > english_count else 'en'

    def extract_context_snippets(self, text: str, term: str, window: int = 50) -> List[str]:
        """Extract context snippets around a term"""
        snippets = []
        term_lower = term.lower()
        text_lower = text.lower()

        pos = 0
        while True:
            idx = text_lower.find(term_lower, pos)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(term) + window)
            snippet = text[start:end].strip()

            if start > 0:
                snippet = '...' + snippet
            if end < len(text):
                snippet = snippet + '...'

            snippets.append(snippet)
            pos = idx + 1

            if len(snippets) >= 3:  # Limit snippets
                break

        return snippets
