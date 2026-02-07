"""
Concept Extractor for WeaveGraph

Extracts concepts from text using NLP techniques:
- Technical term patterns (regex)
- Keyword extraction (TF-IDF)
- Named entity recognition (optional spaCy)
- Domain-specific patterns
- ML-based compound term detection (PMI + semantic filtering)
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field
import math

from .models import ConceptNode, ConceptSource
from .compound_detector import (
    CompoundTermDetector,
    CompoundDetectorConfig,
    PMIConfig,
    CompoundTermResult
)

logger = logging.getLogger(__name__)


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
    # ML-based compound detection
    use_ml_compound_detection: bool = True  # Use PMI-based detection
    ml_min_pmi: float = 1.5  # Minimum PMI score for compound terms
    ml_min_frequency: int = 2  # Minimum frequency for compound terms
    ml_min_combined_score: float = 0.3  # Minimum combined score
    use_semantic_filter: bool = False  # Use embedding-based filtering (slower)


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

        # Technical compounds (lowercase): machine learning, deep learning
        "compound_term": re.compile(r'\b([A-Z]?[a-z]+(?:\s+[a-z]+){1,3})\b'),

        # Title Case Compounds: Machine Learning, Apache Kafka, Message Broker
        "title_case_compound": re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'),

        # Mixed Case Compounds: Apache Kafka, Google Cloud Platform
        "mixed_case_compound": re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){1,4})\b'),

        # Hyphenated: real-time, open-source
        "hyphenated": re.compile(r'\b([a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)?)\b'),

        # Package/module: kafka.consumer, aws.s3
        "package": re.compile(r'\b([a-z]+(?:\.[a-z]+)+)\b'),
    }

    # Known multi-word technical terms (for n-gram boosting)
    KNOWN_COMPOUND_TERMS = {
        # ML/AI
        "machine learning", "deep learning", "neural network", "natural language",
        "computer vision", "reinforcement learning", "transfer learning",
        # Data
        "data pipeline", "data warehouse", "data lake", "data engineering",
        "big data", "data science", "data analytics", "batch processing",
        "stream processing", "real time", "event driven",
        # Cloud/Infra
        "message broker", "message queue", "load balancer", "api gateway",
        "service mesh", "distributed system", "microservices architecture",
        "event sourcing", "command query", "domain driven",
        # Databases
        "primary key", "foreign key", "database schema", "query optimization",
        # DevOps
        "continuous integration", "continuous deployment", "infrastructure code",
        "container orchestration", "blue green", "canary deployment",
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

        # ML-based compound detector
        self._compound_detector: Optional[CompoundTermDetector] = None
        self._learned_compound_terms: Set[str] = set()
        self._is_trained: bool = False

        if self.config.use_ml_compound_detection:
            self._init_compound_detector()

    def _init_compound_detector(self) -> None:
        """Initialize the ML-based compound detector"""
        compound_config = CompoundDetectorConfig(
            pmi_config=PMIConfig(
                min_frequency=self.config.ml_min_frequency,
                min_pmi=self.config.ml_min_pmi,
                max_ngram_size=3
            ),
            min_combined_score=self.config.ml_min_combined_score,
            use_embeddings=self.config.use_semantic_filter
        )
        self._compound_detector = CompoundTermDetector(compound_config)

    def train_on_corpus(self, texts: List[str]) -> int:
        """
        Train the compound detector on a corpus of documents.

        This learns which multi-word terms are significant collocations
        in the corpus, replacing the need for a hardcoded list.

        Args:
            texts: List of document texts

        Returns:
            Number of compound terms learned
        """
        if not self.config.use_ml_compound_detection or self._compound_detector is None:
            logger.debug("ML compound detection disabled, skipping training")
            return 0

        if not texts:
            return 0

        logger.debug(f"Training compound detector on {len(texts)} documents...")
        self._compound_detector.train(texts)

        # Extract learned compound terms
        results = self._compound_detector.detect(top_k=200)
        self._learned_compound_terms = {r.term for r in results}
        self._is_trained = True

        logger.info(f"Learned {len(self._learned_compound_terms)} compound terms from corpus")
        return len(self._learned_compound_terms)

    def get_learned_compound_terms(self) -> Set[str]:
        """Get the set of learned compound terms"""
        return self._learned_compound_terms.copy()

    def add_compound_terms(self, terms: Set[str]) -> None:
        """Manually add compound terms to the learned set"""
        self._learned_compound_terms.update(t.lower() for t in terms)

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

        # Extract title case n-grams (Machine Learning, Apache Kafka, etc.)
        title_case_ngrams = self._extract_title_case_ngrams(text)

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

        # Add title case n-grams (Machine Learning, Apache Kafka, etc.)
        for term, score in title_case_ngrams:
            canonical = self._canonicalize(term)
            if self._is_valid_term(term, canonical):
                if canonical not in all_terms:
                    all_terms[canonical] = {
                        "name": term,  # Preserve original case
                        "source": ConceptSource.NLP_EXTRACTION,
                        "freq": 1,
                        "score": score
                    }
                else:
                    all_terms[canonical]["freq"] += 1
                    # Keep the higher score
                    all_terms[canonical]["score"] = max(
                        all_terms[canonical].get("score", 0), score
                    )

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

    def _extract_title_case_ngrams(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract n-grams that preserve title case.

        Captures compound terms like:
        - "Machine Learning"
        - "Apache Kafka"
        - "Data Pipeline"

        Uses ML-learned compound terms when available, with fallback to
        hardcoded KNOWN_COMPOUND_TERMS.

        Returns list of (ngram, score) tuples.
        """
        results = []

        # Determine which compound term set to use
        # Priority: learned terms > hardcoded terms
        compound_terms = self._get_effective_compound_terms()

        # Split into sentences first to avoid crossing sentence boundaries
        sentences = re.split(r'[.!?;]', text)

        for sentence in sentences:
            # Tokenize preserving case
            words = re.findall(r'\b[A-Za-z][A-Za-z0-9]*\b', sentence)

            # Extract bigrams and trigrams
            for n in [2, 3]:
                for i in range(len(words) - n + 1):
                    ngram_words = words[i:i+n]
                    ngram = ' '.join(ngram_words)
                    ngram_lower = ngram.lower()

                    # Skip if contains stop words
                    if any(w.lower() in self.STOP_WORDS for w in ngram_words):
                        continue

                    # Check if it's a known/learned compound term (high score)
                    if ngram_lower in compound_terms:
                        results.append((ngram, 2.0))  # High score for known terms
                        continue

                    # Check ML detector for real-time scoring if trained
                    if self._is_trained and self._compound_detector is not None:
                        is_compound, score = self._compound_detector.is_compound_term(ngram_lower)
                        if is_compound and score >= self.config.ml_min_combined_score:
                            # Scale score to 1.5-2.0 range
                            scaled_score = 1.5 + (score * 0.5)
                            results.append((ngram, scaled_score))
                            continue

                    # Check if it's title case (Medium score)
                    is_title_case = all(w[0].isupper() for w in ngram_words)
                    if is_title_case:
                        results.append((ngram, 1.5))
                        continue

                    # Check if first word is capitalized (might be start of sentence, lower score)
                    if ngram_words[0][0].isupper() and len(ngram_words[0]) > 2:
                        # Only include if seems technical
                        if any(w.lower() in self._get_all_tech_terms() for w in ngram_words):
                            results.append((ngram, 1.0))

        return results

    def _get_effective_compound_terms(self) -> Set[str]:
        """
        Get the effective set of compound terms to use.

        Priority:
        1. Learned terms from ML detector (if trained) or manually added
        2. Hardcoded KNOWN_COMPOUND_TERMS (fallback)
        3. Merge of both when learned terms exist
        """
        # Include learned terms whether from training or manual addition
        if self._learned_compound_terms:
            return self._learned_compound_terms | self.KNOWN_COMPOUND_TERMS
        return self.KNOWN_COMPOUND_TERMS

    def _get_all_tech_terms(self) -> Set[str]:
        """Get all known technical terms as a flat set"""
        terms = set()
        for domain_terms in self.TECH_DOMAINS.values():
            terms.update(domain_terms)
        # Add hardcoded compound terms
        for compound in self.KNOWN_COMPOUND_TERMS:
            terms.update(compound.split())
        # Add learned compound terms
        for compound in self._learned_compound_terms:
            terms.update(compound.split())
        return terms

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
        """
        Convert term to canonical form.

        Includes:
        - Lowercase normalization
        - Separator normalization (spaces, hyphens, dots → underscore)
        - Singularization to avoid duplicates (DataFrames → dataframe)
        """
        # Lowercase and replace separators with underscore
        canonical = term.lower().strip()
        canonical = re.sub(r'[\s\-\.]+', '_', canonical)
        canonical = re.sub(r'[^a-z0-9_]', '', canonical)

        # Apply singularization to avoid duplicates
        canonical = self._singularize(canonical)

        return canonical

    def _singularize(self, word: str) -> str:
        """
        Simple singularization without external dependencies.

        Handles common English plural patterns:
        - queries → query
        - indexes → index
        - dataframes → dataframe
        - services → service
        - classes → class

        Does NOT singularize:
        - Words ending in 'ss' (class, pass)
        - Words ending in 'is' (analysis, basis)
        - Words ending in 'us' (status, corpus)
        - Short words (< 4 chars)
        """
        if len(word) < 4:
            return word

        # Don't singularize words ending in 'ss', 'is', 'us' (often singular)
        if word.endswith(('ss', 'is', 'us', 'sis', 'xis')):
            return word

        # Special cases that should NOT be singularized
        exceptions = {
            'kubernetes', 'postgres', 'redis', 'aws', 'series',
            'class', 'pass', 'less', 'process', 'access', 'success',
            'analysis', 'basis', 'thesis', 'hypothesis', 'synthesis',
            'status', 'corpus', 'focus', 'radius', 'genius',
        }
        if word in exceptions:
            return word

        # Handle compound words (singularize the last part)
        if '_' in word:
            parts = word.split('_')
            parts[-1] = self._singularize(parts[-1])
            return '_'.join(parts)

        # Rule 1: -ies → -y (queries → query, categories → category)
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'

        # Rule 2: -es after s, x, z, ch, sh → remove -es (indexes → index, classes → class)
        if word.endswith('es') and len(word) > 3:
            if word.endswith(('sses', 'xes', 'zes', 'ches', 'shes')):
                return word[:-2]
            # boxes → box, watches → watch
            if len(word) > 4 and word[-3] in 'xzh':
                return word[:-2]

        # Rule 3: -s → remove (but not if ends in 'ss')
        if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
            # Don't singularize if removing 's' creates invalid word
            candidate = word[:-1]
            # Check it's not a common exception
            if candidate not in ('thi', 'ha', 'wa', 'doe', 'goe'):
                return candidate

        return word

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
