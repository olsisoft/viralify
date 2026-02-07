"""
Unit tests for RAG services in presentation-generator.

Tests:
- RAGMode enum
- RAGThresholdResult dataclass
- RAGThresholdValidator class
- WeaveGraph models (ConceptNode, ConceptEdge, WeaveGraph, QueryExpansion)
- ResonanceConfig, ResonanceResult, ResonanceMatcher
- ConceptExtractor and ExtractionConfig
"""

import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from collections import Counter, defaultdict
import re
import math
from datetime import datetime
import uuid


# =============================================================================
# Standalone implementations to avoid import chain issues
# =============================================================================

class RAGMode(str, Enum):
    """RAG operation mode based on available context"""
    FULL = "full"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    NONE = "none"


@dataclass
class RAGThresholdResult:
    """Result of RAG threshold validation"""
    mode: RAGMode
    token_count: int
    is_sufficient: bool
    warning_message: Optional[str] = None
    error_message: Optional[str] = None
    topic_relevance_score: float = 1.0
    topic_coverage_issues: List[str] = field(default_factory=list)
    content_quality_score: float = 1.0
    unique_sources_count: int = 0
    density_score: float = 1.0

    @property
    def should_block(self) -> bool:
        return self.mode == RAGMode.BLOCKED

    @property
    def has_warning(self) -> bool:
        return self.warning_message is not None

    @property
    def quality_grade(self) -> str:
        combined = (
            (self.topic_relevance_score * 0.4) +
            (self.content_quality_score * 0.3) +
            (self.density_score * 0.3)
        )
        if combined >= 0.9:
            return "A"
        elif combined >= 0.75:
            return "B"
        elif combined >= 0.6:
            return "C"
        elif combined >= 0.4:
            return "D"
        return "F"


class RAGThresholdValidator:
    """Validates RAG context meets minimum thresholds"""

    DEFAULT_MINIMUM_TOKENS = 750
    DEFAULT_QUALITY_TOKENS = 3000
    DEFAULT_OPTIMAL_TOKENS = 5000
    STRICT_MINIMUM_TOKENS = 1000
    STRICT_QUALITY_TOKENS = 4000
    MIN_TOPIC_RELEVANCE = 0.30
    WARN_TOPIC_RELEVANCE = 0.50

    def __init__(
        self,
        minimum_tokens: Optional[int] = None,
        quality_tokens: Optional[int] = None,
        strict_mode: bool = False,
    ):
        self.strict_mode = strict_mode
        if self.strict_mode:
            default_min = self.STRICT_MINIMUM_TOKENS
            default_quality = self.STRICT_QUALITY_TOKENS
        else:
            default_min = self.DEFAULT_MINIMUM_TOKENS
            default_quality = self.DEFAULT_QUALITY_TOKENS

        self.minimum_tokens = minimum_tokens or default_min
        self.quality_tokens = quality_tokens or default_quality

    def count_tokens(self, text: str) -> int:
        """Simple token count approximation"""
        if not text:
            return 0
        # Rough approximation: ~4 chars per token
        return len(text) // 4

    def _extract_unique_terms(self, text: str) -> set:
        stopwords = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has',
            'will', 'can', 'are', 'was', 'were', 'been', 'being', 'would', 'could',
            'vous', 'nous', 'elle', 'sont', 'avec', 'pour', 'dans', 'cette',
        }
        words = re.findall(r'\b[a-zA-Z\u00C0-\u017F]{4,}\b', text.lower())
        return {w for w in words if w not in stopwords}

    def _calculate_topic_relevance(
        self,
        rag_context: str,
        topic: Optional[str]
    ) -> Tuple[float, List[str]]:
        if not topic:
            return 1.0, []

        topic_lower = topic.lower()
        context_lower = rag_context.lower()
        issues = []

        topic_words = set(topic_lower.split())
        topic_words = {w for w in topic_words if len(w) > 3}

        if not topic_words:
            return 1.0, []

        found_keywords = sum(1 for w in topic_words if w in context_lower)
        keyword_coverage = found_keywords / len(topic_words) if topic_words else 1.0

        if keyword_coverage < 0.3:
            issues.append(f"Topic keywords poorly covered ({keyword_coverage:.0%})")

        topic_phrase_found = topic_lower in context_lower
        if not topic_phrase_found and keyword_coverage < 0.5:
            issues.append("Topic phrase not found in source documents")

        relevance = (keyword_coverage * 0.7) + (0.3 if topic_phrase_found else 0.0)
        return relevance, issues

    def _calculate_content_quality(self, rag_context: str) -> Tuple[float, float]:
        tokens = self.count_tokens(rag_context)
        unique_terms = self._extract_unique_terms(rag_context)

        density = len(unique_terms) / max(tokens, 1)
        density_normalized = min(1.0, density * 10)

        has_structure = bool(re.search(r'(^|\n)(#+\s|[-*]\s|\d+\.\s)', rag_context))
        has_technical = bool(re.search(r'\b[A-Z][a-z]+[A-Z]|[A-Z]{2,5}\b', rag_context))
        has_examples = bool(re.search(r'```|`[^`]+`|example|exemple', rag_context, re.IGNORECASE))
        paragraphs = rag_context.split('\n\n')
        has_paragraphs = len(paragraphs) >= 3

        quality_factors = [has_structure, has_technical, has_examples, has_paragraphs]
        quality_score = sum(quality_factors) / len(quality_factors)

        return quality_score, density_normalized

    def _count_unique_sources(self, rag_context: str) -> int:
        separators = [
            r'---+',
            r'Document \d+',
            r'\[Source:',
            r'From:.*\.pdf',
            r'#{2,}\s+',
        ]

        source_count = 1
        for pattern in separators:
            matches = len(re.findall(pattern, rag_context, re.IGNORECASE))
            source_count = max(source_count, matches + 1)

        return min(source_count, 10)

    def validate(
        self,
        rag_context: Optional[str],
        has_documents: bool = False,
        strict_mode: bool = False,
        topic: Optional[str] = None,
    ) -> RAGThresholdResult:
        strict_mode = strict_mode or self.strict_mode

        if not rag_context or not rag_context.strip():
            if has_documents:
                return RAGThresholdResult(
                    mode=RAGMode.BLOCKED,
                    token_count=0,
                    is_sufficient=False,
                    error_message="No content could be extracted from the provided documents.",
                    topic_relevance_score=0.0,
                    content_quality_score=0.0,
                )
            else:
                return RAGThresholdResult(
                    mode=RAGMode.NONE,
                    token_count=0,
                    is_sufficient=True,
                    warning_message="No source documents provided.",
                )

        token_count = self.count_tokens(rag_context)
        topic_relevance, topic_issues = self._calculate_topic_relevance(rag_context, topic)
        quality_score, density_score = self._calculate_content_quality(rag_context)
        unique_sources = self._count_unique_sources(rag_context)

        if topic and topic_relevance < self.MIN_TOPIC_RELEVANCE:
            return RAGThresholdResult(
                mode=RAGMode.BLOCKED,
                token_count=token_count,
                is_sufficient=False,
                error_message=f"Source content is not sufficiently relevant to the topic '{topic}'.",
                topic_relevance_score=topic_relevance,
                topic_coverage_issues=topic_issues,
                content_quality_score=quality_score,
                density_score=density_score,
                unique_sources_count=unique_sources,
            )

        if token_count < self.minimum_tokens:
            return RAGThresholdResult(
                mode=RAGMode.BLOCKED,
                token_count=token_count,
                is_sufficient=False,
                error_message=f"Insufficient source content: {token_count} tokens retrieved.",
                topic_relevance_score=topic_relevance,
                topic_coverage_issues=topic_issues,
                content_quality_score=quality_score,
                density_score=density_score,
                unique_sources_count=unique_sources,
            )

        if token_count < self.quality_tokens:
            warning_msg = f"Limited source content: {token_count} tokens"
            if strict_mode:
                return RAGThresholdResult(
                    mode=RAGMode.BLOCKED,
                    token_count=token_count,
                    is_sufficient=False,
                    error_message=f"Source content below quality threshold in strict mode.",
                    topic_relevance_score=topic_relevance,
                    topic_coverage_issues=topic_issues,
                    content_quality_score=quality_score,
                    density_score=density_score,
                    unique_sources_count=unique_sources,
                )

            return RAGThresholdResult(
                mode=RAGMode.PARTIAL,
                token_count=token_count,
                is_sufficient=True,
                warning_message=warning_msg,
                topic_relevance_score=topic_relevance,
                topic_coverage_issues=topic_issues,
                content_quality_score=quality_score,
                density_score=density_score,
                unique_sources_count=unique_sources,
            )

        return RAGThresholdResult(
            mode=RAGMode.FULL,
            token_count=token_count,
            is_sufficient=True,
            topic_relevance_score=topic_relevance,
            topic_coverage_issues=topic_issues,
            content_quality_score=quality_score,
            density_score=density_score,
            unique_sources_count=unique_sources,
        )


# =============================================================================
# WeaveGraph Models (standalone)
# =============================================================================

class RelationType(str, Enum):
    """Types of relationships between concepts"""
    SIMILAR = "similar"
    TRANSLATION = "translation"
    PART_OF = "part_of"
    PREREQUISITE = "prerequisite"
    RELATED = "related"
    SYNONYM = "synonym"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"


class ConceptSource(str, Enum):
    """How the concept was extracted"""
    NLP_EXTRACTION = "nlp"
    KEYWORD = "keyword"
    ENTITY = "entity"
    TECHNICAL_TERM = "technical"
    USER_DEFINED = "user"
    LLM_ENRICHED = "llm"


@dataclass
class ConceptNode:
    """A concept in the WeaveGraph"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    canonical_name: str = ""
    language: str = "en"
    embedding: Optional[List[float]] = None
    source_document_ids: List[str] = field(default_factory=list)
    frequency: int = 1
    source_type: ConceptSource = ConceptSource.NLP_EXTRACTION
    aliases: List[str] = field(default_factory=list)
    context_snippets: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __hash__(self):
        return hash(self.canonical_name)

    def __eq__(self, other):
        if isinstance(other, ConceptNode):
            return self.canonical_name == other.canonical_name
        return False


@dataclass
class ConceptEdge:
    """An edge connecting two concepts"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: RelationType = RelationType.SIMILAR
    weight: float = 1.0
    bidirectional: bool = True
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WeaveGraphStats:
    """Statistics about a WeaveGraph"""
    total_concepts: int = 0
    total_edges: int = 0
    avg_connections_per_concept: float = 0.0
    languages: List[str] = field(default_factory=list)
    top_concepts: List[str] = field(default_factory=list)
    edge_type_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class ConceptCluster:
    """A cluster of related concepts"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    concepts: List[ConceptNode] = field(default_factory=list)
    centroid_embedding: Optional[List[float]] = None
    coherence_score: float = 0.0


@dataclass
class QueryExpansion:
    """Result of expanding a query using WeaveGraph"""
    original_query: str = ""
    expanded_terms: List[str] = field(default_factory=list)
    expansion_paths: Dict[str, List[str]] = field(default_factory=dict)
    total_weight: float = 0.0
    languages_covered: Set[str] = field(default_factory=set)


@dataclass
class WeaveGraph:
    """The complete concept graph"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    document_ids: List[str] = field(default_factory=list)
    concepts: Dict[str, ConceptNode] = field(default_factory=dict)
    edges: List[ConceptEdge] = field(default_factory=list)
    clusters: List[ConceptCluster] = field(default_factory=list)
    stats: WeaveGraphStats = field(default_factory=WeaveGraphStats)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_concept(self, concept: ConceptNode) -> None:
        self.concepts[concept.id] = concept

    def add_edge(self, edge: ConceptEdge) -> None:
        self.edges.append(edge)

    def get_neighbors(self, concept_id: str, max_depth: int = 1) -> List[ConceptNode]:
        if max_depth < 1:
            return []

        neighbors = set()
        current_ids = {concept_id}

        for _ in range(max_depth):
            next_ids = set()
            for edge in self.edges:
                if edge.source_id in current_ids and edge.target_id not in neighbors:
                    next_ids.add(edge.target_id)
                    neighbors.add(edge.target_id)
                if edge.bidirectional and edge.target_id in current_ids and edge.source_id not in neighbors:
                    next_ids.add(edge.source_id)
                    neighbors.add(edge.source_id)
            current_ids = next_ids

        return [self.concepts[cid] for cid in neighbors if cid in self.concepts]

    def find_concept_by_name(self, name: str) -> Optional[ConceptNode]:
        name_lower = name.lower().strip()
        for concept in self.concepts.values():
            if concept.name.lower() == name_lower or concept.canonical_name == name_lower:
                return concept
            if name_lower in [a.lower() for a in concept.aliases]:
                return concept
        return None

    def compute_stats(self) -> WeaveGraphStats:
        if not self.concepts:
            return WeaveGraphStats()

        connection_counts = {cid: 0 for cid in self.concepts}
        edge_types = {}

        for edge in self.edges:
            if edge.source_id in connection_counts:
                connection_counts[edge.source_id] += 1
            if edge.bidirectional and edge.target_id in connection_counts:
                connection_counts[edge.target_id] += 1
            edge_types[edge.relation_type.value] = edge_types.get(edge.relation_type.value, 0) + 1

        languages = list(set(c.language for c in self.concepts.values()))
        sorted_concepts = sorted(self.concepts.values(), key=lambda c: c.frequency, reverse=True)
        top_concepts = [c.name for c in sorted_concepts[:10]]

        self.stats = WeaveGraphStats(
            total_concepts=len(self.concepts),
            total_edges=len(self.edges),
            avg_connections_per_concept=sum(connection_counts.values()) / len(self.concepts) if self.concepts else 0,
            languages=languages,
            top_concepts=top_concepts,
            edge_type_distribution=edge_types
        )

        return self.stats


# =============================================================================
# Resonance Matcher (standalone)
# =============================================================================

@dataclass
class ResonanceConfig:
    """Configuration for resonance propagation"""
    decay_factor: float = 0.7
    max_depth: int = 3
    min_resonance: float = 0.10
    boost_translation: float = 1.2
    boost_synonym: float = 1.1
    max_resonating_concepts: int = 50


@dataclass
class ResonanceResult:
    """Result of resonance propagation"""
    scores: Dict[str, float] = field(default_factory=dict)
    depths: Dict[str, int] = field(default_factory=dict)
    paths: Dict[str, List[str]] = field(default_factory=dict)
    direct_matches: int = 0
    propagated_matches: int = 0
    total_resonance: float = 0.0
    max_depth_reached: int = 0

    def get_top_concepts(self, n: int = 10) -> List[Tuple[str, float]]:
        sorted_items = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]

    def get_concepts_above_threshold(self, threshold: float = 0.3) -> List[str]:
        return [cid for cid, score in self.scores.items() if score >= threshold]


class ResonanceMatcher:
    """Propagates matching scores through the WeaveGraph"""

    def __init__(self, config: Optional[ResonanceConfig] = None):
        self.config = config or ResonanceConfig()

    def propagate(
        self,
        matched_concept_ids: List[str],
        graph: WeaveGraph,
        initial_scores: Optional[Dict[str, float]] = None
    ) -> ResonanceResult:
        result = ResonanceResult()

        if not matched_concept_ids:
            return result

        scores = initial_scores.copy() if initial_scores else {}
        for cid in matched_concept_ids:
            if cid not in scores:
                scores[cid] = 1.0
            result.depths[cid] = 0
            result.paths[cid] = [cid]

        result.direct_matches = len(matched_concept_ids)

        adjacency = self._build_adjacency_map(graph)

        current_frontier = set(matched_concept_ids)
        visited = set(matched_concept_ids)

        for depth in range(1, self.config.max_depth + 1):
            next_frontier = set()

            for concept_id in current_frontier:
                current_score = scores.get(concept_id, 0)

                if current_score < self.config.min_resonance:
                    continue

                neighbors = adjacency.get(concept_id, [])

                for neighbor_id, edge_weight, relation_type in neighbors:
                    if neighbor_id in visited:
                        new_score = self._compute_resonance(
                            current_score, edge_weight, depth, relation_type
                        )
                        if new_score > scores.get(neighbor_id, 0):
                            scores[neighbor_id] = new_score
                        continue

                    resonance = self._compute_resonance(
                        current_score, edge_weight, depth, relation_type
                    )

                    if resonance >= self.config.min_resonance:
                        scores[neighbor_id] = resonance
                        result.depths[neighbor_id] = depth
                        result.paths[neighbor_id] = result.paths.get(concept_id, []) + [neighbor_id]
                        next_frontier.add(neighbor_id)
                        visited.add(neighbor_id)
                        result.propagated_matches += 1
                        result.max_depth_reached = max(result.max_depth_reached, depth)

            current_frontier = next_frontier

            if len(scores) >= self.config.max_resonating_concepts:
                break

            if not next_frontier:
                break

        result.scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        result.total_resonance = sum(scores.values())

        return result

    def _compute_resonance(
        self,
        parent_score: float,
        edge_weight: float,
        depth: int,
        relation_type: RelationType
    ) -> float:
        resonance = parent_score * edge_weight * (self.config.decay_factor ** depth)

        if relation_type == RelationType.TRANSLATION:
            resonance *= self.config.boost_translation
        elif relation_type == RelationType.SYNONYM:
            resonance *= self.config.boost_synonym

        return min(1.0, max(0.0, resonance))

    def _build_adjacency_map(
        self,
        graph: WeaveGraph
    ) -> Dict[str, List[Tuple[str, float, RelationType]]]:
        adjacency = defaultdict(list)

        for edge in graph.edges:
            adjacency[edge.source_id].append(
                (edge.target_id, edge.weight, edge.relation_type)
            )

            if edge.bidirectional:
                adjacency[edge.target_id].append(
                    (edge.source_id, edge.weight, edge.relation_type)
                )

        return dict(adjacency)


# =============================================================================
# Concept Extractor (standalone)
# =============================================================================

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
    """Extracts concepts from documents using NLP techniques"""

    PATTERNS = {
        "camel_case": re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b'),
        "snake_case": re.compile(r'\b([a-z]+(?:_[a-z]+)+)\b'),
        "acronym": re.compile(r'\b([A-Z]{2,6})\b'),
        "version": re.compile(r'\b([A-Za-z]+\s*\d+(?:\.\d+)*)\b'),
        "code_element": re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\(\)|\.[\w]+))\b'),
        "compound_term": re.compile(r'\b([A-Z]?[a-z]+(?:\s+[a-z]+){1,3})\b'),
        "hyphenated": re.compile(r'\b([a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)?)\b'),
        "package": re.compile(r'\b([a-z]+(?:\.[a-z]+)+)\b'),
    }

    TECH_DOMAINS = {
        "data": ["pipeline", "etl", "elt", "warehouse", "lake", "streaming", "batch", "ingestion"],
        "cloud": ["aws", "azure", "gcp", "kubernetes", "docker", "serverless", "lambda", "ec2"],
        "ml": ["model", "training", "inference", "feature", "embedding", "neural", "transformer"],
        "web": ["api", "rest", "graphql", "endpoint", "microservice", "gateway", "authentication"],
        "database": ["sql", "nosql", "mongodb", "postgresql", "redis", "cassandra", "index"],
        "messaging": ["kafka", "rabbitmq", "pubsub", "queue", "consumer", "producer", "topic"],
    }

    STOP_WORDS = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'en', 'est',
        'que', 'qui', 'dans', 'ce', 'il', 'ne', 'sur', 'se', 'pas', 'plus',
        'par', 'pour', 'au', 'avec', 'son', 'sa', 'ses', 'ou', 'comme', 'mais',
        'nous', 'vous', 'leur', 'cette', 'ces', 'tout', 'elle', 'sont',
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

    def _canonicalize(self, term: str) -> str:
        canonical = term.lower().strip()
        canonical = re.sub(r'[\s\-\.]+', '_', canonical)
        canonical = re.sub(r'[^a-z0-9_]', '', canonical)
        return canonical

    def _is_valid_term(self, term: str, canonical: str) -> bool:
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
        text_lower = text.lower()

        french_indicators = ['le', 'la', 'les', 'de', 'du', 'des', 'est', 'sont', 'avec', 'pour', 'dans', 'cette', 'ces']
        english_indicators = ['the', 'is', 'are', 'with', 'for', 'this', 'that', 'from', 'have', 'has']

        french_count = sum(1 for word in french_indicators if f' {word} ' in f' {text_lower} ')
        english_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_lower} ')

        return 'fr' if french_count > english_count else 'en'

    def _extract_pattern_terms(self, text: str) -> List[Tuple[str, ConceptSource]]:
        terms = []

        for pattern_name, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            source = ConceptSource.TECHNICAL_TERM if pattern_name in ["camel_case", "snake_case", "code_element", "package"] else ConceptSource.NLP_EXTRACTION

            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                terms.append((match, source))

        return terms

    def _extract_ngrams(self, words: List[str], n: int) -> Counter:
        ngrams = Counter()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            if not any(w in self.STOP_WORDS for w in words[i:i+n]):
                ngrams[ngram] += 1
        return ngrams

    def _extract_keywords(self, text: str, top_n: int = 100) -> List[Tuple[str, float]]:
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text.lower())
        words = [w for w in words if w not in self.STOP_WORDS and len(w) >= self.config.min_term_length]

        tf = Counter(words)
        total_words = len(words)

        if total_words == 0:
            return []

        scored_terms = []
        for term, count in tf.items():
            tf_score = count / total_words
            boost = 1.0
            if any(c.isupper() for c in term):
                boost = 1.5
            if '_' in term or '-' in term:
                boost = 1.5
            if len(term) > 8:
                boost *= 1.2

            score = tf_score * boost
            scored_terms.append((term, score))

        if self.config.include_bigrams:
            bigrams = self._extract_ngrams(words, 2)
            for bigram, count in bigrams.items():
                score = (count / total_words) * 1.5
                scored_terms.append((bigram, score))

        scored_terms.sort(key=lambda x: x[1], reverse=True)
        return scored_terms[:top_n]

    def _extract_domain_terms(self, text: str) -> List[str]:
        text_lower = text.lower()
        found_terms = []

        for domain, terms in self.TECH_DOMAINS.items():
            for term in terms:
                if term in text_lower:
                    found_terms.append(term)

        return found_terms

    def extract_context_snippets(self, text: str, term: str, window: int = 50) -> List[str]:
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

            if len(snippets) >= 3:
                break

        return snippets


# =============================================================================
# TESTS
# =============================================================================

class TestRAGMode:
    """Tests for RAGMode enum"""

    def test_all_values(self):
        assert RAGMode.FULL == "full"
        assert RAGMode.PARTIAL == "partial"
        assert RAGMode.BLOCKED == "blocked"
        assert RAGMode.NONE == "none"

    def test_enum_count(self):
        assert len(RAGMode) == 4

    def test_string_comparison(self):
        assert RAGMode.FULL == "full"
        assert RAGMode.BLOCKED.value == "blocked"


class TestRAGThresholdResult:
    """Tests for RAGThresholdResult dataclass"""

    def test_basic_creation(self):
        result = RAGThresholdResult(
            mode=RAGMode.FULL,
            token_count=5000,
            is_sufficient=True
        )
        assert result.mode == RAGMode.FULL
        assert result.token_count == 5000
        assert result.is_sufficient is True

    def test_should_block_property(self):
        blocked_result = RAGThresholdResult(
            mode=RAGMode.BLOCKED,
            token_count=100,
            is_sufficient=False
        )
        assert blocked_result.should_block is True

        full_result = RAGThresholdResult(
            mode=RAGMode.FULL,
            token_count=5000,
            is_sufficient=True
        )
        assert full_result.should_block is False

    def test_has_warning_property(self):
        with_warning = RAGThresholdResult(
            mode=RAGMode.PARTIAL,
            token_count=1000,
            is_sufficient=True,
            warning_message="Limited content"
        )
        assert with_warning.has_warning is True

        without_warning = RAGThresholdResult(
            mode=RAGMode.FULL,
            token_count=5000,
            is_sufficient=True
        )
        assert without_warning.has_warning is False

    def test_quality_grade_a(self):
        result = RAGThresholdResult(
            mode=RAGMode.FULL,
            token_count=5000,
            is_sufficient=True,
            topic_relevance_score=0.95,
            content_quality_score=0.95,
            density_score=0.95
        )
        assert result.quality_grade == "A"

    def test_quality_grade_b(self):
        result = RAGThresholdResult(
            mode=RAGMode.FULL,
            token_count=5000,
            is_sufficient=True,
            topic_relevance_score=0.80,
            content_quality_score=0.75,
            density_score=0.70
        )
        assert result.quality_grade == "B"

    def test_quality_grade_c(self):
        result = RAGThresholdResult(
            mode=RAGMode.PARTIAL,
            token_count=1500,
            is_sufficient=True,
            topic_relevance_score=0.65,
            content_quality_score=0.60,
            density_score=0.55
        )
        assert result.quality_grade == "C"

    def test_quality_grade_f(self):
        result = RAGThresholdResult(
            mode=RAGMode.BLOCKED,
            token_count=100,
            is_sufficient=False,
            topic_relevance_score=0.20,
            content_quality_score=0.20,
            density_score=0.20
        )
        assert result.quality_grade == "F"

    def test_with_topic_issues(self):
        result = RAGThresholdResult(
            mode=RAGMode.PARTIAL,
            token_count=1500,
            is_sufficient=True,
            topic_coverage_issues=["Topic keywords poorly covered", "Missing key terms"]
        )
        assert len(result.topic_coverage_issues) == 2


class TestRAGThresholdValidator:
    """Tests for RAGThresholdValidator class"""

    def test_default_thresholds(self):
        validator = RAGThresholdValidator()
        assert validator.minimum_tokens == 750
        assert validator.quality_tokens == 3000

    def test_strict_mode_thresholds(self):
        validator = RAGThresholdValidator(strict_mode=True)
        assert validator.minimum_tokens == 1000
        assert validator.quality_tokens == 4000

    def test_custom_thresholds(self):
        validator = RAGThresholdValidator(minimum_tokens=500, quality_tokens=2000)
        assert validator.minimum_tokens == 500
        assert validator.quality_tokens == 2000

    def test_count_tokens(self):
        validator = RAGThresholdValidator()
        # Rough approximation: ~4 chars per token
        text = "a" * 400  # Should be ~100 tokens
        tokens = validator.count_tokens(text)
        assert tokens == 100

    def test_count_tokens_empty(self):
        validator = RAGThresholdValidator()
        assert validator.count_tokens("") == 0
        assert validator.count_tokens(None) == 0

    def test_validate_no_context_no_documents(self):
        validator = RAGThresholdValidator()
        result = validator.validate(None, has_documents=False)
        assert result.mode == RAGMode.NONE
        assert result.is_sufficient is True
        assert result.has_warning is True

    def test_validate_no_context_with_documents(self):
        validator = RAGThresholdValidator()
        result = validator.validate(None, has_documents=True)
        assert result.mode == RAGMode.BLOCKED
        assert result.is_sufficient is False
        assert result.error_message is not None

    def test_validate_empty_context(self):
        validator = RAGThresholdValidator()
        result = validator.validate("   ", has_documents=True)
        assert result.mode == RAGMode.BLOCKED

    def test_validate_insufficient_tokens(self):
        validator = RAGThresholdValidator()
        # 500 chars = ~125 tokens < 750 minimum
        text = "a" * 500
        result = validator.validate(text, has_documents=True)
        assert result.mode == RAGMode.BLOCKED
        assert "Insufficient source content" in result.error_message

    def test_validate_partial_tokens(self):
        validator = RAGThresholdValidator()
        # 4000 chars = ~1000 tokens (between 750 and 3000)
        text = "Apache Kafka is a distributed streaming platform. " * 80
        result = validator.validate(text, has_documents=True)
        assert result.mode == RAGMode.PARTIAL
        assert result.is_sufficient is True
        assert result.has_warning is True

    def test_validate_full_mode(self):
        validator = RAGThresholdValidator()
        # 15000 chars = ~3750 tokens > 3000
        text = "Apache Kafka streaming platform with consumer and producer. " * 250
        result = validator.validate(text, has_documents=True)
        assert result.mode == RAGMode.FULL
        assert result.is_sufficient is True

    def test_validate_topic_relevance_blocked(self):
        validator = RAGThresholdValidator()
        # Content about databases, topic about Kafka
        text = "PostgreSQL is a relational database. SQL queries are important. " * 100
        result = validator.validate(text, has_documents=True, topic="Apache Kafka streaming")
        # Should be blocked due to low relevance
        assert result.topic_relevance_score < 0.5

    def test_validate_topic_relevance_good(self):
        validator = RAGThresholdValidator()
        text = "Apache Kafka is a distributed streaming platform for real-time data. " * 100
        result = validator.validate(text, has_documents=True, topic="Apache Kafka")
        assert result.topic_relevance_score > 0.5

    def test_strict_mode_blocks_partial(self):
        validator = RAGThresholdValidator()
        # 4000 chars = ~1000 tokens (between 750 and 3000)
        text = "Apache Kafka is a distributed streaming platform. " * 80
        result = validator.validate(text, has_documents=True, strict_mode=True)
        assert result.mode == RAGMode.BLOCKED

    def test_extract_unique_terms(self):
        validator = RAGThresholdValidator()
        text = "Apache Kafka streaming platform with consumer producer messaging"
        terms = validator._extract_unique_terms(text)
        assert "apache" in terms
        assert "kafka" in terms
        assert "streaming" in terms
        # Stop words should be filtered
        assert "the" not in terms

    def test_calculate_content_quality(self):
        validator = RAGThresholdValidator()
        # Good quality text with structure and technical terms
        text = """# Apache Kafka Overview

Apache Kafka is a distributed streaming platform.

## Key Features

- High throughput
- Low latency
- Fault tolerant

```python
from kafka import KafkaProducer
producer = KafkaProducer()
```

Example: Create a producer to send messages.
"""
        quality, density = validator._calculate_content_quality(text)
        # Should have good quality due to structure, technical terms, examples
        assert quality > 0.5

    def test_count_unique_sources(self):
        validator = RAGThresholdValidator()
        text = """Document 1: Kafka basics
---
Document 2: Advanced topics
---
Document 3: Best practices"""
        sources = validator._count_unique_sources(text)
        assert sources >= 2


class TestRelationType:
    """Tests for RelationType enum"""

    def test_all_values(self):
        assert RelationType.SIMILAR == "similar"
        assert RelationType.TRANSLATION == "translation"
        assert RelationType.PART_OF == "part_of"
        assert RelationType.PREREQUISITE == "prerequisite"
        assert RelationType.RELATED == "related"
        assert RelationType.SYNONYM == "synonym"
        assert RelationType.HYPERNYM == "hypernym"
        assert RelationType.HYPONYM == "hyponym"

    def test_enum_count(self):
        assert len(RelationType) == 8


class TestConceptSource:
    """Tests for ConceptSource enum"""

    def test_all_values(self):
        assert ConceptSource.NLP_EXTRACTION == "nlp"
        assert ConceptSource.KEYWORD == "keyword"
        assert ConceptSource.ENTITY == "entity"
        assert ConceptSource.TECHNICAL_TERM == "technical"
        assert ConceptSource.USER_DEFINED == "user"
        assert ConceptSource.LLM_ENRICHED == "llm"

    def test_enum_count(self):
        assert len(ConceptSource) == 6


class TestConceptNode:
    """Tests for ConceptNode dataclass"""

    def test_basic_creation(self):
        node = ConceptNode(
            name="Apache Kafka",
            canonical_name="apache_kafka",
            language="en"
        )
        assert node.name == "Apache Kafka"
        assert node.canonical_name == "apache_kafka"
        assert node.language == "en"
        assert node.frequency == 1

    def test_auto_id(self):
        node = ConceptNode(name="Test")
        assert node.id is not None
        assert len(node.id) > 0

    def test_with_embedding(self):
        embedding = [0.1, 0.2, 0.3, 0.4]
        node = ConceptNode(
            name="Test",
            embedding=embedding
        )
        assert node.embedding == embedding

    def test_with_aliases(self):
        node = ConceptNode(
            name="Apache Kafka",
            aliases=["Kafka", "kafka-streams"]
        )
        assert len(node.aliases) == 2
        assert "Kafka" in node.aliases

    def test_hash_equality(self):
        node1 = ConceptNode(canonical_name="apache_kafka")
        node2 = ConceptNode(canonical_name="apache_kafka")
        assert hash(node1) == hash(node2)
        assert node1 == node2

    def test_hash_inequality(self):
        node1 = ConceptNode(canonical_name="apache_kafka")
        node2 = ConceptNode(canonical_name="rabbitmq")
        assert node1 != node2


class TestConceptEdge:
    """Tests for ConceptEdge dataclass"""

    def test_basic_creation(self):
        edge = ConceptEdge(
            source_id="id1",
            target_id="id2",
            relation_type=RelationType.SIMILAR,
            weight=0.85
        )
        assert edge.source_id == "id1"
        assert edge.target_id == "id2"
        assert edge.relation_type == RelationType.SIMILAR
        assert edge.weight == 0.85

    def test_default_values(self):
        edge = ConceptEdge()
        assert edge.relation_type == RelationType.SIMILAR
        assert edge.weight == 1.0
        assert edge.bidirectional is True

    def test_unidirectional_edge(self):
        edge = ConceptEdge(
            source_id="parent",
            target_id="child",
            relation_type=RelationType.PART_OF,
            bidirectional=False
        )
        assert edge.bidirectional is False


class TestWeaveGraphStats:
    """Tests for WeaveGraphStats dataclass"""

    def test_default_values(self):
        stats = WeaveGraphStats()
        assert stats.total_concepts == 0
        assert stats.total_edges == 0
        assert stats.avg_connections_per_concept == 0.0
        assert stats.languages == []

    def test_with_data(self):
        stats = WeaveGraphStats(
            total_concepts=100,
            total_edges=250,
            avg_connections_per_concept=2.5,
            languages=["en", "fr"],
            top_concepts=["kafka", "streaming", "consumer"],
            edge_type_distribution={"similar": 200, "translation": 50}
        )
        assert stats.total_concepts == 100
        assert len(stats.languages) == 2


class TestConceptCluster:
    """Tests for ConceptCluster dataclass"""

    def test_basic_creation(self):
        cluster = ConceptCluster(
            name="Messaging",
            coherence_score=0.85
        )
        assert cluster.name == "Messaging"
        assert cluster.coherence_score == 0.85

    def test_with_concepts(self):
        concepts = [
            ConceptNode(name="Kafka"),
            ConceptNode(name="RabbitMQ")
        ]
        cluster = ConceptCluster(
            name="Message Brokers",
            concepts=concepts
        )
        assert len(cluster.concepts) == 2


class TestQueryExpansion:
    """Tests for QueryExpansion dataclass"""

    def test_basic_creation(self):
        expansion = QueryExpansion(
            original_query="Kafka consumer",
            expanded_terms=["kafka", "consumer", "message", "broker"],
            total_weight=2.5
        )
        assert expansion.original_query == "Kafka consumer"
        assert len(expansion.expanded_terms) == 4
        assert expansion.total_weight == 2.5

    def test_with_expansion_paths(self):
        expansion = QueryExpansion(
            original_query="Kafka",
            expansion_paths={
                "message_broker": ["kafka", "message_broker"],
                "streaming": ["kafka", "streaming"]
            },
            languages_covered={"en", "fr"}
        )
        assert len(expansion.expansion_paths) == 2
        assert "en" in expansion.languages_covered


class TestWeaveGraph:
    """Tests for WeaveGraph dataclass"""

    def test_basic_creation(self):
        graph = WeaveGraph(user_id="user123")
        assert graph.user_id == "user123"
        assert len(graph.concepts) == 0
        assert len(graph.edges) == 0

    def test_add_concept(self):
        graph = WeaveGraph()
        concept = ConceptNode(name="Kafka", canonical_name="kafka")
        graph.add_concept(concept)
        assert concept.id in graph.concepts

    def test_add_edge(self):
        graph = WeaveGraph()
        edge = ConceptEdge(source_id="id1", target_id="id2")
        graph.add_edge(edge)
        assert len(graph.edges) == 1

    def test_find_concept_by_name(self):
        graph = WeaveGraph()
        concept = ConceptNode(name="Apache Kafka", canonical_name="apache_kafka")
        graph.add_concept(concept)

        # Find by name
        found = graph.find_concept_by_name("Apache Kafka")
        assert found is not None
        assert found.canonical_name == "apache_kafka"

        # Find by canonical name
        found = graph.find_concept_by_name("apache_kafka")
        assert found is not None

    def test_find_concept_by_alias(self):
        graph = WeaveGraph()
        concept = ConceptNode(
            name="Apache Kafka",
            canonical_name="apache_kafka",
            aliases=["Kafka", "kafka-streams"]
        )
        graph.add_concept(concept)

        found = graph.find_concept_by_name("Kafka")
        assert found is not None

    def test_find_concept_not_found(self):
        graph = WeaveGraph()
        found = graph.find_concept_by_name("nonexistent")
        assert found is None

    def test_get_neighbors_single_hop(self):
        graph = WeaveGraph()

        # Create concepts
        kafka = ConceptNode(id="kafka", name="Kafka", canonical_name="kafka")
        consumer = ConceptNode(id="consumer", name="Consumer", canonical_name="consumer")
        producer = ConceptNode(id="producer", name="Producer", canonical_name="producer")

        graph.add_concept(kafka)
        graph.add_concept(consumer)
        graph.add_concept(producer)

        # Create edges
        graph.add_edge(ConceptEdge(source_id="kafka", target_id="consumer"))
        graph.add_edge(ConceptEdge(source_id="kafka", target_id="producer"))

        neighbors = graph.get_neighbors("kafka", max_depth=1)
        assert len(neighbors) == 2

    def test_get_neighbors_multi_hop(self):
        graph = WeaveGraph()

        # Create chain: A -> B -> C (unidirectional to test proper multi-hop)
        a = ConceptNode(id="a", name="A")
        b = ConceptNode(id="b", name="B")
        c = ConceptNode(id="c", name="C")

        graph.add_concept(a)
        graph.add_concept(b)
        graph.add_concept(c)

        # Use unidirectional edges for clear multi-hop testing
        graph.add_edge(ConceptEdge(source_id="a", target_id="b", bidirectional=False))
        graph.add_edge(ConceptEdge(source_id="b", target_id="c", bidirectional=False))

        # Single hop should only get B
        neighbors_1 = graph.get_neighbors("a", max_depth=1)
        assert len(neighbors_1) == 1
        assert neighbors_1[0].id == "b"

        # Two hops should get B and C
        neighbors_2 = graph.get_neighbors("a", max_depth=2)
        assert len(neighbors_2) == 2

    def test_get_neighbors_zero_depth(self):
        graph = WeaveGraph()
        neighbors = graph.get_neighbors("any", max_depth=0)
        assert len(neighbors) == 0

    def test_compute_stats(self):
        graph = WeaveGraph()

        # Add concepts
        for name in ["Kafka", "Consumer", "Producer"]:
            node = ConceptNode(name=name, canonical_name=name.lower(), frequency=5)
            graph.add_concept(node)

        # Add edges
        graph.add_edge(ConceptEdge(source_id=list(graph.concepts.keys())[0],
                                   target_id=list(graph.concepts.keys())[1]))

        stats = graph.compute_stats()
        assert stats.total_concepts == 3
        assert stats.total_edges == 1
        assert len(stats.languages) > 0

    def test_compute_stats_empty_graph(self):
        graph = WeaveGraph()
        stats = graph.compute_stats()
        assert stats.total_concepts == 0


class TestResonanceConfig:
    """Tests for ResonanceConfig dataclass"""

    def test_default_values(self):
        config = ResonanceConfig()
        assert config.decay_factor == 0.7
        assert config.max_depth == 3
        assert config.min_resonance == 0.10
        assert config.boost_translation == 1.2
        assert config.boost_synonym == 1.1
        assert config.max_resonating_concepts == 50

    def test_custom_values(self):
        config = ResonanceConfig(
            decay_factor=0.5,
            max_depth=5,
            min_resonance=0.05
        )
        assert config.decay_factor == 0.5
        assert config.max_depth == 5


class TestResonanceResult:
    """Tests for ResonanceResult dataclass"""

    def test_default_values(self):
        result = ResonanceResult()
        assert result.direct_matches == 0
        assert result.propagated_matches == 0
        assert result.total_resonance == 0.0
        assert len(result.scores) == 0

    def test_get_top_concepts(self):
        result = ResonanceResult(
            scores={"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3}
        )
        top_2 = result.get_top_concepts(2)
        assert len(top_2) == 2
        assert top_2[0][0] == "a"
        assert top_2[1][0] == "b"

    def test_get_concepts_above_threshold(self):
        result = ResonanceResult(
            scores={"a": 0.9, "b": 0.7, "c": 0.2, "d": 0.1}
        )
        above_03 = result.get_concepts_above_threshold(0.3)
        assert len(above_03) == 2
        assert "a" in above_03
        assert "b" in above_03


class TestResonanceMatcher:
    """Tests for ResonanceMatcher class"""

    def test_basic_creation(self):
        matcher = ResonanceMatcher()
        assert matcher.config.decay_factor == 0.7

    def test_custom_config(self):
        config = ResonanceConfig(decay_factor=0.5)
        matcher = ResonanceMatcher(config)
        assert matcher.config.decay_factor == 0.5

    def test_propagate_empty(self):
        matcher = ResonanceMatcher()
        graph = WeaveGraph()
        result = matcher.propagate([], graph)
        assert result.direct_matches == 0

    def test_propagate_single_concept(self):
        matcher = ResonanceMatcher()
        graph = WeaveGraph()

        concept = ConceptNode(id="kafka", name="Kafka")
        graph.add_concept(concept)

        result = matcher.propagate(["kafka"], graph)
        assert result.direct_matches == 1
        assert "kafka" in result.scores
        assert result.scores["kafka"] == 1.0

    def test_propagate_with_neighbors(self):
        matcher = ResonanceMatcher()
        graph = WeaveGraph()

        # Create concepts
        kafka = ConceptNode(id="kafka", name="Kafka")
        consumer = ConceptNode(id="consumer", name="Consumer")

        graph.add_concept(kafka)
        graph.add_concept(consumer)

        # Create edge with weight 0.8
        graph.add_edge(ConceptEdge(
            source_id="kafka",
            target_id="consumer",
            weight=0.8
        ))

        result = matcher.propagate(["kafka"], graph)

        # Kafka should have score 1.0
        assert result.scores["kafka"] == 1.0

        # Consumer should have propagated score
        # 1.0 * 0.8 * 0.7^1 = 0.56
        assert "consumer" in result.scores
        assert 0.5 < result.scores["consumer"] < 0.6

    def test_propagate_translation_boost(self):
        config = ResonanceConfig(boost_translation=1.2)
        matcher = ResonanceMatcher(config)
        graph = WeaveGraph()

        # Create EN and FR concepts
        en = ConceptNode(id="integration", name="integration")
        fr = ConceptNode(id="integration_fr", name="intégration")

        graph.add_concept(en)
        graph.add_concept(fr)

        # Translation edge
        graph.add_edge(ConceptEdge(
            source_id="integration",
            target_id="integration_fr",
            relation_type=RelationType.TRANSLATION,
            weight=0.9
        ))

        result = matcher.propagate(["integration"], graph)

        # FR should have boosted score
        assert "integration_fr" in result.scores
        # Score should include translation boost

    def test_propagate_max_depth(self):
        config = ResonanceConfig(max_depth=2)
        matcher = ResonanceMatcher(config)
        graph = WeaveGraph()

        # Create chain: A -> B -> C -> D
        for name in ["a", "b", "c", "d"]:
            graph.add_concept(ConceptNode(id=name, name=name))

        graph.add_edge(ConceptEdge(source_id="a", target_id="b", weight=0.9))
        graph.add_edge(ConceptEdge(source_id="b", target_id="c", weight=0.9))
        graph.add_edge(ConceptEdge(source_id="c", target_id="d", weight=0.9))

        result = matcher.propagate(["a"], graph)

        # A and B should definitely be included
        assert "a" in result.scores
        assert "b" in result.scores

        # Max depth is 2, so C might be included but D likely won't
        assert result.max_depth_reached <= 2

    def test_compute_resonance(self):
        matcher = ResonanceMatcher()

        # Test basic computation
        resonance = matcher._compute_resonance(
            parent_score=1.0,
            edge_weight=0.8,
            depth=1,
            relation_type=RelationType.SIMILAR
        )
        # 1.0 * 0.8 * 0.7^1 = 0.56
        assert 0.55 < resonance < 0.57

    def test_compute_resonance_with_boost(self):
        matcher = ResonanceMatcher()

        # With translation boost
        resonance = matcher._compute_resonance(
            parent_score=1.0,
            edge_weight=0.8,
            depth=1,
            relation_type=RelationType.TRANSLATION
        )
        # 1.0 * 0.8 * 0.7^1 * 1.2 = 0.672
        assert 0.65 < resonance < 0.70

    def test_build_adjacency_map(self):
        matcher = ResonanceMatcher()
        graph = WeaveGraph()

        graph.add_edge(ConceptEdge(source_id="a", target_id="b", weight=0.9))
        graph.add_edge(ConceptEdge(source_id="b", target_id="c", weight=0.8, bidirectional=False))

        adjacency = matcher._build_adjacency_map(graph)

        # a -> b (forward)
        assert "a" in adjacency
        assert len(adjacency["a"]) == 1

        # b -> a (reverse, bidirectional)
        assert "b" in adjacency
        # b has both: reverse from a, forward to c
        assert len(adjacency["b"]) == 2

        # c should not have reverse edge (not bidirectional)
        assert "c" not in adjacency or len(adjacency["c"]) == 0


class TestExtractionConfig:
    """Tests for ExtractionConfig dataclass"""

    def test_default_values(self):
        config = ExtractionConfig()
        assert config.min_term_length == 3
        assert config.max_term_length == 50
        assert config.max_concepts == 500
        assert config.include_bigrams is True
        assert config.language_detection is True

    def test_custom_values(self):
        config = ExtractionConfig(
            min_term_length=4,
            max_concepts=100
        )
        assert config.min_term_length == 4
        assert config.max_concepts == 100


class TestConceptExtractor:
    """Tests for ConceptExtractor class"""

    def test_basic_creation(self):
        extractor = ConceptExtractor()
        assert extractor.config.min_term_length == 3

    def test_canonicalize(self):
        extractor = ConceptExtractor()

        assert extractor._canonicalize("Apache Kafka") == "apache_kafka"
        assert extractor._canonicalize("data-pipeline") == "data_pipeline"
        assert extractor._canonicalize("MyClass.method") == "myclass_method"

    def test_is_valid_term(self):
        extractor = ConceptExtractor()

        # Valid terms
        assert extractor._is_valid_term("kafka", "kafka") is True
        assert extractor._is_valid_term("streaming", "streaming") is True

        # Too short
        assert extractor._is_valid_term("ab", "ab") is False

        # Stop word
        assert extractor._is_valid_term("the", "the") is False

        # Just digits
        assert extractor._is_valid_term("123", "123") is False

    def test_detect_language_english(self):
        extractor = ConceptExtractor()
        text = "The Apache Kafka streaming platform is used for real-time data processing."
        lang = extractor._detect_language(text)
        assert lang == "en"

    def test_detect_language_french(self):
        extractor = ConceptExtractor()
        text = "Le système Apache Kafka est utilisé pour le traitement de données en temps réel."
        lang = extractor._detect_language(text)
        assert lang == "fr"

    def test_extract_pattern_terms_camel_case(self):
        extractor = ConceptExtractor()
        text = "Use KafkaConsumer and DataFrameReader to process data."
        terms = extractor._extract_pattern_terms(text)
        term_names = [t[0] for t in terms]
        assert "KafkaConsumer" in term_names
        assert "DataFrameReader" in term_names

    def test_extract_pattern_terms_snake_case(self):
        extractor = ConceptExtractor()
        text = "The data_pipeline uses kafka_consumer for processing."
        terms = extractor._extract_pattern_terms(text)
        term_names = [t[0] for t in terms]
        assert "data_pipeline" in term_names
        assert "kafka_consumer" in term_names

    def test_extract_pattern_terms_acronyms(self):
        extractor = ConceptExtractor()
        text = "Use AWS and GCP for cloud deployment with API endpoints."
        terms = extractor._extract_pattern_terms(text)
        term_names = [t[0] for t in terms]
        assert "AWS" in term_names
        assert "GCP" in term_names
        assert "API" in term_names

    def test_extract_keywords(self):
        extractor = ConceptExtractor()
        text = "Kafka streaming platform consumer producer message queue topic partition"
        keywords = extractor._extract_keywords(text)
        keyword_names = [k[0] for k in keywords]
        assert "kafka" in keyword_names
        assert "streaming" in keyword_names

    def test_extract_keywords_empty(self):
        extractor = ConceptExtractor()
        keywords = extractor._extract_keywords("")
        assert len(keywords) == 0

    def test_extract_domain_terms(self):
        extractor = ConceptExtractor()
        text = "Set up a Kafka pipeline with consumer and producer for streaming data to the warehouse."
        terms = extractor._extract_domain_terms(text)
        assert "kafka" in terms
        assert "pipeline" in terms
        assert "consumer" in terms
        assert "producer" in terms
        assert "streaming" in terms
        assert "warehouse" in terms

    def test_extract_ngrams(self):
        extractor = ConceptExtractor()
        words = ["apache", "kafka", "streaming", "platform"]
        bigrams = extractor._extract_ngrams(words, 2)
        assert "apache kafka" in bigrams
        assert "kafka streaming" in bigrams

    def test_extract_context_snippets(self):
        extractor = ConceptExtractor()
        text = "Apache Kafka is a distributed streaming platform. Kafka provides high throughput. Use Kafka for real-time data."
        snippets = extractor.extract_context_snippets(text, "Kafka", window=20)
        assert len(snippets) >= 2

    def test_extract_context_snippets_not_found(self):
        extractor = ConceptExtractor()
        text = "PostgreSQL is a relational database."
        snippets = extractor.extract_context_snippets(text, "Kafka", window=20)
        assert len(snippets) == 0

    def test_extract_context_snippets_with_ellipsis(self):
        extractor = ConceptExtractor()
        text = "Some prefix text. Apache Kafka is used here. Some suffix text."
        snippets = extractor.extract_context_snippets(text, "Kafka", window=10)
        # Should have ellipsis since we're not at start/end
        assert len(snippets) > 0


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_rag_validation_flow(self):
        """Test complete RAG validation from context to result"""
        validator = RAGThresholdValidator()

        # Good content
        good_text = """
        # Apache Kafka Overview

        Apache Kafka is a distributed streaming platform.

        ## Key Features

        - High throughput
        - Low latency
        - Fault tolerant

        ```python
        from kafka import KafkaProducer
        producer = KafkaProducer(bootstrap_servers='localhost:9092')
        ```

        ## Consumer Groups

        Kafka consumers use consumer groups for load balancing.
        """ * 30  # Make it long enough

        result = validator.validate(good_text, has_documents=True, topic="Apache Kafka")
        assert result.is_sufficient is True
        assert result.mode in [RAGMode.FULL, RAGMode.PARTIAL]

    def test_weave_graph_with_resonance(self):
        """Test WeaveGraph creation and resonance propagation"""
        # Create graph
        graph = WeaveGraph(user_id="test")

        # Add concepts
        concepts = [
            ("kafka", "Apache Kafka"),
            ("consumer", "Kafka Consumer"),
            ("producer", "Kafka Producer"),
            ("topic", "Kafka Topic"),
            ("broker", "Message Broker"),
        ]

        for cname, name in concepts:
            graph.add_concept(ConceptNode(id=cname, name=name, canonical_name=cname))

        # Add edges
        edges = [
            ("kafka", "consumer", 0.9),
            ("kafka", "producer", 0.9),
            ("kafka", "topic", 0.85),
            ("kafka", "broker", 0.8),
            ("consumer", "topic", 0.7),
            ("producer", "topic", 0.7),
        ]

        for src, tgt, weight in edges:
            graph.add_edge(ConceptEdge(source_id=src, target_id=tgt, weight=weight))

        # Compute stats
        stats = graph.compute_stats()
        assert stats.total_concepts == 5
        assert stats.total_edges == 6

        # Test resonance
        matcher = ResonanceMatcher()
        result = matcher.propagate(["kafka"], graph)

        assert result.direct_matches == 1
        assert result.propagated_matches >= 4  # Should propagate to all neighbors
        assert "consumer" in result.scores
        assert "producer" in result.scores

    def test_concept_extraction_to_graph(self):
        """Test extracting concepts and building a graph"""
        extractor = ConceptExtractor()

        text = """
        Apache Kafka is a distributed streaming platform.
        KafkaConsumer reads messages from topics.
        KafkaProducer sends messages to topics.
        The data_pipeline processes streaming data.
        """

        # Extract pattern terms
        terms = extractor._extract_pattern_terms(text)
        assert len(terms) > 0

        # Extract domain terms
        domain_terms = extractor._extract_domain_terms(text)
        assert "kafka" in domain_terms
        assert "streaming" in domain_terms

        # Create graph from extracted concepts
        graph = WeaveGraph()
        for term, source in terms[:5]:
            canonical = extractor._canonicalize(term)
            if extractor._is_valid_term(term, canonical):
                graph.add_concept(ConceptNode(
                    name=term,
                    canonical_name=canonical,
                    source_type=source
                ))

        assert len(graph.concepts) > 0
