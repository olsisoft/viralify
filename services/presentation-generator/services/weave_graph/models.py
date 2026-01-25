"""
WeaveGraph Models

Data models for the concept graph system that discovers
semantic relationships between terms in documents.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
from datetime import datetime
import uuid


class RelationType(str, Enum):
    """Types of relationships between concepts"""
    SIMILAR = "similar"           # Semantic similarity (embedding-based)
    TRANSLATION = "translation"   # Cross-language equivalent
    PART_OF = "part_of"          # Concept is part of another
    PREREQUISITE = "prerequisite" # Concept requires another
    RELATED = "related"          # General relation
    SYNONYM = "synonym"          # Same meaning
    HYPERNYM = "hypernym"        # More general concept
    HYPONYM = "hyponym"          # More specific concept


class ConceptSource(str, Enum):
    """How the concept was extracted"""
    NLP_EXTRACTION = "nlp"        # spaCy/regex extraction
    KEYWORD = "keyword"           # TF-IDF keyword
    ENTITY = "entity"             # Named entity
    TECHNICAL_TERM = "technical"  # Domain-specific term
    USER_DEFINED = "user"         # Manually added
    LLM_ENRICHED = "llm"          # LLM-enhanced


@dataclass
class ConceptNode:
    """
    A concept in the WeaveGraph.

    Represents a single concept extracted from documents,
    with its embedding for similarity search.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""                    # Original form: "Apache Kafka"
    canonical_name: str = ""          # Normalized: "apache_kafka"
    language: str = "en"              # Detected language
    embedding: Optional[List[float]] = None  # E5-large 1024-dim
    source_document_ids: List[str] = field(default_factory=list)
    frequency: int = 1                # How often it appears
    source_type: ConceptSource = ConceptSource.NLP_EXTRACTION
    aliases: List[str] = field(default_factory=list)  # Alternative names
    context_snippets: List[str] = field(default_factory=list)  # Where it appears
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __hash__(self):
        return hash(self.canonical_name)

    def __eq__(self, other):
        if isinstance(other, ConceptNode):
            return self.canonical_name == other.canonical_name
        return False


@dataclass
class ConceptEdge:
    """
    An edge connecting two concepts in the WeaveGraph.

    Represents a relationship discovered between concepts,
    either through embedding similarity or explicit extraction.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""               # Source concept ID
    target_id: str = ""               # Target concept ID
    relation_type: RelationType = RelationType.SIMILAR
    weight: float = 1.0               # Strength of relationship (0-1)
    bidirectional: bool = True        # True for similarity, False for part_of
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WeaveGraphStats:
    """Statistics about a WeaveGraph"""
    total_concepts: int = 0
    total_edges: int = 0
    avg_connections_per_concept: float = 0.0
    languages: List[str] = field(default_factory=list)
    top_concepts: List[str] = field(default_factory=list)  # By frequency
    edge_type_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class ConceptCluster:
    """
    A cluster of related concepts.

    Concepts that are highly interconnected form clusters,
    useful for understanding topic structure.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""                    # Auto-generated from top concept
    concepts: List[ConceptNode] = field(default_factory=list)
    centroid_embedding: Optional[List[float]] = None
    coherence_score: float = 0.0      # How tightly connected


@dataclass
class QueryExpansion:
    """
    Result of expanding a query using WeaveGraph.

    Given a query term, returns related concepts that
    should be included in the search.
    """
    original_query: str = ""
    expanded_terms: List[str] = field(default_factory=list)
    expansion_paths: Dict[str, List[str]] = field(default_factory=dict)  # term -> path
    total_weight: float = 0.0         # Cumulative edge weights
    languages_covered: Set[str] = field(default_factory=set)


@dataclass
class WeaveGraph:
    """
    The complete concept graph for a document set.

    Contains all concepts and their relationships,
    with methods for traversal and query expansion.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    document_ids: List[str] = field(default_factory=list)
    concepts: Dict[str, ConceptNode] = field(default_factory=dict)  # id -> node
    edges: List[ConceptEdge] = field(default_factory=list)
    clusters: List[ConceptCluster] = field(default_factory=list)
    stats: WeaveGraphStats = field(default_factory=WeaveGraphStats)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_concept(self, concept: ConceptNode) -> None:
        """Add a concept to the graph"""
        self.concepts[concept.id] = concept

    def add_edge(self, edge: ConceptEdge) -> None:
        """Add an edge to the graph"""
        self.edges.append(edge)

    def get_neighbors(self, concept_id: str, max_depth: int = 1) -> List[ConceptNode]:
        """Get neighboring concepts up to max_depth hops"""
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
        """Find a concept by its name or canonical name"""
        name_lower = name.lower().strip()
        for concept in self.concepts.values():
            if concept.name.lower() == name_lower or concept.canonical_name == name_lower:
                return concept
            if name_lower in [a.lower() for a in concept.aliases]:
                return concept
        return None

    def compute_stats(self) -> WeaveGraphStats:
        """Compute and return graph statistics"""
        if not self.concepts:
            return WeaveGraphStats()

        # Count connections per concept
        connection_counts = {cid: 0 for cid in self.concepts}
        edge_types = {}

        for edge in self.edges:
            if edge.source_id in connection_counts:
                connection_counts[edge.source_id] += 1
            if edge.bidirectional and edge.target_id in connection_counts:
                connection_counts[edge.target_id] += 1

            edge_types[edge.relation_type.value] = edge_types.get(edge.relation_type.value, 0) + 1

        # Get languages
        languages = list(set(c.language for c in self.concepts.values()))

        # Top concepts by frequency
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
