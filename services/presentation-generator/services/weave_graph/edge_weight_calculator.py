"""
Edge Weight Calculator for WeaveGraph.

Computes edge weights using three complementary signals:
1. Co-occurrence: PMI-based proximity in document chunks
2. Hierarchy: Taxonomic relationships from TECH_DOMAINS
3. Embedding Similarity: Cosine similarity between concept embeddings

The final weight is a weighted fusion of all three signals.
"""

import math
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EdgeWeightConfig:
    """Configuration for edge weight calculation."""

    # Weights for each signal (should sum to 1.0)
    cooccurrence_weight: float = 0.4
    hierarchy_weight: float = 0.3
    embedding_weight: float = 0.3

    # Co-occurrence settings
    window_size: int = 1  # 1 = same chunk, 2 = adjacent chunks too
    min_cooccurrence: int = 1  # Minimum co-occurrence count
    pmi_smoothing: float = 1.0  # Laplace smoothing

    # Hierarchy settings
    parent_child_score: float = 1.0  # Direct parent-child relationship
    same_domain_score: float = 0.5   # Same domain siblings
    related_domain_score: float = 0.25  # Related domains

    # Embedding settings
    min_similarity: float = 0.3  # Minimum cosine similarity to consider

    # Edge filtering
    min_edge_weight: float = 0.1  # Minimum weight to create edge

    def __post_init__(self):
        # Normalize weights to sum to 1.0
        total = self.cooccurrence_weight + self.hierarchy_weight + self.embedding_weight
        if abs(total - 1.0) > 0.01:
            self.cooccurrence_weight /= total
            self.hierarchy_weight /= total
            self.embedding_weight /= total


# =============================================================================
# Tech Domain Hierarchy
# =============================================================================

TECH_HIERARCHY: Dict[str, Dict[str, Any]] = {
    # Data & Analytics
    "data": {
        "parent": None,
        "children": ["data_engineering", "data_science", "analytics", "business_intelligence"],
        "concepts": ["dataset", "schema", "table", "column", "row", "record"]
    },
    "data_engineering": {
        "parent": "data",
        "children": ["etl", "elt", "data_pipeline", "data_warehouse", "data_lake"],
        "concepts": ["ingestion", "transformation", "orchestration", "batch", "streaming"]
    },
    "data_pipeline": {
        "parent": "data_engineering",
        "children": [],
        "concepts": ["dag", "task", "operator", "scheduler", "dependency"]
    },
    "data_warehouse": {
        "parent": "data_engineering",
        "children": [],
        "concepts": ["dimension", "fact", "star_schema", "snowflake", "olap"]
    },
    "data_lake": {
        "parent": "data_engineering",
        "children": [],
        "concepts": ["raw_zone", "curated_zone", "delta", "iceberg", "parquet"]
    },

    # Messaging & Streaming
    "messaging": {
        "parent": "distributed_systems",
        "children": ["kafka", "rabbitmq", "redis_pubsub", "sqs", "pubsub"],
        "concepts": ["message", "queue", "topic", "consumer", "producer", "broker", "partition"]
    },
    "kafka": {
        "parent": "messaging",
        "children": [],
        "concepts": ["consumer_group", "offset", "replication", "zookeeper", "kraft"]
    },
    "rabbitmq": {
        "parent": "messaging",
        "children": [],
        "concepts": ["exchange", "routing_key", "binding", "acknowledgment", "dead_letter"]
    },

    # Distributed Systems
    "distributed_systems": {
        "parent": None,
        "children": ["messaging", "microservices", "containers"],
        "concepts": ["scalability", "availability", "partition_tolerance", "consensus", "replication"]
    },
    "microservices": {
        "parent": "distributed_systems",
        "children": ["api_gateway", "service_mesh"],
        "concepts": ["service", "endpoint", "api", "rest", "grpc", "circuit_breaker"]
    },
    "api_gateway": {
        "parent": "microservices",
        "children": [],
        "concepts": ["routing", "rate_limiting", "authentication", "load_balancing"]
    },

    # Containers & Orchestration
    "containers": {
        "parent": "distributed_systems",
        "children": ["docker", "kubernetes"],
        "concepts": ["container", "image", "registry", "volume", "network"]
    },
    "docker": {
        "parent": "containers",
        "children": [],
        "concepts": ["dockerfile", "compose", "layer", "build", "push", "pull"]
    },
    "kubernetes": {
        "parent": "containers",
        "children": [],
        "concepts": ["pod", "deployment", "service", "ingress", "configmap", "secret", "namespace"]
    },

    # Machine Learning
    "machine_learning": {
        "parent": None,
        "children": ["deep_learning", "nlp", "computer_vision", "mlops"],
        "concepts": ["model", "training", "inference", "feature", "label", "prediction"]
    },
    "deep_learning": {
        "parent": "machine_learning",
        "children": ["neural_network", "transformer"],
        "concepts": ["layer", "neuron", "activation", "backpropagation", "gradient"]
    },
    "neural_network": {
        "parent": "deep_learning",
        "children": [],
        "concepts": ["cnn", "rnn", "lstm", "attention", "encoder", "decoder"]
    },
    "nlp": {
        "parent": "machine_learning",
        "children": [],
        "concepts": ["tokenization", "embedding", "bert", "gpt", "transformer", "attention"]
    },

    # Cloud Providers
    "cloud": {
        "parent": None,
        "children": ["aws", "azure", "gcp"],
        "concepts": ["region", "availability_zone", "vpc", "iam", "storage", "compute"]
    },
    "aws": {
        "parent": "cloud",
        "children": [],
        "concepts": ["ec2", "s3", "lambda", "rds", "dynamodb", "sqs", "sns", "kinesis"]
    },
    "azure": {
        "parent": "cloud",
        "children": [],
        "concepts": ["vm", "blob", "functions", "cosmosdb", "event_hub", "service_bus"]
    },
    "gcp": {
        "parent": "cloud",
        "children": [],
        "concepts": ["compute_engine", "gcs", "bigquery", "pubsub", "dataflow", "vertex"]
    },

    # Databases
    "database": {
        "parent": None,
        "children": ["sql", "nosql"],
        "concepts": ["query", "index", "transaction", "acid", "replication", "sharding"]
    },
    "sql": {
        "parent": "database",
        "children": ["postgresql", "mysql"],
        "concepts": ["select", "join", "where", "group_by", "order_by", "constraint"]
    },
    "postgresql": {
        "parent": "sql",
        "children": [],
        "concepts": ["pgvector", "jsonb", "extension", "vacuum", "wal"]
    },
    "nosql": {
        "parent": "database",
        "children": ["mongodb", "redis", "elasticsearch"],
        "concepts": ["document", "key_value", "graph", "column_family"]
    },
    "redis": {
        "parent": "nosql",
        "children": [],
        "concepts": ["cache", "pub_sub", "sorted_set", "hash", "expire", "cluster"]
    },

    # Programming
    "programming": {
        "parent": None,
        "children": ["python", "javascript", "go", "rust"],
        "concepts": ["variable", "function", "class", "module", "package", "library"]
    },
    "python": {
        "parent": "programming",
        "children": [],
        "concepts": ["pip", "venv", "decorator", "generator", "async", "pandas", "numpy"]
    },
}

# Build reverse lookup for faster queries
_CONCEPT_TO_DOMAIN: Dict[str, str] = {}
_DOMAIN_ANCESTORS: Dict[str, List[str]] = {}

def _build_hierarchy_lookups():
    """Build lookup tables for fast hierarchy queries."""
    global _CONCEPT_TO_DOMAIN, _DOMAIN_ANCESTORS

    # Map concepts to their domains
    for domain, info in TECH_HIERARCHY.items():
        for concept in info.get("concepts", []):
            _CONCEPT_TO_DOMAIN[concept.lower()] = domain
        for child in info.get("children", []):
            _CONCEPT_TO_DOMAIN[child.lower()] = domain
        _CONCEPT_TO_DOMAIN[domain.lower()] = domain

    # Build ancestor chains
    for domain in TECH_HIERARCHY:
        ancestors = []
        current = domain
        while current:
            parent = TECH_HIERARCHY.get(current, {}).get("parent")
            if parent:
                ancestors.append(parent)
            current = parent
        _DOMAIN_ANCESTORS[domain] = ancestors

_build_hierarchy_lookups()


# =============================================================================
# Co-occurrence Calculator
# =============================================================================

class CooccurrenceCalculator:
    """
    Calculates co-occurrence scores using PMI (Pointwise Mutual Information).

    Two concepts that appear together in the same chunk more often than
    expected by chance will have a high PMI score.
    """

    def __init__(self, config: EdgeWeightConfig):
        self.config = config
        self._cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._concept_counts: Dict[str, int] = defaultdict(int)
        self._total_windows: int = 0
        self._is_trained: bool = False

    def train(self, chunks: List[str], concept_extractor=None) -> None:
        """
        Build co-occurrence matrix from document chunks.

        Args:
            chunks: List of text chunks (paragraphs, sections)
            concept_extractor: Optional extractor to get concepts from text
        """
        self._cooccurrence.clear()
        self._concept_counts.clear()
        self._total_windows = 0

        # Extract concepts from each chunk
        chunk_concepts: List[Set[str]] = []
        for chunk in chunks:
            if concept_extractor:
                concepts = concept_extractor.extract_concepts(chunk)
                concept_set = {c.canonical_name for c in concepts}
            else:
                # Simple word extraction as fallback
                words = chunk.lower().split()
                concept_set = {w for w in words if len(w) > 2}
            chunk_concepts.append(concept_set)

        # Build co-occurrence with window
        for i, concepts in enumerate(chunk_concepts):
            # Expand window
            window_concepts = set(concepts)
            for j in range(max(0, i - self.config.window_size),
                          min(len(chunk_concepts), i + self.config.window_size + 1)):
                if i != j:
                    window_concepts.update(chunk_concepts[j])

            # Count individual concepts
            for concept in concepts:
                self._concept_counts[concept] += 1

            # Count co-occurrences (pairs in same window)
            for c1, c2 in combinations(sorted(window_concepts), 2):
                self._cooccurrence[c1][c2] += 1
                self._cooccurrence[c2][c1] += 1

            self._total_windows += 1

        self._is_trained = True
        logger.info(f"[COOC] Trained on {len(chunks)} chunks, "
                   f"{len(self._concept_counts)} concepts, "
                   f"{sum(len(v) for v in self._cooccurrence.values()) // 2} pairs")

    def get_cooccurrence_count(self, concept_a: str, concept_b: str) -> int:
        """Get raw co-occurrence count between two concepts."""
        a, b = concept_a.lower(), concept_b.lower()
        return self._cooccurrence.get(a, {}).get(b, 0)

    def calculate_pmi(self, concept_a: str, concept_b: str) -> float:
        """
        Calculate PMI between two concepts.

        PMI(x,y) = log2(P(x,y) / (P(x) * P(y)))

        Returns normalized PMI in range [0, 1].
        """
        if not self._is_trained or self._total_windows == 0:
            return 0.0

        a, b = concept_a.lower(), concept_b.lower()

        # Get counts with smoothing
        cooc = self._cooccurrence.get(a, {}).get(b, 0) + self.config.pmi_smoothing
        count_a = self._concept_counts.get(a, 0) + self.config.pmi_smoothing
        count_b = self._concept_counts.get(b, 0) + self.config.pmi_smoothing
        total = self._total_windows + self.config.pmi_smoothing * len(self._concept_counts)

        if count_a == 0 or count_b == 0:
            return 0.0

        # Calculate probabilities
        p_ab = cooc / total
        p_a = count_a / total
        p_b = count_b / total

        # PMI
        pmi = math.log2(p_ab / (p_a * p_b)) if p_a * p_b > 0 else 0.0

        # Normalize to [0, 1] using max possible PMI
        # Max PMI = -log2(p_min) where p_min = 1/total
        max_pmi = math.log2(total) if total > 0 else 1.0
        normalized_pmi = max(0.0, min(1.0, (pmi + max_pmi) / (2 * max_pmi)))

        return normalized_pmi

    def get_score(self, concept_a: str, concept_b: str) -> float:
        """
        Get co-occurrence score between two concepts.

        Returns score in range [0, 1].
        """
        if not self._is_trained:
            return 0.0

        # Check minimum co-occurrence
        cooc_count = self.get_cooccurrence_count(concept_a, concept_b)
        if cooc_count < self.config.min_cooccurrence:
            return 0.0

        return self.calculate_pmi(concept_a, concept_b)


# =============================================================================
# Hierarchy Resolver
# =============================================================================

class HierarchyResolver:
    """
    Resolves hierarchical relationships between concepts using TECH_HIERARCHY.

    Assigns scores based on:
    - Parent-child relationships (highest)
    - Same domain siblings (medium)
    - Related domains (low)
    """

    def __init__(self, config: EdgeWeightConfig):
        self.config = config

    def get_domain(self, concept: str) -> Optional[str]:
        """Get the domain a concept belongs to."""
        return _CONCEPT_TO_DOMAIN.get(concept.lower())

    def get_ancestors(self, domain: str) -> List[str]:
        """Get all ancestor domains."""
        return _DOMAIN_ANCESTORS.get(domain, [])

    def is_parent_child(self, concept_a: str, concept_b: str) -> bool:
        """Check if one concept is parent of the other."""
        a, b = concept_a.lower(), concept_b.lower()

        # Check if a is parent of b
        domain_a = self.get_domain(a)
        domain_b = self.get_domain(b)

        if domain_a and domain_b:
            # Direct parent-child via hierarchy
            if domain_a in self.get_ancestors(domain_b):
                return True
            if domain_b in self.get_ancestors(domain_a):
                return True

            # Check if a is in children of b's domain
            info_b = TECH_HIERARCHY.get(domain_b, {})
            if a in [c.lower() for c in info_b.get("children", [])]:
                return True

            # Check if b is in children of a's domain
            info_a = TECH_HIERARCHY.get(domain_a, {})
            if b in [c.lower() for c in info_a.get("children", [])]:
                return True

        return False

    def is_same_domain(self, concept_a: str, concept_b: str) -> bool:
        """Check if two concepts are in the same domain."""
        domain_a = self.get_domain(concept_a.lower())
        domain_b = self.get_domain(concept_b.lower())

        if domain_a and domain_b:
            return domain_a == domain_b
        return False

    def are_related_domains(self, concept_a: str, concept_b: str) -> bool:
        """Check if two concepts are in related (sibling) domains."""
        domain_a = self.get_domain(concept_a.lower())
        domain_b = self.get_domain(concept_b.lower())

        if domain_a and domain_b and domain_a != domain_b:
            # Check if they share a parent
            ancestors_a = set(self.get_ancestors(domain_a))
            ancestors_b = set(self.get_ancestors(domain_b))

            # Share immediate parent
            parent_a = TECH_HIERARCHY.get(domain_a, {}).get("parent")
            parent_b = TECH_HIERARCHY.get(domain_b, {}).get("parent")

            if parent_a and parent_a == parent_b:
                return True

            # Share any ancestor
            if ancestors_a & ancestors_b:
                return True

        return False

    def get_score(self, concept_a: str, concept_b: str) -> float:
        """
        Get hierarchy score between two concepts.

        Returns score in range [0, 1].
        """
        if self.is_parent_child(concept_a, concept_b):
            return self.config.parent_child_score

        if self.is_same_domain(concept_a, concept_b):
            return self.config.same_domain_score

        if self.are_related_domains(concept_a, concept_b):
            return self.config.related_domain_score

        return 0.0


# =============================================================================
# Embedding Similarity Calculator
# =============================================================================

class EmbeddingSimilarityCalculator:
    """
    Calculates cosine similarity between concept embeddings.
    """

    def __init__(self, config: EdgeWeightConfig):
        self.config = config
        self._embeddings: Dict[str, List[float]] = {}

    def set_embeddings(self, embeddings: Dict[str, List[float]]) -> None:
        """Set pre-computed embeddings for concepts."""
        self._embeddings = {k.lower(): v for k, v in embeddings.items()}

    def add_embedding(self, concept: str, embedding: List[float]) -> None:
        """Add embedding for a single concept."""
        self._embeddings[concept.lower()] = embedding

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_score(self, concept_a: str, concept_b: str) -> float:
        """
        Get embedding similarity score between two concepts.

        Returns score in range [0, 1].
        """
        a, b = concept_a.lower(), concept_b.lower()

        emb_a = self._embeddings.get(a)
        emb_b = self._embeddings.get(b)

        if emb_a is None or emb_b is None:
            return 0.0

        similarity = self.cosine_similarity(emb_a, emb_b)

        # Apply minimum threshold
        if similarity < self.config.min_similarity:
            return 0.0

        # Normalize to [0, 1] from [min_similarity, 1]
        normalized = (similarity - self.config.min_similarity) / (1.0 - self.config.min_similarity)
        return max(0.0, min(1.0, normalized))


# =============================================================================
# Edge Weight Calculator (Main Class)
# =============================================================================

@dataclass
class EdgeWeight:
    """Result of edge weight calculation."""
    source: str
    target: str
    weight: float
    cooccurrence_score: float
    hierarchy_score: float
    embedding_score: float

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "components": {
                "cooccurrence": self.cooccurrence_score,
                "hierarchy": self.hierarchy_score,
                "embedding": self.embedding_score
            }
        }


class EdgeWeightCalculator:
    """
    Main class that combines all three signals to calculate edge weights.

    Usage:
        calculator = EdgeWeightCalculator()
        calculator.train_cooccurrence(chunks)
        calculator.set_embeddings(embeddings)

        weight = calculator.calculate_weight("kafka", "consumer")
        # Returns EdgeWeight with combined score
    """

    def __init__(self, config: Optional[EdgeWeightConfig] = None):
        self.config = config or EdgeWeightConfig()
        self.cooccurrence = CooccurrenceCalculator(self.config)
        self.hierarchy = HierarchyResolver(self.config)
        self.embedding = EmbeddingSimilarityCalculator(self.config)

    def train_cooccurrence(self, chunks: List[str], concept_extractor=None) -> None:
        """Train the co-occurrence calculator on document chunks."""
        self.cooccurrence.train(chunks, concept_extractor)

    def set_embeddings(self, embeddings: Dict[str, List[float]]) -> None:
        """Set concept embeddings for similarity calculation."""
        self.embedding.set_embeddings(embeddings)

    def add_embedding(self, concept: str, embedding: List[float]) -> None:
        """Add embedding for a single concept."""
        self.embedding.add_embedding(concept, embedding)

    def calculate_weight(self, concept_a: str, concept_b: str) -> EdgeWeight:
        """
        Calculate the combined edge weight between two concepts.

        Returns EdgeWeight with individual scores and combined weight.
        """
        # Get individual scores
        cooc_score = self.cooccurrence.get_score(concept_a, concept_b)
        hier_score = self.hierarchy.get_score(concept_a, concept_b)
        emb_score = self.embedding.get_score(concept_a, concept_b)

        # Weighted fusion
        combined = (
            self.config.cooccurrence_weight * cooc_score +
            self.config.hierarchy_weight * hier_score +
            self.config.embedding_weight * emb_score
        )

        return EdgeWeight(
            source=concept_a,
            target=concept_b,
            weight=combined,
            cooccurrence_score=cooc_score,
            hierarchy_score=hier_score,
            embedding_score=emb_score
        )

    def should_create_edge(self, concept_a: str, concept_b: str) -> Tuple[bool, EdgeWeight]:
        """
        Determine if an edge should be created between two concepts.

        Returns (should_create, edge_weight).
        """
        weight = self.calculate_weight(concept_a, concept_b)
        should_create = weight.weight >= self.config.min_edge_weight
        return should_create, weight

    def build_weighted_edges(
        self,
        concepts: List[str],
        max_edges_per_concept: int = 10
    ) -> List[EdgeWeight]:
        """
        Build weighted edges for all concept pairs.

        Args:
            concepts: List of concept names
            max_edges_per_concept: Maximum edges per concept (top-k by weight)

        Returns:
            List of EdgeWeight objects for valid edges
        """
        edges: List[EdgeWeight] = []
        concept_edge_counts: Dict[str, int] = defaultdict(int)

        # Calculate all pair weights
        all_weights: List[EdgeWeight] = []
        for c1, c2 in combinations(concepts, 2):
            should_create, weight = self.should_create_edge(c1, c2)
            if should_create:
                all_weights.append(weight)

        # Sort by weight descending
        all_weights.sort(key=lambda w: w.weight, reverse=True)

        # Select top edges respecting max per concept
        for weight in all_weights:
            if (concept_edge_counts[weight.source] < max_edges_per_concept and
                concept_edge_counts[weight.target] < max_edges_per_concept):
                edges.append(weight)
                concept_edge_counts[weight.source] += 1
                concept_edge_counts[weight.target] += 1

        logger.info(f"[EDGE_WEIGHT] Built {len(edges)} weighted edges from {len(concepts)} concepts")
        return edges


# =============================================================================
# Convenience Functions
# =============================================================================

def create_edge_weight_calculator(
    cooccurrence_weight: float = 0.4,
    hierarchy_weight: float = 0.3,
    embedding_weight: float = 0.3,
    **kwargs
) -> EdgeWeightCalculator:
    """
    Create an EdgeWeightCalculator with custom weights.

    Args:
        cooccurrence_weight: Weight for co-occurrence signal
        hierarchy_weight: Weight for hierarchy signal
        embedding_weight: Weight for embedding signal
        **kwargs: Additional EdgeWeightConfig parameters

    Returns:
        Configured EdgeWeightCalculator
    """
    config = EdgeWeightConfig(
        cooccurrence_weight=cooccurrence_weight,
        hierarchy_weight=hierarchy_weight,
        embedding_weight=embedding_weight,
        **kwargs
    )
    return EdgeWeightCalculator(config)
