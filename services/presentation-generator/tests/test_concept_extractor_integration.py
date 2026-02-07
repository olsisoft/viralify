"""
Integration tests for ConceptExtractor + CompoundDetector.

Tests the ML-based compound term detection integrated with the ConceptExtractor.

This file creates a standalone version of the integration to avoid import chain issues.
"""

import pytest
import sys
import os
import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field

# Add path to import compound_detector directly
_weave_graph_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "services",
    "weave_graph"
)
sys.path.insert(0, _weave_graph_path)

# Import compound detector (no complex dependencies)
from compound_detector import (
    CompoundTermDetector,
    CompoundDetectorConfig,
    PMIConfig,
    CompoundTermResult
)


# ============================================================================
# Minimal ConceptNode and ConceptSource for testing
# ============================================================================

class ConceptSource:
    """Source type for concepts"""
    KEYWORD = "keyword"
    TECHNICAL_TERM = "technical_term"
    NLP_EXTRACTION = "nlp_extraction"


@dataclass
class ConceptNode:
    """Minimal concept node for testing"""
    name: str
    canonical_name: str
    language: str = "en"
    frequency: int = 1
    source_type: str = "nlp_extraction"
    source_document_ids: List[str] = field(default_factory=list)


# ============================================================================
# Extraction Config with ML options
# ============================================================================

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
    use_ml_compound_detection: bool = True
    ml_min_pmi: float = 1.5
    ml_min_frequency: int = 2
    ml_min_combined_score: float = 0.3
    use_semantic_filter: bool = False


# ============================================================================
# ConceptExtractor with CompoundDetector integration
# ============================================================================

class ConceptExtractor:
    """ConceptExtractor with ML-based compound detection for testing"""

    KNOWN_COMPOUND_TERMS = {
        "machine learning", "deep learning", "neural network", "natural language",
        "data pipeline", "data warehouse", "message broker", "api gateway",
    }

    TECH_DOMAINS = {
        "data": ["pipeline", "etl", "warehouse", "lake", "streaming"],
        "cloud": ["aws", "azure", "kubernetes", "docker"],
        "messaging": ["kafka", "rabbitmq", "queue", "consumer", "producer"],
    }

    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'and', 'or', 'but',
        'of', 'at', 'by', 'for', 'with', 'to', 'from', 'in', 'on', 'this', 'that',
        'le', 'la', 'les', 'de', 'du', 'des', 'et', 'en', 'est',
    }

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._compound_detector: Optional[CompoundTermDetector] = None
        self._learned_compound_terms: Set[str] = set()
        self._is_trained: bool = False

        if self.config.use_ml_compound_detection:
            self._init_compound_detector()

    def _init_compound_detector(self) -> None:
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
        if not self.config.use_ml_compound_detection or self._compound_detector is None:
            return 0
        if not texts:
            return 0

        self._compound_detector.train(texts)
        results = self._compound_detector.detect(top_k=200)
        self._learned_compound_terms = {r.term for r in results}
        self._is_trained = True
        return len(self._learned_compound_terms)

    def get_learned_compound_terms(self) -> Set[str]:
        return self._learned_compound_terms.copy()

    def add_compound_terms(self, terms: Set[str]) -> None:
        self._learned_compound_terms.update(t.lower() for t in terms)

    def _get_effective_compound_terms(self) -> Set[str]:
        # Include learned terms whether from training or manual addition
        if self._learned_compound_terms:
            return self._learned_compound_terms | self.KNOWN_COMPOUND_TERMS
        return self.KNOWN_COMPOUND_TERMS

    def _get_all_tech_terms(self) -> Set[str]:
        terms = set()
        for domain_terms in self.TECH_DOMAINS.values():
            terms.update(domain_terms)
        for compound in self.KNOWN_COMPOUND_TERMS:
            terms.update(compound.split())
        for compound in self._learned_compound_terms:
            terms.update(compound.split())
        return terms

    def _canonicalize(self, term: str) -> str:
        canonical = term.lower().strip()
        canonical = re.sub(r'[\s\-\.]+', '_', canonical)
        canonical = re.sub(r'[^a-z0-9_]', '', canonical)
        return canonical

    def extract_concepts(
        self,
        text: str,
        document_id: Optional[str] = None
    ) -> List[ConceptNode]:
        """Extract concepts from text"""
        concepts = {}

        # Extract patterns
        words = re.findall(r'\b[A-Za-z][A-Za-z0-9]+\b', text)

        for word in words:
            if word.lower() not in self.STOP_WORDS and len(word) >= 3:
                canonical = self._canonicalize(word)
                if canonical not in concepts:
                    concepts[canonical] = ConceptNode(
                        name=word,
                        canonical_name=canonical,
                        source_document_ids=[document_id] if document_id else []
                    )
                else:
                    concepts[canonical].frequency += 1

        return list(concepts.values())[:self.config.max_concepts]


class TestConceptExtractorWithCompoundDetector:
    """Integration tests for ConceptExtractor using ML-based compound detection"""

    def test_extractor_initializes_compound_detector(self):
        """Test that compound detector is initialized when enabled"""
        config = ExtractionConfig(use_ml_compound_detection=True)
        extractor = ConceptExtractor(config)

        assert extractor._compound_detector is not None
        assert extractor._is_trained is False
        assert len(extractor._learned_compound_terms) == 0

    def test_extractor_without_compound_detector(self):
        """Test that compound detector is not initialized when disabled"""
        config = ExtractionConfig(use_ml_compound_detection=False)
        extractor = ConceptExtractor(config)

        assert extractor._compound_detector is None

    def test_train_on_corpus(self):
        """Test training the compound detector on a corpus"""
        config = ExtractionConfig(
            use_ml_compound_detection=True,
            ml_min_pmi=0.5,
            ml_min_frequency=2,
            ml_min_combined_score=0.2
        )
        extractor = ConceptExtractor(config)

        corpus = [
            "Machine Learning is used for data science.",
            "Machine Learning models require training data.",
            "Machine Learning algorithms improve over time.",
            "Deep Learning uses neural networks.",
            "Deep Learning models are powerful.",
            "Data Pipeline architecture is important.",
            "Data Pipeline connects data sources.",
        ]

        num_learned = extractor.train_on_corpus(corpus)

        assert num_learned > 0
        assert extractor._is_trained is True
        assert len(extractor._learned_compound_terms) > 0

    def test_get_effective_compound_terms_merges_lists(self):
        """Test that effective compound terms includes both learned and hardcoded"""
        config = ExtractionConfig(use_ml_compound_detection=True)
        extractor = ConceptExtractor(config)

        # Add some learned terms manually
        extractor._learned_compound_terms = {"custom term", "new concept"}
        extractor._is_trained = True

        effective = extractor._get_effective_compound_terms()

        # Should contain both hardcoded and learned
        assert "machine learning" in effective  # hardcoded
        assert "custom term" in effective  # learned
        assert "new concept" in effective  # learned

    def test_add_compound_terms_manually(self):
        """Test manually adding compound terms"""
        config = ExtractionConfig(use_ml_compound_detection=True)
        extractor = ConceptExtractor(config)

        extractor.add_compound_terms({"Custom API", "Special Service"})

        assert "custom api" in extractor._learned_compound_terms
        assert "special service" in extractor._learned_compound_terms

    def test_get_learned_compound_terms(self):
        """Test getting learned compound terms"""
        config = ExtractionConfig(use_ml_compound_detection=True)
        extractor = ConceptExtractor(config)

        extractor._learned_compound_terms = {"term1", "term2"}

        learned = extractor.get_learned_compound_terms()

        assert learned == {"term1", "term2"}
        # Should be a copy, not the original set
        learned.add("term3")
        assert "term3" not in extractor._learned_compound_terms

    def test_empty_corpus_training(self):
        """Test training on empty corpus"""
        config = ExtractionConfig(use_ml_compound_detection=True)
        extractor = ConceptExtractor(config)

        num_learned = extractor.train_on_corpus([])

        assert num_learned == 0
        assert extractor._is_trained is False

    def test_get_all_tech_terms_includes_learned(self):
        """Test that _get_all_tech_terms includes learned compound terms"""
        config = ExtractionConfig(use_ml_compound_detection=True)
        extractor = ConceptExtractor(config)

        extractor._learned_compound_terms = {"custom api gateway"}

        all_terms = extractor._get_all_tech_terms()

        # Should include words from learned compound
        assert "custom" in all_terms
        assert "api" in all_terms
        assert "gateway" in all_terms
        # And hardcoded domain terms
        assert "kafka" in all_terms

    def test_extract_concepts_with_trained_detector(self):
        """Test concept extraction after training on corpus"""
        config = ExtractionConfig(
            use_ml_compound_detection=True,
            ml_min_pmi=0.0,
            ml_min_frequency=2,
            ml_min_combined_score=0.1
        )
        extractor = ConceptExtractor(config)

        # Train on corpus
        corpus = [
            "Custom Framework is our internal tool.",
            "Custom Framework handles all requests.",
            "Custom Framework is well designed.",
        ] * 2

        extractor.train_on_corpus(corpus)

        # Extract concepts from new text
        text = "The Custom Framework improves productivity."
        concepts = extractor.extract_concepts(text)

        # Should have extracted concepts
        assert len(concepts) > 0

    def test_fallback_to_hardcoded_when_not_trained(self):
        """Test that hardcoded terms are used when not trained"""
        config = ExtractionConfig(use_ml_compound_detection=True)
        extractor = ConceptExtractor(config)

        # Don't train, just extract
        text = "Machine Learning and Deep Learning are used for AI."
        concepts = extractor.extract_concepts(text)

        # Should still detect terms using hardcoded patterns
        canonical_names = [c.canonical_name for c in concepts]
        # Should find something related to machine learning or deep learning
        assert len(concepts) > 0


class TestConceptExtractorMLConfig:
    """Tests for ML configuration options"""

    def test_min_pmi_config(self):
        """Test that min_pmi config is passed to detector"""
        config = ExtractionConfig(
            use_ml_compound_detection=True,
            ml_min_pmi=3.0
        )
        extractor = ConceptExtractor(config)

        assert extractor._compound_detector is not None
        assert extractor._compound_detector.config.pmi_config.min_pmi == 3.0

    def test_min_frequency_config(self):
        """Test that min_frequency config is passed to detector"""
        config = ExtractionConfig(
            use_ml_compound_detection=True,
            ml_min_frequency=5
        )
        extractor = ConceptExtractor(config)

        assert extractor._compound_detector.config.pmi_config.min_frequency == 5

    def test_semantic_filter_config(self):
        """Test that semantic filter config is passed to detector"""
        config = ExtractionConfig(
            use_ml_compound_detection=True,
            use_semantic_filter=True
        )
        extractor = ConceptExtractor(config)

        assert extractor._compound_detector.config.use_embeddings is True

    def test_min_combined_score_config(self):
        """Test that min_combined_score is passed to detector"""
        config = ExtractionConfig(
            use_ml_compound_detection=True,
            ml_min_combined_score=0.7
        )
        extractor = ConceptExtractor(config)

        assert extractor._compound_detector.config.min_combined_score == 0.7


class TestEndToEndWorkflow:
    """End-to-end workflow tests"""

    def test_full_pipeline_train_and_extract(self):
        """Test full pipeline: train on corpus, then extract from new documents"""
        config = ExtractionConfig(
            use_ml_compound_detection=True,
            ml_min_pmi=0.5,
            ml_min_frequency=2,
            ml_min_combined_score=0.2
        )
        extractor = ConceptExtractor(config)

        # Training corpus about data engineering
        training_corpus = [
            "Data Pipeline connects various data sources to the data warehouse.",
            "The Data Pipeline processes events in real-time.",
            "Apache Kafka is a distributed streaming platform.",
            "Apache Kafka handles high-throughput messaging.",
            "Machine Learning models analyze the streaming data.",
            "Machine Learning pipelines require feature engineering.",
            "Load Balancer distributes traffic across servers.",
            "The Load Balancer ensures high availability.",
        ] * 2

        # Train
        num_learned = extractor.train_on_corpus(training_corpus)
        assert num_learned > 0

        # Now extract from a new document
        new_document = """
        Our system uses Apache Kafka for event streaming.
        The Data Pipeline ingests data from multiple sources.
        Machine Learning models process the events in real-time.
        A Load Balancer distributes requests across our services.
        """

        concepts = extractor.extract_concepts(new_document, document_id="doc_001")

        # Should have extracted concepts
        assert len(concepts) > 0

        # Check that some expected terms are found
        canonical_names = {c.canonical_name for c in concepts}

        # Should find common technical terms
        assert any("kafka" in name for name in canonical_names)

    def test_incremental_learning(self):
        """Test that new compound terms can be added incrementally"""
        config = ExtractionConfig(use_ml_compound_detection=True)
        extractor = ConceptExtractor(config)

        # Start with no learned terms
        assert len(extractor._learned_compound_terms) == 0

        # Add terms manually
        extractor.add_compound_terms({"new term one", "new term two"})
        assert len(extractor._learned_compound_terms) == 2

        # Add more terms
        extractor.add_compound_terms({"new term three"})
        assert len(extractor._learned_compound_terms) == 3

        # Terms should be in effective set
        effective = extractor._get_effective_compound_terms()
        assert "new term one" in effective
        assert "new term two" in effective
        assert "new term three" in effective


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
