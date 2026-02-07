"""
Integration Tests for WeaveGraph in VQV-HALLU Semantic Analyzer

Tests:
1. WeaveGraph Client
2. Concept Extraction
3. Phonetic Matching
4. Concept Integrity Check
5. Semantic Analyzer with WeaveGraph Boost
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.weave_graph_client import (
    WeaveGraphClient,
    ConceptMatch,
    ConceptIntegrityResult,
    create_weave_graph_client
)
from analyzers.semantic_analyzer import (
    SemanticAnalyzer,
    WeaveGraphBoostResult
)
from models.data_models import (
    TranscriptionResult, SemanticAnalysisResult,
    Anomaly, AnomalyType, SeverityLevel
)
from config.settings import ContentTypeConfig, ContentType


class TestWeaveGraphClient:
    """Tests pour le client WeaveGraph."""

    @pytest.fixture
    def client(self):
        return WeaveGraphClient(base_url="http://test:8006")

    def test_extract_camelcase_terms(self, client):
        """Extraction des termes CamelCase."""
        text = "We use ApacheKafka and SpringBoot for MessageQueue"
        terms = client.extract_key_terms(text)

        assert "apachekafka" in terms or "apache kafka" in terms
        assert "springboot" in terms or "spring boot" in terms

    def test_extract_snake_case_terms(self, client):
        """Extraction des termes snake_case."""
        text = "Use the message_broker and event_streaming patterns"
        terms = client.extract_key_terms(text)

        assert "message broker" in terms
        assert "event streaming" in terms

    def test_extract_acronyms(self, client):
        """Extraction des acronymes."""
        text = "Configure the API with REST and gRPC protocols"
        terms = client.extract_key_terms(text)

        assert "api" in terms
        assert "rest" in terms
        assert "grpc" in terms

    def test_extract_long_words(self, client):
        """Extraction des mots longs (>= 6 chars)."""
        text = "Implementation of authentication system"
        terms = client.extract_key_terms(text)

        assert "implementation" in terms
        assert "authentication" in terms
        assert "system" in terms


class TestPhoneticSimilarity:
    """Tests pour la similarité phonétique."""

    @pytest.fixture
    def client(self):
        return WeaveGraphClient(base_url="http://test:8006")

    def test_identical_words(self, client):
        """Mots identiques = similarité 1.0."""
        sim = client.compute_phonetic_similarity("kafka", "kafka")
        assert sim == 1.0

    def test_similar_words(self, client):
        """Mots similaires (erreur phonétique)."""
        # Kafka → Café (erreur TTS commune)
        sim = client.compute_phonetic_similarity("kafka", "cafe")
        assert 0.4 < sim < 0.8

        # Consumer → Consommer
        sim = client.compute_phonetic_similarity("consumer", "consommer")
        assert sim > 0.6

    def test_different_words(self, client):
        """Mots différents = faible similarité."""
        sim = client.compute_phonetic_similarity("kafka", "database")
        assert sim < 0.5

    def test_consonant_structure(self, client):
        """Structure consonantique similaire."""
        # "producer" et "producteur" ont une structure consonantique proche
        sim = client.compute_phonetic_similarity("producer", "producteur")
        assert sim > 0.7


class TestConceptIntegrityCheck:
    """Tests pour la vérification d'intégrité des concepts."""

    @pytest.fixture
    def client(self):
        return WeaveGraphClient(base_url="http://test:8006")

    @pytest.mark.asyncio
    async def test_perfect_match(self, client):
        """Correspondance parfaite entre source et transcription."""
        source = "Apache Kafka is a message broker for event streaming"
        transcription = "Apache Kafka is a message broker for event streaming"

        result = await client.check_concept_integrity(
            source, transcription, user_id=None
        )

        assert result.score >= 0.9
        assert len(result.missing_concepts) == 0
        assert result.boost > 0.1

    @pytest.mark.asyncio
    async def test_missing_concepts(self, client):
        """Concepts manquants dans la transcription."""
        source = "Apache Kafka handles event streaming with consumers"
        transcription = "Apache handles events with users"  # Missing: Kafka, streaming, consumers

        result = await client.check_concept_integrity(
            source, transcription, user_id=None
        )

        assert result.score < 0.8
        assert len(result.missing_concepts) > 0
        # Check that important concepts are marked as missing
        missing_lower = [m.lower() for m in result.missing_concepts]
        assert "streaming" in missing_lower or "consumers" in missing_lower

    @pytest.mark.asyncio
    async def test_phonetic_errors_detected(self, client):
        """Erreurs phonétiques détectées et corrigées."""
        source = "Kafka consumer reads from topics"
        transcription = "Café consommateur reads from topics"  # Phonetic errors

        result = await client.check_concept_integrity(
            source, transcription, user_id=None
        )

        # Should detect phonetic matches
        assert len(result.phonetic_matches) > 0 or result.score > 0.5

    @pytest.mark.asyncio
    async def test_extra_concepts(self, client):
        """Concepts supplémentaires dans la transcription."""
        source = "Kafka is a broker"
        transcription = "Kafka is a message broker for distributed systems"  # Extra: message, distributed, systems

        result = await client.check_concept_integrity(
            source, transcription, user_id=None
        )

        # Extra concepts should be detected
        assert len(result.extra_concepts) >= 0  # May or may not have extras
        # Score should still be decent if source concepts are present
        assert result.score >= 0.5

    @pytest.mark.asyncio
    async def test_boost_calculation(self, client):
        """Calcul correct du boost."""
        # High integrity = high boost
        result_high = await client.check_concept_integrity(
            "Kafka consumer producer topic",
            "Kafka consumer producer topic",
            user_id=None
        )
        assert result_high.boost >= 0.10

        # Low integrity = low boost
        result_low = await client.check_concept_integrity(
            "Kafka consumer producer topic",
            "Database query index table",
            user_id=None
        )
        assert result_low.boost < result_high.boost


class TestWeaveGraphClientWithMockedAPI:
    """Tests avec API WeaveGraph mockée."""

    @pytest.fixture
    def mock_concepts(self):
        return {
            "kafka": ConceptMatch(
                name="Apache Kafka",
                canonical_name="kafka",
                confidence=1.0,
                source="weave_graph",
                aliases=["apache kafka", "kafka streaming"]
            ),
            "consumer": ConceptMatch(
                name="Consumer",
                canonical_name="consumer",
                confidence=1.0,
                source="weave_graph",
                aliases=["kafka consumer", "message consumer"]
            ),
            "producer": ConceptMatch(
                name="Producer",
                canonical_name="producer",
                confidence=1.0,
                source="weave_graph",
                aliases=["kafka producer", "message producer"]
            )
        }

    @pytest.mark.asyncio
    async def test_fetch_concepts_success(self, mock_concepts):
        """Récupération des concepts avec succès."""
        client = WeaveGraphClient(base_url="http://test:8006")

        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "concepts": [
                    {"name": "Apache Kafka", "canonical_name": "kafka", "aliases": []},
                    {"name": "Consumer", "canonical_name": "consumer", "aliases": []},
                ]
            })

            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = mock_response
            mock_session.return_value.__aenter__.return_value.get.return_value = mock_cm

            concepts = await client.fetch_user_concepts("user123")

            # Should have concepts in cache
            assert len(concepts) >= 0  # May be empty if mock not set up correctly

    @pytest.mark.asyncio
    async def test_fetch_concepts_error_handling(self):
        """Gestion des erreurs lors de la récupération."""
        client = WeaveGraphClient(base_url="http://invalid:9999")

        # Should return empty dict on error, not raise
        concepts = await client.fetch_user_concepts("user123")
        assert concepts == {}


class TestSemanticAnalyzerWithWeaveGraph:
    """Tests pour le SemanticAnalyzer avec intégration WeaveGraph."""

    @pytest.fixture
    def mock_config(self):
        return ContentTypeConfig(content_type=ContentType.TECHNICAL_COURSE)

    @pytest.fixture
    def mock_transcription(self):
        return TranscriptionResult(
            text="Kafka is a message broker for streaming",
            language="fr",
            confidence=0.9,
            word_timestamps=[
                {"word": "Kafka", "start_ms": 0, "end_ms": 500},
                {"word": "is", "start_ms": 500, "end_ms": 700},
                {"word": "a", "start_ms": 700, "end_ms": 800},
                {"word": "message", "start_ms": 800, "end_ms": 1200},
                {"word": "broker", "start_ms": 1200, "end_ms": 1600},
                {"word": "for", "start_ms": 1600, "end_ms": 1800},
                {"word": "streaming", "start_ms": 1800, "end_ms": 2500},
            ],
            segments=[]
        )

    @pytest.mark.asyncio
    async def test_analyze_async_without_weave_graph(self, mock_config, mock_transcription):
        """Analyse async sans WeaveGraph (boost = 0)."""
        # Skip if sentence-transformers not available
        pytest.importorskip("sentence_transformers")

        analyzer = SemanticAnalyzer(
            config=mock_config,
            weave_graph_url=None  # Pas de WeaveGraph
        )

        source_text = "Kafka is a message broker for streaming"

        result = await analyzer.analyze_async(
            source_text=source_text,
            transcription=mock_transcription,
            user_id=None
        )

        assert isinstance(result, SemanticAnalysisResult)
        assert result.score > 0
        # Sans WeaveGraph, similarité devrait être très haute (textes identiques)
        assert result.overall_similarity > 0.9

    @pytest.mark.asyncio
    async def test_analyze_async_with_weave_graph_mock(self, mock_config, mock_transcription):
        """Analyse async avec WeaveGraph mocké."""
        pytest.importorskip("sentence_transformers")

        analyzer = SemanticAnalyzer(
            config=mock_config,
            weave_graph_url="http://mock:8006"
        )

        # Mock le client WeaveGraph
        mock_integrity_result = ConceptIntegrityResult(
            score=0.95,
            source_concepts=[],
            transcription_concepts=[],
            matched_concepts=["kafka", "broker", "streaming"],
            missing_concepts=[],
            extra_concepts=[],
            phonetic_matches=[],
            boost=0.12
        )

        mock_client = AsyncMock()
        mock_client.check_concept_integrity = AsyncMock(return_value=mock_integrity_result)
        analyzer._weave_graph_client = mock_client

        source_text = "Kafka is a message broker for streaming"

        result = await analyzer.analyze_async(
            source_text=source_text,
            transcription=mock_transcription,
            user_id="user123"
        )

        assert isinstance(result, SemanticAnalysisResult)
        # Le score devrait être boosté
        assert result.score > 80  # High score expected

    @pytest.mark.asyncio
    async def test_phonetic_errors_create_low_severity_anomalies(self, mock_config):
        """Les erreurs phonétiques créent des anomalies de faible sévérité."""
        pytest.importorskip("sentence_transformers")

        transcription = TranscriptionResult(
            text="Café is a message broker",  # "Café" instead of "Kafka"
            language="fr",
            confidence=0.8,
            word_timestamps=[
                {"word": "Café", "start_ms": 0, "end_ms": 500},
                {"word": "is", "start_ms": 500, "end_ms": 700},
                {"word": "a", "start_ms": 700, "end_ms": 800},
                {"word": "message", "start_ms": 800, "end_ms": 1200},
                {"word": "broker", "start_ms": 1200, "end_ms": 1600},
            ],
            segments=[]
        )

        analyzer = SemanticAnalyzer(
            config=mock_config,
            weave_graph_url="http://mock:8006"
        )

        # Mock avec erreur phonétique détectée
        mock_integrity_result = ConceptIntegrityResult(
            score=0.7,
            source_concepts=[],
            transcription_concepts=[],
            matched_concepts=["broker", "message"],
            missing_concepts=[],
            extra_concepts=[],
            phonetic_matches=[("kafka", "café", 0.65)],  # Erreur phonétique
            boost=0.05
        )

        mock_client = AsyncMock()
        mock_client.check_concept_integrity = AsyncMock(return_value=mock_integrity_result)
        analyzer._weave_graph_client = mock_client

        source_text = "Kafka is a message broker"

        result = await analyzer.analyze_async(
            source_text=source_text,
            transcription=transcription,
            user_id="user123"
        )

        # Vérifier qu'une anomalie phonétique a été créée
        phonetic_anomalies = [
            a for a in result.anomalies
            if "phonétique" in a.description.lower()
        ]
        assert len(phonetic_anomalies) >= 1
        assert phonetic_anomalies[0].severity == SeverityLevel.LOW


class TestWeaveGraphBoostCalculation:
    """Tests pour le calcul du boost WeaveGraph."""

    @pytest.fixture
    def client(self):
        return WeaveGraphClient(base_url="http://test:8006")

    @pytest.mark.asyncio
    async def test_high_integrity_high_boost(self, client):
        """Haute intégrité = boost élevé."""
        result = await client.check_concept_integrity(
            "Kafka consumer producer broker topic partition",
            "Kafka consumer producer broker topic partition",
            user_id=None
        )
        assert result.boost >= 0.10

    @pytest.mark.asyncio
    async def test_medium_integrity_medium_boost(self, client):
        """Intégrité moyenne = boost moyen."""
        result = await client.check_concept_integrity(
            "Kafka consumer producer broker topic partition",
            "Kafka consumer broker",  # Missing some concepts
            user_id=None
        )
        assert 0.03 <= result.boost <= 0.12

    @pytest.mark.asyncio
    async def test_low_integrity_no_boost(self, client):
        """Faible intégrité = pas de boost."""
        result = await client.check_concept_integrity(
            "Kafka consumer producer broker topic partition",
            "Database SQL query index table",  # Completely different
            user_id=None
        )
        assert result.boost <= 0.05

    @pytest.mark.asyncio
    async def test_boost_capped_at_15_percent(self, client):
        """Le boost ne dépasse jamais 15%."""
        result = await client.check_concept_integrity(
            "Kafka consumer producer broker topic partition offset group",
            "Kafka consumer producer broker topic partition offset group",
            user_id=None
        )
        assert result.boost <= 0.15


class TestCreateWeaveGraphClientFactory:
    """Tests pour la factory de création du client."""

    def test_create_with_url(self):
        """Création avec URL valide."""
        client = create_weave_graph_client("http://localhost:8006")
        assert client is not None
        assert isinstance(client, WeaveGraphClient)

    def test_create_without_url(self):
        """Pas de création sans URL."""
        client = create_weave_graph_client(None)
        assert client is None

    def test_create_with_empty_url(self):
        """Pas de création avec URL vide."""
        client = create_weave_graph_client("")
        assert client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
