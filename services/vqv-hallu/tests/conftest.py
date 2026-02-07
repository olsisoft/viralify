"""
Pytest Configuration and Shared Fixtures for VQV-HALLU Tests
"""

import pytest
import asyncio
import tempfile
import os
import wave
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import (
    VQVAnalysisResult, VQVInputMessage, VQVOutputMessage,
    AcousticAnalysisResult, LinguisticAnalysisResult, SemanticAnalysisResult,
    TranscriptionResult, Anomaly, AnomalyType, SeverityLevel, TimeRange,
    WordAlignment
)
from config.settings import VQVHalluConfig, ContentTypeConfig, ContentType


# ============================================
# Event Loop Configuration
# ============================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================
# Configuration Fixtures
# ============================================

@pytest.fixture
def vqv_config():
    """Configuration VQV-HALLU par défaut."""
    return VQVHalluConfig()


@pytest.fixture
def content_config_technical():
    """Configuration pour contenu technique."""
    return ContentTypeConfig(content_type=ContentType.TECHNICAL_COURSE)


@pytest.fixture
def content_config_narrative():
    """Configuration pour contenu narratif."""
    return ContentTypeConfig(content_type=ContentType.NARRATIVE)


# ============================================
# Audio File Fixtures
# ============================================

@pytest.fixture
def valid_audio_file():
    """Crée un fichier audio WAV valide temporaire."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name

    # Créer un fichier WAV avec du contenu audio
    with wave.open(temp_path, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        # 3 secondes d'audio avec signal aléatoire
        frames = np.random.randint(-5000, 5000, 48000, dtype=np.int16)
        wav.writeframes(frames.tobytes())

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def silent_audio_file():
    """Crée un fichier audio silencieux."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name

    with wave.open(temp_path, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        # 2 secondes de silence
        wav.writeframes(b'\x00' * 64000)

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def short_audio_file():
    """Crée un fichier audio trop court (< 500ms)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name

    with wave.open(temp_path, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        # 100ms seulement
        frames = np.random.randint(-1000, 1000, 1600, dtype=np.int16)
        wav.writeframes(frames.tobytes())

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================
# Analysis Result Fixtures
# ============================================

@pytest.fixture
def good_acoustic_result():
    """Résultat acoustique positif."""
    return AcousticAnalysisResult(
        score=85.0,
        anomalies=[],
        spectral_flatness_mean=0.3,
        spectral_flatness_std=0.1,
        silence_ratio=0.1,
        estimated_speech_rate_wpm=150,
        distortion_score=0.05,
        click_count=0,
        spectral_centroid_mean=2000,
        spectral_bandwidth_mean=1500
    )


@pytest.fixture
def bad_acoustic_result():
    """Résultat acoustique négatif."""
    return AcousticAnalysisResult(
        score=25.0,
        anomalies=[
            Anomaly(
                anomaly_type=AnomalyType.DISTORTION,
                severity=SeverityLevel.HIGH,
                time_range=TimeRange(0, 1000),
                confidence=0.9,
                description="Distortion détectée"
            )
        ],
        spectral_flatness_mean=0.1,
        spectral_flatness_std=0.5,
        silence_ratio=0.5,
        estimated_speech_rate_wpm=50,
        distortion_score=0.8,
        click_count=10,
        spectral_centroid_mean=500,
        spectral_bandwidth_mean=200
    )


@pytest.fixture
def good_linguistic_result():
    """Résultat linguistique positif."""
    return LinguisticAnalysisResult(
        score=80.0,
        anomalies=[],
        transcription=TranscriptionResult(
            text="Kafka est un système de messaging distribué",
            language="fr",
            confidence=0.92,
            word_timestamps=[
                {"word": "Kafka", "start_ms": 0, "end_ms": 500},
                {"word": "est", "start_ms": 500, "end_ms": 700},
                {"word": "un", "start_ms": 700, "end_ms": 850},
                {"word": "système", "start_ms": 850, "end_ms": 1300},
                {"word": "de", "start_ms": 1300, "end_ms": 1450},
                {"word": "messaging", "start_ms": 1450, "end_ms": 2100},
                {"word": "distribué", "start_ms": 2100, "end_ms": 2800},
            ],
            segments=[{
                "id": 0,
                "text": "Kafka est un système de messaging distribué",
                "start": 0.0,
                "end": 2.8
            }]
        ),
        mean_word_confidence=0.88,
        phoneme_validity_score=0.85,
        detected_languages=[("fr", 0.95)],
        gibberish_segments=[],
        unknown_phoneme_ratio=0.05,
        word_repetition_count=0
    )


@pytest.fixture
def bad_linguistic_result():
    """Résultat linguistique négatif."""
    return LinguisticAnalysisResult(
        score=35.0,
        anomalies=[
            Anomaly(
                anomaly_type=AnomalyType.GIBBERISH,
                severity=SeverityLevel.HIGH,
                time_range=TimeRange(500, 1500),
                confidence=0.85,
                description="Charabia détecté"
            )
        ],
        transcription=TranscriptionResult(
            text="Kafkak esttt un systt messa",
            language="unknown",
            confidence=0.3,
            word_timestamps=[],
            segments=[]
        ),
        mean_word_confidence=0.35,
        phoneme_validity_score=0.25,
        detected_languages=[("unknown", 0.8)],
        gibberish_segments=[TimeRange(500, 1500)],
        unknown_phoneme_ratio=0.6,
        word_repetition_count=3
    )


@pytest.fixture
def good_semantic_result():
    """Résultat sémantique positif."""
    return SemanticAnalysisResult(
        score=90.0,
        anomalies=[],
        overall_similarity=0.95,
        word_alignments=[
            WordAlignment(
                source_word="kafka",
                transcribed_word="kafka",
                time_range=TimeRange(0, 500),
                confidence=1.0,
                is_match=True,
                phoneme_similarity=1.0
            )
        ],
        hallucination_boundaries=[],
        semantic_drift_score=0.02,
        content_coverage=0.98,
        extra_content_ratio=0.01
    )


@pytest.fixture
def bad_semantic_result():
    """Résultat sémantique négatif."""
    return SemanticAnalysisResult(
        score=40.0,
        anomalies=[
            Anomaly(
                anomaly_type=AnomalyType.HALLUCINATION,
                severity=SeverityLevel.HIGH,
                time_range=TimeRange(1000, 2000),
                confidence=0.9,
                description="Hallucination détectée"
            )
        ],
        overall_similarity=0.4,
        word_alignments=[],
        hallucination_boundaries=[TimeRange(1000, 2000)],
        semantic_drift_score=0.6,
        content_coverage=0.5,
        extra_content_ratio=0.4
    )


# ============================================
# Message Fixtures
# ============================================

@pytest.fixture
def sample_input_message(valid_audio_file):
    """Message d'entrée exemple."""
    return VQVInputMessage(
        audio_id="test_001",
        audio_path=valid_audio_file,
        source_text="Kafka est un système de messaging distribué",
        content_type="technical_course",
        language="fr",
        metadata={"user_id": "user123"}
    )


# ============================================
# Complete Result Fixtures
# ============================================

@pytest.fixture
def complete_vqv_result(
    good_acoustic_result,
    good_linguistic_result,
    good_semantic_result
):
    """Résultat VQV complet positif."""
    return VQVAnalysisResult(
        audio_id="test_001",
        source_text="Kafka est un système de messaging distribué",
        processing_timestamp=datetime.now(),
        final_score=85.0,
        acoustic_score=good_acoustic_result.score,
        linguistic_score=good_linguistic_result.score,
        semantic_score=good_semantic_result.score,
        acoustic_result=good_acoustic_result,
        linguistic_result=good_linguistic_result,
        semantic_result=good_semantic_result,
        is_acceptable=True,
        primary_issues=[],
        recommended_action="accept",
        audio_duration_ms=2800,
        processing_time_ms=1500,
        content_type="technical_course"
    )


# ============================================
# Mock Fixtures
# ============================================

@pytest.fixture
def mock_acoustic_analyzer():
    """Mock de l'analyseur acoustique."""
    mock = Mock()
    mock.analyze.return_value = AcousticAnalysisResult(
        score=80.0,
        anomalies=[],
        spectral_flatness_mean=0.3,
        spectral_flatness_std=0.1,
        silence_ratio=0.1,
        estimated_speech_rate_wpm=150,
        distortion_score=0.05,
        click_count=0,
        spectral_centroid_mean=2000,
        spectral_bandwidth_mean=1500
    )
    return mock


@pytest.fixture
def mock_linguistic_analyzer():
    """Mock de l'analyseur linguistique."""
    mock = Mock()
    mock.analyze.return_value = LinguisticAnalysisResult(
        score=75.0,
        anomalies=[],
        transcription=TranscriptionResult(
            text="Test transcription",
            language="fr",
            confidence=0.85,
            word_timestamps=[],
            segments=[]
        ),
        mean_word_confidence=0.85,
        phoneme_validity_score=0.8,
        detected_languages=[("fr", 0.95)],
        gibberish_segments=[],
        unknown_phoneme_ratio=0.1,
        word_repetition_count=0
    )
    return mock


@pytest.fixture
def mock_semantic_analyzer():
    """Mock de l'analyseur sémantique."""
    mock = Mock()
    mock.analyze.return_value = SemanticAnalysisResult(
        score=82.0,
        anomalies=[],
        overall_similarity=0.88,
        word_alignments=[],
        hallucination_boundaries=[],
        semantic_drift_score=0.05,
        content_coverage=0.92,
        extra_content_ratio=0.03
    )
    mock.analyze_async = AsyncMock(return_value=mock.analyze.return_value)
    return mock


@pytest.fixture
def mock_weave_graph_client():
    """Mock du client WeaveGraph."""
    from clients.weave_graph_client import ConceptIntegrityResult

    mock = AsyncMock()
    mock.check_concept_integrity.return_value = ConceptIntegrityResult(
        score=0.85,
        source_concepts=[],
        transcription_concepts=[],
        matched_concepts=["kafka", "messaging"],
        missing_concepts=[],
        extra_concepts=[],
        phonetic_matches=[],
        boost=0.08
    )
    mock.fetch_user_concepts.return_value = {}
    return mock
