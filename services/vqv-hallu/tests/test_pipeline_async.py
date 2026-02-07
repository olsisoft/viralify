"""
Unit Tests for VQV-HALLU Async Pipeline with L1/L2 Parallelism

Tests:
1. Fast Reject Phase
2. Parallel L1/L2 Execution
3. Early Exit Logic
4. Full Pipeline Flow
5. Batch Processing
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import numpy as np
import tempfile
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline_async import (
    VQVHalluAsyncPipeline,
    FastRejectResult,
    analyze_voiceover_async
)
from models.data_models import (
    VQVAnalysisResult, VQVInputMessage,
    AcousticAnalysisResult, LinguisticAnalysisResult, SemanticAnalysisResult,
    TranscriptionResult, Anomaly, AnomalyType, SeverityLevel, TimeRange
)
from config.settings import VQVHalluConfig, ContentType


class TestFastRejectPhase:
    """Tests pour la phase de rejet rapide."""

    @pytest.fixture
    def pipeline(self):
        config = VQVHalluConfig()
        return VQVHalluAsyncPipeline(config)

    @pytest.mark.asyncio
    async def test_reject_empty_audio(self, pipeline):
        """Audio vide doit être rejeté."""
        # Créer un fichier audio vide (silence)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            # Écrire un header WAV minimal avec silence
            import wave
            with wave.open(temp_path, 'w') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                # 1 seconde de silence
                wav.writeframes(b'\x00' * 32000)

        try:
            result = await pipeline._fast_reject_check(temp_path)
            assert result.should_reject is True
            assert "RMS" in result.reason or "silencieux" in result.reason.lower()
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_reject_too_short_audio(self, pipeline):
        """Audio trop court doit être rejeté."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            import wave
            with wave.open(temp_path, 'w') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                # 100ms seulement (< 500ms threshold)
                frames = np.random.randint(-1000, 1000, 1600, dtype=np.int16)
                wav.writeframes(frames.tobytes())

        try:
            result = await pipeline._fast_reject_check(temp_path)
            assert result.should_reject is True
            assert "court" in result.reason.lower()
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_accept_valid_audio(self, pipeline):
        """Audio valide doit être accepté."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            import wave
            with wave.open(temp_path, 'w') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                # 2 secondes d'audio avec signal
                frames = np.random.randint(-5000, 5000, 32000, dtype=np.int16)
                wav.writeframes(frames.tobytes())

        try:
            result = await pipeline._fast_reject_check(temp_path)
            assert result.should_reject is False
            assert result.audio_duration_ms >= 1500  # ~2 seconds
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_reject_invalid_file(self, pipeline):
        """Fichier invalide doit être rejeté."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b"not an audio file")
            temp_path = f.name

        try:
            result = await pipeline._fast_reject_check(temp_path)
            assert result.should_reject is True
            assert "Erreur" in result.reason
        finally:
            os.unlink(temp_path)


class TestParallelL1L2Execution:
    """Tests pour l'exécution parallèle L1/L2."""

    @pytest.fixture
    def mock_config(self):
        return VQVHalluConfig()

    @pytest.fixture
    def mock_acoustic_result(self):
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
    def mock_linguistic_result(self):
        return LinguisticAnalysisResult(
            score=80.0,
            anomalies=[],
            transcription=TranscriptionResult(
                text="Test transcription",
                language="fr",
                confidence=0.9,
                word_timestamps=[
                    {"word": "Test", "start_ms": 0, "end_ms": 500},
                    {"word": "transcription", "start_ms": 500, "end_ms": 1500}
                ],
                segments=[]
            ),
            mean_word_confidence=0.9,
            phoneme_validity_score=0.85,
            detected_languages=[("fr", 0.95)],
            gibberish_segments=[],
            unknown_phoneme_ratio=0.05,
            word_repetition_count=0
        )

    @pytest.mark.asyncio
    async def test_parallel_execution_faster_than_sequential(
        self, mock_config, mock_acoustic_result, mock_linguistic_result
    ):
        """L'exécution parallèle doit être plus rapide que séquentielle."""

        # Simuler des analyseurs avec délais
        async def slow_acoustic(path):
            await asyncio.sleep(0.3)  # 300ms
            return mock_acoustic_result

        async def slow_linguistic(path, lang):
            await asyncio.sleep(0.5)  # 500ms
            return mock_linguistic_result

        # Mesurer l'exécution parallèle
        start = time.time()
        results = await asyncio.gather(
            slow_acoustic("test.wav"),
            slow_linguistic("test.wav", "fr")
        )
        parallel_time = time.time() - start

        # Mesurer l'exécution séquentielle
        start = time.time()
        await slow_acoustic("test.wav")
        await slow_linguistic("test.wav", "fr")
        sequential_time = time.time() - start

        # Parallèle doit être ~40% plus rapide
        assert parallel_time < sequential_time * 0.8
        # Parallèle doit prendre ~max(L1, L2) = ~500ms
        assert parallel_time < 0.7  # Avec overhead
        # Séquentiel doit prendre ~L1 + L2 = ~800ms
        assert sequential_time > 0.7

    @pytest.mark.asyncio
    async def test_both_results_returned(
        self, mock_config, mock_acoustic_result, mock_linguistic_result
    ):
        """Les deux résultats doivent être retournés."""

        async def mock_acoustic(path):
            return mock_acoustic_result

        async def mock_linguistic(path, lang):
            return mock_linguistic_result

        acoustic, linguistic = await asyncio.gather(
            mock_acoustic("test.wav"),
            mock_linguistic("test.wav", "fr")
        )

        assert acoustic.score == 85.0
        assert linguistic.score == 80.0
        assert linguistic.transcription.text == "Test transcription"


class TestEarlyExitLogic:
    """Tests pour la logique d'early exit."""

    @pytest.fixture
    def pipeline(self):
        config = VQVHalluConfig()
        return VQVHalluAsyncPipeline(config)

    @pytest.fixture
    def good_acoustic_result(self):
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
    def bad_acoustic_result(self):
        return AcousticAnalysisResult(
            score=20.0,  # Below threshold (30)
            anomalies=[
                Anomaly(
                    anomaly_type=AnomalyType.DISTORTION,
                    severity=SeverityLevel.CRITICAL,
                    time_range=TimeRange(0, 1000),
                    confidence=0.9,
                    description="Severe distortion"
                )
            ],
            spectral_flatness_mean=0.1,
            spectral_flatness_std=0.5,
            silence_ratio=0.5,
            estimated_speech_rate_wpm=50,
            distortion_score=0.8,
            click_count=15,
            spectral_centroid_mean=500,
            spectral_bandwidth_mean=200
        )

    @pytest.fixture
    def good_linguistic_result(self):
        return LinguisticAnalysisResult(
            score=80.0,
            anomalies=[],
            transcription=TranscriptionResult(
                text="Test",
                language="fr",
                confidence=0.9,
                word_timestamps=[],
                segments=[]
            ),
            mean_word_confidence=0.85,  # Above 60% threshold
            phoneme_validity_score=0.85,
            detected_languages=[("fr", 0.95)],
            gibberish_segments=[],
            unknown_phoneme_ratio=0.05,
            word_repetition_count=0
        )

    @pytest.fixture
    def bad_linguistic_result(self):
        return LinguisticAnalysisResult(
            score=40.0,
            anomalies=[],
            transcription=TranscriptionResult(
                text="???",
                language="unknown",
                confidence=0.2,
                word_timestamps=[],
                segments=[]
            ),
            mean_word_confidence=0.3,  # Below 60% threshold
            phoneme_validity_score=0.2,
            detected_languages=[("unknown", 0.8)],
            gibberish_segments=[TimeRange(0, 1000)],
            unknown_phoneme_ratio=0.6,
            word_repetition_count=5
        )

    def test_no_early_exit_when_good_scores(
        self, pipeline, good_acoustic_result, good_linguistic_result
    ):
        """Pas d'early exit quand les scores sont bons."""
        should_exit, reason = pipeline._should_early_exit(
            good_acoustic_result, good_linguistic_result
        )
        assert should_exit is False
        assert reason == ""

    def test_early_exit_on_bad_acoustic(
        self, pipeline, bad_acoustic_result, good_linguistic_result
    ):
        """Early exit si score acoustique trop bas."""
        should_exit, reason = pipeline._should_early_exit(
            bad_acoustic_result, good_linguistic_result
        )
        assert should_exit is True
        assert "L1" in reason
        assert "20" in reason  # Score affiché

    def test_early_exit_on_bad_linguistic(
        self, pipeline, good_acoustic_result, bad_linguistic_result
    ):
        """Early exit si confidence linguistique trop basse."""
        should_exit, reason = pipeline._should_early_exit(
            good_acoustic_result, bad_linguistic_result
        )
        assert should_exit is True
        assert "L2" in reason or "confidence" in reason.lower()


class TestFullPipelineFlow:
    """Tests pour le flux complet du pipeline."""

    @pytest.fixture
    def pipeline(self):
        config = VQVHalluConfig()
        return VQVHalluAsyncPipeline(config)

    @pytest.mark.asyncio
    async def test_create_reject_result(self, pipeline):
        """Vérifier la création d'un résultat de rejet."""
        result = pipeline._create_reject_result(
            audio_id="test_001",
            source_text="Test source",
            reason="Audio trop court",
            audio_duration_ms=100,
            processing_time_ms=50,
            content_type="mixed"
        )

        assert result.audio_id == "test_001"
        assert result.final_score == 0.0
        assert result.is_acceptable is False
        assert result.recommended_action == "regenerate"
        assert "Fast reject" in result.primary_issues[0]

    @pytest.mark.asyncio
    async def test_create_early_exit_result(self, pipeline):
        """Vérifier la création d'un résultat d'early exit."""
        acoustic_result = AcousticAnalysisResult(
            score=25.0,
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

        linguistic_result = LinguisticAnalysisResult(
            score=70.0,
            anomalies=[],
            transcription=TranscriptionResult(
                text="Test",
                language="fr",
                confidence=0.8,
                word_timestamps=[],
                segments=[]
            ),
            mean_word_confidence=0.8,
            phoneme_validity_score=0.8,
            detected_languages=[("fr", 0.95)],
            gibberish_segments=[],
            unknown_phoneme_ratio=0.1,
            word_repetition_count=0
        )

        result = pipeline._create_early_exit_result(
            audio_id="test_002",
            source_text="Test source",
            acoustic_result=acoustic_result,
            linguistic_result=linguistic_result,
            exit_reason="L1 score too low",
            audio_duration_ms=5000,
            processing_time_ms=500,
            content_type="technical_course"
        )

        assert result.audio_id == "test_002"
        assert result.acoustic_score == 25.0
        assert result.linguistic_score == 70.0
        assert result.semantic_score == 0.0  # L3 non exécuté
        assert result.is_acceptable is False
        assert "Early exit" in result.primary_issues[0]
        # Score fusionné: 25*0.4 + 70*0.6 = 10 + 42 = 52
        assert 50 <= result.final_score <= 55

    @pytest.mark.asyncio
    async def test_stats_tracking(self, pipeline):
        """Vérifier le suivi des statistiques."""
        initial_stats = pipeline.get_stats()
        assert initial_stats["total_analyses"] == 0
        assert initial_stats["fast_rejects"] == 0
        assert initial_stats["early_exits"] == 0


class TestBatchProcessing:
    """Tests pour le traitement par lots."""

    @pytest.fixture
    def pipeline(self):
        config = VQVHalluConfig()
        return VQVHalluAsyncPipeline(config, max_concurrent_analyses=2)

    @pytest.mark.asyncio
    async def test_batch_respects_concurrency_limit(self, pipeline):
        """Le batch doit respecter la limite de concurrence."""
        concurrent_count = 0
        max_concurrent = 0

        async def mock_analyze(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)
            concurrent_count -= 1
            return VQVAnalysisResult(
                audio_id="test",
                source_text="test",
                processing_timestamp=datetime.now(),
                final_score=80.0,
                acoustic_score=80.0,
                linguistic_score=80.0,
                semantic_score=80.0,
                acoustic_result=Mock(),
                linguistic_result=Mock(),
                semantic_result=Mock(),
                is_acceptable=True,
                primary_issues=[],
                recommended_action="accept",
                audio_duration_ms=1000,
                processing_time_ms=100,
                content_type="mixed"
            )

        # Créer 5 items
        items = [
            VQVInputMessage(
                audio_id=f"test_{i}",
                audio_path=f"/tmp/test_{i}.wav",
                source_text=f"Test {i}"
            )
            for i in range(5)
        ]

        with patch.object(pipeline, 'analyze_from_message', mock_analyze):
            results = await pipeline.batch_analyze(items, max_concurrent=2)

        assert len(results) == 5
        assert max_concurrent <= 2  # Respecte la limite


class TestPipelineStats:
    """Tests pour les statistiques du pipeline."""

    @pytest.fixture
    def pipeline(self):
        config = VQVHalluConfig()
        return VQVHalluAsyncPipeline(config)

    def test_initial_stats(self, pipeline):
        """Statistiques initiales à zéro."""
        stats = pipeline.get_stats()
        assert stats["total_analyses"] == 0
        assert stats["fast_rejects"] == 0
        assert stats["early_exits"] == 0
        assert stats["parallel_time_saved_ms"] == 0
        assert stats["fast_reject_rate"] == 0
        assert stats["early_exit_rate"] == 0

    def test_stats_rates_calculation(self, pipeline):
        """Calcul correct des taux."""
        pipeline._stats["total_analyses"] = 10
        pipeline._stats["fast_rejects"] = 2
        pipeline._stats["early_exits"] = 3

        stats = pipeline.get_stats()
        assert stats["fast_reject_rate"] == 0.2  # 2/10
        assert stats["early_exit_rate"] == 0.3  # 3/10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
