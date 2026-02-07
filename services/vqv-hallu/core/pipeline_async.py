"""
VQV-HALLU Async Pipeline with L1/L2 Parallelism
Orchestration optimisée avec exécution parallèle des couches indépendantes
"""

import asyncio
import time
import logging
from typing import Optional, Tuple
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import aiohttp
import aiofiles
from urllib.parse import urlparse

from analyzers.acoustic_analyzer import AcousticAnalyzer
from analyzers.linguistic_analyzer import LinguisticAnalyzer
from analyzers.semantic_analyzer import SemanticAnalyzer
from core.score_fusion import ScoreFusionEngine, AdaptiveScoreFusion
from models.data_models import (
    VQVAnalysisResult, VQVInputMessage, VQVOutputMessage,
    AcousticAnalysisResult, LinguisticAnalysisResult, SemanticAnalysisResult
)
from config.settings import (
    VQVHalluConfig, ContentType, ContentTypeConfig,
    get_config_for_content_type
)


logger = logging.getLogger(__name__)


class FastRejectResult:
    """Résultat de la phase de rejet rapide"""
    def __init__(self, should_reject: bool, reason: str = "",
                 audio_duration_ms: int = 0):
        self.should_reject = should_reject
        self.reason = reason
        self.audio_duration_ms = audio_duration_ms


class VQVHalluAsyncPipeline:
    """
    Pipeline VQV-HALLU asynchrone avec exécution parallèle L1/L2.

    Optimisations:
    1. Fast Reject: Validation rapide avant analyse complète
    2. Parallel L1/L2: Acoustic et Linguistic en parallèle
    3. Early Exit: Sortie anticipée si L1 ou L2 trop bas
    4. WeaveGraph Integration: Boost sémantique via graphe de concepts
    """

    # Seuils pour early exit (avant L3)
    EARLY_EXIT_L1_THRESHOLD = 30.0  # Score L1 minimum
    EARLY_EXIT_L2_WER_THRESHOLD = 0.40  # WER maximum (40%)

    # Seuils pour fast reject
    MIN_AUDIO_DURATION_MS = 500  # Minimum 500ms
    MAX_AUDIO_DURATION_MS = 600000  # Maximum 10 minutes
    MIN_RMS_THRESHOLD = 0.001  # Énergie minimum

    def __init__(self,
                 global_config: VQVHalluConfig,
                 use_adaptive_fusion: bool = True,
                 max_concurrent_analyses: int = 5,
                 weave_graph_url: Optional[str] = None):
        """
        Initialise le pipeline async.

        Args:
            global_config: Configuration globale VQV-HALLU
            use_adaptive_fusion: Utiliser la fusion adaptative des scores
            max_concurrent_analyses: Nombre max d'analyses parallèles
            weave_graph_url: URL du service WeaveGraph (optionnel)
        """
        self.global_config = global_config
        self.use_adaptive_fusion = use_adaptive_fusion
        self.weave_graph_url = weave_graph_url

        # Thread pool pour les opérations bloquantes (librosa, whisper)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_analyses)

        # Cache des analyseurs par content type
        self._acoustic_analyzers = {}
        self._linguistic_analyzers = {}
        self._semantic_analyzers = {}
        self._fusion_engines = {}

        # Créer le répertoire temporaire
        Path(global_config.temp_dir).mkdir(parents=True, exist_ok=True)

        # Statistiques
        self._stats = {
            "total_analyses": 0,
            "fast_rejects": 0,
            "early_exits": 0,
            "parallel_time_saved_ms": 0,
        }

        logger.info("Pipeline VQV-HALLU Async initialisé")

    def _get_content_config(self, content_type_str: str) -> ContentTypeConfig:
        """Récupère la configuration pour un type de contenu."""
        try:
            content_type = ContentType(content_type_str)
        except ValueError:
            logger.warning(f"Type de contenu inconnu: {content_type_str}, utilisation de MIXED")
            content_type = ContentType.MIXED
        return get_config_for_content_type(content_type)

    def _get_acoustic_analyzer(self, config: ContentTypeConfig) -> AcousticAnalyzer:
        """Récupère ou crée l'analyseur acoustique."""
        key = config.content_type.value
        if key not in self._acoustic_analyzers:
            self._acoustic_analyzers[key] = AcousticAnalyzer(config)
        return self._acoustic_analyzers[key]

    def _get_linguistic_analyzer(self, config: ContentTypeConfig) -> LinguisticAnalyzer:
        """Récupère ou crée l'analyseur linguistique."""
        key = config.content_type.value
        if key not in self._linguistic_analyzers:
            self._linguistic_analyzers[key] = LinguisticAnalyzer(
                config,
                whisper_model=self.global_config.asr_model
            )
        return self._linguistic_analyzers[key]

    def _get_semantic_analyzer(self, config: ContentTypeConfig) -> SemanticAnalyzer:
        """Récupère ou crée l'analyseur sémantique."""
        key = config.content_type.value
        if key not in self._semantic_analyzers:
            self._semantic_analyzers[key] = SemanticAnalyzer(
                config,
                embedding_model=self.global_config.embedding_model,
                weave_graph_url=self.weave_graph_url
            )
        return self._semantic_analyzers[key]

    def _get_fusion_engine(self, config: ContentTypeConfig) -> ScoreFusionEngine:
        """Récupère ou crée le moteur de fusion."""
        key = config.content_type.value
        if key not in self._fusion_engines:
            if self.use_adaptive_fusion:
                self._fusion_engines[key] = AdaptiveScoreFusion(config)
            else:
                self._fusion_engines[key] = ScoreFusionEngine(config)
        return self._fusion_engines[key]

    async def _fast_reject_check(self, audio_path: str) -> FastRejectResult:
        """
        Phase 1: Vérification rapide pour rejet immédiat.

        Vérifie:
        - Durée audio (trop court/long)
        - Énergie RMS (audio vide)
        - Format valide

        Returns:
            FastRejectResult avec should_reject et reason
        """
        import librosa
        import numpy as np

        try:
            # Charger l'audio en basse qualité pour vérification rapide
            loop = asyncio.get_event_loop()

            def load_and_check():
                y, sr = librosa.load(audio_path, sr=16000, mono=True)
                duration_ms = int(len(y) / sr * 1000)
                rms = float(np.sqrt(np.mean(y ** 2)))
                return duration_ms, rms

            duration_ms, rms = await loop.run_in_executor(
                self._executor, load_and_check
            )

            # Vérifier la durée
            if duration_ms < self.MIN_AUDIO_DURATION_MS:
                return FastRejectResult(
                    should_reject=True,
                    reason=f"Audio trop court: {duration_ms}ms < {self.MIN_AUDIO_DURATION_MS}ms",
                    audio_duration_ms=duration_ms
                )

            if duration_ms > self.MAX_AUDIO_DURATION_MS:
                return FastRejectResult(
                    should_reject=True,
                    reason=f"Audio trop long: {duration_ms}ms > {self.MAX_AUDIO_DURATION_MS}ms",
                    audio_duration_ms=duration_ms
                )

            # Vérifier l'énergie RMS
            if rms < self.MIN_RMS_THRESHOLD:
                return FastRejectResult(
                    should_reject=True,
                    reason=f"Audio vide ou silencieux: RMS={rms:.6f}",
                    audio_duration_ms=duration_ms
                )

            return FastRejectResult(
                should_reject=False,
                audio_duration_ms=duration_ms
            )

        except Exception as e:
            return FastRejectResult(
                should_reject=True,
                reason=f"Erreur lecture audio: {str(e)}"
            )

    async def _run_acoustic_analysis(
        self,
        analyzer: AcousticAnalyzer,
        audio_path: str
    ) -> AcousticAnalysisResult:
        """Exécute L1 dans un thread séparé."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            analyzer.analyze,
            audio_path
        )

    async def _run_linguistic_analysis(
        self,
        analyzer: LinguisticAnalyzer,
        audio_path: str,
        language: str
    ) -> LinguisticAnalysisResult:
        """Exécute L2 dans un thread séparé."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(analyzer.analyze, audio_path, language)
        )

    async def _run_semantic_analysis(
        self,
        analyzer: SemanticAnalyzer,
        source_text: str,
        linguistic_result: LinguisticAnalysisResult,
        user_id: Optional[str] = None
    ) -> SemanticAnalysisResult:
        """Exécute L3 (peut utiliser WeaveGraph async)."""
        # Si WeaveGraph est configuré, utiliser la méthode async
        if self.weave_graph_url and user_id:
            return await analyzer.analyze_async(
                source_text,
                linguistic_result.transcription,
                user_id=user_id
            )
        else:
            # Fallback sur la version synchrone dans un thread
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                partial(analyzer.analyze, source_text, linguistic_result.transcription)
            )

    def _should_early_exit(
        self,
        acoustic_result: AcousticAnalysisResult,
        linguistic_result: LinguisticAnalysisResult
    ) -> Tuple[bool, str]:
        """
        Vérifie si on doit sortir avant L3 (early exit).

        Returns:
            (should_exit, reason)
        """
        # Vérifier score L1
        if acoustic_result.score < self.EARLY_EXIT_L1_THRESHOLD:
            return True, f"L1 score trop bas: {acoustic_result.score:.1f} < {self.EARLY_EXIT_L1_THRESHOLD}"

        # Vérifier WER de L2
        # WER approximatif basé sur la confidence moyenne des mots
        if linguistic_result.mean_word_confidence < (1 - self.EARLY_EXIT_L2_WER_THRESHOLD):
            return True, f"L2 confidence trop basse: {linguistic_result.mean_word_confidence:.1%}"

        return False, ""

    async def analyze(
        self,
        audio_path: str,
        source_text: str,
        audio_id: str,
        content_type: str = "mixed",
        language: str = "fr",
        user_id: Optional[str] = None
    ) -> VQVAnalysisResult:
        """
        Analyse complète d'un fichier audio avec parallélisme L1/L2.

        Args:
            audio_path: Chemin vers le fichier audio (local ou URL)
            source_text: Texte source qui a généré l'audio
            audio_id: Identifiant unique
            content_type: Type de contenu
            language: Langue attendue
            user_id: ID utilisateur pour WeaveGraph (optionnel)

        Returns:
            VQVAnalysisResult complet
        """
        start_time = time.time()
        self._stats["total_analyses"] += 1

        # Obtenir la configuration
        config = self._get_content_config(content_type)

        # Télécharger si URL
        local_path = await self._ensure_local_file(audio_path)

        try:
            # ============================================
            # PHASE 1: FAST REJECT (< 100ms)
            # ============================================
            logger.debug("Phase 1: Fast Reject Check...")
            fast_reject = await self._fast_reject_check(local_path)

            if fast_reject.should_reject:
                self._stats["fast_rejects"] += 1
                logger.warning(f"Fast reject: {fast_reject.reason}")

                # Retourner un résultat d'échec rapide
                return self._create_reject_result(
                    audio_id=audio_id,
                    source_text=source_text,
                    reason=fast_reject.reason,
                    audio_duration_ms=fast_reject.audio_duration_ms,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    content_type=content_type
                )

            audio_duration_ms = fast_reject.audio_duration_ms
            logger.info(f"Analyse de {audio_id} ({audio_duration_ms/1000:.1f}s, {content_type})")

            # ============================================
            # PHASE 2: PARALLEL L1 & L2
            # ============================================
            logger.debug("Phase 2: Parallel L1/L2 Analysis...")
            phase2_start = time.time()

            acoustic_analyzer = self._get_acoustic_analyzer(config)
            linguistic_analyzer = self._get_linguistic_analyzer(config)

            # Exécution parallèle avec asyncio.gather
            acoustic_result, linguistic_result = await asyncio.gather(
                self._run_acoustic_analysis(acoustic_analyzer, local_path),
                self._run_linguistic_analysis(linguistic_analyzer, local_path, language)
            )

            phase2_time = (time.time() - phase2_start) * 1000

            # Calculer le temps économisé (vs séquentiel)
            # En séquentiel: L1_time + L2_time, en parallèle: max(L1_time, L2_time)
            # On estime ~30% d'économie en moyenne
            self._stats["parallel_time_saved_ms"] += int(phase2_time * 0.3)

            logger.info(f"  L1 Acoustic: {acoustic_result.score:.1f} | L2 Linguistic: {linguistic_result.score:.1f}")

            # ============================================
            # EARLY EXIT CHECK
            # ============================================
            should_exit, exit_reason = self._should_early_exit(acoustic_result, linguistic_result)

            if should_exit:
                self._stats["early_exits"] += 1
                logger.warning(f"Early exit: {exit_reason}")

                return self._create_early_exit_result(
                    audio_id=audio_id,
                    source_text=source_text,
                    acoustic_result=acoustic_result,
                    linguistic_result=linguistic_result,
                    exit_reason=exit_reason,
                    audio_duration_ms=audio_duration_ms,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    content_type=content_type
                )

            # ============================================
            # PHASE 3: SEMANTIC ANALYSIS (depends on L2)
            # ============================================
            logger.debug("Phase 3: Semantic Analysis...")
            semantic_analyzer = self._get_semantic_analyzer(config)

            semantic_result = await self._run_semantic_analysis(
                semantic_analyzer,
                source_text,
                linguistic_result,
                user_id=user_id
            )

            logger.info(f"  L3 Semantic: {semantic_result.score:.1f}")

            # ============================================
            # PHASE 4: SCORE FUSION
            # ============================================
            logger.debug("Phase 4: Score Fusion...")
            fusion_engine = self._get_fusion_engine(config)

            processing_time_ms = int((time.time() - start_time) * 1000)

            result = fusion_engine.fuse(
                audio_id=audio_id,
                source_text=source_text,
                acoustic_result=acoustic_result,
                linguistic_result=linguistic_result,
                semantic_result=semantic_result,
                audio_duration_ms=audio_duration_ms,
                processing_time_ms=processing_time_ms
            )

            logger.info(
                f"Analyse terminée: score={result.final_score:.1f}, "
                f"acceptable={result.is_acceptable}, action={result.recommended_action}"
            )

            return result

        finally:
            # Nettoyer le fichier temporaire si téléchargé
            if local_path != audio_path and Path(local_path).exists():
                Path(local_path).unlink()

    async def _ensure_local_file(self, path: str) -> str:
        """S'assure que le fichier est local. Télécharge si URL."""
        parsed = urlparse(path)

        if parsed.scheme in ('http', 'https'):
            logger.debug(f"Téléchargement de {path}")

            async with aiohttp.ClientSession() as session:
                async with session.get(path, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    response.raise_for_status()

                    content_type = response.headers.get('content-type', '')
                    if 'wav' in content_type:
                        ext = '.wav'
                    elif 'mp3' in content_type:
                        ext = '.mp3'
                    elif 'ogg' in content_type:
                        ext = '.ogg'
                    else:
                        ext = '.wav'

                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=ext,
                        dir=self.global_config.temp_dir
                    )

                    async with aiofiles.open(temp_file.name, 'wb') as f:
                        await f.write(await response.read())

                    return temp_file.name

        return path

    def _create_reject_result(
        self,
        audio_id: str,
        source_text: str,
        reason: str,
        audio_duration_ms: int,
        processing_time_ms: int,
        content_type: str
    ) -> VQVAnalysisResult:
        """Crée un résultat pour un rejet rapide."""
        from datetime import datetime
        from models.data_models import (
            AcousticAnalysisResult, LinguisticAnalysisResult,
            SemanticAnalysisResult, TranscriptionResult,
            Anomaly, AnomalyType, SeverityLevel, TimeRange
        )

        # Créer des résultats vides/minimaux
        acoustic_result = AcousticAnalysisResult(
            score=0.0,
            anomalies=[Anomaly(
                anomaly_type=AnomalyType.SPECTRAL_ANOMALY,
                severity=SeverityLevel.CRITICAL,
                time_range=TimeRange(0, 0),
                confidence=1.0,
                description=f"Fast reject: {reason}"
            )],
            spectral_flatness_mean=0.0,
            spectral_flatness_std=0.0,
            silence_ratio=1.0,
            estimated_speech_rate_wpm=0.0,
            distortion_score=1.0,
            click_count=0,
            spectral_centroid_mean=0.0,
            spectral_bandwidth_mean=0.0
        )

        linguistic_result = LinguisticAnalysisResult(
            score=0.0,
            anomalies=[],
            transcription=TranscriptionResult(
                text="",
                language="unknown",
                confidence=0.0,
                word_timestamps=[],
                segments=[]
            ),
            mean_word_confidence=0.0,
            phoneme_validity_score=0.0,
            detected_languages=[],
            gibberish_segments=[],
            unknown_phoneme_ratio=1.0,
            word_repetition_count=0
        )

        semantic_result = SemanticAnalysisResult(
            score=0.0,
            anomalies=[],
            overall_similarity=0.0,
            word_alignments=[],
            hallucination_boundaries=[],
            semantic_drift_score=1.0,
            content_coverage=0.0,
            extra_content_ratio=0.0
        )

        return VQVAnalysisResult(
            audio_id=audio_id,
            source_text=source_text,
            processing_timestamp=datetime.now(),
            final_score=0.0,
            acoustic_score=0.0,
            linguistic_score=0.0,
            semantic_score=0.0,
            acoustic_result=acoustic_result,
            linguistic_result=linguistic_result,
            semantic_result=semantic_result,
            is_acceptable=False,
            primary_issues=[f"Fast reject: {reason}"],
            recommended_action="regenerate",
            audio_duration_ms=audio_duration_ms,
            processing_time_ms=processing_time_ms,
            content_type=content_type
        )

    def _create_early_exit_result(
        self,
        audio_id: str,
        source_text: str,
        acoustic_result: AcousticAnalysisResult,
        linguistic_result: LinguisticAnalysisResult,
        exit_reason: str,
        audio_duration_ms: int,
        processing_time_ms: int,
        content_type: str
    ) -> VQVAnalysisResult:
        """Crée un résultat pour un early exit (avant L3)."""
        from datetime import datetime
        from models.data_models import SemanticAnalysisResult

        # Créer un résultat sémantique minimal (L3 non exécuté)
        semantic_result = SemanticAnalysisResult(
            score=0.0,
            anomalies=[],
            overall_similarity=0.0,
            word_alignments=[],
            hallucination_boundaries=[],
            semantic_drift_score=0.0,
            content_coverage=0.0,
            extra_content_ratio=0.0
        )

        # Score fusionné basé uniquement sur L1 et L2
        # Pondération: L1=40%, L2=60% (L3 non disponible)
        fused_score = (acoustic_result.score * 0.4) + (linguistic_result.score * 0.6)

        return VQVAnalysisResult(
            audio_id=audio_id,
            source_text=source_text,
            processing_timestamp=datetime.now(),
            final_score=fused_score,
            acoustic_score=acoustic_result.score,
            linguistic_score=linguistic_result.score,
            semantic_score=0.0,
            acoustic_result=acoustic_result,
            linguistic_result=linguistic_result,
            semantic_result=semantic_result,
            is_acceptable=False,
            primary_issues=[f"Early exit: {exit_reason}"],
            recommended_action="regenerate",
            audio_duration_ms=audio_duration_ms,
            processing_time_ms=processing_time_ms,
            content_type=content_type
        )

    async def analyze_from_message(self, message: VQVInputMessage) -> VQVOutputMessage:
        """Analyse à partir d'un message d'entrée."""
        try:
            result = await self.analyze(
                audio_path=message.audio_path,
                source_text=message.source_text,
                audio_id=message.audio_id,
                content_type=message.content_type,
                language=message.language,
                user_id=message.metadata.get("user_id")
            )

            return VQVOutputMessage(
                audio_id=message.audio_id,
                status="success",
                result=result
            )

        except Exception as e:
            logger.exception(f"Erreur lors de l'analyse de {message.audio_id}")
            return VQVOutputMessage(
                audio_id=message.audio_id,
                status="error",
                result=None,
                error_message=str(e)
            )

    async def batch_analyze(
        self,
        items: list,
        max_concurrent: int = 3
    ) -> list:
        """
        Analyse un lot d'items avec parallélisme contrôlé.

        Args:
            items: Liste de VQVInputMessage
            max_concurrent: Nombre max d'analyses simultanées

        Returns:
            Liste de VQVOutputMessage
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(item):
            async with semaphore:
                return await self.analyze_from_message(item)

        tasks = [analyze_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict:
        """Retourne les statistiques du pipeline."""
        return {
            **self._stats,
            "fast_reject_rate": (
                self._stats["fast_rejects"] / self._stats["total_analyses"]
                if self._stats["total_analyses"] > 0 else 0
            ),
            "early_exit_rate": (
                self._stats["early_exits"] / self._stats["total_analyses"]
                if self._stats["total_analyses"] > 0 else 0
            ),
        }

    async def close(self):
        """Ferme le pipeline et libère les ressources."""
        self._executor.shutdown(wait=True)


# Interface simplifiée pour utilisation directe
async def analyze_voiceover_async(
    audio_path: str,
    source_text: str,
    audio_id: str = "default",
    content_type: str = "mixed",
    language: str = "fr",
    user_id: Optional[str] = None,
    config: Optional[VQVHalluConfig] = None,
    weave_graph_url: Optional[str] = None
) -> VQVAnalysisResult:
    """
    Interface simplifiée async pour analyser un voiceover.

    Args:
        audio_path: Chemin vers le fichier audio
        source_text: Texte source
        audio_id: Identifiant (optionnel)
        content_type: Type de contenu
        language: Langue
        user_id: ID utilisateur pour WeaveGraph
        config: Configuration (optionnel)
        weave_graph_url: URL WeaveGraph (optionnel)

    Returns:
        Résultat d'analyse
    """
    if config is None:
        config = VQVHalluConfig()

    pipeline = VQVHalluAsyncPipeline(
        config,
        weave_graph_url=weave_graph_url
    )

    try:
        return await pipeline.analyze(
            audio_path=audio_path,
            source_text=source_text,
            audio_id=audio_id,
            content_type=content_type,
            language=language,
            user_id=user_id
        )
    finally:
        await pipeline.close()
