"""
SSVS Calibrator - Correction des décalages audio-vidéo

Ce module corrige les 5 sources principales de désynchronisation:
1. OFFSET GLOBAL: Décalage de base dans les timestamps
2. LATENCE STT: Délai de traitement Whisper
3. ANTICIPATION SÉMANTIQUE: La slide doit apparaître AVANT le mot-clé
4. DURÉE DE TRANSITION: Les animations visuelles prennent du temps
5. INERTIE VISUELLE: Temps de transition/animation non compensé

SOLUTION: Calibration multi-niveaux avec offsets adaptatifs
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum

from .ssvs_algorithm import VoiceSegment, SynchronizationResult, Slide


# ==============================================================================
# SECTION 1: CONFIGURATION DE CALIBRATION
# ==============================================================================

@dataclass
class CalibrationConfig:
    """
    Configuration de calibration pour corriger les décalages.

    PARAMÈTRES CLÉS À AJUSTER:
    - global_offset_ms: Le plus important! Négatif = slide plus tôt
    - semantic_anticipation_ms: Anticiper l'apparition avant le mot-clé
    - stt_latency_compensation_ms: Compenser le délai Whisper
    """

    # ═══════════════════════════════════════════════════════════════════
    # OFFSET GLOBAL (le plus important!)
    # ═══════════════════════════════════════════════════════════════════
    global_offset_ms: float = -300.0
    """
    Décalage global appliqué à TOUTES les transitions.
    Négatif = slide apparaît plus tôt (recommandé)
    Positif = slide apparaît plus tard

    GUIDE:
    - Si la slide apparaît TROP TARD: diminuer (ex: -300 → -500)
    - Si la slide apparaît TROP TÔT: augmenter (ex: -300 → -100)
    """

    # ═══════════════════════════════════════════════════════════════════
    # ANTICIPATION SÉMANTIQUE
    # ═══════════════════════════════════════════════════════════════════
    semantic_anticipation_ms: float = -150.0
    """
    Anticipation pour les transitions basées sur des mots-clés.
    La slide doit apparaître AVANT que le narrateur dise le mot-clé.

    Ex: "Passons maintenant au diagramme" → slide change AVANT "diagramme"
    """

    # ═══════════════════════════════════════════════════════════════════
    # COMPENSATION TRANSITIONS
    # ═══════════════════════════════════════════════════════════════════
    transition_duration_ms: float = 200.0
    """Durée moyenne des transitions visuelles (fade, slide, etc.)"""

    transition_compensation: float = 0.5
    """
    Facteur de compensation pour les transitions.
    0.0 = pas de compensation
    1.0 = compensation complète de transition_duration_ms
    """

    # ═══════════════════════════════════════════════════════════════════
    # COMPENSATION STT (Speech-to-Text)
    # ═══════════════════════════════════════════════════════════════════
    stt_latency_compensation_ms: float = -50.0
    """
    Compensation pour la latence du STT (Whisper).
    Whisper peut avoir un léger délai dans la détection des mots.
    """

    # ═══════════════════════════════════════════════════════════════════
    # ALIGNEMENT SUR DÉBUT DE PHRASE
    # ═══════════════════════════════════════════════════════════════════
    align_to_sentence_start: bool = True
    """
    Si True, aligne les transitions sur le début des phrases.
    Évite les changements au milieu d'une phrase.
    """

    sentence_start_markers: List[str] = field(default_factory=lambda: [
        # Français
        "maintenant", "ensuite", "puis", "passons", "voyons", "regardons",
        "commençons", "abordons", "examinons", "prenons", "considérons",
        "voici", "voilà", "premièrement", "deuxièmement", "finalement",
        "d'abord", "après", "enfin", "pour conclure", "en résumé",
        # English
        "now", "next", "then", "let's", "moving on", "first", "second",
        "finally", "let me", "we'll", "here's", "starting with",
        "looking at", "consider", "notice", "observe", "as you can see",
    ])
    """Marqueurs indiquant le début d'une nouvelle section/phrase."""

    # ═══════════════════════════════════════════════════════════════════
    # DÉTECTION DE PAUSES
    # ═══════════════════════════════════════════════════════════════════
    use_pause_detection: bool = True
    """
    Si True, utilise les pauses naturelles comme points de transition.
    Les pauses sont souvent de meilleurs moments pour changer de slide.
    """

    min_pause_duration_ms: float = 300.0
    """Durée minimum pour qu'un silence soit considéré comme une pause."""

    snap_to_pause_threshold_ms: float = 500.0
    """Si une pause est à moins de X ms, snap la transition sur la pause."""

    # ═══════════════════════════════════════════════════════════════════
    # ADAPTATION À LA VITESSE DE PAROLE
    # ═══════════════════════════════════════════════════════════════════
    adapt_to_speech_rate: bool = True
    """
    Si True, adapte les offsets selon la vitesse de parole.
    Parle vite → plus d'anticipation
    Parle lentement → moins d'anticipation
    """

    reference_speech_rate: float = 150.0
    """Vitesse de parole de référence en mots/minute."""

    # ═══════════════════════════════════════════════════════════════════
    # DURÉE MINIMUM PAR SLIDE
    # ═══════════════════════════════════════════════════════════════════
    min_slide_duration_ms: float = 2000.0
    """Durée minimum d'affichage d'une slide (évite le flickering)."""

    max_slide_duration_ms: float = 120000.0
    """Durée maximum (2 minutes) avant alerte."""


# ==============================================================================
# SECTION 2: DÉTECTEUR DE PAUSES
# ==============================================================================

class PauseDetector:
    """
    Détecte les pauses naturelles dans la narration.
    Les pauses sont d'excellents points de transition.
    """

    def __init__(self, min_pause_ms: float = 300.0):
        self.min_pause_ms = min_pause_ms

    def detect_pauses(self, segments: List[VoiceSegment]) -> List[Tuple[float, float]]:
        """
        Détecte les pauses entre les segments.

        Returns:
            Liste de (start_time, duration) pour chaque pause
        """
        pauses = []

        for i in range(len(segments) - 1):
            gap_start = segments[i].end_time
            gap_end = segments[i + 1].start_time
            gap_duration = (gap_end - gap_start) * 1000  # en ms

            if gap_duration >= self.min_pause_ms:
                pauses.append((gap_start, gap_duration))

        return pauses

    def find_nearest_pause(self,
                           timestamp: float,
                           pauses: List[Tuple[float, float]],
                           max_distance_ms: float = 1000.0) -> Optional[float]:
        """
        Trouve la pause la plus proche d'un timestamp donné.

        Returns:
            Timestamp de la pause ou None si aucune pause proche
        """
        best_pause = None
        best_distance = float('inf')

        for pause_time, _ in pauses:
            distance = abs(pause_time - timestamp) * 1000
            if distance < best_distance and distance <= max_distance_ms:
                best_distance = distance
                best_pause = pause_time

        return best_pause


# ==============================================================================
# SECTION 3: ANALYSEUR DE VITESSE DE PAROLE
# ==============================================================================

class SpeechRateAnalyzer:
    """
    Analyse la vitesse de parole pour adapter les offsets.
    """

    def __init__(self, reference_rate: float = 150.0):
        self.reference_rate = reference_rate  # mots/minute

    def compute_speech_rate(self, segments: List[VoiceSegment]) -> float:
        """Calcule la vitesse de parole moyenne en mots/minute."""
        total_words = 0
        total_duration = 0.0

        for seg in segments:
            words = len(seg.text.split())
            duration = seg.end_time - seg.start_time
            total_words += words
            total_duration += duration

        if total_duration == 0:
            return self.reference_rate

        # mots par seconde -> mots par minute
        return (total_words / total_duration) * 60

    def compute_rate_factor(self, segments: List[VoiceSegment]) -> float:
        """
        Calcule un facteur d'ajustement basé sur la vitesse de parole.

        Returns:
            Facteur multiplicatif (>1 si parle vite, <1 si parle lentement)
        """
        actual_rate = self.compute_speech_rate(segments)
        return actual_rate / self.reference_rate

    def compute_local_rate(self, segment: VoiceSegment) -> float:
        """Calcule la vitesse de parole pour un segment spécifique."""
        words = len(segment.text.split())
        duration = segment.end_time - segment.start_time

        if duration == 0:
            return self.reference_rate

        return (words / duration) * 60


# ==============================================================================
# SECTION 4: ALIGNEUR SUR DÉBUT DE PHRASE
# ==============================================================================

class SentenceAligner:
    """
    Aligne les transitions sur le début des phrases/propositions,
    pas sur la détection des mots-clés au milieu.
    """

    def __init__(self, markers: List[str]):
        self.markers = [m.lower() for m in markers]

    def find_sentence_start(self, segment: VoiceSegment) -> Optional[float]:
        """
        Trouve le timestamp du début de phrase dans un segment.

        Pour une implémentation précise, il faudrait les word-level timestamps
        de Whisper. Ici on approxime.
        """
        text_lower = segment.text.lower()

        # Cherche un marqueur de début
        for marker in self.markers:
            if text_lower.startswith(marker):
                # Le marqueur est au début -> retourne le début du segment
                return segment.start_time

            pos = text_lower.find(marker)
            if pos > 0 and pos < len(text_lower) * 0.3:
                # Marqueur trouvé dans le premier tiers
                # Approximation: proportion linéaire
                ratio = pos / len(text_lower)
                duration = segment.end_time - segment.start_time
                return segment.start_time + (ratio * duration * 0.5)  # On recule un peu

        return None

    def adjust_timestamp(self,
                         original_timestamp: float,
                         segment: VoiceSegment) -> float:
        """
        Ajuste un timestamp pour l'aligner sur le début de phrase.
        """
        sentence_start = self.find_sentence_start(segment)

        if sentence_start and sentence_start < original_timestamp:
            return sentence_start

        return original_timestamp


# ==============================================================================
# SECTION 5: CALIBRATEUR PRINCIPAL
# ==============================================================================

class SSVSCalibrator:
    """
    Calibrateur principal pour corriger les décalages de synchronisation.

    Applique les corrections dans l'ordre:
    1. Offset global
    2. Compensation STT
    3. Anticipation sémantique
    4. Compensation de transition
    5. Alignement sur début de phrase
    6. Snap sur les pauses
    7. Ajustement vitesse de parole
    8. Validation durée minimum
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.pause_detector = PauseDetector(self.config.min_pause_duration_ms)
        self.speech_analyzer = SpeechRateAnalyzer(self.config.reference_speech_rate)
        self.sentence_aligner = SentenceAligner(self.config.sentence_start_markers)

    def calibrate(self,
                  results: List[SynchronizationResult],
                  segments: List[VoiceSegment]) -> List[SynchronizationResult]:
        """
        Applique toutes les corrections de calibration.

        Args:
            results: Résultats de synchronisation originaux
            segments: Segments audio pour analyse

        Returns:
            Résultats calibrés
        """
        if not results:
            return results

        print(f"[SSVS_CALIBRATOR] Calibrating {len(results)} results with config:", flush=True)
        print(f"  global_offset: {self.config.global_offset_ms}ms", flush=True)
        print(f"  semantic_anticipation: {self.config.semantic_anticipation_ms}ms", flush=True)
        print(f"  stt_compensation: {self.config.stt_latency_compensation_ms}ms", flush=True)

        # Analyses préliminaires
        pauses = self.pause_detector.detect_pauses(segments) if self.config.use_pause_detection else []
        rate_factor = self.speech_analyzer.compute_rate_factor(segments) if self.config.adapt_to_speech_rate else 1.0

        print(f"  speech_rate_factor: {rate_factor:.2f}", flush=True)
        print(f"  pauses_detected: {len(pauses)}", flush=True)

        # Création du mapping segment_id -> segment
        segment_map = {s.id: s for s in segments}

        calibrated = []

        for i, result in enumerate(results):
            # Copie du résultat
            original_start = result.start_time
            original_end = result.end_time
            original_duration = original_end - original_start
            new_start = original_start

            # ─────────────────────────────────────────────────────────────
            # ÉTAPE 1: Offset global
            # ─────────────────────────────────────────────────────────────
            offset_seconds = self.config.global_offset_ms / 1000.0
            new_start += offset_seconds

            # ─────────────────────────────────────────────────────────────
            # ÉTAPE 2: Compensation STT
            # ─────────────────────────────────────────────────────────────
            stt_offset = self.config.stt_latency_compensation_ms / 1000.0
            new_start += stt_offset

            # ─────────────────────────────────────────────────────────────
            # ÉTAPE 3: Anticipation sémantique
            # ─────────────────────────────────────────────────────────────
            sem_offset = self.config.semantic_anticipation_ms / 1000.0
            new_start += sem_offset

            # ─────────────────────────────────────────────────────────────
            # ÉTAPE 4: Compensation de transition
            # ─────────────────────────────────────────────────────────────
            trans_offset = (self.config.transition_duration_ms / 1000.0) * self.config.transition_compensation
            new_start -= trans_offset

            # ─────────────────────────────────────────────────────────────
            # ÉTAPE 5: Ajustement selon vitesse de parole
            # ─────────────────────────────────────────────────────────────
            if self.config.adapt_to_speech_rate and rate_factor != 1.0:
                # Si parle vite (rate_factor > 1), augmenter l'anticipation
                adjustment = (rate_factor - 1.0) * 0.1  # 10% d'ajustement par facteur
                new_start -= adjustment

            # ─────────────────────────────────────────────────────────────
            # ÉTAPE 6: Alignement sur début de phrase
            # ─────────────────────────────────────────────────────────────
            if self.config.align_to_sentence_start and result.segment_ids:
                first_seg_id = result.segment_ids[0]
                if first_seg_id in segment_map:
                    segment = segment_map[first_seg_id]
                    new_start = self.sentence_aligner.adjust_timestamp(new_start, segment)

            # ─────────────────────────────────────────────────────────────
            # ÉTAPE 7: Snap sur les pauses
            # ─────────────────────────────────────────────────────────────
            if self.config.use_pause_detection and pauses:
                nearest_pause = self.pause_detector.find_nearest_pause(
                    new_start, pauses, self.config.snap_to_pause_threshold_ms
                )
                if nearest_pause is not None:
                    new_start = nearest_pause

            # ─────────────────────────────────────────────────────────────
            # ÉTAPE 8: Appliquer le même offset à end_time (ANTI-DRIFT FIX)
            # ─────────────────────────────────────────────────────────────
            # CRITICAL: Pour éviter le drift cumulatif, on applique le même
            # offset total à end_time qu'à start_time. Cela préserve la durée
            # originale de chaque slide.
            total_offset = new_start - original_start
            new_end = original_end + total_offset

            # ─────────────────────────────────────────────────────────────
            # ÉTAPE 9: Validation et contraintes
            # ─────────────────────────────────────────────────────────────
            # Ne pas commencer avant 0
            if new_start < 0.0:
                # Shift both by the same amount to maintain duration
                shift = -new_start
                new_start = 0.0
                new_end = min(new_end + shift, original_end + shift)

            # Durée minimum
            min_duration = self.config.min_slide_duration_ms / 1000.0
            actual_duration = new_end - new_start
            if actual_duration < min_duration:
                new_end = new_start + min_duration

            # Ne pas chevaucher la slide précédente
            if calibrated and new_start < calibrated[-1].end_time:
                # Ajuster la fin de la slide précédente
                prev_result = calibrated[-1]
                adjusted_end = new_start - 0.01  # Petit gap de 10ms

                # Vérifier que la slide précédente garde sa durée minimum
                if adjusted_end - prev_result.start_time >= min_duration:
                    calibrated[-1] = SynchronizationResult(
                        slide_id=prev_result.slide_id,
                        slide_index=prev_result.slide_index,
                        segment_ids=prev_result.segment_ids,
                        start_time=prev_result.start_time,
                        end_time=adjusted_end,
                        semantic_score=prev_result.semantic_score,
                        temporal_score=prev_result.temporal_score,
                        combined_score=prev_result.combined_score,
                        transition_words=prev_result.transition_words
                    )

            # Créer le résultat calibré
            calibrated_result = SynchronizationResult(
                slide_id=result.slide_id,
                slide_index=result.slide_index,
                segment_ids=result.segment_ids,
                start_time=round(new_start, 3),
                end_time=round(new_end, 3),
                semantic_score=result.semantic_score,
                temporal_score=result.temporal_score,
                combined_score=result.combined_score,
                transition_words=result.transition_words
            )
            calibrated.append(calibrated_result)

            delta_ms = (new_start - result.start_time) * 1000
            duration_delta_ms = ((new_end - new_start) - original_duration) * 1000
            print(f"  Slide {result.slide_id}: {result.start_time:.2f}s → {new_start:.2f}s (Δ{delta_ms:+.0f}ms, dur Δ{duration_delta_ms:+.0f}ms)", flush=True)

        return calibrated

    def auto_calibrate(self,
                       results: List[SynchronizationResult],
                       segments: List[VoiceSegment],
                       feedback: Optional[List[Tuple[int, float]]] = None) -> CalibrationConfig:
        """
        Auto-calibration basée sur le feedback utilisateur.

        Args:
            feedback: Liste de (slide_index, décalage_observé_ms)
                      Positif = slide trop tard, Négatif = slide trop tôt

        Returns:
            Configuration optimisée
        """
        if not feedback:
            return self.config

        # Calculer l'offset moyen observé
        avg_offset = np.mean([f[1] for f in feedback])

        print(f"[SSVS_CALIBRATOR] Auto-calibrating based on feedback: avg_offset={avg_offset:.0f}ms", flush=True)

        # Ajuster l'offset global
        new_config = CalibrationConfig(
            global_offset_ms=self.config.global_offset_ms - avg_offset,
            semantic_anticipation_ms=self.config.semantic_anticipation_ms,
            transition_duration_ms=self.config.transition_duration_ms,
            transition_compensation=self.config.transition_compensation,
            stt_latency_compensation_ms=self.config.stt_latency_compensation_ms,
            align_to_sentence_start=self.config.align_to_sentence_start,
            sentence_start_markers=self.config.sentence_start_markers,
            use_pause_detection=self.config.use_pause_detection,
            min_pause_duration_ms=self.config.min_pause_duration_ms,
            snap_to_pause_threshold_ms=self.config.snap_to_pause_threshold_ms,
            adapt_to_speech_rate=self.config.adapt_to_speech_rate,
            reference_speech_rate=self.config.reference_speech_rate,
            min_slide_duration_ms=self.config.min_slide_duration_ms,
            max_slide_duration_ms=self.config.max_slide_duration_ms
        )

        return new_config


# ==============================================================================
# SECTION 6: PRESETS DE CONFIGURATION
# ==============================================================================

class CalibrationPresets:
    """Configurations pré-définies pour différents cas d'usage."""

    @staticmethod
    def default() -> CalibrationConfig:
        """Configuration par défaut équilibrée."""
        return CalibrationConfig()

    @staticmethod
    def fast_speech() -> CalibrationConfig:
        """Pour les narrateurs qui parlent vite."""
        return CalibrationConfig(
            global_offset_ms=-500.0,
            semantic_anticipation_ms=-250.0,
            adapt_to_speech_rate=True,
            reference_speech_rate=150.0
        )

    @staticmethod
    def slow_speech() -> CalibrationConfig:
        """Pour les narrateurs qui parlent lentement."""
        return CalibrationConfig(
            global_offset_ms=-150.0,
            semantic_anticipation_ms=-100.0,
            min_slide_duration_ms=3000.0
        )

    @staticmethod
    def technical_content() -> CalibrationConfig:
        """Pour du contenu technique avec diagrammes complexes."""
        return CalibrationConfig(
            global_offset_ms=-600.0,
            semantic_anticipation_ms=-300.0,
            transition_duration_ms=300.0,
            min_slide_duration_ms=3000.0
        )

    @staticmethod
    def simple_slides() -> CalibrationConfig:
        """Pour des slides simples avec peu de texte."""
        return CalibrationConfig(
            global_offset_ms=-200.0,
            semantic_anticipation_ms=-100.0,
            transition_duration_ms=150.0,
            min_slide_duration_ms=1500.0
        )

    @staticmethod
    def live_presentation() -> CalibrationConfig:
        """Pour une présentation en direct (moins d'anticipation)."""
        return CalibrationConfig(
            global_offset_ms=-100.0,
            semantic_anticipation_ms=-50.0,
            use_pause_detection=True,
            min_pause_duration_ms=200.0
        )

    @staticmethod
    def training_course() -> CalibrationConfig:
        """Pour les cours de formation (contexte Viralify)."""
        return CalibrationConfig(
            global_offset_ms=-400.0,
            semantic_anticipation_ms=-200.0,
            transition_duration_ms=250.0,
            transition_compensation=0.6,
            min_slide_duration_ms=2500.0,
            use_pause_detection=True,
            adapt_to_speech_rate=True
        )


# ==============================================================================
# SECTION 7: DIAGNOSTIC DE DÉCALAGE
# ==============================================================================

class SyncDiagnostic:
    """
    Outils de diagnostic pour identifier les causes de décalage.
    """

    @staticmethod
    def analyze_timing(results: List[SynchronizationResult],
                       segments: List[VoiceSegment]) -> Dict:
        """
        Analyse les timings pour identifier les problèmes potentiels.
        """
        segment_map = {s.id: s for s in segments}

        issues = []
        stats = {
            "total_slides": len(results),
            "total_duration": 0.0,
            "avg_slide_duration": 0.0,
            "min_slide_duration": float('inf'),
            "max_slide_duration": 0.0,
            "short_slides": 0,
            "long_slides": 0,
            "gaps": [],
            "overlaps": [],
            "speech_rate": 0.0
        }

        durations = []

        for i, result in enumerate(results):
            duration = result.end_time - result.start_time
            durations.append(duration)
            stats["total_duration"] += duration
            stats["min_slide_duration"] = min(stats["min_slide_duration"], duration)
            stats["max_slide_duration"] = max(stats["max_slide_duration"], duration)

            # Slides trop courtes
            if duration < 2.0:
                stats["short_slides"] += 1
                issues.append(f"Slide {result.slide_id}: durée trop courte ({duration:.1f}s)")

            # Slides trop longues
            if duration > 60.0:
                stats["long_slides"] += 1
                issues.append(f"Slide {result.slide_id}: durée très longue ({duration:.1f}s)")

            # Vérifier les gaps
            if i > 0:
                gap = result.start_time - results[i-1].end_time
                if gap > 0.5:
                    stats["gaps"].append((i-1, i, gap))
                    issues.append(f"Gap de {gap:.2f}s entre slides {results[i-1].slide_id} et {result.slide_id}")
                elif gap < -0.1:
                    stats["overlaps"].append((i-1, i, -gap))
                    issues.append(f"Chevauchement de {-gap:.2f}s entre slides {results[i-1].slide_id} et {result.slide_id}")

        if durations:
            stats["avg_slide_duration"] = np.mean(durations)

        if stats["min_slide_duration"] == float('inf'):
            stats["min_slide_duration"] = 0.0

        # Analyser la vitesse de parole
        analyzer = SpeechRateAnalyzer()
        stats["speech_rate"] = analyzer.compute_speech_rate(segments)

        if stats["speech_rate"] > 180:
            issues.append(f"Vitesse de parole élevée ({stats['speech_rate']:.0f} mots/min) - augmenter l'anticipation")
        elif stats["speech_rate"] < 120:
            issues.append(f"Vitesse de parole lente ({stats['speech_rate']:.0f} mots/min) - réduire l'anticipation")

        return {
            "stats": stats,
            "issues": issues,
            "recommendation": SyncDiagnostic._generate_recommendation(stats, issues)
        }

    @staticmethod
    def _generate_recommendation(stats: Dict, issues: List[str]) -> str:
        """Génère une recommandation basée sur l'analyse."""

        if not issues:
            return "Aucun problème détecté. Si un décalage persiste, ajustez global_offset_ms."

        recommendations = []

        if stats["speech_rate"] > 180:
            recommendations.append("Utilisez CalibrationPresets.fast_speech()")
        elif stats["speech_rate"] < 120:
            recommendations.append("Utilisez CalibrationPresets.slow_speech()")

        if stats["short_slides"] > len(issues) * 0.3:
            recommendations.append("Augmentez min_slide_duration_ms à 3000")

        if stats["gaps"]:
            recommendations.append("Activez use_pause_detection pour utiliser les pauses")

        if not recommendations:
            recommendations.append("Ajustez global_offset_ms (-500ms pour plus d'anticipation)")

        return "\n".join(recommendations)
