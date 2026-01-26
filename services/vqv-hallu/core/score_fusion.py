"""
VQV-HALLU Layer 4: Score Fusion Engine
Fusion des scores multi-couches avec calibration de confiance
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

from models.data_models import (
    VQVAnalysisResult, Anomaly, AnomalyType, SeverityLevel,
    AcousticAnalysisResult, LinguisticAnalysisResult, SemanticAnalysisResult
)
from config.settings import ContentTypeConfig, ContentType


logger = logging.getLogger(__name__)


class ScoreFusionEngine:
    """
    Moteur de fusion multi-couches pour calcul du score final VQV-HALLU.
    
    Implémente:
    - Fusion pondérée des scores par couche
    - Calibration de confiance basée sur les anomalies
    - Détection de patterns d'hallucination cross-layer
    - Génération du verdict et recommandations
    """
    
    # Bonus/malus pour patterns cross-layer
    CROSS_LAYER_PATTERNS = {
        # (acoustic_anomaly, linguistic_anomaly) -> malus
        (AnomalyType.DISTORTION, AnomalyType.GIBBERISH): 10,
        (AnomalyType.SPECTRAL_ANOMALY, AnomalyType.UNKNOWN_PHONEMES): 8,
        (AnomalyType.PACE_TOO_FAST, AnomalyType.LOW_ASR_CONFIDENCE): 5,
        (AnomalyType.SILENCE_EXCESSIVE, AnomalyType.WORD_REPETITION): 7,
    }
    
    # Combinaisons linguistique + sémantique
    LINGUISTIC_SEMANTIC_PATTERNS = {
        (AnomalyType.LANGUAGE_SWITCH, AnomalyType.HALLUCINATION): 15,
        (AnomalyType.GIBBERISH, AnomalyType.HALLUCINATION): 12,
        (AnomalyType.WORD_REPETITION, AnomalyType.EXTRA_CONTENT): 8,
        (AnomalyType.LOW_ASR_CONFIDENCE, AnomalyType.MISSING_CONTENT): 6,
    }
    
    def __init__(self, config: ContentTypeConfig):
        self.config = config
        self.weight_acoustic = config.weight_acoustic
        self.weight_linguistic = config.weight_linguistic
        self.weight_semantic = config.weight_semantic
        self.min_acceptable_score = config.min_acceptable_score
    
    def fuse(self,
             audio_id: str,
             source_text: str,
             acoustic_result: AcousticAnalysisResult,
             linguistic_result: LinguisticAnalysisResult,
             semantic_result: SemanticAnalysisResult,
             audio_duration_ms: int,
             processing_time_ms: int) -> VQVAnalysisResult:
        """
        Fusionne les résultats des trois couches en un score final.
        
        Args:
            audio_id: Identifiant unique de l'audio
            source_text: Texte source original
            acoustic_result: Résultat Layer 1
            linguistic_result: Résultat Layer 2
            semantic_result: Résultat Layer 3
            audio_duration_ms: Durée de l'audio en ms
            processing_time_ms: Temps de traitement en ms
            
        Returns:
            VQVAnalysisResult complet avec verdict
        """
        # 1. Fusion pondérée de base
        base_score = self._compute_weighted_score(
            acoustic_result.score,
            linguistic_result.score,
            semantic_result.score
        )
        
        # 2. Détection de patterns cross-layer
        cross_layer_penalty = self._detect_cross_layer_patterns(
            acoustic_result.anomalies,
            linguistic_result.anomalies,
            semantic_result.anomalies
        )
        
        # 3. Calibration de confiance
        confidence_adjustment = self._calibrate_confidence(
            acoustic_result,
            linguistic_result,
            semantic_result
        )
        
        # 4. Score final
        final_score = max(0.0, min(100.0, 
            base_score - cross_layer_penalty + confidence_adjustment
        ))
        
        # 5. Générer le verdict
        is_acceptable = final_score >= self.min_acceptable_score
        primary_issues = self._identify_primary_issues(
            acoustic_result, linguistic_result, semantic_result
        )
        recommended_action = self._determine_action(
            final_score, primary_issues
        )
        
        return VQVAnalysisResult(
            audio_id=audio_id,
            source_text=source_text,
            processing_timestamp=datetime.now(),
            final_score=final_score,
            acoustic_score=acoustic_result.score,
            linguistic_score=linguistic_result.score,
            semantic_score=semantic_result.score,
            acoustic_result=acoustic_result,
            linguistic_result=linguistic_result,
            semantic_result=semantic_result,
            is_acceptable=is_acceptable,
            primary_issues=primary_issues,
            recommended_action=recommended_action,
            audio_duration_ms=audio_duration_ms,
            processing_time_ms=processing_time_ms,
            content_type=self.config.content_type.value,
        )
    
    def _compute_weighted_score(self, acoustic_score: float,
                                 linguistic_score: float,
                                 semantic_score: float) -> float:
        """
        Calcule le score pondéré de base.
        
        Les poids sont configurables par type de contenu.
        """
        weighted = (
            acoustic_score * self.weight_acoustic +
            linguistic_score * self.weight_linguistic +
            semantic_score * self.weight_semantic
        )
        
        # Vérifier que les poids somment à 1
        total_weight = self.weight_acoustic + self.weight_linguistic + self.weight_semantic
        if total_weight != 1.0:
            weighted /= total_weight
        
        return weighted
    
    def _detect_cross_layer_patterns(self,
                                      acoustic_anomalies: List[Anomaly],
                                      linguistic_anomalies: List[Anomaly],
                                      semantic_anomalies: List[Anomaly]) -> float:
        """
        Détecte les patterns d'anomalies corrélées entre couches.
        
        Certaines combinaisons d'anomalies sont plus graves que la somme
        de leurs parties car elles indiquent un problème systémique.
        """
        total_penalty = 0.0
        
        acoustic_types = {a.anomaly_type for a in acoustic_anomalies}
        linguistic_types = {a.anomaly_type for a in linguistic_anomalies}
        semantic_types = {a.anomaly_type for a in semantic_anomalies}
        
        # Patterns acoustique-linguistique
        for (a_type, l_type), penalty in self.CROSS_LAYER_PATTERNS.items():
            if a_type in acoustic_types and l_type in linguistic_types:
                # Vérifier le chevauchement temporel
                overlap = self._check_temporal_overlap(
                    [a for a in acoustic_anomalies if a.anomaly_type == a_type],
                    [a for a in linguistic_anomalies if a.anomaly_type == l_type]
                )
                if overlap:
                    total_penalty += penalty * overlap
                    logger.debug(f"Pattern cross-layer détecté: {a_type.value} + {l_type.value}")
        
        # Patterns linguistique-sémantique
        for (l_type, s_type), penalty in self.LINGUISTIC_SEMANTIC_PATTERNS.items():
            if l_type in linguistic_types and s_type in semantic_types:
                overlap = self._check_temporal_overlap(
                    [a for a in linguistic_anomalies if a.anomaly_type == l_type],
                    [a for a in semantic_anomalies if a.anomaly_type == s_type]
                )
                if overlap:
                    total_penalty += penalty * overlap
                    logger.debug(f"Pattern cross-layer détecté: {l_type.value} + {s_type.value}")
        
        # Pattern triple: anomalie dans les 3 couches au même endroit
        triple_penalty = self._detect_triple_pattern(
            acoustic_anomalies, linguistic_anomalies, semantic_anomalies
        )
        total_penalty += triple_penalty
        
        return total_penalty
    
    def _check_temporal_overlap(self, anomalies1: List[Anomaly],
                                 anomalies2: List[Anomaly]) -> float:
        """
        Vérifie le chevauchement temporel entre deux listes d'anomalies.
        
        Retourne un facteur 0-1 indiquant le degré de chevauchement.
        """
        if not anomalies1 or not anomalies2:
            return 0.0
        
        overlap_count = 0
        for a1 in anomalies1:
            for a2 in anomalies2:
                if a1.time_range.overlaps(a2.time_range):
                    overlap_count += 1
        
        max_overlaps = min(len(anomalies1), len(anomalies2))
        return overlap_count / max_overlaps if max_overlaps > 0 else 0.0
    
    def _detect_triple_pattern(self,
                                acoustic_anomalies: List[Anomaly],
                                linguistic_anomalies: List[Anomaly],
                                semantic_anomalies: List[Anomaly]) -> float:
        """
        Détecte les segments avec des anomalies dans les 3 couches simultanément.
        
        C'est un indicateur fort d'hallucination sévère.
        """
        penalty = 0.0
        
        for a_anom in acoustic_anomalies:
            for l_anom in linguistic_anomalies:
                if not a_anom.time_range.overlaps(l_anom.time_range):
                    continue
                    
                for s_anom in semantic_anomalies:
                    if (a_anom.time_range.overlaps(s_anom.time_range) and
                        l_anom.time_range.overlaps(s_anom.time_range)):
                        # Triple chevauchement!
                        severity_factor = max(
                            self._severity_to_factor(a_anom.severity),
                            self._severity_to_factor(l_anom.severity),
                            self._severity_to_factor(s_anom.severity)
                        )
                        penalty += 20 * severity_factor
                        logger.warning(
                            f"Triple pattern détecté: {a_anom.anomaly_type.value} + "
                            f"{l_anom.anomaly_type.value} + {s_anom.anomaly_type.value}"
                        )
        
        return penalty
    
    def _severity_to_factor(self, severity: SeverityLevel) -> float:
        """Convertit un niveau de sévérité en facteur multiplicatif."""
        return {
            SeverityLevel.LOW: 0.5,
            SeverityLevel.MEDIUM: 0.75,
            SeverityLevel.HIGH: 1.0,
            SeverityLevel.CRITICAL: 1.5,
        }.get(severity, 1.0)
    
    def _calibrate_confidence(self,
                               acoustic_result: AcousticAnalysisResult,
                               linguistic_result: LinguisticAnalysisResult,
                               semantic_result: SemanticAnalysisResult) -> float:
        """
        Ajuste le score basé sur la cohérence inter-couches.
        
        Si les trois couches concordent, on peut être plus confiant.
        Si elles divergent significativement, le score est moins fiable.
        """
        scores = [
            acoustic_result.score,
            linguistic_result.score,
            semantic_result.score
        ]
        
        # Écart-type des scores
        score_std = np.std(scores)
        
        # Si les scores sont cohérents (std faible), bonus
        # Si incohérents (std élevé), pas d'ajustement
        if score_std < 5:
            # Très cohérent - bonus de confiance
            return 3.0
        elif score_std < 10:
            return 1.0
        elif score_std < 20:
            return 0.0
        else:
            # Très incohérent - légère pénalité
            return -2.0
    
    def _identify_primary_issues(self,
                                  acoustic_result: AcousticAnalysisResult,
                                  linguistic_result: LinguisticAnalysisResult,
                                  semantic_result: SemanticAnalysisResult) -> List[str]:
        """
        Identifie les problèmes principaux à reporter.
        """
        issues = []
        
        all_anomalies = (
            acoustic_result.anomalies +
            linguistic_result.anomalies +
            semantic_result.anomalies
        )
        
        # Trier par sévérité puis par confiance
        sorted_anomalies = sorted(
            all_anomalies,
            key=lambda a: (
                -list(SeverityLevel).index(a.severity),
                -a.confidence
            )
        )
        
        # Prendre les top 5 problèmes uniques
        seen_types = set()
        for anomaly in sorted_anomalies:
            if anomaly.anomaly_type not in seen_types:
                issues.append(anomaly.description)
                seen_types.add(anomaly.anomaly_type)
                if len(issues) >= 5:
                    break
        
        # Ajouter des métriques importantes si problématiques
        if linguistic_result.mean_word_confidence < 0.6:
            issues.append(
                f"Confiance ASR globale faible ({linguistic_result.mean_word_confidence:.0%})"
            )
        
        if semantic_result.overall_similarity < 0.7:
            issues.append(
                f"Alignement sémantique faible ({semantic_result.overall_similarity:.0%})"
            )
        
        if acoustic_result.distortion_score > 0.15:
            issues.append(
                f"Distorsion audio détectée (THD={acoustic_result.distortion_score:.1%})"
            )
        
        return issues[:5]  # Maximum 5 issues
    
    def _determine_action(self, score: float, issues: List[str]) -> str:
        """
        Détermine l'action recommandée basée sur le score et les problèmes.
        """
        if score >= 85:
            return "accept"
        elif score >= self.min_acceptable_score:
            # Acceptable mais avec réserves
            if any("hallucination" in issue.lower() for issue in issues):
                return "manual_review"
            return "accept"
        elif score >= 50:
            return "regenerate"
        elif score >= 30:
            # Score bas mais potentiellement récupérable
            return "regenerate"
        else:
            # Score très bas - probablement irrécupérable
            return "manual_review"


class AdaptiveScoreFusion(ScoreFusionEngine):
    """
    Version adaptative du moteur de fusion qui ajuste les poids
    dynamiquement basé sur la fiabilité de chaque couche.
    """
    
    def __init__(self, config: ContentTypeConfig):
        super().__init__(config)
        self.base_weights = {
            'acoustic': config.weight_acoustic,
            'linguistic': config.weight_linguistic,
            'semantic': config.weight_semantic,
        }
    
    def _compute_weighted_score(self, acoustic_score: float,
                                 linguistic_score: float,
                                 semantic_score: float) -> float:
        """
        Calcule le score pondéré avec poids adaptatifs.
        
        Si une couche a un score très bas ou très haut (extrême),
        son poids est réduit car elle est moins fiable.
        """
        # Calculer la fiabilité de chaque couche (proche de 50 = plus fiable)
        def reliability(score):
            # Score extrêmes (0 ou 100) = moins fiable
            return 1 - abs(score - 50) / 50 * 0.5
        
        reliabilities = {
            'acoustic': reliability(acoustic_score),
            'linguistic': reliability(linguistic_score),
            'semantic': reliability(semantic_score),
        }
        
        # Ajuster les poids
        adjusted_weights = {
            k: self.base_weights[k] * reliabilities[k]
            for k in self.base_weights
        }
        
        # Normaliser
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        else:
            adjusted_weights = self.base_weights
        
        # Calculer le score
        weighted = (
            acoustic_score * adjusted_weights['acoustic'] +
            linguistic_score * adjusted_weights['linguistic'] +
            semantic_score * adjusted_weights['semantic']
        )
        
        logger.debug(f"Poids adaptatifs: {adjusted_weights}")
        
        return weighted
