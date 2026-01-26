"""
VQV-HALLU Data Models
Structures de données pour les analyses et résultats
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import json


class AnomalyType(Enum):
    """Types d'anomalies détectées"""
    # Acoustiques
    DISTORTION = "distortion"
    CLICK_POP = "click_pop"
    SILENCE_EXCESSIVE = "silence_excessive"
    PACE_TOO_FAST = "pace_too_fast"
    PACE_TOO_SLOW = "pace_too_slow"
    SPECTRAL_ANOMALY = "spectral_anomaly"
    
    # Linguistiques
    GIBBERISH = "gibberish"
    UNKNOWN_PHONEMES = "unknown_phonemes"
    LOW_ASR_CONFIDENCE = "low_asr_confidence"
    LANGUAGE_SWITCH = "language_switch"
    WORD_REPETITION = "word_repetition"
    
    # Sémantiques
    HALLUCINATION = "hallucination"
    SEMANTIC_DRIFT = "semantic_drift"
    MISSING_CONTENT = "missing_content"
    EXTRA_CONTENT = "extra_content"


class SeverityLevel(Enum):
    """Niveaux de sévérité des anomalies"""
    LOW = "low"           # Score impact: 5-10%
    MEDIUM = "medium"     # Score impact: 10-25%
    HIGH = "high"         # Score impact: 25-50%
    CRITICAL = "critical" # Score impact: >50%


@dataclass
class TimeRange:
    """Plage temporelle dans l'audio"""
    start_ms: int
    end_ms: int
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
    
    def overlaps(self, other: "TimeRange") -> bool:
        return self.start_ms < other.end_ms and self.end_ms > other.start_ms
    
    def to_dict(self) -> Dict[str, int]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms}


@dataclass
class Anomaly:
    """Représentation d'une anomalie détectée"""
    anomaly_type: AnomalyType
    severity: SeverityLevel
    time_range: TimeRange
    confidence: float           # 0.0 - 1.0
    description: str
    raw_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "time_range": self.time_range.to_dict(),
            "confidence": self.confidence,
            "description": self.description,
            "raw_data": self.raw_data,
        }


@dataclass
class WordAlignment:
    """Alignement mot-à-mot entre source et transcription"""
    source_word: str
    transcribed_word: Optional[str]
    time_range: Optional[TimeRange]
    confidence: float
    is_match: bool
    phoneme_similarity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_word": self.source_word,
            "transcribed_word": self.transcribed_word,
            "time_range": self.time_range.to_dict() if self.time_range else None,
            "confidence": self.confidence,
            "is_match": self.is_match,
            "phoneme_similarity": self.phoneme_similarity,
        }


@dataclass 
class TranscriptionResult:
    """Résultat de la transcription ASR inverse"""
    text: str
    language: str
    confidence: float
    word_timestamps: List[Dict[str, Any]]
    segments: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "word_timestamps": self.word_timestamps,
            "segments": self.segments,
        }


@dataclass
class AcousticAnalysisResult:
    """Résultat de l'analyse acoustique (Layer 1)"""
    score: float                    # 0-100
    anomalies: List[Anomaly]
    
    # Métriques détaillées
    spectral_flatness_mean: float
    spectral_flatness_std: float
    silence_ratio: float
    estimated_speech_rate_wpm: float
    distortion_score: float
    click_count: int
    
    # Statistiques spectrales
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "metrics": {
                "spectral_flatness_mean": self.spectral_flatness_mean,
                "spectral_flatness_std": self.spectral_flatness_std,
                "silence_ratio": self.silence_ratio,
                "estimated_speech_rate_wpm": self.estimated_speech_rate_wpm,
                "distortion_score": self.distortion_score,
                "click_count": self.click_count,
                "spectral_centroid_mean": self.spectral_centroid_mean,
                "spectral_bandwidth_mean": self.spectral_bandwidth_mean,
            }
        }


@dataclass
class LinguisticAnalysisResult:
    """Résultat de l'analyse linguistique (Layer 2)"""
    score: float                    # 0-100
    anomalies: List[Anomaly]
    transcription: TranscriptionResult
    
    # Métriques détaillées
    mean_word_confidence: float
    phoneme_validity_score: float
    detected_languages: List[Tuple[str, float]]  # (lang_code, ratio)
    gibberish_segments: List[TimeRange]
    unknown_phoneme_ratio: float
    word_repetition_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "transcription": self.transcription.to_dict(),
            "metrics": {
                "mean_word_confidence": self.mean_word_confidence,
                "phoneme_validity_score": self.phoneme_validity_score,
                "detected_languages": self.detected_languages,
                "gibberish_segments": [s.to_dict() for s in self.gibberish_segments],
                "unknown_phoneme_ratio": self.unknown_phoneme_ratio,
                "word_repetition_count": self.word_repetition_count,
            }
        }


@dataclass
class SemanticAnalysisResult:
    """Résultat de l'analyse sémantique (Layer 3)"""
    score: float                    # 0-100
    anomalies: List[Anomaly]
    
    # Métriques détaillées
    overall_similarity: float       # Similarité embedding globale
    word_alignments: List[WordAlignment]
    hallucination_boundaries: List[TimeRange]
    semantic_drift_score: float
    content_coverage: float         # % du texte source couvert
    extra_content_ratio: float      # % de contenu non-source
    
    # Embeddings (optionnel, pour debug)
    source_embedding: Optional[List[float]] = None
    transcript_embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "metrics": {
                "overall_similarity": self.overall_similarity,
                "word_alignments": [w.to_dict() for w in self.word_alignments],
                "hallucination_boundaries": [h.to_dict() for h in self.hallucination_boundaries],
                "semantic_drift_score": self.semantic_drift_score,
                "content_coverage": self.content_coverage,
                "extra_content_ratio": self.extra_content_ratio,
            }
        }


@dataclass
class VQVAnalysisResult:
    """Résultat complet de l'analyse VQV-HALLU"""
    # Identifiants
    audio_id: str
    source_text: str
    processing_timestamp: datetime
    
    # Scores
    final_score: float              # 0-100 (score fusionné)
    acoustic_score: float
    linguistic_score: float
    semantic_score: float
    
    # Résultats détaillés par couche
    acoustic_result: AcousticAnalysisResult
    linguistic_result: LinguisticAnalysisResult
    semantic_result: SemanticAnalysisResult
    
    # Verdict
    is_acceptable: bool
    primary_issues: List[str]
    recommended_action: str         # "accept", "regenerate", "manual_review"
    
    # Métadonnées
    audio_duration_ms: int
    processing_time_ms: int
    content_type: str
    config_version: str = "1.0.0"
    
    def get_all_anomalies(self) -> List[Anomaly]:
        """Retourne toutes les anomalies détectées"""
        return (
            self.acoustic_result.anomalies +
            self.linguistic_result.anomalies +
            self.semantic_result.anomalies
        )
    
    def get_critical_anomalies(self) -> List[Anomaly]:
        """Retourne les anomalies critiques"""
        return [a for a in self.get_all_anomalies() 
                if a.severity == SeverityLevel.CRITICAL]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_id": self.audio_id,
            "source_text": self.source_text,
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "scores": {
                "final": self.final_score,
                "acoustic": self.acoustic_score,
                "linguistic": self.linguistic_score,
                "semantic": self.semantic_score,
            },
            "layers": {
                "acoustic": self.acoustic_result.to_dict(),
                "linguistic": self.linguistic_result.to_dict(),
                "semantic": self.semantic_result.to_dict(),
            },
            "verdict": {
                "is_acceptable": self.is_acceptable,
                "primary_issues": self.primary_issues,
                "recommended_action": self.recommended_action,
            },
            "metadata": {
                "audio_duration_ms": self.audio_duration_ms,
                "processing_time_ms": self.processing_time_ms,
                "content_type": self.content_type,
                "config_version": self.config_version,
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class VQVInputMessage:
    """Message d'entrée pour le traitement"""
    audio_id: str
    audio_path: str                 # Chemin ou URL vers le fichier audio
    source_text: str                # Texte source qui a généré l'audio
    content_type: str = "mixed"     # Type de contenu
    language: str = "fr"            # Langue attendue
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> "VQVInputMessage":
        data = json.loads(json_str)
        return cls(**data)
    
    def to_json(self) -> str:
        return json.dumps({
            "audio_id": self.audio_id,
            "audio_path": self.audio_path,
            "source_text": self.source_text,
            "content_type": self.content_type,
            "language": self.language,
            "metadata": self.metadata,
        })


@dataclass
class VQVOutputMessage:
    """Message de sortie après traitement"""
    audio_id: str
    status: str                     # "success", "failed", "error"
    result: Optional[VQVAnalysisResult]
    error_message: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps({
            "audio_id": self.audio_id,
            "status": self.status,
            "result": self.result.to_dict() if self.result else None,
            "error_message": self.error_message,
        }, ensure_ascii=False)
