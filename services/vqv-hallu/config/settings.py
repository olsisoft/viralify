"""
VQV-HALLU Configuration Settings
Seuils configurables par type de contenu et paramètres globaux
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import json
import os


class ContentType(Enum):
    """Types de contenu avec seuils différenciés"""
    TECHNICAL_COURSE = "technical_course"      # Cours techniques (code, maths, etc.)
    NARRATIVE = "narrative"                     # Contenu narratif/storytelling
    CONVERSATIONAL = "conversational"           # Style conversationnel
    FORMAL_PRESENTATION = "formal_presentation" # Présentations formelles
    MIXED = "mixed"                             # Contenu mixte (défaut)


@dataclass
class AcousticThresholds:
    """Seuils pour l'analyse acoustique"""
    min_spectral_flatness: float = 0.1          # Détection distorsion
    max_spectral_flatness: float = 0.9          # Détection bruit blanc
    silence_threshold_db: float = -40.0         # Seuil silence en dB
    max_silence_ratio: float = 0.3              # Ratio max de silence
    min_speech_rate_wpm: float = 80.0           # Mots/min minimum
    max_speech_rate_wpm: float = 220.0          # Mots/min maximum
    distortion_threshold: float = 0.15          # Seuil THD (Total Harmonic Distortion)
    click_detection_threshold: float = 3.0      # Écart-type pour détection clics


@dataclass
class LinguisticThresholds:
    """Seuils pour l'analyse linguistique"""
    min_word_confidence: float = 0.6            # Confiance ASR minimum par mot
    min_phoneme_validity: float = 0.7           # Score validité phonème
    language_switch_penalty: float = 0.3        # Pénalité changement langue
    max_unknown_phoneme_ratio: float = 0.1      # Ratio phonèmes inconnus max
    gibberish_detection_threshold: float = 0.4  # Seuil détection charabia


@dataclass
class SemanticThresholds:
    """Seuils pour l'alignement sémantique"""
    min_embedding_similarity: float = 0.75      # Similarité cosinus minimum
    hallucination_boundary_threshold: float = 0.5  # Seuil détection frontière hallucination
    sentence_alignment_tolerance: float = 0.2   # Tolérance alignement phrase
    semantic_drift_max: float = 0.4             # Dérive sémantique maximale


@dataclass
class ContentTypeConfig:
    """Configuration complète pour un type de contenu"""
    content_type: ContentType
    acoustic: AcousticThresholds
    linguistic: LinguisticThresholds
    semantic: SemanticThresholds
    
    # Poids pour la fusion des scores (doivent sommer à 1.0)
    weight_acoustic: float = 0.25
    weight_linguistic: float = 0.35
    weight_semantic: float = 0.40
    
    # Score minimum acceptable
    min_acceptable_score: float = 70.0
    
    # Nombre de tentatives de régénération
    max_regeneration_attempts: int = 3


# Configurations prédéfinies par type de contenu
CONTENT_TYPE_CONFIGS: Dict[ContentType, ContentTypeConfig] = {
    
    ContentType.TECHNICAL_COURSE: ContentTypeConfig(
        content_type=ContentType.TECHNICAL_COURSE,
        acoustic=AcousticThresholds(
            min_speech_rate_wpm=100.0,  # Plus lent pour technique
            max_speech_rate_wpm=180.0,
            max_silence_ratio=0.35,     # Plus de pauses acceptées
        ),
        linguistic=LinguisticThresholds(
            min_word_confidence=0.7,    # Plus strict sur les termes
            gibberish_detection_threshold=0.5,
        ),
        semantic=SemanticThresholds(
            min_embedding_similarity=0.8,  # Alignement strict
            hallucination_boundary_threshold=0.6,
        ),
        weight_acoustic=0.20,
        weight_linguistic=0.40,  # Importance des termes techniques
        weight_semantic=0.40,
        min_acceptable_score=75.0,
    ),
    
    ContentType.NARRATIVE: ContentTypeConfig(
        content_type=ContentType.NARRATIVE,
        acoustic=AcousticThresholds(
            min_speech_rate_wpm=90.0,
            max_speech_rate_wpm=200.0,
            max_silence_ratio=0.25,
        ),
        linguistic=LinguisticThresholds(
            min_word_confidence=0.55,   # Plus flexible
            language_switch_penalty=0.4,
        ),
        semantic=SemanticThresholds(
            min_embedding_similarity=0.7,
            semantic_drift_max=0.5,     # Plus de flexibilité narrative
        ),
        weight_acoustic=0.30,
        weight_linguistic=0.30,
        weight_semantic=0.40,
        min_acceptable_score=65.0,
    ),
    
    ContentType.CONVERSATIONAL: ContentTypeConfig(
        content_type=ContentType.CONVERSATIONAL,
        acoustic=AcousticThresholds(
            min_speech_rate_wpm=100.0,
            max_speech_rate_wpm=220.0,
            max_silence_ratio=0.2,
        ),
        linguistic=LinguisticThresholds(
            min_word_confidence=0.5,
            gibberish_detection_threshold=0.35,
        ),
        semantic=SemanticThresholds(
            min_embedding_similarity=0.65,
        ),
        weight_acoustic=0.35,
        weight_linguistic=0.30,
        weight_semantic=0.35,
        min_acceptable_score=60.0,
    ),
    
    ContentType.FORMAL_PRESENTATION: ContentTypeConfig(
        content_type=ContentType.FORMAL_PRESENTATION,
        acoustic=AcousticThresholds(
            min_speech_rate_wpm=110.0,
            max_speech_rate_wpm=170.0,
            max_silence_ratio=0.3,
            distortion_threshold=0.1,   # Très strict sur la qualité
        ),
        linguistic=LinguisticThresholds(
            min_word_confidence=0.75,
            min_phoneme_validity=0.8,
        ),
        semantic=SemanticThresholds(
            min_embedding_similarity=0.85,
            hallucination_boundary_threshold=0.65,
        ),
        weight_acoustic=0.25,
        weight_linguistic=0.35,
        weight_semantic=0.40,
        min_acceptable_score=80.0,
        max_regeneration_attempts=5,
    ),
    
    ContentType.MIXED: ContentTypeConfig(
        content_type=ContentType.MIXED,
        acoustic=AcousticThresholds(),      # Défauts
        linguistic=LinguisticThresholds(),  # Défauts
        semantic=SemanticThresholds(),      # Défauts
        weight_acoustic=0.25,
        weight_linguistic=0.35,
        weight_semantic=0.40,
        min_acceptable_score=70.0,
    ),
}


@dataclass
class VQVHalluConfig:
    """Configuration globale VQV-HALLU"""
    
    # RabbitMQ
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_queue_input: str = "vqv_hallu_input"
    rabbitmq_queue_output: str = "vqv_hallu_output"
    rabbitmq_queue_failed: str = "vqv_hallu_failed"
    
    # Traitement
    batch_size: int = 10
    max_workers: int = 4
    processing_timeout_seconds: int = 300
    
    # Modèles
    asr_model: str = "openai/whisper-large-v3"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    language_detection_model: str = "facebook/fasttext-language-identification"
    
    # Langues supportées
    supported_languages: list = field(default_factory=lambda: ["fr", "en"])
    primary_language: str = "fr"
    
    # Chemins
    temp_dir: str = "/tmp/vqv_hallu"
    log_dir: str = "/var/log/vqv_hallu"
    
    # Cache
    enable_embedding_cache: bool = True
    cache_ttl_seconds: int = 3600
    
    @classmethod
    def from_env(cls) -> "VQVHalluConfig":
        """Charge la configuration depuis les variables d'environnement"""
        return cls(
            rabbitmq_host=os.getenv("RABBITMQ_HOST", "localhost"),
            rabbitmq_port=int(os.getenv("RABBITMQ_PORT", "5672")),
            rabbitmq_user=os.getenv("RABBITMQ_USER", "guest"),
            rabbitmq_password=os.getenv("RABBITMQ_PASSWORD", "guest"),
            rabbitmq_queue_input=os.getenv("VQV_QUEUE_INPUT", "vqv_hallu_input"),
            rabbitmq_queue_output=os.getenv("VQV_QUEUE_OUTPUT", "vqv_hallu_output"),
            rabbitmq_queue_failed=os.getenv("VQV_QUEUE_FAILED", "vqv_hallu_failed"),
            batch_size=int(os.getenv("VQV_BATCH_SIZE", "10")),
            max_workers=int(os.getenv("VQV_MAX_WORKERS", "4")),
            asr_model=os.getenv("VQV_ASR_MODEL", "openai/whisper-large-v3"),
            embedding_model=os.getenv("VQV_EMBEDDING_MODEL", 
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            primary_language=os.getenv("VQV_PRIMARY_LANGUAGE", "fr"),
        )
    
    @classmethod
    def from_json(cls, path: str) -> "VQVHalluConfig":
        """Charge la configuration depuis un fichier JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def get_config_for_content_type(content_type: ContentType) -> ContentTypeConfig:
    """Récupère la configuration pour un type de contenu"""
    return CONTENT_TYPE_CONFIGS.get(content_type, CONTENT_TYPE_CONFIGS[ContentType.MIXED])
