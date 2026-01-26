"""
VQV-HALLU Main Pipeline
Orchestration des analyseurs multi-couches
"""

import time
import logging
from typing import Optional
from pathlib import Path
import tempfile
import requests
from urllib.parse import urlparse

from analyzers.acoustic_analyzer import AcousticAnalyzer
from analyzers.linguistic_analyzer import LinguisticAnalyzer
from analyzers.semantic_analyzer import SemanticAnalyzer
from core.score_fusion import ScoreFusionEngine, AdaptiveScoreFusion
from models.data_models import (
    VQVAnalysisResult, VQVInputMessage, VQVOutputMessage
)
from config.settings import (
    VQVHalluConfig, ContentType, ContentTypeConfig,
    get_config_for_content_type
)


logger = logging.getLogger(__name__)


class VQVHalluPipeline:
    """
    Pipeline principal VQV-HALLU pour l'analyse de qualité vocale.
    
    Orchestre les 4 couches:
    1. Acoustic Analyzer
    2. Linguistic Analyzer  
    3. Semantic Analyzer
    4. Score Fusion Engine
    """
    
    def __init__(self, 
                 global_config: VQVHalluConfig,
                 use_adaptive_fusion: bool = True):
        """
        Initialise le pipeline.
        
        Args:
            global_config: Configuration globale VQV-HALLU
            use_adaptive_fusion: Utiliser la fusion adaptative des scores
        """
        self.global_config = global_config
        self.use_adaptive_fusion = use_adaptive_fusion
        
        # Les analyseurs seront initialisés à la demande avec la config du content type
        self._acoustic_analyzers = {}
        self._linguistic_analyzers = {}
        self._semantic_analyzers = {}
        self._fusion_engines = {}
        
        # Créer le répertoire temporaire
        Path(global_config.temp_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Pipeline VQV-HALLU initialisé")
    
    def _get_content_config(self, content_type_str: str) -> ContentTypeConfig:
        """Récupère la configuration pour un type de contenu."""
        try:
            content_type = ContentType(content_type_str)
        except ValueError:
            logger.warning(f"Type de contenu inconnu: {content_type_str}, utilisation de MIXED")
            content_type = ContentType.MIXED
        
        return get_config_for_content_type(content_type)
    
    def _get_acoustic_analyzer(self, config: ContentTypeConfig) -> AcousticAnalyzer:
        """Récupère ou crée l'analyseur acoustique pour une config."""
        key = config.content_type.value
        if key not in self._acoustic_analyzers:
            self._acoustic_analyzers[key] = AcousticAnalyzer(config)
        return self._acoustic_analyzers[key]
    
    def _get_linguistic_analyzer(self, config: ContentTypeConfig) -> LinguisticAnalyzer:
        """Récupère ou crée l'analyseur linguistique pour une config."""
        key = config.content_type.value
        if key not in self._linguistic_analyzers:
            self._linguistic_analyzers[key] = LinguisticAnalyzer(
                config,
                whisper_model=self.global_config.asr_model
            )
        return self._linguistic_analyzers[key]
    
    def _get_semantic_analyzer(self, config: ContentTypeConfig) -> SemanticAnalyzer:
        """Récupère ou crée l'analyseur sémantique pour une config."""
        key = config.content_type.value
        if key not in self._semantic_analyzers:
            self._semantic_analyzers[key] = SemanticAnalyzer(
                config,
                embedding_model=self.global_config.embedding_model
            )
        return self._semantic_analyzers[key]
    
    def _get_fusion_engine(self, config: ContentTypeConfig) -> ScoreFusionEngine:
        """Récupère ou crée le moteur de fusion pour une config."""
        key = config.content_type.value
        if key not in self._fusion_engines:
            if self.use_adaptive_fusion:
                self._fusion_engines[key] = AdaptiveScoreFusion(config)
            else:
                self._fusion_engines[key] = ScoreFusionEngine(config)
        return self._fusion_engines[key]
    
    def analyze(self, 
                audio_path: str,
                source_text: str,
                audio_id: str,
                content_type: str = "mixed",
                language: str = "fr") -> VQVAnalysisResult:
        """
        Analyse complète d'un fichier audio.
        
        Args:
            audio_path: Chemin vers le fichier audio (local ou URL)
            source_text: Texte source qui a généré l'audio
            audio_id: Identifiant unique
            content_type: Type de contenu
            language: Langue attendue
            
        Returns:
            VQVAnalysisResult complet
        """
        start_time = time.time()
        
        # Obtenir la configuration
        config = self._get_content_config(content_type)
        
        # Télécharger si URL
        local_path = self._ensure_local_file(audio_path)
        
        try:
            # Obtenir la durée audio
            import librosa
            audio_duration = librosa.get_duration(path=local_path)
            audio_duration_ms = int(audio_duration * 1000)
            
            logger.info(f"Analyse de {audio_id} ({audio_duration:.1f}s, {content_type})")
            
            # Layer 1: Analyse acoustique
            logger.debug("Layer 1: Analyse acoustique...")
            acoustic_analyzer = self._get_acoustic_analyzer(config)
            acoustic_result = acoustic_analyzer.analyze(local_path)
            logger.info(f"  Acoustic score: {acoustic_result.score:.1f}")
            
            # Layer 2: Analyse linguistique
            logger.debug("Layer 2: Analyse linguistique...")
            linguistic_analyzer = self._get_linguistic_analyzer(config)
            linguistic_result = linguistic_analyzer.analyze(local_path, language)
            logger.info(f"  Linguistic score: {linguistic_result.score:.1f}")
            
            # Layer 3: Analyse sémantique
            logger.debug("Layer 3: Analyse sémantique...")
            semantic_analyzer = self._get_semantic_analyzer(config)
            semantic_result = semantic_analyzer.analyze(
                source_text, linguistic_result.transcription
            )
            logger.info(f"  Semantic score: {semantic_result.score:.1f}")
            
            # Layer 4: Fusion des scores
            logger.debug("Layer 4: Fusion des scores...")
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
    
    def analyze_from_message(self, message: VQVInputMessage) -> VQVOutputMessage:
        """
        Analyse à partir d'un message d'entrée.
        
        Args:
            message: Message d'entrée VQV
            
        Returns:
            Message de sortie avec résultat ou erreur
        """
        try:
            result = self.analyze(
                audio_path=message.audio_path,
                source_text=message.source_text,
                audio_id=message.audio_id,
                content_type=message.content_type,
                language=message.language
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
    
    def _ensure_local_file(self, path: str) -> str:
        """
        S'assure que le fichier est local.
        
        Si c'est une URL, télécharge le fichier.
        """
        parsed = urlparse(path)
        
        if parsed.scheme in ('http', 'https'):
            # Télécharger
            logger.debug(f"Téléchargement de {path}")
            
            response = requests.get(path, timeout=60)
            response.raise_for_status()
            
            # Déterminer l'extension
            content_type = response.headers.get('content-type', '')
            if 'wav' in content_type:
                ext = '.wav'
            elif 'mp3' in content_type:
                ext = '.mp3'
            elif 'ogg' in content_type:
                ext = '.ogg'
            else:
                ext = '.wav'  # Par défaut
            
            # Sauvegarder temporairement
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=ext,
                dir=self.global_config.temp_dir
            )
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name
        
        return path
    
    def batch_analyze(self, items: list) -> list:
        """
        Analyse un lot d'items.
        
        Args:
            items: Liste de VQVInputMessage
            
        Returns:
            Liste de VQVOutputMessage
        """
        results = []
        for item in items:
            result = self.analyze_from_message(item)
            results.append(result)
        return results


# Interface simplifiée pour utilisation directe
def analyze_voiceover(audio_path: str,
                      source_text: str,
                      audio_id: str = "default",
                      content_type: str = "mixed",
                      language: str = "fr",
                      config: Optional[VQVHalluConfig] = None) -> VQVAnalysisResult:
    """
    Interface simplifiée pour analyser un voiceover.
    
    Args:
        audio_path: Chemin vers le fichier audio
        source_text: Texte source
        audio_id: Identifiant (optionnel)
        content_type: Type de contenu
        language: Langue
        config: Configuration (optionnel)
        
    Returns:
        Résultat d'analyse
    """
    if config is None:
        config = VQVHalluConfig()
    
    pipeline = VQVHalluPipeline(config)
    return pipeline.analyze(
        audio_path=audio_path,
        source_text=source_text,
        audio_id=audio_id,
        content_type=content_type,
        language=language
    )
