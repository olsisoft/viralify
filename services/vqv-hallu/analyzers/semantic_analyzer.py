"""
VQV-HALLU Layer 3: Semantic Analyzer
Alignement sémantique et détection des frontières d'hallucination
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import logging
from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

from models.data_models import (
    SemanticAnalysisResult, Anomaly, AnomalyType,
    SeverityLevel, TimeRange, WordAlignment, TranscriptionResult
)
from config.settings import SemanticThresholds, ContentTypeConfig


logger = logging.getLogger(__name__)


@dataclass
class AlignmentSegment:
    """Segment d'alignement entre source et transcription"""
    source_text: str
    transcript_text: str
    similarity: float
    time_range: Optional[TimeRange]
    is_hallucination: bool
    is_missing: bool


class SemanticAnalyzer:
    """
    Analyseur sémantique pour alignement texte source ↔ transcription.
    
    Implémente:
    - Calcul de similarité par embeddings (sentence-transformers)
    - Alignement mot-à-mot avec distance de Levenshtein
    - Détection des frontières d'hallucination
    - Calcul de la dérive sémantique
    - Détection de contenu manquant/ajouté
    """
    
    def __init__(self, config: ContentTypeConfig,
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = None):
        self.config = config
        self.thresholds = config.semantic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialiser le modèle d'embeddings
        logger.info(f"Chargement du modèle d'embeddings: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        
        # Cache pour les embeddings
        self._embedding_cache = {}
    
    def analyze(self, source_text: str,
                transcription: TranscriptionResult) -> SemanticAnalysisResult:
        """
        Analyse sémantique complète comparant source et transcription.
        
        Args:
            source_text: Texte source qui a généré l'audio
            transcription: Résultat de la transcription ASR
            
        Returns:
            SemanticAnalysisResult avec score et anomalies
        """
        anomalies = []
        transcript_text = transcription.text
        
        # 1. Calcul de la similarité globale par embeddings
        overall_similarity = self._compute_embedding_similarity(
            source_text, transcript_text
        )
        
        # 2. Alignement mot-à-mot
        word_alignments = self._align_words(
            source_text, transcript_text, transcription.word_timestamps
        )
        
        # 3. Détection des frontières d'hallucination
        hallucination_result = self._detect_hallucination_boundaries(
            source_text, transcript_text, transcription
        )
        anomalies.extend(hallucination_result['anomalies'])
        
        # 4. Calcul de la dérive sémantique
        semantic_drift = self._compute_semantic_drift(
            source_text, transcript_text
        )
        
        # 5. Analyse de couverture du contenu
        coverage_result = self._analyze_content_coverage(
            source_text, transcript_text, word_alignments
        )
        anomalies.extend(coverage_result['anomalies'])
        
        # 6. Détection de contenu extra
        extra_result = self._detect_extra_content(
            source_text, transcript_text, word_alignments
        )
        anomalies.extend(extra_result['anomalies'])
        
        # Vérifier le seuil de similarité global
        if overall_similarity < self.thresholds.min_embedding_similarity:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.SEMANTIC_DRIFT,
                severity=SeverityLevel.HIGH if overall_similarity < 0.5 else SeverityLevel.MEDIUM,
                time_range=TimeRange(0, 0),
                confidence=1.0 - overall_similarity,
                description=f"Similarité sémantique globale faible: {overall_similarity:.1%}",
            ))
        
        # Calculer le score final
        score = self._compute_semantic_score(
            anomalies,
            overall_similarity,
            semantic_drift,
            coverage_result['content_coverage'],
        )
        
        return SemanticAnalysisResult(
            score=score,
            anomalies=anomalies,
            overall_similarity=overall_similarity,
            word_alignments=word_alignments,
            hallucination_boundaries=hallucination_result['boundaries'],
            semantic_drift_score=semantic_drift,
            content_coverage=coverage_result['content_coverage'],
            extra_content_ratio=extra_result['extra_ratio'],
        )
    
    def _compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité cosinus entre les embeddings de deux textes.
        """
        if not text1 or not text2:
            return 0.0
        
        # Vérifier le cache
        cache_key = (text1[:100], text2[:100])  # Clé tronquée
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Calculer les embeddings
        embeddings = self.embedding_model.encode(
            [text1, text2],
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Similarité cosinus
        similarity = float(cosine_similarity(
            embeddings[0].cpu().numpy().reshape(1, -1),
            embeddings[1].cpu().numpy().reshape(1, -1)
        )[0, 0])
        
        # Mettre en cache
        self._embedding_cache[cache_key] = similarity
        
        return similarity
    
    def _align_words(self, source_text: str, transcript_text: str,
                     word_timestamps: List[Dict]) -> List[WordAlignment]:
        """
        Aligne les mots du texte source avec la transcription.
        
        Utilise une combinaison de:
        - Distance de Levenshtein pour similarité lexicale
        - Embeddings pour similarité sémantique
        """
        source_words = source_text.lower().split()
        transcript_words = transcript_text.lower().split()
        
        alignments = []
        
        # Utiliser SequenceMatcher pour l'alignement global
        matcher = SequenceMatcher(None, source_words, transcript_words)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Mots identiques
                for k, src_word in enumerate(source_words[i1:i2]):
                    trans_word = transcript_words[j1 + k]
                    ts = word_timestamps[j1 + k] if j1 + k < len(word_timestamps) else None
                    
                    alignments.append(WordAlignment(
                        source_word=src_word,
                        transcribed_word=trans_word,
                        time_range=TimeRange(ts['start_ms'], ts['end_ms']) if ts else None,
                        confidence=1.0,
                        is_match=True,
                        phoneme_similarity=1.0,
                    ))
            
            elif tag == 'replace':
                # Mots différents - calculer similarité
                for k, src_word in enumerate(source_words[i1:i2]):
                    if j1 + k < j2:
                        trans_word = transcript_words[j1 + k]
                        ts = word_timestamps[j1 + k] if j1 + k < len(word_timestamps) else None
                        
                        # Similarité Levenshtein
                        lev_sim = 1 - Levenshtein.distance(src_word, trans_word) / max(len(src_word), len(trans_word))
                        
                        alignments.append(WordAlignment(
                            source_word=src_word,
                            transcribed_word=trans_word,
                            time_range=TimeRange(ts['start_ms'], ts['end_ms']) if ts else None,
                            confidence=lev_sim,
                            is_match=lev_sim > 0.8,
                            phoneme_similarity=lev_sim,
                        ))
                    else:
                        # Mot source sans correspondance
                        alignments.append(WordAlignment(
                            source_word=src_word,
                            transcribed_word=None,
                            time_range=None,
                            confidence=0.0,
                            is_match=False,
                            phoneme_similarity=0.0,
                        ))
            
            elif tag == 'delete':
                # Mots dans la source mais pas dans la transcription
                for src_word in source_words[i1:i2]:
                    alignments.append(WordAlignment(
                        source_word=src_word,
                        transcribed_word=None,
                        time_range=None,
                        confidence=0.0,
                        is_match=False,
                        phoneme_similarity=0.0,
                    ))
            
            elif tag == 'insert':
                # Mots dans la transcription mais pas dans la source (hallucination potentielle)
                for k in range(j1, j2):
                    trans_word = transcript_words[k]
                    ts = word_timestamps[k] if k < len(word_timestamps) else None
                    
                    alignments.append(WordAlignment(
                        source_word="",  # Pas de mot source
                        transcribed_word=trans_word,
                        time_range=TimeRange(ts['start_ms'], ts['end_ms']) if ts else None,
                        confidence=0.0,
                        is_match=False,
                        phoneme_similarity=0.0,
                    ))
        
        return alignments
    
    def _detect_hallucination_boundaries(self, source_text: str,
                                          transcript_text: str,
                                          transcription: TranscriptionResult) -> dict:
        """
        Détecte les frontières où le contenu diverge significativement (hallucinations).
        
        Principe de "triangulation phonético-sémantique":
        - Divise le texte en segments
        - Compare la similarité sémantique de chaque segment
        - Identifie les segments où la similarité chute brutalement
        """
        anomalies = []
        hallucination_boundaries = []
        
        # Diviser en phrases/segments
        source_sentences = self._split_into_sentences(source_text)
        transcript_sentences = self._split_into_sentences(transcript_text)
        
        if not source_sentences or not transcript_sentences:
            return {'boundaries': [], 'anomalies': []}
        
        # Calculer les embeddings de toutes les phrases
        source_embeddings = self.embedding_model.encode(
            source_sentences, convert_to_tensor=True, show_progress_bar=False
        )
        transcript_embeddings = self.embedding_model.encode(
            transcript_sentences, convert_to_tensor=True, show_progress_bar=False
        )
        
        # Matrice de similarité
        sim_matrix = cosine_similarity(
            source_embeddings.cpu().numpy(),
            transcript_embeddings.cpu().numpy()
        )
        
        # Pour chaque phrase de la transcription, trouver la meilleure correspondance
        word_timestamps = transcription.word_timestamps
        char_to_timestamp = self._build_char_timestamp_map(
            transcript_text, word_timestamps
        )
        
        current_pos = 0
        for i, trans_sent in enumerate(transcript_sentences):
            # Meilleure similarité avec une phrase source
            best_sim = float(np.max(sim_matrix[:, i]))
            
            # Position dans le texte
            sent_start = transcript_text.find(trans_sent, current_pos)
            sent_end = sent_start + len(trans_sent)
            current_pos = sent_end
            
            # Obtenir les timestamps
            time_range = self._get_timestamp_range(
                char_to_timestamp, sent_start, sent_end
            )
            
            if best_sim < self.thresholds.hallucination_boundary_threshold:
                # Hallucination détectée
                hallucination_boundaries.append(time_range)
                
                severity = (SeverityLevel.CRITICAL if best_sim < 0.3
                           else SeverityLevel.HIGH if best_sim < 0.5
                           else SeverityLevel.MEDIUM)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.HALLUCINATION,
                    severity=severity,
                    time_range=time_range,
                    confidence=1.0 - best_sim,
                    description=f"Hallucination probable: '{trans_sent[:50]}...' (sim={best_sim:.1%})",
                    raw_data={
                        'transcript_segment': trans_sent,
                        'best_similarity': best_sim,
                    }
                ))
        
        # Détecter les transitions brusques (frontières)
        for i in range(1, len(transcript_sentences)):
            prev_sim = float(np.max(sim_matrix[:, i-1]))
            curr_sim = float(np.max(sim_matrix[:, i]))
            
            # Chute de similarité > 30%
            if prev_sim - curr_sim > 0.3:
                # Trouver la position de transition
                sent_start = transcript_text.find(transcript_sentences[i])
                time_range = self._get_timestamp_range(
                    char_to_timestamp, sent_start, sent_start + 10
                )
                
                if time_range not in hallucination_boundaries:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.HALLUCINATION,
                        severity=SeverityLevel.HIGH,
                        time_range=time_range,
                        confidence=prev_sim - curr_sim,
                        description=f"Frontière d'hallucination détectée (chute de {prev_sim:.0%} à {curr_sim:.0%})",
                    ))
        
        return {
            'boundaries': hallucination_boundaries,
            'anomalies': anomalies,
        }
    
    def _compute_semantic_drift(self, source_text: str, 
                                 transcript_text: str) -> float:
        """
        Calcule la dérive sémantique progressive entre source et transcription.
        
        Divise les textes en quartiles et compare la dérive de similarité.
        """
        source_parts = self._split_into_parts(source_text, 4)
        transcript_parts = self._split_into_parts(transcript_text, 4)
        
        if len(source_parts) < 2 or len(transcript_parts) < 2:
            return 0.0
        
        similarities = []
        for sp, tp in zip(source_parts, transcript_parts):
            if sp and tp:
                sim = self._compute_embedding_similarity(sp, tp)
                similarities.append(sim)
        
        if len(similarities) < 2:
            return 0.0
        
        # La dérive est la différence entre le début et la fin
        drift = similarities[0] - similarities[-1]
        
        # Aussi considérer la variance (instabilité)
        variance = np.var(similarities)
        
        # Score combiné: dérive + instabilité
        drift_score = max(0, drift) + variance
        
        return float(min(1.0, drift_score))
    
    def _analyze_content_coverage(self, source_text: str, 
                                   transcript_text: str,
                                   alignments: List[WordAlignment]) -> dict:
        """
        Analyse le pourcentage du contenu source couvert par la transcription.
        """
        anomalies = []
        
        # Compter les mots source couverts
        source_words = source_text.lower().split()
        covered_words = sum(1 for a in alignments if a.is_match and a.source_word)
        
        if not source_words:
            return {'content_coverage': 1.0, 'anomalies': []}
        
        content_coverage = covered_words / len(source_words)
        
        # Identifier les mots manquants
        missing_words = [a.source_word for a in alignments 
                        if not a.is_match and a.source_word and not a.transcribed_word]
        
        if content_coverage < 0.8:
            severity = (SeverityLevel.HIGH if content_coverage < 0.5
                       else SeverityLevel.MEDIUM)
            
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.MISSING_CONTENT,
                severity=severity,
                time_range=TimeRange(0, 0),
                confidence=1.0 - content_coverage,
                description=f"Contenu source non couvert: {1-content_coverage:.0%} ({len(missing_words)} mots)",
                raw_data={'missing_words': missing_words[:20]},  # Limiter pour éviter messages trop longs
            ))
        
        return {
            'content_coverage': content_coverage,
            'anomalies': anomalies,
        }
    
    def _detect_extra_content(self, source_text: str,
                               transcript_text: str,
                               alignments: List[WordAlignment]) -> dict:
        """
        Détecte le contenu dans la transcription qui n'est pas dans la source.
        """
        anomalies = []
        
        # Mots dans la transcription sans correspondance source
        extra_words = [a.transcribed_word for a in alignments 
                      if not a.is_match and a.transcribed_word and not a.source_word]
        
        transcript_words = transcript_text.lower().split()
        if not transcript_words:
            return {'extra_ratio': 0.0, 'anomalies': []}
        
        extra_ratio = len(extra_words) / len(transcript_words)
        
        # Regrouper les mots extra consécutifs
        extra_segments = []
        current_segment = []
        
        for a in alignments:
            if not a.is_match and a.transcribed_word and not a.source_word:
                current_segment.append(a)
            elif current_segment:
                if len(current_segment) >= 3:  # Au moins 3 mots extra consécutifs
                    extra_segments.append(current_segment)
                current_segment = []
        
        if current_segment and len(current_segment) >= 3:
            extra_segments.append(current_segment)
        
        for segment in extra_segments:
            words = ' '.join(a.transcribed_word for a in segment)
            first_ts = segment[0].time_range
            last_ts = segment[-1].time_range
            
            if first_ts and last_ts:
                time_range = TimeRange(first_ts.start_ms, last_ts.end_ms)
            else:
                time_range = TimeRange(0, 0)
            
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.EXTRA_CONTENT,
                severity=SeverityLevel.HIGH if len(segment) >= 5 else SeverityLevel.MEDIUM,
                time_range=time_range,
                confidence=min(1.0, len(segment) / 10),
                description=f"Contenu ajouté non présent dans la source: '{words[:50]}...'",
            ))
        
        return {
            'extra_ratio': extra_ratio,
            'anomalies': anomalies,
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Divise un texte en phrases."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_parts(self, text: str, n: int) -> List[str]:
        """Divise un texte en n parties égales."""
        words = text.split()
        if len(words) < n:
            return [text]
        
        part_size = len(words) // n
        parts = []
        for i in range(n):
            start = i * part_size
            end = start + part_size if i < n - 1 else len(words)
            parts.append(' '.join(words[start:end]))
        
        return parts
    
    def _build_char_timestamp_map(self, text: str,
                                   word_timestamps: List[Dict]) -> Dict[int, int]:
        """
        Construit une map caractère -> timestamp approximatif.
        """
        char_map = {}
        current_pos = 0
        
        for wt in word_timestamps:
            word = wt['word']
            word_start = text.find(word, current_pos)
            if word_start == -1:
                continue
            
            word_end = word_start + len(word)
            
            # Interpolation linéaire pour chaque caractère
            duration = wt['end_ms'] - wt['start_ms']
            for i, pos in enumerate(range(word_start, word_end)):
                char_map[pos] = wt['start_ms'] + int(i / len(word) * duration)
            
            current_pos = word_end
        
        return char_map
    
    def _get_timestamp_range(self, char_map: Dict[int, int],
                              start_char: int, end_char: int) -> TimeRange:
        """Obtient la plage temporelle pour une plage de caractères."""
        start_ms = 0
        end_ms = 0
        
        for pos in range(start_char, end_char):
            if pos in char_map:
                if start_ms == 0:
                    start_ms = char_map[pos]
                end_ms = char_map[pos]
        
        return TimeRange(start_ms, end_ms)
    
    def _compute_semantic_score(self, anomalies: List[Anomaly],
                                 overall_similarity: float,
                                 semantic_drift: float,
                                 content_coverage: float) -> float:
        """
        Calcule le score sémantique final.
        """
        # Score de base
        base_score = (
            overall_similarity * 40 +        # 40% similarité globale
            (1 - semantic_drift) * 20 +      # 20% stabilité
            content_coverage * 40             # 40% couverture
        )
        
        # Pénalités
        severity_penalties = {
            SeverityLevel.LOW: 3,
            SeverityLevel.MEDIUM: 8,
            SeverityLevel.HIGH: 15,
            SeverityLevel.CRITICAL: 30,
        }
        
        for anomaly in anomalies:
            penalty = severity_penalties[anomaly.severity] * anomaly.confidence
            base_score -= penalty
        
        return max(0.0, min(100.0, base_score))
