"""
VQV-HALLU Layer 2: Linguistic Analyzer
Analyse linguistique et cohérence via transcription inverse
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import logging
from dataclasses import dataclass

# ASR et NLP
import torch
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    pipeline
)
import epitran
from phonemizer import phonemize
from langdetect import detect_langs, LangDetectException

from models.data_models import (
    LinguisticAnalysisResult, Anomaly, AnomalyType,
    SeverityLevel, TimeRange, TranscriptionResult
)
from config.settings import LinguisticThresholds, ContentTypeConfig


logger = logging.getLogger(__name__)


@dataclass
class PhonemeAnalysis:
    """Résultat de l'analyse phonémique"""
    phonemes: List[str]
    validity_scores: List[float]
    unknown_ratio: float
    language: str


class LinguisticAnalyzer:
    """
    Analyseur linguistique pour détection d'hallucinations vocales.
    
    Implémente:
    - Transcription ASR inverse (Whisper)
    - Détection automatique de langue
    - Analyse de validité phonémique
    - Détection de charabia (gibberish)
    - Détection de répétitions anormales
    """
    
    # Phonèmes valides par langue (IPA)
    VALID_PHONEMES = {
        'fr': set([
            'a', 'ɑ', 'e', 'ɛ', 'i', 'o', 'ɔ', 'u', 'y', 'ø', 'œ', 'ə',
            'ɑ̃', 'ɛ̃', 'ɔ̃', 'œ̃',  # Voyelles nasales
            'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 's', 'z', 'ʃ', 'ʒ',
            'm', 'n', 'ɲ', 'ŋ', 'l', 'ʁ', 'w', 'ɥ', 'j',
        ]),
        'en': set([
            'æ', 'ɑ', 'ɒ', 'ʌ', 'ə', 'ɜ', 'ɪ', 'i', 'ʊ', 'u', 'e', 'ɛ', 'ɔ',
            'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ',  # Diphtongues
            'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z',
            'ʃ', 'ʒ', 'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j',
        ]),
    }
    
    # Patterns de charabia courants (regex)
    GIBBERISH_PATTERNS = [
        r'(.)\1{4,}',                    # Répétition de caractère 4+ fois
        r'(\b\w{1,2}\b\s*){5,}',         # Séquence de mots très courts
        r'[^aeiouAEIOUàâäéèêëïîôùûüœæ\s]{6,}',  # 6+ consonnes consécutives
        r'(\b\w+\b)\s+\1(\s+\1){2,}',    # Répétition de mot 3+ fois
    ]
    
    def __init__(self, config: ContentTypeConfig, 
                 whisper_model: str = "openai/whisper-large-v3",
                 device: str = None):
        self.config = config
        self.thresholds = config.linguistic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialiser Whisper
        logger.info(f"Chargement du modèle Whisper: {whisper_model}")
        self.processor = WhisperProcessor.from_pretrained(whisper_model)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            whisper_model
        ).to(self.device)
        
        # Initialiser les outils phonémiques
        self.epitran_fr = epitran.Epitran('fra-Latn')
        self.epitran_en = epitran.Epitran('eng-Latn')
        
        # Compiler les patterns de charabia
        self.gibberish_patterns = [re.compile(p, re.IGNORECASE) 
                                   for p in self.GIBBERISH_PATTERNS]
    
    def analyze(self, audio_path: str, 
                expected_language: str = "fr") -> LinguisticAnalysisResult:
        """
        Analyse linguistique complète d'un fichier audio.
        
        Args:
            audio_path: Chemin vers le fichier audio
            expected_language: Langue attendue ("fr" ou "en")
            
        Returns:
            LinguisticAnalysisResult avec score et anomalies
        """
        anomalies = []
        
        # 1. Transcription ASR inverse avec Whisper
        transcription = self._transcribe(audio_path, expected_language)
        
        # 2. Analyse de la confiance par mot
        confidence_result = self._analyze_word_confidence(transcription)
        anomalies.extend(confidence_result['anomalies'])
        
        # 3. Détection des langues
        language_result = self._detect_languages(transcription, expected_language)
        anomalies.extend(language_result['anomalies'])
        
        # 4. Analyse phonémique
        phoneme_result = self._analyze_phonemes(
            transcription.text, expected_language
        )
        anomalies.extend(phoneme_result['anomalies'])
        
        # 5. Détection de charabia
        gibberish_result = self._detect_gibberish(transcription)
        anomalies.extend(gibberish_result['anomalies'])
        
        # 6. Détection de répétitions
        repetition_result = self._detect_repetitions(transcription)
        anomalies.extend(repetition_result['anomalies'])
        
        # Calculer le score final
        score = self._compute_linguistic_score(
            anomalies,
            confidence_result['mean_confidence'],
            phoneme_result['validity_score'],
            language_result['language_consistency'],
        )
        
        return LinguisticAnalysisResult(
            score=score,
            anomalies=anomalies,
            transcription=transcription,
            mean_word_confidence=confidence_result['mean_confidence'],
            phoneme_validity_score=phoneme_result['validity_score'],
            detected_languages=language_result['detected_languages'],
            gibberish_segments=gibberish_result['segments'],
            unknown_phoneme_ratio=phoneme_result['unknown_ratio'],
            word_repetition_count=repetition_result['repetition_count'],
        )
    
    def _transcribe(self, audio_path: str, 
                    expected_language: str) -> TranscriptionResult:
        """
        Transcription avec Whisper incluant timestamps par mot.
        """
        import librosa
        
        # Charger l'audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Préparer l'entrée
        input_features = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Générer avec timestamps
        with torch.no_grad():
            generated_ids = self.whisper_model.generate(
                input_features,
                language=expected_language,
                task="transcribe",
                return_timestamps=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Décoder
        transcription = self.processor.batch_decode(
            generated_ids.sequences, 
            skip_special_tokens=True,
            output_word_offsets=True,
        )[0]
        
        # Extraire les segments avec timestamps
        # Note: Whisper retourne les timestamps dans un format spécifique
        segments = self._extract_segments_with_timestamps(
            generated_ids, audio_path
        )
        
        # Calculer la confiance moyenne
        if hasattr(generated_ids, 'scores') and generated_ids.scores:
            scores = torch.stack(generated_ids.scores, dim=1)
            probs = torch.softmax(scores, dim=-1)
            confidence = float(probs.max(dim=-1).values.mean())
        else:
            confidence = 0.8  # Valeur par défaut
        
        return TranscriptionResult(
            text=transcription.strip(),
            language=expected_language,
            confidence=confidence,
            word_timestamps=self._extract_word_timestamps(segments),
            segments=segments,
        )
    
    def _extract_segments_with_timestamps(self, generated_ids, 
                                          audio_path: str) -> List[Dict]:
        """
        Extrait les segments avec leurs timestamps.
        """
        # Utiliser le pipeline pour une extraction plus robuste
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=self.device,
        )
        
        result = pipe(
            audio_path,
            return_timestamps="word",
            generate_kwargs={"language": "french"},
        )
        
        segments = []
        if 'chunks' in result:
            for chunk in result['chunks']:
                segments.append({
                    'text': chunk['text'],
                    'start': chunk['timestamp'][0] if chunk['timestamp'][0] else 0,
                    'end': chunk['timestamp'][1] if chunk['timestamp'][1] else 0,
                })
        
        return segments
    
    def _extract_word_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """Convertit les segments en liste de mots avec timestamps"""
        word_timestamps = []
        for seg in segments:
            word_timestamps.append({
                'word': seg['text'].strip(),
                'start_ms': int(seg['start'] * 1000) if seg['start'] else 0,
                'end_ms': int(seg['end'] * 1000) if seg['end'] else 0,
            })
        return word_timestamps
    
    def _analyze_word_confidence(self, transcription: TranscriptionResult) -> dict:
        """
        Analyse la confiance ASR par mot.
        """
        anomalies = []
        word_confidences = []
        
        # La confiance globale de Whisper
        mean_confidence = transcription.confidence
        
        # Simuler des confidences par mot basées sur la confiance globale
        # et des heuristiques (mots courts = moins fiables, etc.)
        for wt in transcription.word_timestamps:
            word = wt['word']
            
            # Heuristiques de confiance
            word_conf = mean_confidence
            
            # Mots très courts = potentiellement du bruit
            if len(word) <= 2:
                word_conf *= 0.8
            
            # Mots avec caractères inhabituels
            if re.search(r'[^a-zA-ZàâäéèêëïîôùûüœæçÀÂÄÉÈÊËÏÎÔÙÛÜŒÆÇ\'-]', word):
                word_conf *= 0.7
            
            word_confidences.append(word_conf)
            
            if word_conf < self.thresholds.min_word_confidence:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.LOW_ASR_CONFIDENCE,
                    severity=SeverityLevel.MEDIUM if word_conf < 0.4 else SeverityLevel.LOW,
                    time_range=TimeRange(wt['start_ms'], wt['end_ms']),
                    confidence=1.0 - word_conf,
                    description=f"Mot à faible confiance: '{word}' ({word_conf:.1%})",
                ))
        
        return {
            'mean_confidence': float(np.mean(word_confidences)) if word_confidences else mean_confidence,
            'anomalies': anomalies,
        }
    
    def _detect_languages(self, transcription: TranscriptionResult,
                          expected_language: str) -> dict:
        """
        Détecte les changements de langue dans la transcription.
        """
        anomalies = []
        detected_languages = []
        
        text = transcription.text
        if not text:
            return {
                'detected_languages': [(expected_language, 1.0)],
                'language_consistency': 1.0,
                'anomalies': [],
            }
        
        try:
            # Détection globale
            langs = detect_langs(text)
            detected_languages = [(l.lang, l.prob) for l in langs]
            
            # Vérifier la langue principale
            primary_lang = detected_languages[0][0] if detected_languages else expected_language
            primary_prob = detected_languages[0][1] if detected_languages else 1.0
            
            # Détecter les changements de langue par segment
            for seg in transcription.segments:
                seg_text = seg['text']
                if len(seg_text) < 10:  # Trop court pour détection fiable
                    continue
                
                try:
                    seg_langs = detect_langs(seg_text)
                    seg_lang = seg_langs[0].lang if seg_langs else expected_language
                    
                    if seg_lang != expected_language:
                        start_ms = int(seg['start'] * 1000)
                        end_ms = int(seg['end'] * 1000)
                        
                        anomalies.append(Anomaly(
                            anomaly_type=AnomalyType.LANGUAGE_SWITCH,
                            severity=SeverityLevel.HIGH,
                            time_range=TimeRange(start_ms, end_ms),
                            confidence=seg_langs[0].prob if seg_langs else 0.5,
                            description=f"Changement de langue détecté: {seg_lang} (attendu: {expected_language})",
                            raw_data={'detected': seg_lang, 'expected': expected_language},
                        ))
                except LangDetectException:
                    pass
            
            # Calculer la cohérence linguistique
            expected_ratio = sum(p for l, p in detected_languages if l == expected_language)
            language_consistency = expected_ratio
            
        except LangDetectException:
            detected_languages = [(expected_language, 1.0)]
            language_consistency = 1.0
        
        return {
            'detected_languages': detected_languages,
            'language_consistency': language_consistency,
            'anomalies': anomalies,
        }
    
    def _analyze_phonemes(self, text: str, language: str) -> dict:
        """
        Analyse la validité phonémique du texte transcrit.
        """
        if not text:
            return {
                'validity_score': 1.0,
                'unknown_ratio': 0.0,
                'anomalies': [],
            }
        
        anomalies = []
        
        # Obtenir les phonèmes
        try:
            if language == 'fr':
                phonemes = self.epitran_fr.transliterate(text)
            else:
                phonemes = self.epitran_en.transliterate(text)
        except Exception as e:
            logger.warning(f"Erreur phonémisation: {e}")
            return {
                'validity_score': 0.8,
                'unknown_ratio': 0.1,
                'anomalies': [],
            }
        
        # Valider les phonèmes
        valid_set = self.VALID_PHONEMES.get(language, self.VALID_PHONEMES['en'])
        
        phoneme_list = list(phonemes)
        valid_count = sum(1 for p in phoneme_list if p in valid_set or p.isspace())
        total_count = len(phoneme_list)
        
        if total_count == 0:
            return {
                'validity_score': 1.0,
                'unknown_ratio': 0.0,
                'anomalies': [],
            }
        
        validity_score = valid_count / total_count
        unknown_ratio = 1.0 - validity_score
        
        # Détecter les séquences de phonèmes invalides
        invalid_sequence = []
        for i, p in enumerate(phoneme_list):
            if p not in valid_set and not p.isspace():
                invalid_sequence.append((i, p))
            elif invalid_sequence:
                # Fin d'une séquence invalide
                if len(invalid_sequence) >= 3:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.UNKNOWN_PHONEMES,
                        severity=SeverityLevel.MEDIUM,
                        time_range=TimeRange(0, 0),  # Pas de timestamp précis
                        confidence=len(invalid_sequence) / 10,
                        description=f"Séquence phonémique invalide: {''.join(p for _, p in invalid_sequence)}",
                    ))
                invalid_sequence = []
        
        if unknown_ratio > self.thresholds.max_unknown_phoneme_ratio:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.UNKNOWN_PHONEMES,
                severity=SeverityLevel.HIGH,
                time_range=TimeRange(0, 0),
                confidence=unknown_ratio,
                description=f"Ratio de phonèmes inconnus élevé: {unknown_ratio:.1%}",
            ))
        
        return {
            'validity_score': validity_score,
            'unknown_ratio': unknown_ratio,
            'anomalies': anomalies,
        }
    
    def _detect_gibberish(self, transcription: TranscriptionResult) -> dict:
        """
        Détecte les segments de charabia (gibberish) dans la transcription.
        
        Utilise plusieurs heuristiques:
        - Patterns regex de charabia
        - Ratio voyelles/consonnes anormal
        - Longueur moyenne des mots anormale
        """
        anomalies = []
        gibberish_segments = []
        
        text = transcription.text
        if not text:
            return {'segments': [], 'anomalies': []}
        
        # Vérifier les patterns de charabia
        for pattern in self.gibberish_patterns:
            for match in pattern.finditer(text):
                # Trouver le timestamp correspondant
                start_char = match.start()
                end_char = match.end()
                
                time_range = self._find_timestamp_for_text_range(
                    transcription, start_char, end_char
                )
                
                gibberish_segments.append(time_range)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.GIBBERISH,
                    severity=SeverityLevel.HIGH,
                    time_range=time_range,
                    confidence=0.9,
                    description=f"Charabia détecté: '{match.group()[:30]}...'",
                ))
        
        # Analyse par segment
        for seg in transcription.segments:
            seg_text = seg['text']
            gibberish_score = self._compute_gibberish_score(seg_text)
            
            if gibberish_score > self.thresholds.gibberish_detection_threshold:
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                time_range = TimeRange(start_ms, end_ms)
                
                # Éviter les doublons
                if not any(tr.overlaps(time_range) for tr in gibberish_segments):
                    gibberish_segments.append(time_range)
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.GIBBERISH,
                        severity=SeverityLevel.HIGH if gibberish_score > 0.7 else SeverityLevel.MEDIUM,
                        time_range=time_range,
                        confidence=gibberish_score,
                        description=f"Segment incohérent détecté (score={gibberish_score:.2f})",
                    ))
        
        return {
            'segments': gibberish_segments,
            'anomalies': anomalies,
        }
    
    def _compute_gibberish_score(self, text: str) -> float:
        """
        Calcule un score de "gibberish" pour un texte.
        
        Score élevé = plus probable que ce soit du charabia.
        """
        if not text or len(text) < 5:
            return 0.0
        
        score = 0.0
        
        # 1. Ratio voyelles/consonnes (doit être autour de 0.4-0.6)
        vowels = set('aeiouyàâäéèêëïîôùûüœæAEIOUYÀÂÄÉÈÊËÏÎÔÙÛÜŒÆ')
        letters = [c for c in text if c.isalpha()]
        if letters:
            vowel_ratio = sum(1 for c in letters if c in vowels) / len(letters)
            if vowel_ratio < 0.2 or vowel_ratio > 0.8:
                score += 0.3
        
        # 2. Longueur moyenne des mots
        words = text.split()
        if words:
            avg_word_len = np.mean([len(w) for w in words])
            if avg_word_len < 2 or avg_word_len > 15:
                score += 0.2
        
        # 3. Répétition de caractères
        for i in range(len(text) - 3):
            if text[i] == text[i+1] == text[i+2] == text[i+3]:
                score += 0.2
                break
        
        # 4. Présence de caractères non-alphabétiques étranges
        strange_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,!?;:\'-"')
        if len(text) > 0:
            strange_ratio = strange_chars / len(text)
            if strange_ratio > 0.1:
                score += 0.3
        
        return min(1.0, score)
    
    def _detect_repetitions(self, transcription: TranscriptionResult) -> dict:
        """
        Détecte les répétitions anormales de mots ou phrases.
        """
        anomalies = []
        words = transcription.text.lower().split()
        
        if len(words) < 3:
            return {'repetition_count': 0, 'anomalies': []}
        
        # Compter les répétitions consécutives
        repetition_count = 0
        i = 0
        while i < len(words) - 1:
            if words[i] == words[i + 1]:
                # Trouver la longueur de la répétition
                rep_len = 2
                while i + rep_len < len(words) and words[i] == words[i + rep_len]:
                    rep_len += 1
                
                if rep_len >= 3:  # 3+ répétitions = anormal
                    repetition_count += 1
                    
                    # Trouver les timestamps
                    word_ts = transcription.word_timestamps
                    if i < len(word_ts) and i + rep_len - 1 < len(word_ts):
                        start_ms = word_ts[i]['start_ms']
                        end_ms = word_ts[i + rep_len - 1]['end_ms']
                        
                        anomalies.append(Anomaly(
                            anomaly_type=AnomalyType.WORD_REPETITION,
                            severity=SeverityLevel.HIGH if rep_len >= 5 else SeverityLevel.MEDIUM,
                            time_range=TimeRange(start_ms, end_ms),
                            confidence=min(1.0, rep_len / 5),
                            description=f"Mot répété {rep_len} fois: '{words[i]}'",
                        ))
                
                i += rep_len
            else:
                i += 1
        
        # Détecter les répétitions de n-grams (2-3 mots)
        for n in [2, 3]:
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            
            for ngram, count in ngram_counts.items():
                if count >= 3:  # Même n-gram 3+ fois
                    repetition_count += 1
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.WORD_REPETITION,
                        severity=SeverityLevel.MEDIUM,
                        time_range=TimeRange(0, 0),  # Pas de timestamp précis
                        confidence=min(1.0, count / 5),
                        description=f"Séquence répétée {count} fois: '{ngram}'",
                    ))
        
        return {
            'repetition_count': repetition_count,
            'anomalies': anomalies,
        }
    
    def _find_timestamp_for_text_range(self, transcription: TranscriptionResult,
                                        start_char: int, end_char: int) -> TimeRange:
        """
        Trouve le timestamp correspondant à une plage de caractères dans le texte.
        """
        # Approximation basée sur la position relative
        text_len = len(transcription.text)
        if text_len == 0:
            return TimeRange(0, 0)
        
        word_timestamps = transcription.word_timestamps
        if not word_timestamps:
            return TimeRange(0, 0)
        
        total_duration = word_timestamps[-1]['end_ms'] if word_timestamps else 0
        
        # Estimation linéaire
        start_ms = int(start_char / text_len * total_duration)
        end_ms = int(end_char / text_len * total_duration)
        
        return TimeRange(start_ms, end_ms)
    
    def _compute_linguistic_score(self, anomalies: List[Anomaly],
                                   mean_confidence: float,
                                   phoneme_validity: float,
                                   language_consistency: float) -> float:
        """
        Calcule le score linguistique final.
        """
        # Score de base combinant les métriques
        base_score = (
            mean_confidence * 30 +          # 30% confiance ASR
            phoneme_validity * 30 +          # 30% validité phonèmes
            language_consistency * 40        # 40% cohérence langue
        )
        
        # Pénalités pour anomalies
        severity_penalties = {
            SeverityLevel.LOW: 3,
            SeverityLevel.MEDIUM: 8,
            SeverityLevel.HIGH: 15,
            SeverityLevel.CRITICAL: 25,
        }
        
        for anomaly in anomalies:
            penalty = severity_penalties[anomaly.severity] * anomaly.confidence
            base_score -= penalty
        
        return max(0.0, min(100.0, base_score))
