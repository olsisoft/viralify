"""
VQV-HALLU Layer 1: Acoustic Analyzer
Analyse spectrale et détection d'anomalies acoustiques
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import librosa
from typing import List, Tuple, Optional
import logging

from models.data_models import (
    AcousticAnalysisResult, Anomaly, AnomalyType, 
    SeverityLevel, TimeRange
)
from config.settings import AcousticThresholds, ContentTypeConfig


logger = logging.getLogger(__name__)


class AcousticAnalyzer:
    """
    Analyseur acoustique multi-dimension pour détection d'anomalies vocales.
    
    Implémente:
    - Détection de distorsion harmonique (THD)
    - Détection de clics et pops
    - Analyse de la flatness spectrale
    - Détection de silences anormaux
    - Estimation du débit de parole
    """
    
    def __init__(self, config: ContentTypeConfig):
        self.config = config
        self.thresholds = config.acoustic
        self.sample_rate = 16000  # Standard pour l'analyse vocale
        
    def analyze(self, audio_path: str) -> AcousticAnalysisResult:
        """
        Analyse acoustique complète d'un fichier audio.
        
        Args:
            audio_path: Chemin vers le fichier audio
            
        Returns:
            AcousticAnalysisResult avec score et anomalies détectées
        """
        # Charger l'audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        duration_ms = int(len(audio) / sr * 1000)
        
        anomalies = []
        
        # 1. Analyse spectrale
        spectral_metrics = self._analyze_spectral(audio, sr)
        anomalies.extend(spectral_metrics['anomalies'])
        
        # 2. Détection de distorsion
        distortion_result = self._detect_distortion(audio, sr)
        anomalies.extend(distortion_result['anomalies'])
        
        # 3. Détection de clics/pops
        click_result = self._detect_clicks(audio, sr)
        anomalies.extend(click_result['anomalies'])
        
        # 4. Analyse des silences
        silence_result = self._analyze_silence(audio, sr)
        anomalies.extend(silence_result['anomalies'])
        
        # 5. Estimation du débit de parole
        pace_result = self._estimate_speech_pace(audio, sr)
        anomalies.extend(pace_result['anomalies'])
        
        # Calculer le score final
        score = self._compute_acoustic_score(anomalies, duration_ms)
        
        return AcousticAnalysisResult(
            score=score,
            anomalies=anomalies,
            spectral_flatness_mean=spectral_metrics['flatness_mean'],
            spectral_flatness_std=spectral_metrics['flatness_std'],
            silence_ratio=silence_result['silence_ratio'],
            estimated_speech_rate_wpm=pace_result['estimated_wpm'],
            distortion_score=distortion_result['thd_score'],
            click_count=click_result['click_count'],
            spectral_centroid_mean=spectral_metrics['centroid_mean'],
            spectral_bandwidth_mean=spectral_metrics['bandwidth_mean'],
        )
    
    def _analyze_spectral(self, audio: np.ndarray, sr: int) -> dict:
        """
        Analyse les caractéristiques spectrales pour détecter anomalies.
        
        La flatness spectrale proche de 1 = bruit blanc (mauvais)
        La flatness spectrale proche de 0 = signal tonal pur
        """
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculer la flatness spectrale par frame
        flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=frame_length, hop_length=hop_length
        )[0]
        
        # Calculer le centroid et bandwidth
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length
        )[0]
        
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=frame_length, hop_length=hop_length
        )[0]
        
        anomalies = []
        
        # Détecter les segments avec flatness anormale
        for i, f in enumerate(flatness):
            if f > self.thresholds.max_spectral_flatness:
                # Segment ressemblant à du bruit
                start_ms = int(i * hop_length / sr * 1000)
                end_ms = int((i + 1) * hop_length / sr * 1000)
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.SPECTRAL_ANOMALY,
                    severity=SeverityLevel.MEDIUM,
                    time_range=TimeRange(start_ms, end_ms),
                    confidence=float(f),
                    description=f"Segment ressemblant à du bruit (flatness={f:.2f})",
                ))
            elif f < self.thresholds.min_spectral_flatness:
                # Signal anormalement tonal (potentielle distorsion)
                start_ms = int(i * hop_length / sr * 1000)
                end_ms = int((i + 1) * hop_length / sr * 1000)
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.SPECTRAL_ANOMALY,
                    severity=SeverityLevel.LOW,
                    time_range=TimeRange(start_ms, end_ms),
                    confidence=1.0 - float(f),
                    description=f"Signal anormalement tonal (flatness={f:.2f})",
                ))
        
        # Fusionner les anomalies adjacentes
        anomalies = self._merge_adjacent_anomalies(anomalies)
        
        return {
            'flatness_mean': float(np.mean(flatness)),
            'flatness_std': float(np.std(flatness)),
            'centroid_mean': float(np.mean(centroid)),
            'bandwidth_mean': float(np.mean(bandwidth)),
            'anomalies': anomalies,
        }
    
    def _detect_distortion(self, audio: np.ndarray, sr: int) -> dict:
        """
        Détecte la distorsion harmonique totale (THD).
        
        Principe: Dans un signal propre, l'énergie se concentre sur les
        harmoniques fondamentales. La distorsion ajoute des harmoniques
        parasites.
        """
        # Analyser par segments
        segment_length = int(0.5 * sr)  # 500ms segments
        hop = int(0.25 * sr)            # 250ms hop
        
        thd_scores = []
        anomalies = []
        
        for start in range(0, len(audio) - segment_length, hop):
            segment = audio[start:start + segment_length]
            
            # FFT du segment
            spectrum = np.abs(fft(segment))[:len(segment)//2]
            freqs = fftfreq(len(segment), 1/sr)[:len(segment)//2]
            
            # Trouver la fréquence fondamentale (F0)
            # Chercher dans la plage vocale typique (80-400 Hz)
            voice_range = (freqs >= 80) & (freqs <= 400)
            if not np.any(voice_range):
                continue
                
            voice_spectrum = spectrum.copy()
            voice_spectrum[~voice_range] = 0
            
            f0_idx = np.argmax(voice_spectrum)
            f0 = freqs[f0_idx]
            
            if f0 < 80:  # Pas de fondamentale claire
                continue
            
            # Calculer l'énergie des harmoniques (2f0, 3f0, ...)
            fundamental_energy = spectrum[f0_idx] ** 2
            harmonic_energy = 0
            
            for h in range(2, 6):  # Harmoniques 2 à 5
                h_freq = f0 * h
                if h_freq > sr / 2:
                    break
                h_idx = np.argmin(np.abs(freqs - h_freq))
                harmonic_energy += spectrum[h_idx] ** 2
            
            # THD = sqrt(somme harmoniques / fondamentale)
            if fundamental_energy > 0:
                thd = np.sqrt(harmonic_energy / fundamental_energy)
                thd_scores.append(thd)
                
                if thd > self.thresholds.distortion_threshold:
                    start_ms = int(start / sr * 1000)
                    end_ms = int((start + segment_length) / sr * 1000)
                    
                    severity = SeverityLevel.HIGH if thd > 0.3 else SeverityLevel.MEDIUM
                    
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.DISTORTION,
                        severity=severity,
                        time_range=TimeRange(start_ms, end_ms),
                        confidence=min(1.0, thd / 0.5),
                        description=f"Distorsion harmonique détectée (THD={thd:.2%})",
                        raw_data={"thd": float(thd), "f0": float(f0)},
                    ))
        
        mean_thd = float(np.mean(thd_scores)) if thd_scores else 0.0
        
        return {
            'thd_score': mean_thd,
            'anomalies': self._merge_adjacent_anomalies(anomalies),
        }
    
    def _detect_clicks(self, audio: np.ndarray, sr: int) -> dict:
        """
        Détecte les clics et pops dans l'audio.
        
        Principe: Les clics sont des transitoires rapides avec une
        énergie concentrée sur une très courte durée.
        """
        # Calculer la dérivée première (différence entre samples)
        diff = np.abs(np.diff(audio))
        
        # Statistiques pour normalisation
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        if std_diff == 0:
            return {'click_count': 0, 'anomalies': []}
        
        # Détecter les pics (clics) dépassant le seuil
        threshold = mean_diff + self.thresholds.click_detection_threshold * std_diff
        click_indices = np.where(diff > threshold)[0]
        
        # Regrouper les indices proches (même clic)
        clicks = []
        anomalies = []
        
        if len(click_indices) > 0:
            groups = [[click_indices[0]]]
            for idx in click_indices[1:]:
                if idx - groups[-1][-1] < sr * 0.01:  # 10ms gap
                    groups[-1].append(idx)
                else:
                    groups.append([idx])
            
            for group in groups:
                center_sample = int(np.mean(group))
                center_ms = int(center_sample / sr * 1000)
                
                # Calculer l'intensité du clic
                intensity = float(np.max(diff[group]) / threshold)
                
                severity = (SeverityLevel.HIGH if intensity > 2 
                           else SeverityLevel.MEDIUM if intensity > 1.5 
                           else SeverityLevel.LOW)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.CLICK_POP,
                    severity=severity,
                    time_range=TimeRange(max(0, center_ms - 5), center_ms + 5),
                    confidence=min(1.0, intensity / 3),
                    description=f"Clic/pop détecté (intensité={intensity:.2f}x)",
                ))
                clicks.append(center_ms)
        
        return {
            'click_count': len(clicks),
            'anomalies': anomalies,
        }
    
    def _analyze_silence(self, audio: np.ndarray, sr: int) -> dict:
        """
        Analyse les silences et pauses dans l'audio.
        """
        # Calculer l'énergie RMS par frame
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)    # 10ms
        
        rms = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]
        
        # Convertir en dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms))
        
        # Détecter les frames silencieuses
        silence_mask = rms_db < self.thresholds.silence_threshold_db
        silence_ratio = float(np.mean(silence_mask))
        
        anomalies = []
        
        # Trouver les segments de silence prolongés
        silence_segments = self._find_contiguous_segments(silence_mask)
        
        for start_frame, end_frame in silence_segments:
            duration_frames = end_frame - start_frame
            duration_ms = int(duration_frames * hop_length / sr * 1000)
            
            # Silences > 500ms sont suspects
            if duration_ms > 500:
                start_ms = int(start_frame * hop_length / sr * 1000)
                end_ms = int(end_frame * hop_length / sr * 1000)
                
                severity = (SeverityLevel.HIGH if duration_ms > 2000
                           else SeverityLevel.MEDIUM if duration_ms > 1000
                           else SeverityLevel.LOW)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.SILENCE_EXCESSIVE,
                    severity=severity,
                    time_range=TimeRange(start_ms, end_ms),
                    confidence=min(1.0, duration_ms / 3000),
                    description=f"Silence prolongé ({duration_ms}ms)",
                ))
        
        # Vérifier le ratio global de silence
        if silence_ratio > self.thresholds.max_silence_ratio:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.SILENCE_EXCESSIVE,
                severity=SeverityLevel.MEDIUM,
                time_range=TimeRange(0, int(len(audio) / sr * 1000)),
                confidence=silence_ratio,
                description=f"Ratio de silence excessif ({silence_ratio:.1%})",
            ))
        
        return {
            'silence_ratio': silence_ratio,
            'anomalies': anomalies,
        }
    
    def _estimate_speech_pace(self, audio: np.ndarray, sr: int) -> dict:
        """
        Estime le débit de parole en mots par minute.
        
        Utilise la détection d'onset pour estimer les syllabes,
        puis extrapole en mots.
        """
        # Détecter les onsets (débuts de sons)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, units='time'
        )
        
        duration_min = len(audio) / sr / 60
        
        if duration_min == 0 or len(onsets) < 2:
            return {'estimated_wpm': 0, 'anomalies': []}
        
        # Estimation: ~1.5 syllabes par mot en français
        syllables_per_minute = len(onsets) / duration_min
        estimated_wpm = syllables_per_minute / 1.5
        
        anomalies = []
        
        # Analyser les variations locales de vitesse
        if len(onsets) > 10:
            # Calculer les intervalles inter-onset
            intervals = np.diff(onsets)
            
            # Chercher les variations brusques
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            for i, interval in enumerate(intervals):
                local_wpm = 60 / (interval * 1.5) if interval > 0 else 0
                
                if local_wpm > self.thresholds.max_speech_rate_wpm:
                    start_ms = int(onsets[i] * 1000)
                    end_ms = int(onsets[i + 1] * 1000)
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.PACE_TOO_FAST,
                        severity=SeverityLevel.MEDIUM,
                        time_range=TimeRange(start_ms, end_ms),
                        confidence=min(1.0, local_wpm / 300),
                        description=f"Débit trop rapide ({local_wpm:.0f} mots/min)",
                    ))
                elif local_wpm < self.thresholds.min_speech_rate_wpm and local_wpm > 0:
                    start_ms = int(onsets[i] * 1000)
                    end_ms = int(onsets[i + 1] * 1000)
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.PACE_TOO_SLOW,
                        severity=SeverityLevel.LOW,
                        time_range=TimeRange(start_ms, end_ms),
                        confidence=1.0 - (local_wpm / self.thresholds.min_speech_rate_wpm),
                        description=f"Débit trop lent ({local_wpm:.0f} mots/min)",
                    ))
        
        # Vérifier le débit global
        if estimated_wpm > self.thresholds.max_speech_rate_wpm:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.PACE_TOO_FAST,
                severity=SeverityLevel.HIGH,
                time_range=TimeRange(0, int(len(audio) / sr * 1000)),
                confidence=min(1.0, estimated_wpm / 300),
                description=f"Débit global trop rapide ({estimated_wpm:.0f} mots/min)",
            ))
        elif estimated_wpm < self.thresholds.min_speech_rate_wpm:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.PACE_TOO_SLOW,
                severity=SeverityLevel.MEDIUM,
                time_range=TimeRange(0, int(len(audio) / sr * 1000)),
                confidence=1.0 - (estimated_wpm / self.thresholds.min_speech_rate_wpm),
                description=f"Débit global trop lent ({estimated_wpm:.0f} mots/min)",
            ))
        
        return {
            'estimated_wpm': float(estimated_wpm),
            'anomalies': self._merge_adjacent_anomalies(anomalies),
        }
    
    def _find_contiguous_segments(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Trouve les segments contigus dans un masque booléen"""
        segments = []
        in_segment = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                segments.append((start, i))
                in_segment = False
        
        if in_segment:
            segments.append((start, len(mask)))
        
        return segments
    
    def _merge_adjacent_anomalies(self, anomalies: List[Anomaly], 
                                   gap_threshold_ms: int = 100) -> List[Anomaly]:
        """Fusionne les anomalies adjacentes du même type"""
        if not anomalies:
            return []
        
        # Trier par type puis par temps de début
        sorted_anomalies = sorted(anomalies, 
            key=lambda a: (a.anomaly_type.value, a.time_range.start_ms))
        
        merged = []
        current = sorted_anomalies[0]
        
        for anomaly in sorted_anomalies[1:]:
            if (anomaly.anomaly_type == current.anomaly_type and
                anomaly.time_range.start_ms - current.time_range.end_ms < gap_threshold_ms):
                # Fusionner
                current = Anomaly(
                    anomaly_type=current.anomaly_type,
                    severity=max(current.severity, anomaly.severity, 
                                key=lambda s: list(SeverityLevel).index(s)),
                    time_range=TimeRange(
                        current.time_range.start_ms,
                        anomaly.time_range.end_ms
                    ),
                    confidence=max(current.confidence, anomaly.confidence),
                    description=current.description,
                )
            else:
                merged.append(current)
                current = anomaly
        
        merged.append(current)
        return merged
    
    def _compute_acoustic_score(self, anomalies: List[Anomaly], 
                                 duration_ms: int) -> float:
        """
        Calcule le score acoustique final basé sur les anomalies.
        
        Score = 100 - pénalités
        """
        base_score = 100.0
        
        # Pénalités par sévérité
        severity_penalties = {
            SeverityLevel.LOW: 2,
            SeverityLevel.MEDIUM: 5,
            SeverityLevel.HIGH: 15,
            SeverityLevel.CRITICAL: 30,
        }
        
        for anomaly in anomalies:
            # Pénalité de base
            penalty = severity_penalties[anomaly.severity]
            
            # Pondérer par la confiance
            penalty *= anomaly.confidence
            
            # Pondérer par la durée relative de l'anomalie
            anomaly_duration = anomaly.time_range.duration_ms
            duration_factor = min(1.0, anomaly_duration / duration_ms * 10)
            penalty *= (0.5 + 0.5 * duration_factor)
            
            base_score -= penalty
        
        return max(0.0, min(100.0, base_score))
