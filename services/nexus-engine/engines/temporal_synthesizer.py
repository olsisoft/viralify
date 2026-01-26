"""
NEXUS Temporal Code Synthesizer
Adaptation du code au temps alloué

Innovation: Génère du code qui s'adapte dynamiquement au temps disponible
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import math

from models.data_models import (
    CognitiveBlueprint, CognitiveStep, CodeSegment,
    ArchitectureDNA, NexusRequest, CodeVerbosity, TargetAudience
)


logger = logging.getLogger(__name__)


@dataclass
class TimeSlot:
    """Slot de temps pour un segment"""
    step_id: str
    allocated_seconds: int
    priority: float  # 0-1, plus haut = plus important
    can_compress: bool = True
    min_seconds: int = 15


@dataclass
class TimeBudget:
    """Budget de temps pour le projet"""
    total_seconds: int
    slots: List[TimeSlot]
    buffer_seconds: int = 30  # Marge de sécurité
    
    @property
    def allocated_seconds(self) -> int:
        return sum(s.allocated_seconds for s in self.slots)
    
    @property
    def remaining_seconds(self) -> int:
        return self.total_seconds - self.allocated_seconds - self.buffer_seconds
    
    def is_over_budget(self) -> bool:
        return self.remaining_seconds < 0


class TemporalCodeSynthesizer:
    """
    Synthétiseur de code adaptatif au temps.
    
    Stratégies d'adaptation:
    1. COMPRESSION: Réduire la verbosité des segments moins critiques
    2. AGGREGATION: Fusionner des segments similaires
    3. PRIORITIZATION: Focus sur les concepts essentiels
    4. PARALLEL: Générer des versions différentes selon le temps
    """
    
    # Durées de référence par type de composant (secondes)
    COMPONENT_DURATIONS = {
        "model": 30,
        "repository": 45,
        "service": 60,
        "controller": 45,
        "api_endpoint": 40,
        "config": 20,
        "test": 30,
        "utility": 25,
        "none": 15,
    }
    
    # Facteurs de verbosité
    VERBOSITY_FACTORS = {
        CodeVerbosity.MINIMAL: 0.6,
        CodeVerbosity.STANDARD: 1.0,
        CodeVerbosity.VERBOSE: 1.5,
        CodeVerbosity.PRODUCTION: 1.3,
    }
    
    # Priorité par phase cognitive
    PHASE_PRIORITIES = {
        "analysis": 0.6,
        "design": 0.8,
        "implementation": 1.0,
        "validation": 0.7,
    }
    
    def __init__(self):
        pass
    
    def create_time_budget(self, request: NexusRequest, 
                           blueprint: CognitiveBlueprint) -> TimeBudget:
        """
        Crée un budget de temps pour le projet.
        
        Distribue le temps alloué entre les différentes étapes
        en tenant compte de la complexité et de la priorité.
        """
        total_time = request.allocated_time_seconds
        verbosity_factor = self.VERBOSITY_FACTORS.get(
            request.verbosity, 1.0
        )
        
        # Calculer les durées de base
        slots = []
        
        for phase_name, steps in [
            ("analysis", blueprint.analysis_phase),
            ("design", blueprint.design_phase),
            ("implementation", blueprint.implementation_phase),
            ("validation", blueprint.validation_phase),
        ]:
            phase_priority = self.PHASE_PRIORITIES.get(phase_name, 0.8)
            
            for step in steps:
                base_duration = self.COMPONENT_DURATIONS.get(
                    step.code_component, 30
                )
                
                # Ajuster selon la verbosité
                adjusted_duration = int(base_duration * verbosity_factor)
                
                # Ajuster selon la complexité estimée du step
                if "complex" in step.thought.lower() or "advanced" in step.thought.lower():
                    adjusted_duration = int(adjusted_duration * 1.3)
                
                slot = TimeSlot(
                    step_id=step.id,
                    allocated_seconds=adjusted_duration,
                    priority=phase_priority,
                    can_compress=step.code_component not in ["model", "service"],
                    min_seconds=15 if step.code_component == "none" else 20,
                )
                slots.append(slot)
        
        budget = TimeBudget(
            total_seconds=total_time,
            slots=slots,
        )
        
        # Si over budget, compresser
        if budget.is_over_budget():
            budget = self._compress_budget(budget)
        
        return budget
    
    def _compress_budget(self, budget: TimeBudget) -> TimeBudget:
        """
        Compresse le budget pour rentrer dans le temps alloué.
        
        Stratégie:
        1. D'abord réduire les éléments à basse priorité
        2. Puis réduire proportionnellement
        """
        target = budget.total_seconds - budget.buffer_seconds
        current = budget.allocated_seconds
        
        if current <= target:
            return budget
        
        # Calculer le ratio de compression nécessaire
        compression_ratio = target / current
        
        # Appliquer la compression en tenant compte des priorités
        for slot in budget.slots:
            if not slot.can_compress:
                continue
            
            # Les éléments basse priorité sont plus compressés
            priority_factor = 0.5 + (slot.priority * 0.5)  # 0.5 à 1.0
            slot_ratio = compression_ratio / priority_factor
            
            new_duration = max(
                slot.min_seconds,
                int(slot.allocated_seconds * slot_ratio)
            )
            slot.allocated_seconds = new_duration
        
        return budget
    
    def adapt_segment_to_time(self, segment: CodeSegment,
                              time_slot: TimeSlot,
                              request: NexusRequest) -> CodeSegment:
        """
        Adapte un segment de code au temps alloué.
        
        Si le temps est serré:
        - Réduit les commentaires
        - Simplifie les noms de variables
        - Supprime les exemples secondaires
        """
        target_seconds = time_slot.allocated_seconds
        
        # Estimer la durée actuelle
        estimated_seconds = self._estimate_segment_duration(segment)
        
        if estimated_seconds <= target_seconds:
            # OK, pas besoin d'adapter
            segment.duration_seconds = estimated_seconds
            return segment
        
        # Besoin de compresser
        ratio = target_seconds / estimated_seconds
        
        if ratio > 0.8:
            # Compression légère: juste réduire les commentaires
            segment.code = self._reduce_comments(segment.code, ratio)
        elif ratio > 0.5:
            # Compression moyenne: simplifier aussi les noms
            segment.code = self._reduce_comments(segment.code, ratio)
            segment.common_mistakes = segment.common_mistakes[:1]  # Garder juste 1
        else:
            # Compression forte: version minimale
            segment.code = self._create_minimal_version(segment)
            segment.common_mistakes = []
        
        segment.duration_seconds = target_seconds
        return segment
    
    def _estimate_segment_duration(self, segment: CodeSegment) -> int:
        """
        Estime la durée de présentation d'un segment.
        
        Basé sur:
        - Nombre de lignes
        - Complexité du code
        - Longueur de la narration
        """
        # Base: 2 secondes par ligne de code
        line_count = segment.line_count
        base_duration = line_count * 2
        
        # Ajout pour la narration (3 mots/seconde environ)
        if segment.narration_script:
            word_count = len(segment.narration_script.split())
            base_duration += word_count // 3
        
        # Minimum 15 secondes
        return max(15, base_duration)
    
    def _reduce_comments(self, code: str, ratio: float) -> str:
        """Réduit les commentaires dans le code selon le ratio"""
        lines = code.split('\n')
        result = []
        comment_count = 0
        max_comments = max(1, int(len([l for l in lines if '#' in l or '//' in l]) * ratio))
        
        for line in lines:
            stripped = line.strip()
            
            # Ligne de commentaire pure
            if stripped.startswith('#') or stripped.startswith('//'):
                if comment_count < max_comments:
                    result.append(line)
                    comment_count += 1
                # Sinon on skip
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def _create_minimal_version(self, segment: CodeSegment) -> str:
        """Crée une version minimale du code (squelette)"""
        lines = segment.code.split('\n')
        result = []
        
        for line in lines:
            stripped = line.strip()
            
            # Garder les définitions (classes, fonctions)
            if any(stripped.startswith(kw) for kw in 
                   ['class ', 'def ', 'function ', 'const ', 'let ', 'var ', 'import ', 'from ']):
                result.append(line)
            # Garder les décorateurs
            elif stripped.startswith('@'):
                result.append(line)
            # Garder les return/raise
            elif stripped.startswith('return ') or stripped.startswith('raise '):
                result.append(line)
            # Skip les commentaires
            elif stripped.startswith('#') or stripped.startswith('//'):
                continue
            # Garder si c'est une ligne très courte (probablement importante)
            elif len(stripped) < 30:
                result.append(line)
        
        return '\n'.join(result)
    
    def generate_progressive_versions(self, segment: CodeSegment,
                                       dna: ArchitectureDNA) -> List[CodeSegment]:
        """
        Génère des versions progressives du code (v1 → v2 → v3).
        
        Utile quand request.show_evolution = True
        """
        versions = []
        
        # V1: Version naïve/simple
        v1 = CodeSegment(
            id=f"{segment.id}_v1",
            cognitive_step_id=segment.cognitive_step_id,
            code=self._create_naive_version(segment, dna),
            language=segment.language,
            filename=segment.filename.replace('.', '_v1.'),
            component_type=segment.component_type,
            explanation="Version initiale - approche simple",
            key_concepts=segment.key_concepts[:2],
            display_order=segment.display_order,
            duration_seconds=segment.duration_seconds // 3,
        )
        versions.append(v1)
        
        # V2: Version améliorée
        v2 = CodeSegment(
            id=f"{segment.id}_v2",
            cognitive_step_id=segment.cognitive_step_id,
            code=self._create_improved_version(segment, dna),
            language=segment.language,
            filename=segment.filename.replace('.', '_v2.'),
            component_type=segment.component_type,
            explanation="Version améliorée - patterns appliqués",
            key_concepts=segment.key_concepts,
            display_order=segment.display_order + 0.3,
            duration_seconds=segment.duration_seconds // 3,
        )
        versions.append(v2)
        
        # V3: Version finale (originale)
        v3 = CodeSegment(
            id=f"{segment.id}_v3",
            cognitive_step_id=segment.cognitive_step_id,
            code=segment.code,
            language=segment.language,
            filename=segment.filename,
            component_type=segment.component_type,
            explanation="Version finale - production ready",
            key_concepts=segment.key_concepts,
            common_mistakes=segment.common_mistakes,
            display_order=segment.display_order + 0.6,
            duration_seconds=segment.duration_seconds // 3,
        )
        versions.append(v3)
        
        return versions
    
    def _create_naive_version(self, segment: CodeSegment, 
                              dna: ArchitectureDNA) -> str:
        """Crée une version naïve/débutant du code"""
        # Simplifier le code: variables simples, pas de patterns
        code = segment.code
        
        # Remplacer les noms sophistiqués par des noms simples
        replacements = [
            ('Repository', 'Data'),
            ('Service', 'Helper'),
            ('Controller', 'Handler'),
            ('Factory', 'Creator'),
        ]
        
        for old, new in replacements:
            code = code.replace(old, new)
        
        return f"# Version simple - à améliorer\n{code}"
    
    def _create_improved_version(self, segment: CodeSegment,
                                  dna: ArchitectureDNA) -> str:
        """Crée une version intermédiaire du code"""
        return f"# Version améliorée - patterns partiellement appliqués\n{segment.code}"
