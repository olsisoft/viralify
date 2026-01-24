"""
SSVS - Semantic Slide-Voiceover Synchronization

This package implements the SSVS algorithm for precise audio-video synchronization
using semantic alignment instead of simple proportional distribution.

Key components:
- SSVSSynchronizer: Main algorithm for slide-voiceover alignment
- DiagramAwareSynchronizer: Extension for diagram element focus
- SemanticEmbeddingEngine: TF-IDF or Sentence-BERT embeddings
- FocusAnimationGenerator: Generates animation keyframes
"""

from .ssvs_algorithm import (
    SSVSSynchronizer,
    VoiceSegment,
    Slide,
    SynchronizationResult,
    SemanticEmbeddingEngine,
)

from .diagram_synchronizer import (
    DiagramAwareSynchronizer,
    Diagram,
    DiagramElement,
    DiagramElementType,
    BoundingBox,
    DiagramFocusPoint,
    DiagramSyncResult,
    FocusAnimationGenerator,
)

__all__ = [
    # Core SSVS
    'SSVSSynchronizer',
    'VoiceSegment',
    'Slide',
    'SynchronizationResult',
    'SemanticEmbeddingEngine',
    # Diagram extension
    'DiagramAwareSynchronizer',
    'Diagram',
    'DiagramElement',
    'DiagramElementType',
    'BoundingBox',
    'DiagramFocusPoint',
    'DiagramSyncResult',
    'FocusAnimationGenerator',
]
