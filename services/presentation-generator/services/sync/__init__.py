"""
SSVS - Semantic Slide-Voiceover Synchronization

This package implements the SSVS algorithm for precise audio-video synchronization
using semantic alignment instead of simple proportional distribution.

Key components:
- SSVSSynchronizer: Main algorithm for slide-voiceover alignment
- SSVSCalibrator: Calibration for fixing audio-video offset issues
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

from .ssvs_calibrator import (
    SSVSCalibrator,
    CalibrationConfig,
    CalibrationPresets,
    SyncDiagnostic,
    PauseDetector,
    SpeechRateAnalyzer,
    SentenceAligner,
)

__all__ = [
    # Core SSVS
    'SSVSSynchronizer',
    'VoiceSegment',
    'Slide',
    'SynchronizationResult',
    'SemanticEmbeddingEngine',
    # Calibration
    'SSVSCalibrator',
    'CalibrationConfig',
    'CalibrationPresets',
    'SyncDiagnostic',
    'PauseDetector',
    'SpeechRateAnalyzer',
    'SentenceAligner',
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
