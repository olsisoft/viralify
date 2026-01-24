"""
SSVS - Semantic Slide-Voiceover Synchronization

This package implements the SSVS algorithm for precise audio-video synchronization
using semantic alignment instead of simple proportional distribution.

Key components:
- SSVSSynchronizer: Main algorithm for slide-voiceover alignment
- SSVSCalibrator: Calibration for fixing audio-video offset issues
- DiagramAwareSynchronizer: Extension for diagram element focus
- EmbeddingEngineFactory: Factory for creating embedding engines
- FocusAnimationGenerator: Generates animation keyframes

Embedding backends (set via SSVS_EMBEDDING_BACKEND env var):
- "auto" (default): MiniLM with TF-IDF fallback
- "minilm": all-MiniLM-L6-v2 (384 dims, fast, good quality)
- "bge-m3": BAAI/bge-m3 (1024 dims, best multilingual, slower)
- "tfidf": TF-IDF (no dependencies, vocabulary-based)
"""

from .ssvs_algorithm import (
    SSVSSynchronizer,
    VoiceSegment,
    Slide,
    SynchronizationResult,
    SyncAnchor,  # Hard constraint for forced alignment
    SemanticEmbeddingEngine,  # Legacy alias for TFIDFEmbeddingEngine
)

from .embedding_engine import (
    EmbeddingEngineFactory,
    EmbeddingEngineBase,
    EmbeddingBackend,
    TFIDFEmbeddingEngine,
    get_embedding_engine,
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
    'SyncAnchor',
    # Embedding engines
    'EmbeddingEngineFactory',
    'EmbeddingEngineBase',
    'EmbeddingBackend',
    'TFIDFEmbeddingEngine',
    'SemanticEmbeddingEngine',  # Legacy alias
    'get_embedding_engine',
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
