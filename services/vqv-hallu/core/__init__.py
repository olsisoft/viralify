"""VQV-HALLU Core Package"""
from .pipeline import VQVHalluPipeline, analyze_voiceover
from .pipeline_async import (
    VQVHalluAsyncPipeline,
    FastRejectResult,
    analyze_voiceover_async
)
from .score_fusion import ScoreFusionEngine, AdaptiveScoreFusion

__all__ = [
    # Synchronous pipeline (legacy)
    'VQVHalluPipeline',
    'analyze_voiceover',
    # Asynchronous pipeline with L1/L2 parallelism
    'VQVHalluAsyncPipeline',
    'FastRejectResult',
    'analyze_voiceover_async',
    # Score fusion
    'ScoreFusionEngine',
    'AdaptiveScoreFusion'
]
