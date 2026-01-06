"""
Models for the intelligent visual generation system.
"""

from .visual_types import (
    VisualType,
    DiagramType,
    VisualAnalysis,
    DiagramResult,
    VisualGenerationRequest,
)

from .avatar_models import (
    AvatarStyle,
    PredefinedAvatar,
    AvatarVideoRequest,
    AvatarVideoResult,
    CustomAvatarRequest,
)

__all__ = [
    # Visual types
    "VisualType",
    "DiagramType",
    "VisualAnalysis",
    "DiagramResult",
    "VisualGenerationRequest",
    # Avatar models
    "AvatarStyle",
    "PredefinedAvatar",
    "AvatarVideoRequest",
    "AvatarVideoResult",
    "CustomAvatarRequest",
]
