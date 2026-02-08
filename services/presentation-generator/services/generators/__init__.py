"""
Sequential Generators for Presentation Creation

Ce module découple le PresentationPlanner monolithique en générateurs séquentiels:
1. StructureGenerator: Génère la structure du cours (titres, types, durées)
2. VoiceoverGenerator: Génère les voiceovers slide par slide
3. ContentGenerator: Génère le contenu selon le type de slide

Architecture:
    Request → StructureGenerator → VoiceoverGenerator → ContentGenerator → Slides
"""

from .structure_generator import StructureGenerator, SlideStructure, get_structure_generator
from .voiceover_generator import VoiceoverGenerator, get_voiceover_generator
from .content_generator import ContentGenerator, SlideContent, get_content_generator

__all__ = [
    # Structure
    "StructureGenerator",
    "SlideStructure",
    "get_structure_generator",
    # Voiceover
    "VoiceoverGenerator",
    "get_voiceover_generator",
    # Content
    "ContentGenerator",
    "SlideContent",
    "get_content_generator",
]
