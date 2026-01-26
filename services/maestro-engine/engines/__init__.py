"""
MAESTRO Engines

5-layer pipeline for course generation:
1. Domain Discovery - Analyze domain structure
2. Knowledge Graph - Build prerequisite graph
3. Difficulty Calibration - 4D difficulty vectors
4. Curriculum Sequencing - Smooth progression
5. Content Generation - Generate lessons
"""

from engines.domain_discovery import DomainDiscoveryEngine, discover_domain
from engines.knowledge_graph import KnowledgeGraphEngine, build_knowledge_graph
from engines.difficulty_calibrator import DifficultyCalibratorEngine, calibrate_concepts
from engines.curriculum_sequencer import CurriculumSequencerEngine, sequence_curriculum

__all__ = [
    "DomainDiscoveryEngine",
    "discover_domain",
    "KnowledgeGraphEngine",
    "build_knowledge_graph",
    "DifficultyCalibratorEngine",
    "calibrate_concepts",
    "CurriculumSequencerEngine",
    "sequence_curriculum",
]
