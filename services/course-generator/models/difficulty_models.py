"""
Difficulty Models - Multi-dimensional difficulty calibration

This module implements a 4D difficulty vector system inspired by MAESTRO
for calibrating concept difficulty in course generation.

The 4 dimensions are:
1. Conceptual Complexity - Level of abstraction required
2. Prerequisites Depth - How many prerequisite concepts needed
3. Information Density - Amount of information to process
4. Cognitive Load - Mental effort required (Bloom's taxonomy level)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class BloomLevel(str, Enum):
    """Bloom's Taxonomy cognitive levels"""
    REMEMBER = "remember"       # Recall facts and basic concepts
    UNDERSTAND = "understand"   # Explain ideas or concepts
    APPLY = "apply"            # Use information in new situations
    ANALYZE = "analyze"        # Draw connections among ideas
    EVALUATE = "evaluate"      # Justify a decision or course of action
    CREATE = "create"          # Produce new or original work


class SkillLevel(str, Enum):
    """Skill levels mapped from difficulty scores"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    VERY_ADVANCED = "very_advanced"
    EXPERT = "expert"


# Bloom level to cognitive load score mapping
BLOOM_TO_COGNITIVE_LOAD: Dict[BloomLevel, float] = {
    BloomLevel.REMEMBER: 0.1,
    BloomLevel.UNDERSTAND: 0.25,
    BloomLevel.APPLY: 0.45,
    BloomLevel.ANALYZE: 0.6,
    BloomLevel.EVALUATE: 0.8,
    BloomLevel.CREATE: 0.95,
}

# Skill level to Bloom levels mapping
SKILL_TO_BLOOM: Dict[SkillLevel, List[BloomLevel]] = {
    SkillLevel.BEGINNER: [BloomLevel.REMEMBER, BloomLevel.UNDERSTAND],
    SkillLevel.INTERMEDIATE: [BloomLevel.APPLY, BloomLevel.ANALYZE],
    SkillLevel.ADVANCED: [BloomLevel.ANALYZE, BloomLevel.EVALUATE],
    SkillLevel.VERY_ADVANCED: [BloomLevel.EVALUATE, BloomLevel.CREATE],
    SkillLevel.EXPERT: [BloomLevel.CREATE],
}


@dataclass
class DifficultyVector:
    """
    4-dimensional difficulty vector for concept calibration.

    Each dimension is scored 0.0 to 1.0:
    - conceptual_complexity: How abstract/complex the concept is
    - prerequisites_depth: How many prerequisites are needed
    - information_density: Amount of information to process
    - cognitive_load: Mental effort required (Bloom's level)

    The composite score is a weighted average used for:
    - Skill level assignment
    - Difficulty progression planning
    - Content adaptation
    """
    conceptual_complexity: float = 0.5
    prerequisites_depth: float = 0.5
    information_density: float = 0.5
    cognitive_load: float = 0.5

    # Weights for composite score calculation
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "conceptual_complexity": 0.25,
        "prerequisites_depth": 0.20,
        "information_density": 0.25,
        "cognitive_load": 0.30,
    })

    def __post_init__(self):
        """Validate all scores are in range [0.0, 1.0]"""
        for attr in ["conceptual_complexity", "prerequisites_depth",
                     "information_density", "cognitive_load"]:
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                setattr(self, attr, max(0.0, min(1.0, value)))

    @property
    def composite_score(self) -> float:
        """Calculate weighted composite difficulty score"""
        return (
            self.WEIGHTS["conceptual_complexity"] * self.conceptual_complexity +
            self.WEIGHTS["prerequisites_depth"] * self.prerequisites_depth +
            self.WEIGHTS["information_density"] * self.information_density +
            self.WEIGHTS["cognitive_load"] * self.cognitive_load
        )

    @property
    def skill_level(self) -> SkillLevel:
        """Convert composite score to skill level"""
        score = self.composite_score
        if score < 0.20:
            return SkillLevel.BEGINNER
        elif score < 0.40:
            return SkillLevel.INTERMEDIATE
        elif score < 0.60:
            return SkillLevel.ADVANCED
        elif score < 0.80:
            return SkillLevel.VERY_ADVANCED
        else:
            return SkillLevel.EXPERT

    @property
    def bloom_level(self) -> BloomLevel:
        """Derive Bloom level from cognitive load"""
        load = self.cognitive_load
        if load < 0.15:
            return BloomLevel.REMEMBER
        elif load < 0.35:
            return BloomLevel.UNDERSTAND
        elif load < 0.50:
            return BloomLevel.APPLY
        elif load < 0.70:
            return BloomLevel.ANALYZE
        elif load < 0.85:
            return BloomLevel.EVALUATE
        else:
            return BloomLevel.CREATE

    @classmethod
    def from_bloom_level(cls, bloom: BloomLevel, base_complexity: float = 0.5) -> "DifficultyVector":
        """Create a difficulty vector from a Bloom level"""
        cognitive_load = BLOOM_TO_COGNITIVE_LOAD.get(bloom, 0.5)
        # Scale other dimensions based on Bloom level
        return cls(
            conceptual_complexity=base_complexity * (1 + cognitive_load * 0.5),
            prerequisites_depth=cognitive_load * 0.8,
            information_density=base_complexity,
            cognitive_load=cognitive_load,
        )

    @classmethod
    def from_skill_level(cls, skill: SkillLevel) -> "DifficultyVector":
        """Create a difficulty vector from a skill level"""
        skill_scores = {
            SkillLevel.BEGINNER: 0.15,
            SkillLevel.INTERMEDIATE: 0.35,
            SkillLevel.ADVANCED: 0.55,
            SkillLevel.VERY_ADVANCED: 0.75,
            SkillLevel.EXPERT: 0.90,
        }
        base_score = skill_scores.get(skill, 0.5)
        return cls(
            conceptual_complexity=base_score,
            prerequisites_depth=base_score * 0.9,
            information_density=base_score * 0.85,
            cognitive_load=base_score * 1.1,  # Slightly higher cognitive load
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "conceptual_complexity": round(self.conceptual_complexity, 3),
            "prerequisites_depth": round(self.prerequisites_depth, 3),
            "information_density": round(self.information_density, 3),
            "cognitive_load": round(self.cognitive_load, 3),
            "composite_score": round(self.composite_score, 3),
            "skill_level": self.skill_level.value,
            "bloom_level": self.bloom_level.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DifficultyVector":
        """Create from dictionary"""
        return cls(
            conceptual_complexity=data.get("conceptual_complexity", 0.5),
            prerequisites_depth=data.get("prerequisites_depth", 0.5),
            information_density=data.get("information_density", 0.5),
            cognitive_load=data.get("cognitive_load", 0.5),
        )

    def difficulty_delta(self, other: "DifficultyVector") -> float:
        """Calculate difficulty delta between two vectors"""
        return abs(self.composite_score - other.composite_score)

    def is_smooth_progression(self, next_vector: "DifficultyVector", max_jump: float = 0.15) -> bool:
        """Check if progression to next vector is smooth (within max_jump)"""
        return self.difficulty_delta(next_vector) <= max_jump


@dataclass
class CalibratedConcept:
    """A concept with its difficulty calibration"""
    concept_id: str
    name: str
    description: str
    difficulty: DifficultyVector
    prerequisites: List[str] = field(default_factory=list)
    estimated_duration_minutes: int = 10
    keywords: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)  # RAG source references

    @property
    def skill_level(self) -> SkillLevel:
        return self.difficulty.skill_level

    @property
    def bloom_level(self) -> BloomLevel:
        return self.difficulty.bloom_level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty.to_dict(),
            "prerequisites": self.prerequisites,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "keywords": self.keywords,
            "source_ids": self.source_ids,
            "skill_level": self.skill_level.value,
            "bloom_level": self.bloom_level.value,
        }


@dataclass
class DifficultyProgression:
    """Tracks difficulty progression through a course"""
    concepts: List[CalibratedConcept]
    max_difficulty_jump: float = 0.15

    @property
    def difficulty_curve(self) -> List[float]:
        """Get the difficulty curve as list of composite scores"""
        return [c.difficulty.composite_score for c in self.concepts]

    @property
    def is_smooth(self) -> bool:
        """Check if entire progression is smooth"""
        for i in range(1, len(self.concepts)):
            if not self.concepts[i-1].difficulty.is_smooth_progression(
                self.concepts[i].difficulty, self.max_difficulty_jump
            ):
                return False
        return True

    @property
    def difficulty_jumps(self) -> List[Dict[str, Any]]:
        """Find all difficulty jumps that exceed threshold"""
        jumps = []
        for i in range(1, len(self.concepts)):
            delta = self.concepts[i-1].difficulty.difficulty_delta(self.concepts[i].difficulty)
            if delta > self.max_difficulty_jump:
                jumps.append({
                    "from_concept": self.concepts[i-1].name,
                    "to_concept": self.concepts[i].name,
                    "delta": round(delta, 3),
                    "index": i,
                })
        return jumps

    @property
    def average_difficulty(self) -> float:
        """Calculate average difficulty"""
        if not self.concepts:
            return 0.0
        return sum(self.difficulty_curve) / len(self.concepts)

    @property
    def difficulty_range(self) -> Dict[str, float]:
        """Get min/max difficulty range"""
        if not self.concepts:
            return {"min": 0.0, "max": 0.0}
        curve = self.difficulty_curve
        return {
            "min": round(min(curve), 3),
            "max": round(max(curve), 3),
        }

    def summary(self) -> Dict[str, Any]:
        """Get progression summary"""
        return {
            "total_concepts": len(self.concepts),
            "is_smooth": self.is_smooth,
            "difficulty_jumps": self.difficulty_jumps,
            "average_difficulty": round(self.average_difficulty, 3),
            "difficulty_range": self.difficulty_range,
            "skill_distribution": self._skill_distribution(),
        }

    def _skill_distribution(self) -> Dict[str, int]:
        """Count concepts per skill level"""
        distribution = {level.value: 0 for level in SkillLevel}
        for concept in self.concepts:
            distribution[concept.skill_level.value] += 1
        return distribution
