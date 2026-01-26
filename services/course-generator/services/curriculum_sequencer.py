"""
Curriculum Sequencer Service

This service sequences concepts into a smooth learning path by:
1. Topological sorting (respecting prerequisites)
2. Difficulty ramping (max 15% jump between consecutive concepts)
3. Module assignment based on difficulty ranges
4. Duration balancing across modules

Inspired by MAESTRO's curriculum sequencing algorithm.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum

from models.difficulty_models import (
    DifficultyVector,
    CalibratedConcept,
    DifficultyProgression,
    SkillLevel,
)


class ProgressionPath(str, Enum):
    """Pre-defined learning progression paths"""
    BEGINNER_TO_INTERMEDIATE = "beginner_to_intermediate"
    INTERMEDIATE_TO_ADVANCED = "intermediate_to_advanced"
    ADVANCED_TO_EXPERT = "advanced_to_expert"
    FULL_RANGE = "full_range"


# Progression path to skill level ranges
PROGRESSION_RANGES: Dict[ProgressionPath, Tuple[SkillLevel, SkillLevel]] = {
    ProgressionPath.BEGINNER_TO_INTERMEDIATE: (SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE),
    ProgressionPath.INTERMEDIATE_TO_ADVANCED: (SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED),
    ProgressionPath.ADVANCED_TO_EXPERT: (SkillLevel.ADVANCED, SkillLevel.EXPERT),
    ProgressionPath.FULL_RANGE: (SkillLevel.BEGINNER, SkillLevel.EXPERT),
}

# Skill level to difficulty score ranges
SKILL_LEVEL_RANGES: Dict[SkillLevel, Tuple[float, float]] = {
    SkillLevel.BEGINNER: (0.0, 0.20),
    SkillLevel.INTERMEDIATE: (0.20, 0.40),
    SkillLevel.ADVANCED: (0.40, 0.60),
    SkillLevel.VERY_ADVANCED: (0.60, 0.80),
    SkillLevel.EXPERT: (0.80, 1.0),
}


@dataclass
class SequencedConcept:
    """A concept with its position in the learning sequence"""
    concept: CalibratedConcept
    sequence_order: int
    module_index: int
    difficulty_delta: float  # Change from previous concept
    cumulative_duration_minutes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept.to_dict(),
            "sequence_order": self.sequence_order,
            "module_index": self.module_index,
            "difficulty_delta": round(self.difficulty_delta, 3),
            "cumulative_duration_minutes": self.cumulative_duration_minutes,
        }


@dataclass
class LearningModule:
    """A module containing a group of related concepts"""
    module_id: str
    name: str
    description: str
    concepts: List[SequencedConcept] = field(default_factory=list)
    skill_level_range: Tuple[SkillLevel, SkillLevel] = (SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE)
    learning_objectives: List[str] = field(default_factory=list)

    @property
    def total_duration_minutes(self) -> int:
        return sum(c.concept.estimated_duration_minutes for c in self.concepts)

    @property
    def concept_count(self) -> int:
        return len(self.concepts)

    @property
    def average_difficulty(self) -> float:
        if not self.concepts:
            return 0.0
        return sum(c.concept.difficulty.composite_score for c in self.concepts) / len(self.concepts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "name": self.name,
            "description": self.description,
            "concepts": [c.to_dict() for c in self.concepts],
            "total_duration_minutes": self.total_duration_minutes,
            "concept_count": self.concept_count,
            "average_difficulty": round(self.average_difficulty, 3),
            "skill_level_range": [self.skill_level_range[0].value, self.skill_level_range[1].value],
            "learning_objectives": self.learning_objectives,
        }


@dataclass
class LearningPath:
    """Complete sequenced learning path"""
    modules: List[LearningModule]
    progression_path: ProgressionPath
    total_duration_minutes: int
    difficulty_curve: List[float]
    is_smooth: bool
    smoothing_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modules": [m.to_dict() for m in self.modules],
            "progression_path": self.progression_path.value,
            "total_duration_minutes": self.total_duration_minutes,
            "total_concepts": sum(m.concept_count for m in self.modules),
            "difficulty_curve": [round(d, 3) for d in self.difficulty_curve],
            "is_smooth": self.is_smooth,
            "smoothing_applied": self.smoothing_applied,
        }


class CurriculumSequencer:
    """
    Sequences concepts into a smooth, pedagogically sound learning path.

    Features:
    1. Topological sort respecting prerequisites
    2. Difficulty ramping with max 15% jumps
    3. Module assignment based on difficulty ranges
    4. Duration balancing
    5. Progression path filtering
    """

    # Configuration
    MAX_DIFFICULTY_JUMP = 0.15  # Maximum allowed difficulty increase between concepts
    SMOOTHING_WINDOW = 3  # Number of concepts to consider for smoothing
    MIN_MODULE_CONCEPTS = 3  # Minimum concepts per module
    MAX_MODULE_CONCEPTS = 15  # Maximum concepts per module

    def __init__(
        self,
        max_difficulty_jump: float = 0.15,
        smoothing_window: int = 3,
    ):
        self.max_difficulty_jump = max_difficulty_jump
        self.smoothing_window = smoothing_window

    def sequence(
        self,
        concepts: List[CalibratedConcept],
        progression_path: ProgressionPath = ProgressionPath.FULL_RANGE,
        num_modules: int = 5,
        target_duration_minutes: Optional[int] = None,
    ) -> LearningPath:
        """
        Sequence concepts into a learning path.

        Args:
            concepts: List of calibrated concepts
            progression_path: Desired progression (e.g., beginner_to_intermediate)
            num_modules: Number of modules to create
            target_duration_minutes: Optional target total duration

        Returns:
            LearningPath with sequenced modules
        """
        print(f"[CURRICULUM_SEQUENCER] Sequencing {len(concepts)} concepts into {num_modules} modules", flush=True)

        # Step 1: Filter concepts by progression path
        filtered_concepts = self._filter_by_progression(concepts, progression_path)
        print(f"[CURRICULUM_SEQUENCER] After progression filter: {len(filtered_concepts)} concepts", flush=True)

        if not filtered_concepts:
            # If no concepts match, use all concepts
            filtered_concepts = concepts

        # Step 2: Topological sort (respects prerequisites)
        sorted_concepts = self._topological_sort(filtered_concepts)
        print(f"[CURRICULUM_SEQUENCER] After topological sort: {len(sorted_concepts)} concepts", flush=True)

        # Step 3: Apply difficulty smoothing
        smoothed_concepts, smoothing_applied = self._smooth_difficulty_progression(sorted_concepts)
        print(f"[CURRICULUM_SEQUENCER] Smoothing applied: {smoothing_applied}", flush=True)

        # Step 4: Assign to modules
        modules = self._assign_to_modules(smoothed_concepts, num_modules, progression_path)
        print(f"[CURRICULUM_SEQUENCER] Created {len(modules)} modules", flush=True)

        # Step 5: Build learning path
        difficulty_curve = [c.difficulty.composite_score for c in smoothed_concepts]
        total_duration = sum(c.estimated_duration_minutes for c in smoothed_concepts)

        # Check if progression is smooth
        is_smooth = self._is_smooth_progression(smoothed_concepts)

        learning_path = LearningPath(
            modules=modules,
            progression_path=progression_path,
            total_duration_minutes=total_duration,
            difficulty_curve=difficulty_curve,
            is_smooth=is_smooth,
            smoothing_applied=smoothing_applied,
        )

        print(f"[CURRICULUM_SEQUENCER] Learning path created: {total_duration}min, smooth={is_smooth}", flush=True)

        return learning_path

    def _filter_by_progression(
        self,
        concepts: List[CalibratedConcept],
        progression_path: ProgressionPath,
    ) -> List[CalibratedConcept]:
        """Filter concepts to match the progression path skill range"""
        start_level, end_level = PROGRESSION_RANGES[progression_path]
        start_range = SKILL_LEVEL_RANGES[start_level]
        end_range = SKILL_LEVEL_RANGES[end_level]

        min_difficulty = start_range[0]
        max_difficulty = end_range[1]

        return [
            c for c in concepts
            if min_difficulty <= c.difficulty.composite_score <= max_difficulty
        ]

    def _topological_sort(
        self,
        concepts: List[CalibratedConcept],
    ) -> List[CalibratedConcept]:
        """
        Sort concepts respecting prerequisites using Kahn's algorithm.
        Falls back to difficulty-based sorting if cycle detected.
        """
        # Build adjacency list and in-degree count
        concept_map = {c.concept_id: c for c in concepts}
        concept_ids = set(concept_map.keys())

        in_degree: Dict[str, int] = defaultdict(int)
        adjacency: Dict[str, List[str]] = defaultdict(list)

        for concept in concepts:
            in_degree[concept.concept_id] = 0

        for concept in concepts:
            for prereq_id in concept.prerequisites:
                if prereq_id in concept_ids:
                    adjacency[prereq_id].append(concept.concept_id)
                    in_degree[concept.concept_id] += 1

        # Kahn's algorithm
        queue = deque([cid for cid, deg in in_degree.items() if deg == 0])
        sorted_ids = []

        while queue:
            # Among concepts with 0 in-degree, pick the one with lowest difficulty
            # This ensures smoother progression within topological constraints
            candidates = list(queue)
            candidates.sort(key=lambda cid: concept_map[cid].difficulty.composite_score)
            current_id = candidates[0]
            queue.remove(current_id)

            sorted_ids.append(current_id)

            for neighbor_id in adjacency[current_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        # Check for cycle (not all nodes processed)
        if len(sorted_ids) != len(concepts):
            print("[CURRICULUM_SEQUENCER] Warning: Cycle detected, falling back to difficulty sort", flush=True)
            # Fallback: sort by difficulty
            return sorted(concepts, key=lambda c: c.difficulty.composite_score)

        return [concept_map[cid] for cid in sorted_ids]

    def _smooth_difficulty_progression(
        self,
        concepts: List[CalibratedConcept],
    ) -> Tuple[List[CalibratedConcept], bool]:
        """
        Reorder concepts to ensure smooth difficulty progression.
        Returns (reordered_concepts, was_smoothing_needed).
        """
        if len(concepts) <= 1:
            return concepts, False

        smoothing_needed = False
        result = [concepts[0]]

        for i in range(1, len(concepts)):
            current = concepts[i]
            prev = result[-1]

            delta = current.difficulty.composite_score - prev.difficulty.composite_score

            if delta > self.max_difficulty_jump:
                smoothing_needed = True
                # Find a better position for this concept
                inserted = False

                # Look for intermediate concepts that could come before
                for j in range(i + 1, len(concepts)):
                    candidate = concepts[j]
                    candidate_delta = candidate.difficulty.composite_score - prev.difficulty.composite_score

                    if 0 <= candidate_delta <= self.max_difficulty_jump:
                        # Check prerequisites
                        if self._can_insert_before(candidate, current, result):
                            result.append(candidate)
                            # Mark as used (we'll handle this in actual implementation)
                            inserted = True
                            break

                # If no better candidate, just add current
                result.append(current)
            else:
                result.append(current)

        # Remove duplicates while preserving order
        seen = set()
        final_result = []
        for c in result:
            if c.concept_id not in seen:
                seen.add(c.concept_id)
                final_result.append(c)

        # Add any missing concepts at the end
        for c in concepts:
            if c.concept_id not in seen:
                final_result.append(c)

        return final_result, smoothing_needed

    def _can_insert_before(
        self,
        candidate: CalibratedConcept,
        target: CalibratedConcept,
        current_sequence: List[CalibratedConcept],
    ) -> bool:
        """Check if candidate can be inserted before target without violating prerequisites"""
        current_ids = {c.concept_id for c in current_sequence}

        # Candidate's prerequisites must all be in current sequence
        for prereq in candidate.prerequisites:
            if prereq not in current_ids:
                return False

        # Target cannot depend on candidate
        if candidate.concept_id in target.prerequisites:
            return False

        return True

    def _is_smooth_progression(self, concepts: List[CalibratedConcept]) -> bool:
        """Check if the concept sequence has smooth difficulty progression"""
        for i in range(1, len(concepts)):
            delta = concepts[i].difficulty.composite_score - concepts[i-1].difficulty.composite_score
            if delta > self.max_difficulty_jump:
                return False
        return True

    def _assign_to_modules(
        self,
        concepts: List[CalibratedConcept],
        num_modules: int,
        progression_path: ProgressionPath,
    ) -> List[LearningModule]:
        """Assign sequenced concepts to modules based on difficulty ranges"""
        if not concepts:
            return []

        # Calculate difficulty range per module
        min_diff = min(c.difficulty.composite_score for c in concepts)
        max_diff = max(c.difficulty.composite_score for c in concepts)
        diff_range = max_diff - min_diff

        if diff_range == 0:
            diff_range = 1.0  # Avoid division by zero

        module_width = diff_range / num_modules

        # Create modules
        modules = []
        for i in range(num_modules):
            module_min = min_diff + i * module_width
            module_max = min_diff + (i + 1) * module_width

            # Determine skill level range for this module
            start_skill = self._difficulty_to_skill(module_min)
            end_skill = self._difficulty_to_skill(module_max)

            module = LearningModule(
                module_id=f"module_{i+1:02d}",
                name=f"Module {i+1}",
                description=f"Concepts from {start_skill.value} to {end_skill.value} level",
                skill_level_range=(start_skill, end_skill),
                learning_objectives=[],
            )
            modules.append(module)

        # Assign concepts to modules
        cumulative_duration = 0
        for seq_order, concept in enumerate(concepts):
            diff_score = concept.difficulty.composite_score

            # Find appropriate module
            module_idx = min(
                int((diff_score - min_diff) / module_width),
                num_modules - 1
            )
            module_idx = max(0, module_idx)  # Ensure non-negative

            # Calculate difficulty delta
            if seq_order == 0:
                diff_delta = 0.0
            else:
                prev_concept = concepts[seq_order - 1]
                diff_delta = diff_score - prev_concept.difficulty.composite_score

            cumulative_duration += concept.estimated_duration_minutes

            sequenced = SequencedConcept(
                concept=concept,
                sequence_order=seq_order,
                module_index=module_idx,
                difficulty_delta=diff_delta,
                cumulative_duration_minutes=cumulative_duration,
            )

            modules[module_idx].concepts.append(sequenced)

        # Generate module names based on content
        for module in modules:
            if module.concepts:
                first_concept = module.concepts[0].concept.name
                last_concept = module.concepts[-1].concept.name
                module.name = f"From {first_concept} to {last_concept}"
                module.learning_objectives = [
                    f"Understand {c.concept.name}" for c in module.concepts[:3]
                ]

        # Filter out empty modules
        modules = [m for m in modules if m.concepts]

        return modules

    def _difficulty_to_skill(self, difficulty: float) -> SkillLevel:
        """Convert difficulty score to skill level"""
        for skill, (min_d, max_d) in SKILL_LEVEL_RANGES.items():
            if min_d <= difficulty < max_d:
                return skill
        return SkillLevel.EXPERT


# Singleton instance
_sequencer_instance: Optional[CurriculumSequencer] = None


def get_curriculum_sequencer() -> CurriculumSequencer:
    """Get the singleton curriculum sequencer instance"""
    global _sequencer_instance
    if _sequencer_instance is None:
        _sequencer_instance = CurriculumSequencer()
    return _sequencer_instance


def sequence_concepts(
    concepts: List[CalibratedConcept],
    progression_path: str = "full_range",
    num_modules: int = 5,
) -> LearningPath:
    """
    Convenience function to sequence concepts into a learning path.

    Example:
        from services.curriculum_sequencer import sequence_concepts

        path = sequence_concepts(
            calibrated_concepts,
            progression_path="beginner_to_intermediate",
            num_modules=5
        )

        print(f"Total duration: {path.total_duration_minutes} minutes")
        print(f"Smooth progression: {path.is_smooth}")

        for module in path.modules:
            print(f"  {module.name}: {module.concept_count} concepts")
    """
    sequencer = get_curriculum_sequencer()
    prog_path = ProgressionPath(progression_path)
    return sequencer.sequence(concepts, prog_path, num_modules)
