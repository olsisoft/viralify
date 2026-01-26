"""
Curriculum Sequencer Engine

Layer 4 of the MAESTRO pipeline.
Sequences concepts into modules with smooth difficulty progression.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from models.data_models import (
    Concept,
    Module,
    ProgressionPath,
    SkillLevel,
    PROGRESSION_RANGES,
)
from engines.knowledge_graph import KnowledgeGraphEngine


# Maximum allowed difficulty jump between consecutive concepts
MAX_DIFFICULTY_JUMP = 0.15

# Target concepts per module
CONCEPTS_PER_MODULE = 5


@dataclass
class SequencedConcept:
    """A concept with its position in the learning sequence"""
    concept: Concept
    sequence_order: int
    module_index: int
    difficulty_delta: float = 0.0  # Change from previous concept


@dataclass
class LearningPath:
    """A complete learning path with modules and concepts"""
    modules: List[Module]
    sequenced_concepts: List[SequencedConcept]
    total_duration_minutes: int = 0
    difficulty_range: Tuple[float, float] = (0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modules": [m.to_dict() for m in self.modules],
            "sequenced_concepts": [
                {
                    "concept": sc.concept.to_dict(),
                    "sequence_order": sc.sequence_order,
                    "module_index": sc.module_index,
                    "difficulty_delta": round(sc.difficulty_delta, 3),
                }
                for sc in self.sequenced_concepts
            ],
            "total_duration_minutes": self.total_duration_minutes,
            "difficulty_range": [round(d, 3) for d in self.difficulty_range],
        }


class CurriculumSequencerEngine:
    """
    Sequences concepts into a structured curriculum.

    Features:
    - Topological prerequisite ordering
    - Smooth difficulty progression (max 15% jump)
    - Module grouping by theme/level
    - Duration optimization
    """

    def __init__(self, max_difficulty_jump: float = MAX_DIFFICULTY_JUMP):
        self.max_difficulty_jump = max_difficulty_jump

    def sequence_curriculum(
        self,
        concepts: List[Concept],
        knowledge_graph: KnowledgeGraphEngine,
        progression_path: ProgressionPath,
        target_modules: int = 5,
    ) -> LearningPath:
        """
        Sequence concepts into a learning path.

        Args:
            concepts: All concepts to sequence
            knowledge_graph: The prerequisite graph
            progression_path: Target progression
            target_modules: Desired number of modules

        Returns:
            LearningPath with sequenced modules and concepts
        """
        print(f"[CURRICULUM_SEQUENCER] Sequencing {len(concepts)} concepts into {target_modules} modules", flush=True)

        # Get topological order from knowledge graph
        ordered_concepts = knowledge_graph.get_learning_order()

        # Apply difficulty smoothing
        smoothed_concepts = self._smooth_difficulty_progression(ordered_concepts)

        # Group into modules
        modules = self._create_modules(
            smoothed_concepts,
            target_modules,
            progression_path,
        )

        # Create sequenced concepts
        sequenced = []
        prev_difficulty = 0.0

        for module_idx, module in enumerate(modules):
            for concept in [c for c in smoothed_concepts if c.id in module.concept_ids]:
                current_difficulty = concept.difficulty.composite_score
                sequenced.append(SequencedConcept(
                    concept=concept,
                    sequence_order=len(sequenced),
                    module_index=module_idx,
                    difficulty_delta=current_difficulty - prev_difficulty,
                ))
                prev_difficulty = current_difficulty

        # Calculate totals
        total_duration = sum(c.estimated_duration_minutes for c in concepts)
        difficulties = [c.difficulty.composite_score for c in concepts]

        learning_path = LearningPath(
            modules=modules,
            sequenced_concepts=sequenced,
            total_duration_minutes=total_duration,
            difficulty_range=(min(difficulties), max(difficulties)) if difficulties else (0.0, 1.0),
        )

        print(f"[CURRICULUM_SEQUENCER] Created {len(modules)} modules with {len(sequenced)} sequenced concepts", flush=True)
        return learning_path

    def _smooth_difficulty_progression(
        self,
        concepts: List[Concept],
    ) -> List[Concept]:
        """
        Reorder concepts to ensure smooth difficulty progression.

        Maintains prerequisite ordering while minimizing difficulty jumps.
        """
        if not concepts:
            return concepts

        # Sort by difficulty while respecting prerequisites
        # This is a simplified version - full implementation would use
        # constraint satisfaction or dynamic programming

        result = []
        remaining = list(concepts)
        prev_difficulty = 0.0

        while remaining:
            # Find best next concept
            best_concept = None
            best_score = float('inf')

            for concept in remaining:
                # Check if prerequisites are satisfied
                prereqs_satisfied = all(
                    prereq_id in [c.id for c in result]
                    for prereq_id in concept.prerequisites
                )

                if prereqs_satisfied:
                    difficulty = concept.difficulty.composite_score
                    jump = abs(difficulty - prev_difficulty)

                    # Score: prefer smaller jumps, but also consider natural order
                    score = jump

                    if score < best_score:
                        best_score = score
                        best_concept = concept

            if best_concept is None:
                # No valid concept found, take first remaining
                best_concept = remaining[0]

            result.append(best_concept)
            remaining.remove(best_concept)
            prev_difficulty = best_concept.difficulty.composite_score

        return result

    def _create_modules(
        self,
        concepts: List[Concept],
        target_modules: int,
        progression_path: ProgressionPath,
    ) -> List[Module]:
        """
        Group concepts into modules.

        Args:
            concepts: Ordered list of concepts
            target_modules: Desired number of modules
            progression_path: Target progression path

        Returns:
            List of Module objects
        """
        if not concepts:
            return []

        # Calculate concepts per module
        concepts_per_module = max(1, len(concepts) // target_modules)
        modules = []

        current_concepts = []
        current_module_idx = 0

        for concept in concepts:
            current_concepts.append(concept)

            # Start new module when we have enough concepts
            # or when there's a significant skill level change
            should_split = (
                len(current_concepts) >= concepts_per_module and
                current_module_idx < target_modules - 1
            )

            if should_split:
                module = self._create_module(
                    current_concepts,
                    current_module_idx,
                    progression_path,
                )
                modules.append(module)
                current_concepts = []
                current_module_idx += 1

        # Add remaining concepts to final module
        if current_concepts:
            module = self._create_module(
                current_concepts,
                current_module_idx,
                progression_path,
            )
            modules.append(module)

        return modules

    def _create_module(
        self,
        concepts: List[Concept],
        module_index: int,
        progression_path: ProgressionPath,
    ) -> Module:
        """Create a module from a group of concepts"""
        if not concepts:
            return Module()

        # Determine skill level range
        skill_levels = [c.skill_level for c in concepts]
        min_level = min(skill_levels, key=lambda x: list(SkillLevel).index(x))
        max_level = max(skill_levels, key=lambda x: list(SkillLevel).index(x))

        # Generate module name based on concepts
        concept_names = [c.name for c in concepts[:3]]
        module_name = f"Module {module_index + 1}: {', '.join(concept_names[:2])}"
        if len(concepts) > 2:
            module_name += "..."

        # Generate learning objectives
        objectives = [
            f"Understand {c.name}" for c in concepts[:3]
        ]

        module = Module(
            name=module_name,
            description=f"Covers {len(concepts)} concepts from {min_level.value} to {max_level.value} level",
            learning_objectives=objectives,
            concept_ids=[c.id for c in concepts],
            skill_level_range=(min_level, max_level),
        )

        return module

    def validate_sequence(
        self,
        learning_path: LearningPath,
    ) -> List[Dict[str, Any]]:
        """
        Validate the learning path for issues.

        Returns:
            List of validation issues
        """
        issues = []

        prev_difficulty = 0.0
        for sc in learning_path.sequenced_concepts:
            # Check difficulty jumps
            difficulty = sc.concept.difficulty.composite_score
            jump = difficulty - prev_difficulty

            if jump > self.max_difficulty_jump:
                issues.append({
                    "type": "excessive_difficulty_jump",
                    "concept_id": sc.concept.id,
                    "jump": round(jump, 3),
                    "max_allowed": self.max_difficulty_jump,
                })

            prev_difficulty = difficulty

        # Check module balance
        module_sizes = [
            len([sc for sc in learning_path.sequenced_concepts if sc.module_index == i])
            for i in range(len(learning_path.modules))
        ]

        if module_sizes:
            avg_size = sum(module_sizes) / len(module_sizes)
            for i, size in enumerate(module_sizes):
                if abs(size - avg_size) > avg_size * 0.5:
                    issues.append({
                        "type": "unbalanced_module",
                        "module_index": i,
                        "size": size,
                        "average_size": round(avg_size, 1),
                    })

        return issues


def sequence_curriculum(
    concepts: List[Concept],
    knowledge_graph: KnowledgeGraphEngine,
    progression_path: str = "beginner_to_intermediate",
    target_modules: int = 5,
) -> LearningPath:
    """
    Convenience function to sequence a curriculum.

    Example:
        learning_path = sequence_curriculum(
            concepts=concepts,
            knowledge_graph=graph,
            progression_path="beginner_to_intermediate",
            target_modules=5,
        )
    """
    engine = CurriculumSequencerEngine()
    path = ProgressionPath(progression_path)

    return engine.sequence_curriculum(
        concepts=concepts,
        knowledge_graph=knowledge_graph,
        progression_path=path,
        target_modules=target_modules,
    )
