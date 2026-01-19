"""
Exercise Service

Manages exercise library and selection.
Now integrates with ExerciseGeneratorService for dynamic course-based exercises.
"""

import json
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from models.practice_models import (
    Exercise,
    ExerciseType,
    ExerciseCategory,
    DifficultyLevel,
    LearnerProgress,
)
from agents.exercise_selector import ExerciseSelector
from services.exercise_generator import get_exercise_generator, ExerciseGenerationConfig


class ExerciseService:
    """
    Manages the exercise library and exercise selection.

    Features:
    - Load exercises from JSON files (legacy, for static exercises)
    - Dynamic exercise generation from course content via LLM
    - Filter and search exercises
    - Smart exercise selection
    - Automatic generation when accessing course-specific exercises
    """

    def __init__(self):
        self.exercises: Dict[str, Exercise] = {}
        self.selector = ExerciseSelector()
        self._generator = get_exercise_generator()
        self._load_exercises()

    def _load_exercises(self):
        """Load exercises from the exercises directory"""
        exercises_dir = Path(__file__).parent.parent / "exercises"

        if not exercises_dir.exists():
            print(f"[EXERCISE_SERVICE] Exercises directory not found: {exercises_dir}")
            return

        # Load all JSON files in subdirectories
        for category_dir in exercises_dir.iterdir():
            if category_dir.is_dir():
                for exercise_file in category_dir.glob("*.json"):
                    try:
                        with open(exercise_file, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Handle single exercise or list
                        if isinstance(data, list):
                            for ex_data in data:
                                exercise = Exercise(**ex_data)
                                self.exercises[exercise.id] = exercise
                        else:
                            exercise = Exercise(**data)
                            self.exercises[exercise.id] = exercise

                    except Exception as e:
                        print(f"[EXERCISE_SERVICE] Error loading {exercise_file}: {e}")

        print(f"[EXERCISE_SERVICE] Loaded {len(self.exercises)} exercises")

    def get_exercise(self, exercise_id: str) -> Optional[Exercise]:
        """Get an exercise by ID"""
        return self.exercises.get(exercise_id)

    def list_exercises(
        self,
        category: Optional[ExerciseCategory] = None,
        difficulty: Optional[DifficultyLevel] = None,
        exercise_type: Optional[ExerciseType] = None,
        course_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Exercise]:
        """List exercises with optional filters"""
        exercises = list(self.exercises.values())

        # Apply filters
        if category:
            exercises = [e for e in exercises if e.category == category]
        if difficulty:
            exercises = [e for e in exercises if e.difficulty == difficulty]
        if exercise_type:
            exercises = [e for e in exercises if e.type == exercise_type]
        if course_id:
            exercises = [e for e in exercises if e.course_id == course_id]
        if tags:
            exercises = [e for e in exercises if any(t in e.tags for t in tags)]

        # Sort by difficulty, then points
        difficulty_order = {
            DifficultyLevel.BEGINNER: 1,
            DifficultyLevel.INTERMEDIATE: 2,
            DifficultyLevel.ADVANCED: 3,
            DifficultyLevel.EXPERT: 4,
        }
        exercises.sort(key=lambda e: (difficulty_order.get(e.difficulty, 0), -e.points))

        return exercises[offset:offset + limit]

    async def select_next_exercise(
        self,
        completed_exercises: List[str],
        difficulty: str = "beginner",
        categories: List[str] = None,
        course_id: Optional[str] = None,
    ) -> Optional[Exercise]:
        """
        Select the next best exercise for a learner.

        If course_id is provided and no exercises exist for it,
        automatically generates exercises using the LLM-powered generator.
        """
        # Create a mock progress object
        progress = LearnerProgress(
            user_id="temp",
            completed_exercises=completed_exercises,
        )

        # Map difficulty string to enum
        try:
            diff_level = DifficultyLevel(difficulty)
        except ValueError:
            diff_level = DifficultyLevel.BEGINNER

        # Map categories
        cat_enums = []
        if categories:
            for cat in categories:
                try:
                    cat_enums.append(ExerciseCategory(cat))
                except ValueError:
                    pass

        # Get available exercises
        available = self.list_exercises(
            difficulty=diff_level,
            course_id=course_id,
        )

        if not available:
            # Try without difficulty filter
            available = self.list_exercises(course_id=course_id)

        # If no exercises for this course, try to generate them dynamically
        if not available and course_id:
            print(f"[EXERCISE_SERVICE] No exercises for course {course_id}, generating dynamically...", flush=True)
            try:
                generated = await self.generate_exercises_for_course(course_id)
                if generated:
                    available = generated
                    print(f"[EXERCISE_SERVICE] Generated {len(generated)} exercises for course {course_id}", flush=True)
            except Exception as e:
                print(f"[EXERCISE_SERVICE] Failed to generate exercises: {e}", flush=True)

        if not available:
            return None

        # Use selector for smart selection
        return await self.selector.select_next_exercise(
            available_exercises=available,
            learner_progress=progress,
            preferred_category=cat_enums[0] if cat_enums else None,
            preferred_difficulty=diff_level,
        )

    def search_exercises(self, query: str, limit: int = 20) -> List[Exercise]:
        """Search exercises by text query"""
        query_lower = query.lower()
        results = []

        for exercise in self.exercises.values():
            score = 0

            # Title match (highest weight)
            if query_lower in exercise.title.lower():
                score += 10

            # Description match
            if query_lower in exercise.description.lower():
                score += 5

            # Instructions match
            if query_lower in exercise.instructions.lower():
                score += 3

            # Tags match
            for tag in exercise.tags:
                if query_lower in tag.lower():
                    score += 2

            if score > 0:
                results.append((exercise, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in results[:limit]]

    def get_exercises_by_category(self, category: ExerciseCategory) -> List[Exercise]:
        """Get all exercises for a category"""
        return [e for e in self.exercises.values() if e.category == category]

    def get_exercise_stats(self) -> Dict[str, Any]:
        """Get statistics about the exercise library"""
        stats = {
            "total": len(self.exercises),
            "by_category": {},
            "by_difficulty": {},
            "by_type": {},
        }

        for exercise in self.exercises.values():
            # By category
            cat = exercise.category.value
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # By difficulty
            diff = exercise.difficulty.value
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

            # By type
            etype = exercise.type.value
            stats["by_type"][etype] = stats["by_type"].get(etype, 0) + 1

        return stats

    def add_exercise(self, exercise: Exercise) -> bool:
        """Add a new exercise to the library"""
        if exercise.id in self.exercises:
            return False
        self.exercises[exercise.id] = exercise
        return True

    def update_exercise(self, exercise_id: str, updates: Dict[str, Any]) -> Optional[Exercise]:
        """Update an existing exercise"""
        if exercise_id not in self.exercises:
            return None

        exercise = self.exercises[exercise_id]
        for key, value in updates.items():
            if hasattr(exercise, key):
                setattr(exercise, key, value)

        return exercise

    async def generate_exercises_for_course(
        self,
        course_id: str,
        config: Optional[ExerciseGenerationConfig] = None,
        force_regenerate: bool = False,
    ) -> List[Exercise]:
        """
        Generate exercises for a course using the LLM-powered generator.

        This method:
        1. Fetches course content from course-generator service
        2. Retrieves RAG context from uploaded documents
        3. Uses GPT-4 to generate tailored exercises
        4. Adds them to the local exercise library

        Args:
            course_id: The course job ID
            config: Optional generation configuration
            force_regenerate: If True, regenerate even if exercises exist

        Returns:
            List of generated exercises
        """
        try:
            # Use the generator service
            exercise_set = await self._generator.generate_exercises_for_course(
                course_id=course_id,
                config=config,
                force_regenerate=force_regenerate,
            )

            # Add all generated exercises to our local store
            for exercise in exercise_set.exercises:
                self.add_exercise(exercise)

            return exercise_set.exercises

        except Exception as e:
            print(f"[EXERCISE_SERVICE] Error generating exercises: {e}", flush=True)
            raise

    async def get_course_exercises(
        self,
        course_id: str,
        difficulty: Optional[DifficultyLevel] = None,
        exercise_type: Optional[ExerciseType] = None,
        auto_generate: bool = True,
    ) -> List[Exercise]:
        """
        Get exercises for a specific course.

        If auto_generate is True and no exercises exist, generates them dynamically.

        Args:
            course_id: The course job ID
            difficulty: Optional filter by difficulty
            exercise_type: Optional filter by type
            auto_generate: Whether to generate if none exist

        Returns:
            List of exercises for the course
        """
        # First check local store
        exercises = self.list_exercises(
            course_id=course_id,
            difficulty=difficulty,
            exercise_type=exercise_type,
        )

        # Generate if needed
        if not exercises and auto_generate:
            print(f"[EXERCISE_SERVICE] Auto-generating exercises for course {course_id}", flush=True)
            exercises = await self.generate_exercises_for_course(course_id)

            # Apply filters to generated exercises
            if difficulty:
                exercises = [e for e in exercises if e.difficulty == difficulty]
            if exercise_type:
                exercises = [e for e in exercises if e.type == exercise_type]

        return exercises

    async def regenerate_exercise(
        self,
        course_id: str,
        exercise_id: str,
        feedback: Optional[str] = None,
    ) -> Optional[Exercise]:
        """
        Regenerate a specific exercise based on feedback.

        Args:
            course_id: The course job ID
            exercise_id: The exercise to regenerate
            feedback: Optional feedback for improvement

        Returns:
            The regenerated exercise or None
        """
        new_exercise = await self._generator.regenerate_exercise(
            course_id=course_id,
            exercise_id=exercise_id,
            feedback=feedback,
        )

        if new_exercise:
            # Update local store
            self.exercises[new_exercise.id] = new_exercise
            # Remove old exercise if different ID
            if exercise_id != new_exercise.id and exercise_id in self.exercises:
                del self.exercises[exercise_id]

        return new_exercise

    def clear_course_exercises(self, course_id: str):
        """Remove all exercises for a course (to force regeneration)"""
        to_remove = [eid for eid, ex in self.exercises.items() if ex.course_id == course_id]
        for eid in to_remove:
            del self.exercises[eid]
        self._generator.clear_cache(course_id)
        print(f"[EXERCISE_SERVICE] Cleared {len(to_remove)} exercises for course {course_id}", flush=True)

    # Legacy method for backwards compatibility
    async def generate_from_course(
        self,
        course_id: str,
        course_content: str,
        category: ExerciseCategory,
        count: int = 5,
    ) -> List[Exercise]:
        """
        Legacy method - now delegates to generate_exercises_for_course.

        For backwards compatibility only.
        """
        return await self.generate_exercises_for_course(course_id)
