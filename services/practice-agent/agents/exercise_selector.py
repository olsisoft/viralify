"""
Exercise Selector

Intelligent selection of exercises based on learner progress and course content.
"""

import random
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from models.practice_models import (
    Exercise,
    DifficultyLevel,
    ExerciseCategory,
    LearnerProgress,
)

try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    _USE_SHARED_LLM = False


class ExerciseSelector:
    """
    Selects appropriate exercises based on:
    - Learner's current level and progress
    - Course content (via RAG)
    - Spaced repetition for review
    - Adaptive difficulty
    """

    def __init__(self):
        self.llm = ChatOpenAI(model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini", temperature=0.3)

    async def select_next_exercise(
        self,
        available_exercises: List[Exercise],
        learner_progress: LearnerProgress,
        course_id: Optional[str] = None,
        preferred_category: Optional[ExerciseCategory] = None,
        preferred_difficulty: Optional[DifficultyLevel] = None,
    ) -> Optional[Exercise]:
        """
        Select the most appropriate next exercise for the learner.
        """
        if not available_exercises:
            return None

        # Filter out completed exercises
        completed_ids = set(learner_progress.completed_exercises)
        candidates = [e for e in available_exercises if e.id not in completed_ids]

        if not candidates:
            # All exercises completed - offer review
            return self._select_review_exercise(available_exercises, learner_progress)

        # Apply filters
        if preferred_category:
            category_filtered = [e for e in candidates if e.category == preferred_category]
            if category_filtered:
                candidates = category_filtered

        if preferred_difficulty:
            difficulty_filtered = [e for e in candidates if e.difficulty == preferred_difficulty]
            if difficulty_filtered:
                candidates = difficulty_filtered

        # Score and rank candidates
        scored_candidates = await self._score_exercises(candidates, learner_progress)

        # Select from top candidates with some randomness
        top_candidates = scored_candidates[:5]
        weights = [score for _, score in top_candidates]

        if not top_candidates:
            return random.choice(candidates) if candidates else None

        selected_exercise, _ = random.choices(top_candidates, weights=weights, k=1)[0]
        return selected_exercise

    async def _score_exercises(
        self,
        exercises: List[Exercise],
        learner_progress: LearnerProgress,
    ) -> List[tuple[Exercise, float]]:
        """Score exercises based on relevance to learner"""
        scored = []

        for exercise in exercises:
            score = 0.0

            # Base score from difficulty match
            learner_level = self._estimate_learner_level(learner_progress)
            difficulty_map = {
                DifficultyLevel.BEGINNER: 1,
                DifficultyLevel.INTERMEDIATE: 2,
                DifficultyLevel.ADVANCED: 3,
                DifficultyLevel.EXPERT: 4,
            }
            exercise_level = difficulty_map.get(exercise.difficulty, 1)

            # Prefer exercises slightly above current level (zone of proximal development)
            level_diff = exercise_level - learner_level
            if level_diff == 0:
                score += 30
            elif level_diff == 1:
                score += 40  # Slightly challenging is best
            elif level_diff == -1:
                score += 20  # Review is okay
            else:
                score += 10

            # Bonus for categories where learner is weak
            category_progress = learner_progress.category_progress.get(exercise.category.value, {})
            if category_progress.get("completed", 0) < 3:
                score += 20  # Encourage variety

            # Bonus for prerequisite completion
            prereqs_met = all(
                p in learner_progress.completed_exercises
                for p in exercise.prerequisite_exercises
            )
            if prereqs_met:
                score += 15
            elif exercise.prerequisite_exercises:
                score -= 50  # Penalize if prerequisites not met

            # Points value consideration
            score += min(exercise.points / 10, 10)  # Up to 10 points bonus

            scored.append((exercise, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _estimate_learner_level(self, progress: LearnerProgress) -> float:
        """Estimate learner's current skill level (1-4)"""
        if progress.total_exercises_completed == 0:
            return 1.0

        # Calculate from difficulty distribution
        total = sum(progress.difficulty_stats.values())
        if total == 0:
            return 1.0

        weighted_sum = (
            progress.difficulty_stats.get("beginner", 0) * 1 +
            progress.difficulty_stats.get("intermediate", 0) * 2 +
            progress.difficulty_stats.get("advanced", 0) * 3 +
            progress.difficulty_stats.get("expert", 0) * 4
        )

        return min(weighted_sum / total + 0.5, 4.0)  # Slightly above average completed

    def _select_review_exercise(
        self,
        exercises: List[Exercise],
        progress: LearnerProgress,
    ) -> Optional[Exercise]:
        """Select an exercise for review based on spaced repetition"""
        # For now, simple random selection from completed
        completed_ids = set(progress.completed_exercises)
        review_candidates = [e for e in exercises if e.id in completed_ids]

        if review_candidates:
            return random.choice(review_candidates)
        return None

    async def generate_exercise_from_course(
        self,
        course_content: str,
        lecture_content: str,
        category: ExerciseCategory,
        difficulty: DifficultyLevel,
    ) -> Optional[Exercise]:
        """
        Generate a new exercise based on course content using LLM.
        """
        prompt = f"""
Génère un exercice pratique basé sur ce contenu de cours.

Contenu du cours:
{course_content[:2000]}

Contenu de la leçon:
{lecture_content[:1500]}

Catégorie: {category.value}
Difficulté: {difficulty.value}

Génère un exercice au format JSON avec:
{{
    "title": "Titre court et descriptif",
    "description": "Description brève",
    "instructions": "Instructions détaillées en markdown",
    "starter_code": "Code de départ si applicable",
    "hints": ["Indice 1", "Indice 2", "Indice 3"],
    "solution": "Solution complète",
    "solution_explanation": "Explication de la solution",
    "validation_checks": [
        {{"name": "check1", "description": "Ce que ça vérifie"}}
    ],
    "estimated_minutes": 15,
    "points": 100
}}

L'exercice doit être pratique et directement lié au contenu.
Réponds uniquement avec le JSON.
"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="Tu génères des exercices pratiques DevOps."),
                HumanMessage(content=prompt)
            ])

            # Parse JSON response
            import json
            exercise_data = json.loads(response.content)

            return Exercise(
                title=exercise_data["title"],
                description=exercise_data.get("description", ""),
                instructions=exercise_data["instructions"],
                category=category,
                difficulty=difficulty,
                starter_code=exercise_data.get("starter_code"),
                hints=exercise_data.get("hints", []),
                solution=exercise_data.get("solution"),
                solution_explanation=exercise_data.get("solution_explanation"),
                estimated_minutes=exercise_data.get("estimated_minutes", 15),
                points=exercise_data.get("points", 100),
            )
        except Exception as e:
            print(f"[EXERCISE_SELECTOR] Failed to generate exercise: {e}")
            return None

    async def get_exercises_for_concept(
        self,
        concept: str,
        available_exercises: List[Exercise],
        count: int = 3,
    ) -> List[Exercise]:
        """Find exercises that teach a specific concept"""
        # Use LLM to match exercises to concept
        exercise_summaries = [
            f"- {e.id}: {e.title} ({e.category.value}, {e.difficulty.value})"
            for e in available_exercises[:50]  # Limit for context
        ]

        prompt = f"""
Parmi ces exercices, lesquels sont les plus pertinents pour apprendre le concept: "{concept}"?

Exercices disponibles:
{chr(10).join(exercise_summaries)}

Retourne les IDs des {count} exercices les plus pertinents, séparés par des virgules.
Réponds uniquement avec les IDs.
"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            selected_ids = [id.strip() for id in response.content.split(",")]

            id_to_exercise = {e.id: e for e in available_exercises}
            return [id_to_exercise[id] for id in selected_ids if id in id_to_exercise]
        except Exception:
            # Fallback to random selection
            return random.sample(available_exercises, min(count, len(available_exercises)))
