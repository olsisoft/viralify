"""
Progress Service

Tracks and manages learner progress across sessions and exercises.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from models.practice_models import (
    LearnerProgress,
    ExerciseCategory,
    DifficultyLevel,
    PracticeSession,
)
from models.assessment_models import (
    ProgressAssessment,
    UnderstandingLevel,
)


class ProgressRepository:
    """In-memory progress storage (replace with DB in production)"""

    def __init__(self):
        self._progress: Dict[str, LearnerProgress] = {}

    async def get(self, user_id: str) -> Optional[LearnerProgress]:
        return self._progress.get(user_id)

    async def save(self, progress: LearnerProgress) -> LearnerProgress:
        self._progress[progress.user_id] = progress
        return progress

    async def get_or_create(self, user_id: str) -> LearnerProgress:
        if user_id not in self._progress:
            self._progress[user_id] = LearnerProgress(user_id=user_id)
        return self._progress[user_id]


class ProgressService:
    """
    Manages learner progress tracking.

    Features:
    - Progress persistence
    - Statistics calculation
    - Streak tracking
    - Badge/achievement management
    """

    # Badge definitions
    BADGES = {
        "first_exercise": {
            "name": "Premier pas",
            "description": "Compléter votre premier exercice",
            "condition": lambda p: p.total_exercises_completed >= 1,
        },
        "docker_apprentice": {
            "name": "Apprenti Docker",
            "description": "Compléter 5 exercices Docker",
            "condition": lambda p: p.category_progress.get("docker", {}).get("completed", 0) >= 5,
        },
        "k8s_explorer": {
            "name": "Explorateur K8s",
            "description": "Compléter 5 exercices Kubernetes",
            "condition": lambda p: p.category_progress.get("kubernetes", {}).get("completed", 0) >= 5,
        },
        "week_streak": {
            "name": "Semaine de feu",
            "description": "7 jours consécutifs de pratique",
            "condition": lambda p: p.current_streak >= 7,
        },
        "hundred_club": {
            "name": "Club des 100",
            "description": "Atteindre 100 points",
            "condition": lambda p: p.total_points >= 100,
        },
        "perfectionist": {
            "name": "Perfectionniste",
            "description": "Compléter 10 exercices sans utiliser d'indice",
            "condition": lambda p: p.average_hints_used == 0 and p.total_exercises_completed >= 10,
        },
        "intermediate_level": {
            "name": "Niveau intermédiaire",
            "description": "Compléter 5 exercices de niveau intermédiaire",
            "condition": lambda p: p.difficulty_stats.get("intermediate", 0) >= 5,
        },
        "advanced_level": {
            "name": "Niveau avancé",
            "description": "Compléter 3 exercices de niveau avancé",
            "condition": lambda p: p.difficulty_stats.get("advanced", 0) >= 3,
        },
    }

    def __init__(self):
        self.repository = ProgressRepository()

    async def get_progress(self, user_id: str) -> LearnerProgress:
        """Get learner progress, creating if needed"""
        return await self.repository.get_or_create(user_id)

    async def record_completion(
        self,
        user_id: str,
        exercise_id: str,
        category: str,
        difficulty: str,
        points: int,
        hints_used: int,
    ) -> LearnerProgress:
        """Record an exercise completion"""
        progress = await self.repository.get_or_create(user_id)

        # Update basic stats
        if exercise_id not in progress.completed_exercises:
            progress.completed_exercises.append(exercise_id)
            progress.total_exercises_completed += 1

        progress.total_points += points

        # Update category progress
        if category not in progress.category_progress:
            progress.category_progress[category] = {"completed": 0, "total": 0, "points": 0}
        progress.category_progress[category]["completed"] += 1
        progress.category_progress[category]["points"] += points

        # Update difficulty stats
        if difficulty in progress.difficulty_stats:
            progress.difficulty_stats[difficulty] += 1

        # Update averages
        total = progress.total_exercises_completed
        progress.average_hints_used = (
            (progress.average_hints_used * (total - 1) + hints_used) / total
        )

        # Update streak
        await self._update_streak(progress)

        # Check for new badges
        await self._check_badges(progress)

        # Update timestamps
        if not progress.first_exercise_at:
            progress.first_exercise_at = datetime.utcnow()
        progress.last_exercise_at = datetime.utcnow()
        progress.updated_at = datetime.utcnow()

        await self.repository.save(progress)
        return progress

    async def _update_streak(self, progress: LearnerProgress):
        """Update the practice streak"""
        now = datetime.utcnow()
        last = progress.last_exercise_at

        if last:
            days_since = (now.date() - last.date()).days
            if days_since == 0:
                # Same day, no change
                pass
            elif days_since == 1:
                # Consecutive day
                progress.current_streak += 1
                progress.longest_streak = max(progress.longest_streak, progress.current_streak)
            else:
                # Streak broken
                progress.current_streak = 1
        else:
            # First exercise
            progress.current_streak = 1

    async def _check_badges(self, progress: LearnerProgress):
        """Check and award new badges"""
        for badge_id, badge_def in self.BADGES.items():
            if badge_id not in progress.badges:
                if badge_def["condition"](progress):
                    progress.badges.append(badge_id)
                    print(f"[PROGRESS] Badge awarded: {badge_def['name']} to {progress.user_id}")

    async def get_progress_assessment(self, user_id: str) -> ProgressAssessment:
        """Generate a comprehensive progress assessment"""
        progress = await self.repository.get_or_create(user_id)

        # Calculate skill levels by category
        skill_levels = {}
        for category, stats in progress.category_progress.items():
            completed = stats.get("completed", 0)
            if completed >= 10:
                skill_levels[category] = UnderstandingLevel.PROFICIENT
            elif completed >= 5:
                skill_levels[category] = UnderstandingLevel.FUNCTIONAL
            elif completed >= 2:
                skill_levels[category] = UnderstandingLevel.DEVELOPING
            elif completed >= 1:
                skill_levels[category] = UnderstandingLevel.SURFACE
            else:
                skill_levels[category] = UnderstandingLevel.NONE

        # Identify areas
        improving_areas = []
        struggling_areas = []
        mastered_areas = []

        for category, level in skill_levels.items():
            if level in [UnderstandingLevel.PROFICIENT, UnderstandingLevel.EXPERT]:
                mastered_areas.append(category)
            elif level == UnderstandingLevel.FUNCTIONAL:
                improving_areas.append(category)
            elif level in [UnderstandingLevel.NONE, UnderstandingLevel.SURFACE]:
                struggling_areas.append(category)

        # Calculate engagement score
        engagement_score = min(100, (
            progress.current_streak * 5 +
            progress.total_exercises_completed * 2 +
            len(progress.badges) * 10
        ))

        # Persistence score based on average attempts
        persistence_score = min(100, max(0, 100 - (progress.average_hints_used * 10)))

        return ProgressAssessment(
            user_id=user_id,
            skill_levels=skill_levels,
            improving_areas=improving_areas,
            struggling_areas=struggling_areas,
            mastered_areas=mastered_areas,
            engagement_score=engagement_score,
            persistence_score=persistence_score,
            ready_for_next_level=len(mastered_areas) >= 2,
        )

    async def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top learners by points"""
        all_progress = list(self.repository._progress.values())
        sorted_progress = sorted(all_progress, key=lambda p: p.total_points, reverse=True)

        return [
            {
                "rank": i + 1,
                "user_id": p.user_id,
                "total_points": p.total_points,
                "exercises_completed": p.total_exercises_completed,
                "badges": len(p.badges),
                "streak": p.current_streak,
            }
            for i, p in enumerate(sorted_progress[:limit])
        ]

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a user"""
        progress = await self.repository.get_or_create(user_id)

        return {
            "user_id": user_id,
            "total_points": progress.total_points,
            "total_exercises": progress.total_exercises_completed,
            "current_streak": progress.current_streak,
            "longest_streak": progress.longest_streak,
            "badges_earned": progress.badges,
            "badges_count": len(progress.badges),
            "difficulty_distribution": progress.difficulty_stats,
            "category_progress": progress.category_progress,
            "average_hints_used": round(progress.average_hints_used, 2),
            "first_exercise_at": progress.first_exercise_at.isoformat() if progress.first_exercise_at else None,
            "last_exercise_at": progress.last_exercise_at.isoformat() if progress.last_exercise_at else None,
            "days_active": self._calculate_days_active(progress),
        }

    def _calculate_days_active(self, progress: LearnerProgress) -> int:
        """Calculate total days the user has been active"""
        if not progress.first_exercise_at:
            return 0
        return (datetime.utcnow() - progress.first_exercise_at).days + 1

    async def get_badge_info(self, badge_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a badge"""
        badge_def = self.BADGES.get(badge_id)
        if not badge_def:
            return None

        return {
            "id": badge_id,
            "name": badge_def["name"],
            "description": badge_def["description"],
        }

    async def get_all_badges(self) -> List[Dict[str, Any]]:
        """Get all available badges"""
        return [
            {
                "id": badge_id,
                "name": badge_def["name"],
                "description": badge_def["description"],
            }
            for badge_id, badge_def in self.BADGES.items()
        ]
