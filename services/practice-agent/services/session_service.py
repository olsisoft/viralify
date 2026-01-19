"""
Session Service

Manages practice sessions.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from models.practice_models import (
    PracticeSession,
    SessionStatus,
    Exercise,
    ExerciseAttempt,
    Message,
    DifficultyLevel,
    ExerciseCategory,
    CreateSessionRequest,
    CreateSessionResponse,
)


class SessionRepository:
    """In-memory session storage (replace with DB in production)"""

    def __init__(self):
        self._sessions: Dict[str, PracticeSession] = {}

    async def save(self, session: PracticeSession) -> PracticeSession:
        self._sessions[session.id] = session
        return session

    async def get(self, session_id: str) -> Optional[PracticeSession]:
        return self._sessions.get(session_id)

    async def get_by_user(self, user_id: str, active_only: bool = True) -> List[PracticeSession]:
        sessions = [s for s in self._sessions.values() if s.user_id == user_id]
        if active_only:
            sessions = [s for s in sessions if s.status == SessionStatus.ACTIVE]
        return sessions

    async def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


class SessionService:
    """
    Manages practice sessions lifecycle.

    Features:
    - Session creation and management
    - State persistence
    - Activity tracking
    - Multi-session support
    """

    def __init__(self):
        self.repository = SessionRepository()

    async def create_session(
        self,
        request: CreateSessionRequest,
    ) -> CreateSessionResponse:
        """Create a new practice session"""
        from services.exercise_service import ExerciseService

        exercise_service = ExerciseService()

        # Create session
        session = PracticeSession(
            user_id=request.user_id,
            course_id=request.course_id,
            difficulty_preference=request.difficulty_preference,
            categories_focus=request.categories_focus,
            pair_programming_enabled=request.pair_programming_enabled,
            voice_enabled=request.voice_enabled,
        )

        # Get first exercise
        first_exercise = await exercise_service.select_next_exercise(
            completed_exercises=[],
            difficulty=request.difficulty_preference.value,
            categories=[c.value for c in request.categories_focus] if request.categories_focus else None,
            course_id=request.course_id,
        )

        if first_exercise:
            session.current_exercise = first_exercise

        # Save session
        await self.repository.save(session)

        return CreateSessionResponse(
            session_id=session.id,
            status=session.status.value,
            first_exercise=first_exercise,
            message="Session créée avec succès. Prêt à commencer !",
        )

    async def get_session(self, session_id: str) -> Optional[PracticeSession]:
        """Get a session by ID"""
        return await self.repository.get(session_id)

    async def get_user_sessions(
        self,
        user_id: str,
        active_only: bool = True,
    ) -> List[PracticeSession]:
        """Get all sessions for a user"""
        return await self.repository.get_by_user(user_id, active_only)

    async def update_session(
        self,
        session_id: str,
        updates: Dict,
    ) -> Optional[PracticeSession]:
        """Update session state"""
        session = await self.repository.get(session_id)
        if not session:
            return None

        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)

        session.last_activity_at = datetime.utcnow()
        await self.repository.save(session)
        return session

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict = None,
    ) -> Optional[PracticeSession]:
        """Add a message to the session conversation"""
        session = await self.repository.get(session_id)
        if not session:
            return None

        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        session.conversation_history.append(message)
        session.last_activity_at = datetime.utcnow()

        await self.repository.save(session)
        return session

    async def set_current_exercise(
        self,
        session_id: str,
        exercise: Exercise,
    ) -> Optional[PracticeSession]:
        """Set the current exercise for a session"""
        session = await self.repository.get(session_id)
        if not session:
            return None

        session.current_exercise = exercise
        session.current_attempt = None
        session.last_activity_at = datetime.utcnow()

        await self.repository.save(session)
        return session

    async def record_attempt(
        self,
        session_id: str,
        attempt: ExerciseAttempt,
    ) -> Optional[PracticeSession]:
        """Record an exercise attempt"""
        session = await self.repository.get(session_id)
        if not session:
            return None

        session.current_attempt = attempt

        # Add to attempts history
        exercise_id = attempt.exercise_id
        if exercise_id not in session.exercises_attempted:
            session.exercises_attempted[exercise_id] = []
        session.exercises_attempted[exercise_id].append(attempt)

        session.last_activity_at = datetime.utcnow()
        await self.repository.save(session)
        return session

    async def complete_exercise(
        self,
        session_id: str,
        exercise_id: str,
        points: int,
    ) -> Optional[PracticeSession]:
        """Mark an exercise as completed"""
        session = await self.repository.get(session_id)
        if not session:
            return None

        if exercise_id not in session.exercises_completed:
            session.exercises_completed.append(exercise_id)

        session.points_earned += points
        session.current_exercise = None
        session.current_attempt = None
        session.last_activity_at = datetime.utcnow()

        await self.repository.save(session)
        return session

    async def use_hint(self, session_id: str) -> Optional[PracticeSession]:
        """Record hint usage"""
        session = await self.repository.get(session_id)
        if not session:
            return None

        session.hints_used_total += 1
        session.last_activity_at = datetime.utcnow()

        await self.repository.save(session)
        return session

    async def end_session(
        self,
        session_id: str,
        status: SessionStatus = SessionStatus.COMPLETED,
    ) -> Optional[PracticeSession]:
        """End a practice session"""
        session = await self.repository.get(session_id)
        if not session:
            return None

        session.status = status
        session.completed_at = datetime.utcnow()

        await self.repository.save(session)
        return session

    async def pause_session(self, session_id: str) -> Optional[PracticeSession]:
        """Pause a session"""
        return await self.end_session(session_id, SessionStatus.PAUSED)

    async def resume_session(self, session_id: str) -> Optional[PracticeSession]:
        """Resume a paused session"""
        session = await self.repository.get(session_id)
        if not session:
            return None

        if session.status == SessionStatus.PAUSED:
            session.status = SessionStatus.ACTIVE
            session.last_activity_at = datetime.utcnow()
            await self.repository.save(session)

        return session

    async def get_session_summary(self, session_id: str) -> Dict:
        """Get a summary of the session"""
        session = await self.repository.get(session_id)
        if not session:
            return {}

        duration = None
        if session.started_at:
            end_time = session.completed_at or datetime.utcnow()
            duration = (end_time - session.started_at).total_seconds() / 60  # minutes

        return {
            "session_id": session.id,
            "status": session.status.value,
            "exercises_completed": len(session.exercises_completed),
            "points_earned": session.points_earned,
            "hints_used": session.hints_used_total,
            "messages_exchanged": len(session.conversation_history),
            "duration_minutes": round(duration, 1) if duration else None,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
        }
