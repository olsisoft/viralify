"""
Practice Agent - FastAPI Application

Provides REST API and WebSocket endpoints for the practice agent.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Models
from models.practice_models import (
    PracticeSession,
    Exercise,
    LearnerProgress,
    CreateSessionRequest,
    CreateSessionResponse,
    SubmitCodeRequest,
    SubmitCodeResponse,
    ChatRequest,
    ChatResponse,
    HintRequest,
    HintResponse,
    DifficultyLevel,
    ExerciseCategory,
)
from models.assessment_models import AssessmentResult, ProgressAssessment

# Services
from services.session_service import SessionService
from services.exercise_service import ExerciseService
from services.sandbox_manager import SandboxManager, get_sandbox_manager
from services.assessment_service import AssessmentService
from services.progress_service import ProgressService

# Agents
from agents.practice_graph import create_practice_agent


# Startup/Shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    print("[STARTUP] Practice Agent Service starting...", flush=True)

    # Initialize services
    sandbox_manager = get_sandbox_manager()
    await sandbox_manager.start()

    # Initialize exercise service to load exercises
    exercise_service = ExerciseService()
    stats = exercise_service.get_exercise_stats()
    print(f"[STARTUP] Loaded {stats['total']} exercises", flush=True)

    print("[STARTUP] Practice Agent Service ready!", flush=True)

    yield

    # Shutdown
    print("[SHUTDOWN] Practice Agent Service stopping...", flush=True)
    await sandbox_manager.stop()
    print("[SHUTDOWN] Practice Agent Service stopped", flush=True)


# Create FastAPI app
app = FastAPI(
    title="Practice Agent API",
    description="API for the Practice Agent - Interactive DevOps training",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service instances
session_service = SessionService()
exercise_service = ExerciseService()
assessment_service = AssessmentService()
progress_service = ProgressService()


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "practice-agent",
    }


# ==================== SESSION ENDPOINTS ====================

@app.post("/api/v1/practice/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new practice session"""
    try:
        response = await session_service.create_session(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/practice/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/api/v1/practice/sessions")
async def list_user_sessions(
    user_id: str = Query(..., description="User ID"),
    active_only: bool = Query(True, description="Only active sessions"),
):
    """List sessions for a user"""
    sessions = await session_service.get_user_sessions(user_id, active_only)
    return {"sessions": sessions, "total": len(sessions)}


@app.delete("/api/v1/practice/sessions/{session_id}")
async def end_session(session_id: str):
    """End a practice session"""
    session = await session_service.end_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ended", "summary": await session_service.get_session_summary(session_id)}


@app.get("/api/v1/practice/sessions/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get session summary"""
    summary = await session_service.get_session_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary


# ==================== EXERCISE ENDPOINTS ====================

@app.get("/api/v1/practice/exercises")
async def list_exercises(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    course_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List available exercises with filters"""
    cat_enum = ExerciseCategory(category) if category else None
    diff_enum = DifficultyLevel(difficulty) if difficulty else None

    exercises = exercise_service.list_exercises(
        category=cat_enum,
        difficulty=diff_enum,
        course_id=course_id,
        limit=limit,
        offset=offset,
    )
    return {"exercises": exercises, "total": len(exercises)}


@app.get("/api/v1/practice/exercises/{exercise_id}")
async def get_exercise(exercise_id: str):
    """Get exercise details"""
    exercise = exercise_service.get_exercise(exercise_id)
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")
    return exercise


@app.get("/api/v1/practice/exercises/search")
async def search_exercises(q: str = Query(..., min_length=2)):
    """Search exercises by text"""
    exercises = exercise_service.search_exercises(q)
    return {"exercises": exercises, "total": len(exercises)}


@app.get("/api/v1/practice/exercises/stats")
async def get_exercise_stats():
    """Get exercise library statistics"""
    return exercise_service.get_exercise_stats()


# ==================== DYNAMIC EXERCISE GENERATION ENDPOINTS ====================

class GenerateExercisesRequest(BaseModel):
    """Request to generate exercises for a course"""
    exercises_per_section: int = 3
    include_coding: bool = True
    include_debugging: bool = True
    include_quiz: bool = True
    include_architecture: bool = False
    difficulty_progression: bool = True
    force_regenerate: bool = False


class GenerateExercisesResponse(BaseModel):
    """Response with generated exercises"""
    course_id: str
    course_title: str
    exercises_count: int
    exercises: List[Dict[str, Any]]
    message: str


class RegenerateExerciseRequest(BaseModel):
    """Request to regenerate a specific exercise"""
    feedback: Optional[str] = None


@app.post("/api/v1/practice/exercises/generate/{course_id}", response_model=GenerateExercisesResponse)
async def generate_exercises_for_course(
    course_id: str,
    request: Optional[GenerateExercisesRequest] = None,
):
    """
    Generate practice exercises for a course using AI.

    This endpoint:
    1. Fetches course content from course-generator service
    2. Retrieves RAG context from any uploaded source documents
    3. Uses GPT-4 to generate tailored, practical exercises
    4. Creates validation checks and sandbox configurations

    The exercises are cached and will be returned on subsequent calls
    unless force_regenerate is True.
    """
    from services.exercise_generator import ExerciseGenerationConfig

    try:
        config = None
        force_regenerate = False

        if request:
            config = ExerciseGenerationConfig(
                exercises_per_section=request.exercises_per_section,
                include_coding=request.include_coding,
                include_debugging=request.include_debugging,
                include_quiz=request.include_quiz,
                include_architecture=request.include_architecture,
                difficulty_progression=request.difficulty_progression,
            )
            force_regenerate = request.force_regenerate

        exercises = await exercise_service.generate_exercises_for_course(
            course_id=course_id,
            config=config,
            force_regenerate=force_regenerate,
        )

        # Get course title from first exercise or generator cache
        from services.exercise_generator import get_exercise_generator
        generator = get_exercise_generator()
        exercise_set = generator._exercise_cache.get(course_id)
        course_title = exercise_set.course_title if exercise_set else "Unknown Course"

        return GenerateExercisesResponse(
            course_id=course_id,
            course_title=course_title,
            exercises_count=len(exercises),
            exercises=[e.model_dump() for e in exercises],
            message=f"Successfully generated {len(exercises)} exercises for the course",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[API] Error generating exercises: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate exercises: {str(e)}")


@app.get("/api/v1/practice/exercises/course/{course_id}")
async def get_course_exercises(
    course_id: str,
    difficulty: Optional[str] = None,
    exercise_type: Optional[str] = None,
    auto_generate: bool = Query(True, description="Auto-generate if no exercises exist"),
):
    """
    Get exercises for a specific course.

    If auto_generate is True (default) and no exercises exist for the course,
    they will be automatically generated using AI.
    """
    diff_enum = DifficultyLevel(difficulty) if difficulty else None
    type_enum = None
    if exercise_type:
        try:
            from models.practice_models import ExerciseType
            type_enum = ExerciseType(exercise_type)
        except ValueError:
            pass

    exercises = await exercise_service.get_course_exercises(
        course_id=course_id,
        difficulty=diff_enum,
        exercise_type=type_enum,
        auto_generate=auto_generate,
    )

    return {
        "course_id": course_id,
        "exercises": [e.model_dump() for e in exercises],
        "total": len(exercises),
        "auto_generated": auto_generate and len(exercises) > 0,
    }


@app.post("/api/v1/practice/exercises/{exercise_id}/regenerate")
async def regenerate_exercise(
    exercise_id: str,
    request: RegenerateExerciseRequest,
):
    """
    Regenerate a specific exercise based on feedback.

    Use this when an exercise needs improvement, such as:
    - Unclear instructions
    - Validation checks not working properly
    - Difficulty level mismatch
    """
    # Find the exercise to get its course_id
    exercise = exercise_service.get_exercise(exercise_id)
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")

    if not exercise.course_id:
        raise HTTPException(status_code=400, detail="Cannot regenerate static exercises")

    new_exercise = await exercise_service.regenerate_exercise(
        course_id=exercise.course_id,
        exercise_id=exercise_id,
        feedback=request.feedback,
    )

    if not new_exercise:
        raise HTTPException(status_code=500, detail="Failed to regenerate exercise")

    return {
        "message": "Exercise regenerated successfully",
        "exercise": new_exercise.model_dump(),
    }


@app.delete("/api/v1/practice/exercises/course/{course_id}")
async def clear_course_exercises(course_id: str):
    """
    Clear all exercises for a course.

    This forces regeneration on the next request.
    Useful when course content has been updated.
    """
    exercise_service.clear_course_exercises(course_id)
    return {
        "message": f"Cleared all exercises for course {course_id}",
        "course_id": course_id,
    }


# ==================== PRACTICE INTERACTION ENDPOINTS ====================

class InteractionRequest(BaseModel):
    """Request for agent interaction"""
    session_id: str
    message: Optional[str] = None
    code: Optional[str] = None


class InteractionResponse(BaseModel):
    """Response from agent interaction"""
    response: str
    exercise: Optional[Dict[str, Any]] = None
    assessment: Optional[Dict[str, Any]] = None
    next_action: Optional[str] = None


@app.post("/api/v1/practice/interact", response_model=InteractionResponse)
async def interact_with_agent(request: InteractionRequest):
    """
    Main interaction endpoint with the practice agent.

    Send messages or code submissions and get responses.
    """
    session = await session_service.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Create agent and invoke
    agent = create_practice_agent()

    result = await agent.invoke(
        session_id=request.session_id,
        user_id=session.user_id,
        message=request.message,
        code=request.code,
        course_id=session.course_id,
        difficulty=session.difficulty_preference.value,
        categories=[c.value for c in session.categories_focus] if session.categories_focus else [],
        exercises_completed=session.exercises_completed,
        hints_used=session.hints_used_total,
        total_points=session.points_earned,
        current_exercise=session.current_exercise.model_dump() if session.current_exercise else None,
        pair_programming_mode=session.pair_programming_enabled,
    )

    # Extract response
    response_text = result.get("feedback", "")

    # Update session state if needed
    if result.get("current_exercise"):
        await session_service.update_session(
            request.session_id,
            {"current_exercise": Exercise(**result["current_exercise"])}
        )

    if result.get("exercises_completed"):
        await session_service.update_session(
            request.session_id,
            {
                "exercises_completed": result["exercises_completed"],
                "points_earned": result.get("total_points", session.points_earned),
            }
        )

    # Add message to history
    if request.message:
        await session_service.add_message(request.session_id, "user", request.message)
    await session_service.add_message(request.session_id, "assistant", response_text)

    return InteractionResponse(
        response=response_text,
        exercise=result.get("current_exercise"),
        assessment=result.get("assessment"),
        next_action=result.get("next_action"),
    )


@app.post("/api/v1/practice/sessions/{session_id}/submit")
async def submit_code(session_id: str, request: SubmitCodeRequest):
    """Submit code for evaluation"""
    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.current_exercise:
        raise HTTPException(status_code=400, detail="No active exercise")

    # Execute and evaluate
    sandbox_manager = get_sandbox_manager()
    exercise_dict = session.current_exercise.model_dump()

    sandbox_result = await sandbox_manager.execute(
        sandbox_type=exercise_dict.get("sandbox_type", "docker"),
        code=request.code,
        exercise_config=exercise_dict.get("sandbox_config", {}),
        timeout=exercise_dict.get("timeout_seconds", 60),
        user_id=session.user_id,
    )

    # Assess
    assessment = await assessment_service.evaluate(
        exercise=exercise_dict,
        submitted_code=request.code,
        execution_result=sandbox_result.model_dump() if sandbox_result else None,
    )

    # Update progress if passed
    next_exercise = None
    if assessment.passed:
        await progress_service.record_completion(
            user_id=session.user_id,
            exercise_id=session.current_exercise.id,
            category=session.current_exercise.category.value,
            difficulty=session.current_exercise.difficulty.value,
            points=assessment.score,
            hints_used=session.hints_used_total,
        )

        await session_service.complete_exercise(
            session_id,
            session.current_exercise.id,
            assessment.score,
        )

        # Get next exercise
        next_exercise = await exercise_service.select_next_exercise(
            completed_exercises=session.exercises_completed + [session.current_exercise.id],
            difficulty=session.difficulty_preference.value,
            categories=[c.value for c in session.categories_focus] if session.categories_focus else None,
            course_id=session.course_id,
        )

    return SubmitCodeResponse(
        passed=assessment.passed,
        score=assessment.score,
        feedback=assessment.summary_feedback,
        checks_passed=assessment.checks_passed,
        checks_failed=assessment.checks_failed,
        execution_output=sandbox_result.execution.combined_output if sandbox_result else None,
        next_exercise=next_exercise,
    )


@app.post("/api/v1/practice/sessions/{session_id}/hint")
async def get_hint(session_id: str, request: HintRequest):
    """Get a hint for the current exercise"""
    session = await session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.current_exercise:
        raise HTTPException(status_code=400, detail="No active exercise")

    exercise = session.current_exercise
    hints = exercise.hints
    hints_used = session.hints_used_total

    # Get hint at requested level
    hint_index = min(request.hint_level - 1, len(hints) - 1)
    hint_text = hints[hint_index] if hints and hint_index >= 0 else "Pas d'indice disponible."

    # Record hint usage
    await session_service.use_hint(session_id)

    return HintResponse(
        hint=hint_text,
        hint_number=hint_index + 1,
        hints_remaining=max(0, len(hints) - hint_index - 1),
        points_deduction=10,  # Standard deduction per hint
    )


# ==================== PROGRESS ENDPOINTS ====================

@app.get("/api/v1/practice/progress/{user_id}")
async def get_user_progress(user_id: str):
    """Get user's overall progress"""
    progress = await progress_service.get_progress(user_id)
    return progress


@app.get("/api/v1/practice/progress/{user_id}/stats")
async def get_user_stats(user_id: str):
    """Get detailed user statistics"""
    stats = await progress_service.get_user_stats(user_id)
    return stats


@app.get("/api/v1/practice/progress/{user_id}/assessment")
async def get_progress_assessment(user_id: str):
    """Get comprehensive progress assessment"""
    assessment = await progress_service.get_progress_assessment(user_id)
    return assessment


@app.get("/api/v1/practice/leaderboard")
async def get_leaderboard(limit: int = Query(10, ge=1, le=100)):
    """Get top learners leaderboard"""
    leaderboard = await progress_service.get_leaderboard(limit)
    return {"leaderboard": leaderboard}


@app.get("/api/v1/practice/badges")
async def get_all_badges():
    """Get all available badges"""
    badges = await progress_service.get_all_badges()
    return {"badges": badges}


# ==================== WEBSOCKET FOR REAL-TIME PRACTICE ====================

class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


connection_manager = ConnectionManager()


@app.websocket("/ws/practice/{session_id}")
async def practice_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time practice interaction.

    Message types:
    - message: Chat message
    - code: Code submission
    - hint: Request hint
    - pair_start: Start pair programming
    - pair_stop: Stop pair programming
    """
    await connection_manager.connect(session_id, websocket)

    try:
        session = await session_service.get_session(session_id)
        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return

        agent = create_practice_agent()

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")
            content = data.get("content", "")
            code = data.get("code")

            # Process based on type
            if msg_type == "message":
                result = await agent.invoke(
                    session_id=session_id,
                    user_id=session.user_id,
                    message=content,
                    current_exercise=session.current_exercise.model_dump() if session.current_exercise else None,
                )
                await websocket.send_json({
                    "type": "response",
                    "content": result.get("feedback", ""),
                    "exercise": result.get("current_exercise"),
                })

            elif msg_type == "code":
                result = await agent.invoke(
                    session_id=session_id,
                    user_id=session.user_id,
                    code=code,
                    current_exercise=session.current_exercise.model_dump() if session.current_exercise else None,
                )
                await websocket.send_json({
                    "type": "evaluation",
                    "content": result.get("feedback", ""),
                    "assessment": result.get("assessment"),
                    "passed": result.get("assessment", {}).get("passed", False),
                })

            elif msg_type == "hint":
                hints = session.current_exercise.hints if session.current_exercise else []
                hint_index = min(session.hints_used_total, len(hints) - 1)
                hint = hints[hint_index] if hints and hint_index >= 0 else "Pas d'indice."
                await session_service.use_hint(session_id)
                await websocket.send_json({
                    "type": "hint",
                    "content": hint,
                    "hint_number": hint_index + 1,
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)
    except Exception as e:
        print(f"[WS] Error: {e}", flush=True)
        connection_manager.disconnect(session_id)


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
