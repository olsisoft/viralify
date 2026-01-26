"""
MAESTRO Adapter Service

Integrates MAESTRO Engine with the course-generator service.
Provides dual-mode generation: RAG mode (existing) and MAESTRO mode.
"""

import os
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class GenerationMode(str, Enum):
    """Course generation modes"""
    RAG = "rag"           # Use documents with RAG (existing system)
    MAESTRO = "maestro"   # Use MAESTRO 5-layer pipeline (no documents)
    HYBRID = "hybrid"     # Future: Combine RAG context with MAESTRO structure


@dataclass
class MaestroJobResponse:
    """Response from MAESTRO engine job creation"""
    job_id: str
    status: str
    progress: float
    stage: str
    message: str


@dataclass
class MaestroCourseResult:
    """Result from MAESTRO course generation"""
    course_id: str
    title: str
    description: str
    modules: List[Dict[str, Any]]
    concepts: List[Dict[str, Any]]
    total_duration_minutes: int
    generation_time_seconds: float


class MaestroAdapterService:
    """
    Adapter for MAESTRO Engine integration.

    Handles communication with the maestro-engine microservice
    and converts between Viralify and MAESTRO data formats.
    """

    def __init__(self, maestro_url: Optional[str] = None):
        self.maestro_url = maestro_url or os.getenv("MAESTRO_ENGINE_URL", "http://maestro-engine:8008")
        self.timeout = httpx.Timeout(300.0, connect=10.0)

    async def is_available(self) -> bool:
        """Check if MAESTRO engine is available"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.maestro_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def generate_course(
        self,
        subject: str,
        progression_path: str = "beginner_to_intermediate",
        total_duration_hours: float = 5.0,
        num_modules: int = 5,
        language: str = "en",
        include_quizzes: bool = True,
        include_exercises: bool = True,
    ) -> MaestroJobResponse:
        """
        Start course generation via MAESTRO engine.

        Args:
            subject: Course subject
            progression_path: Difficulty progression
            total_duration_hours: Target duration
            num_modules: Number of modules
            language: Content language
            include_quizzes: Include quiz questions
            include_exercises: Include practical exercises

        Returns:
            MaestroJobResponse with job_id for tracking
        """
        print(f"[MAESTRO_ADAPTER] Starting generation for '{subject}'", flush=True)

        request_data = {
            "subject": subject,
            "progression_path": progression_path,
            "total_duration_hours": total_duration_hours,
            "num_modules": num_modules,
            "language": language,
            "include_quizzes": include_quizzes,
            "include_exercises": include_exercises,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.maestro_url}/api/v1/courses/generate",
                json=request_data,
            )
            response.raise_for_status()

            data = response.json()
            return MaestroJobResponse(
                job_id=data["job_id"],
                status=data["status"],
                progress=data["progress"],
                stage=data["stage"],
                message=data["message"],
            )

    async def get_job_status(self, job_id: str) -> MaestroJobResponse:
        """
        Get the status of a MAESTRO generation job.

        Args:
            job_id: The job ID from generate_course

        Returns:
            MaestroJobResponse with current status
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.maestro_url}/api/v1/courses/jobs/{job_id}"
            )
            response.raise_for_status()

            data = response.json()
            return MaestroJobResponse(
                job_id=data["job_id"],
                status=data["status"],
                progress=data["progress"],
                stage=data["stage"],
                message=data["message"],
            )

    async def get_course(self, course_id: str) -> MaestroCourseResult:
        """
        Get a generated course from MAESTRO engine.

        Args:
            course_id: The course ID

        Returns:
            MaestroCourseResult with full course data
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.maestro_url}/api/v1/courses/{course_id}"
            )
            response.raise_for_status()

            data = response.json()
            return MaestroCourseResult(
                course_id=data["id"],
                title=data["title"],
                description=data["description"],
                modules=data["modules"],
                concepts=data["concepts"],
                total_duration_minutes=data["total_duration_minutes"],
                generation_time_seconds=data["generation_time_seconds"],
            )

    async def analyze_domain(
        self,
        subject: str,
        progression_path: str = "beginner_to_intermediate",
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Analyze a domain without generating a full course.

        Useful for previewing the domain structure.

        Args:
            subject: Subject to analyze
            progression_path: Target progression
            language: Content language

        Returns:
            Domain analysis with themes and objectives
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.maestro_url}/api/v1/domain/analyze",
                params={
                    "subject": subject,
                    "progression_path": progression_path,
                    "language": language,
                }
            )
            response.raise_for_status()
            return response.json()

    async def get_progression_paths(self) -> List[Dict[str, Any]]:
        """Get available progression paths from MAESTRO"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.maestro_url}/api/v1/config/progression-paths"
            )
            response.raise_for_status()
            return response.json()["paths"]

    async def get_skill_levels(self) -> List[Dict[str, Any]]:
        """Get available skill levels from MAESTRO"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.maestro_url}/api/v1/config/skill-levels"
            )
            response.raise_for_status()
            return response.json()["levels"]

    async def get_bloom_levels(self) -> List[Dict[str, Any]]:
        """Get Bloom's taxonomy levels from MAESTRO"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.maestro_url}/api/v1/config/bloom-levels"
            )
            response.raise_for_status()
            return response.json()["levels"]

    def convert_to_viralify_format(
        self,
        maestro_course: MaestroCourseResult,
    ) -> Dict[str, Any]:
        """
        Convert MAESTRO course format to Viralify course format.

        Args:
            maestro_course: Course from MAESTRO engine

        Returns:
            Course in Viralify format (compatible with existing frontend)
        """
        # Convert modules to Viralify sections format
        sections = []
        for module in maestro_course.modules:
            lectures = []
            for lesson in module.get("lessons", []):
                # Convert script segments to voiceover text
                voiceover_text = " ".join(
                    seg.get("content", "")
                    for seg in lesson.get("script_segments", [])
                )

                lectures.append({
                    "id": lesson.get("id", ""),
                    "title": lesson.get("title", ""),
                    "description": lesson.get("description", ""),
                    "voiceover_text": voiceover_text or lesson.get("script", ""),
                    "duration_minutes": lesson.get("estimated_duration_minutes", 10),
                    "skill_level": lesson.get("skill_level", "intermediate"),
                    "bloom_level": lesson.get("bloom_level", "understand"),
                    "quiz_questions": lesson.get("quiz_questions", []),
                    "exercises": lesson.get("exercises", []),
                    "key_takeaways": lesson.get("key_takeaways", []),
                    "script_segments": lesson.get("script_segments", []),
                })

            sections.append({
                "id": module.get("id", ""),
                "title": module.get("name", ""),
                "description": module.get("description", ""),
                "learning_objectives": module.get("learning_objectives", []),
                "lectures": lectures,
                "total_duration_minutes": module.get("total_duration_minutes", 0),
            })

        return {
            "course_id": maestro_course.course_id,
            "title": maestro_course.title,
            "description": maestro_course.description,
            "sections": sections,
            "concepts": maestro_course.concepts,
            "total_duration_minutes": maestro_course.total_duration_minutes,
            "generation_time_seconds": maestro_course.generation_time_seconds,
            "generation_mode": "maestro",
        }


# Singleton instance
_maestro_adapter: Optional[MaestroAdapterService] = None


def get_maestro_adapter() -> MaestroAdapterService:
    """Get singleton MAESTRO adapter instance"""
    global _maestro_adapter
    if _maestro_adapter is None:
        _maestro_adapter = MaestroAdapterService()
    return _maestro_adapter
