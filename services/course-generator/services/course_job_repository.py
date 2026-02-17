"""
Course Job Repository

PostgreSQL-backed persistence for course generation jobs.
Provides disk-based storage that survives container restarts.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import asyncpg

from models.course_models import (
    CourseJob,
    CourseOutline,
    CourseStage,
    GenerateCourseRequest,
)


class CourseJobRepository:
    """
    Repository for storing and retrieving course jobs from PostgreSQL.

    Provides:
    - Full disk persistence (survives restarts)
    - Efficient queries by user, status, date
    - Automatic updated_at via trigger
    """

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://tiktok_user:tiktok_secure_2024@postgres:5432/tiktok_platform"
        )
        self._pool: Optional[asyncpg.Pool] = None

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10
            )
        return self._pool

    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def ensure_table(self):
        """Create table if it doesn't exist (for development)"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS course_jobs (
                    id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255),
                    topic VARCHAR(500) NOT NULL,
                    status VARCHAR(50) DEFAULT 'queued',
                    current_stage VARCHAR(50) DEFAULT 'queued',
                    progress INTEGER DEFAULT 0,
                    message TEXT,
                    error TEXT,
                    request_json JSONB NOT NULL,
                    outline_json JSONB,
                    lectures_total INTEGER DEFAULT 0,
                    lectures_completed INTEGER DEFAULT 0,
                    lectures_failed INTEGER DEFAULT 0,
                    output_urls JSONB DEFAULT '[]',
                    zip_url TEXT,
                    is_distributed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    started_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)

    async def save(self, job: CourseJob) -> str:
        """
        Save or update a course job.

        Uses UPSERT to handle both insert and update.
        """
        pool = await self.get_pool()

        # Serialize data
        request_json = job.request.model_dump_json() if job.request else "{}"
        outline_json = job.outline.model_dump_json() if job.outline else None
        output_urls = json.dumps(job.output_urls or [])

        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO course_jobs (
                    id, user_id, topic, status, current_stage, progress, message, error,
                    request_json, outline_json, lectures_total, lectures_completed, lectures_failed,
                    output_urls, zip_url, is_distributed, created_at, started_at, completed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    current_stage = EXCLUDED.current_stage,
                    progress = EXCLUDED.progress,
                    message = EXCLUDED.message,
                    error = EXCLUDED.error,
                    outline_json = EXCLUDED.outline_json,
                    lectures_total = EXCLUDED.lectures_total,
                    lectures_completed = EXCLUDED.lectures_completed,
                    lectures_failed = EXCLUDED.lectures_failed,
                    output_urls = EXCLUDED.output_urls,
                    zip_url = EXCLUDED.zip_url,
                    is_distributed = EXCLUDED.is_distributed,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    updated_at = NOW()
            """,
                job.job_id,
                job.request.profile_id if job.request else None,
                job.request.topic if job.request else "Unknown",
                job.current_stage.value if isinstance(job.current_stage, CourseStage) else str(job.current_stage),
                job.current_stage.value if isinstance(job.current_stage, CourseStage) else str(job.current_stage),
                job.progress,
                job.message,
                job.error,
                request_json,
                outline_json,
                job.lectures_total or 0,
                job.lectures_completed or 0,
                job.lectures_failed or 0,
                output_urls,
                job.zip_url,
                getattr(job, 'is_distributed', False),
                job.created_at,
                job.started_at,
                job.completed_at
            )

        return job.job_id

    async def get_by_id(self, job_id: str) -> Optional[CourseJob]:
        """Get a course job by ID"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM course_jobs WHERE id = $1",
                job_id
            )

        if not row:
            return None

        return self._row_to_job(row)

    async def get_by_user(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        status: str = None
    ) -> List[CourseJob]:
        """Get course jobs for a user"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    """
                    SELECT * FROM course_jobs
                    WHERE user_id = $1 AND status = $2
                    ORDER BY created_at DESC
                    LIMIT $3 OFFSET $4
                    """,
                    user_id, status, limit, offset
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM course_jobs
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $3 OFFSET $4
                    """,
                    user_id, limit, offset
                )

        return [self._row_to_job(row) for row in rows]

    async def get_pending_jobs(self, limit: int = 100) -> List[CourseJob]:
        """Get jobs that are still in progress"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM course_jobs
                WHERE status IN ('queued', 'planning', 'generating_lectures', 'compiling')
                ORDER BY created_at ASC
                LIMIT $1
                """,
                limit
            )

        return [self._row_to_job(row) for row in rows]

    async def update_progress(
        self,
        job_id: str,
        stage: CourseStage,
        progress: int,
        message: str = None
    ):
        """Update job progress"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE course_jobs
                SET current_stage = $2, status = $2, progress = $3, message = $4, updated_at = NOW()
                WHERE id = $1
                """,
                job_id,
                stage.value,
                progress,
                message
            )

    async def update_lectures_progress(
        self,
        job_id: str,
        completed: int,
        failed: int,
        total: int
    ):
        """Update lecture generation progress"""
        pool = await self.get_pool()

        # Calculate overall progress (planning=10%, lectures=80%, finalizing=10%)
        if total > 0:
            lecture_progress = (completed / total) * 80
            progress = 10 + int(lecture_progress)
        else:
            progress = 10

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE course_jobs
                SET lectures_completed = $2, lectures_failed = $3, lectures_total = $4,
                    progress = $5, updated_at = NOW()
                WHERE id = $1
                """,
                job_id, completed, failed, total, progress
            )

    async def mark_completed(
        self,
        job_id: str,
        output_urls: List[str] = None,
        zip_url: str = None
    ):
        """Mark job as completed"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE course_jobs
                SET status = 'completed', current_stage = 'completed', progress = 100,
                    output_urls = $2, zip_url = $3, completed_at = NOW(), updated_at = NOW()
                WHERE id = $1
                """,
                job_id,
                json.dumps(output_urls or []),
                zip_url
            )

    async def mark_failed(self, job_id: str, error: str):
        """Mark job as failed"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE course_jobs
                SET status = 'failed', current_stage = 'failed', error = $2,
                    completed_at = NOW(), updated_at = NOW()
                WHERE id = $1
                """,
                job_id, error
            )

    async def save_outline(self, job_id: str, outline: CourseOutline):
        """Save course outline"""
        pool = await self.get_pool()

        outline_json = outline.model_dump_json()
        total_lectures = sum(len(section.lectures) for section in outline.sections)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE course_jobs
                SET outline_json = $2, lectures_total = $3, updated_at = NOW()
                WHERE id = $1
                """,
                job_id, outline_json, total_lectures
            )

    async def delete(self, job_id: str) -> bool:
        """Delete a course job"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM course_jobs WHERE id = $1",
                job_id
            )

        return "DELETE 1" in result

    async def cleanup_old_jobs(self, days: int = 30) -> int:
        """Delete jobs older than N days"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM course_jobs
                WHERE created_at < NOW() - INTERVAL '%s days'
                AND status IN ('completed', 'failed')
                """,
                days
            )

        # Extract count from "DELETE N"
        try:
            return int(result.split()[1])
        except:
            return 0

    def _row_to_job(self, row: asyncpg.Record) -> CourseJob:
        """Convert database row to CourseJob"""
        # Parse request
        request_data = row["request_json"]
        if isinstance(request_data, str):
            request_data = json.loads(request_data)

        # Create minimal request for reconstruction
        from models.course_models import CourseStructureConfig
        request = GenerateCourseRequest(
            profile_id=row["user_id"] or "unknown",
            topic=row["topic"],
            structure=CourseStructureConfig(
                number_of_sections=request_data.get("structure", {}).get("number_of_sections", 5),
                lectures_per_section=request_data.get("structure", {}).get("lectures_per_section", 3)
            )
        )

        # Create job
        job = CourseJob(request=request, job_id=row["id"])

        # Set fields
        job.current_stage = CourseStage(row["current_stage"]) if row["current_stage"] else CourseStage.QUEUED
        job.progress = row["progress"] or 0
        job.message = row["message"]
        job.error = row["error"]
        job.lectures_total = row["lectures_total"] or 0
        job.lectures_completed = row["lectures_completed"] or 0
        job.lectures_failed = row["lectures_failed"] or 0
        job.zip_url = row["zip_url"]
        job.created_at = row["created_at"]
        job.started_at = row["started_at"]
        job.completed_at = row["completed_at"]
        job.updated_at = row["updated_at"]

        # Parse output URLs
        output_urls = row["output_urls"]
        if output_urls:
            if isinstance(output_urls, str):
                job.output_urls = json.loads(output_urls)
            else:
                job.output_urls = output_urls

        # Parse outline
        outline_json = row["outline_json"]
        if outline_json:
            if isinstance(outline_json, str):
                outline_data = json.loads(outline_json)
            else:
                outline_data = outline_json
            try:
                job.outline = CourseOutline(**outline_data)
            except Exception:
                pass

        return job


# Singleton instance
_repository: Optional[CourseJobRepository] = None


async def get_course_job_repository() -> CourseJobRepository:
    """Get the singleton repository instance"""
    global _repository
    if _repository is None:
        _repository = CourseJobRepository()
        # Ensure table exists
        try:
            await _repository.ensure_table()
        except Exception as e:
            print(f"[REPO] Warning: Could not ensure table: {e}", flush=True)
    return _repository
