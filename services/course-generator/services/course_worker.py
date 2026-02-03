"""
Course Worker Service

Background worker that processes course generation jobs from the RabbitMQ queue.
Runs independently and can be scaled horizontally.
"""
import asyncio
import os
import signal
from datetime import datetime
from typing import Optional

from services.course_queue import CourseQueueService, QueuedCourseJob, get_queue_service
from services.course_compositor import CourseCompositor
from services.course_planner import CoursePlanner
from services.retrieval_service import RAGService
from models.course_models import (
    CourseJob,
    CourseStage,
    GenerateCourseRequest,
    PreviewOutlineRequest,
    CourseStructureConfig,
    CourseContext,
    DifficultyLevel,
    ProfileCategory,
    QuizConfigRequest,
    QuizFrequencyConfig,
)


# Status enum for Redis storage (simplified)
class CourseJobStatus:
    QUEUED = "queued"
    GENERATING_OUTLINE = "generating_outline"
    GENERATING_LECTURES = "generating_lectures"
    CREATING_PACKAGE = "creating_package"
    COMPLETED = "completed"
    FAILED = "failed"


class CourseWorker:
    """
    Worker that processes course generation jobs from the queue.

    Features:
    - Processes one job at a time by default
    - Updates job status in Redis for client polling
    - Handles graceful shutdown
    - Automatic retry on failure
    """

    def __init__(
        self,
        queue_service: CourseQueueService = None,
        max_concurrent_jobs: int = 1
    ):
        self.queue_service = queue_service or get_queue_service()
        self.max_concurrent_jobs = max_concurrent_jobs

        # Initialize services with environment variables
        # Only pass openai_api_key if NOT using groq provider (to let CoursePlanner use shared LLM)
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        openai_api_key = None if llm_provider == "groq" else os.getenv("OPENAI_API_KEY")
        presentation_generator_url = os.getenv("PRESENTATION_GENERATOR_URL", "http://presentation-generator:8006")
        media_generator_url = os.getenv("MEDIA_GENERATOR_URL", "http://media-generator:8004")

        self.compositor = CourseCompositor(
            presentation_generator_url=presentation_generator_url,
            media_generator_url=media_generator_url,
            max_parallel_lectures=3
        )
        self.planner = CoursePlanner(openai_api_key=openai_api_key)
        self._running = False
        self._current_jobs: dict = {}

        # Redis for job status
        self._redis = None

    async def _get_redis(self):
        """Get Redis connection for job status updates"""
        if self._redis is None:
            import redis.asyncio as redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/7")
            self._redis = redis.from_url(redis_url)
        return self._redis

    async def _update_job_status(
        self,
        job_id: str,
        status: CourseJobStatus,
        progress: float = 0,
        error: str = None,
        output_urls: dict = None,
        outline: dict = None
    ):
        """Update job status in Redis (FIX: ERR-001 - improved error handling)"""
        import json
        try:
            redis = await self._get_redis()

            job_data = {
                "job_id": job_id,
                "status": status.value if hasattr(status, 'value') else status,
                "progress": progress,
                "updated_at": datetime.utcnow().isoformat()
            }

            if error:
                job_data["error"] = error
            if output_urls:
                job_data["output_urls"] = json.dumps(output_urls)
            if outline:
                outline_json = json.dumps(outline)
                job_data["outline"] = outline_json
                print(f"[WORKER] Redis update: status={job_data['status']}, outline={len(outline_json)} bytes", flush=True)

            await redis.hset(f"course_job:{job_id}", mapping={
                k: str(v) if not isinstance(v, str) else v
                for k, v in job_data.items()
            })

            # Set expiry (48 hours - increased from 24 to handle long jobs)
            await redis.expire(f"course_job:{job_id}", 172800)
        except Exception as redis_err:
            print(f"[WORKER] ERROR: Redis update failed for {job_id}: {redis_err}", flush=True)
            raise

    async def process_job(self, queued_job: QueuedCourseJob) -> None:
        """
        Process a single course generation job.

        This is the main job processing logic.
        """
        job_id = queued_job.job_id
        self._current_jobs[job_id] = queued_job

        try:
            print(f"[WORKER] Starting job {job_id}: {queued_job.topic}", flush=True)

            # Update status to processing
            await self._update_job_status(
                job_id,
                CourseJobStatus.GENERATING_OUTLINE,
                progress=5
            )

            # Build quiz config if provided - use QuizConfigRequest for GenerateCourseRequest
            quiz_config = QuizConfigRequest()  # Default config
            if queued_job.quiz_config:
                # Parse frequency
                frequency_str = queued_job.quiz_config.get("frequency", "per_section")
                try:
                    frequency = QuizFrequencyConfig(frequency_str)
                except ValueError:
                    frequency = QuizFrequencyConfig.PER_SECTION

                quiz_config = QuizConfigRequest(
                    enabled=queued_job.quiz_config.get("enabled", True),
                    frequency=frequency,
                    custom_frequency=queued_job.quiz_config.get("custom_frequency"),
                    questions_per_quiz=queued_job.quiz_config.get("questions_per_quiz", 5),
                    passing_score=queued_job.quiz_config.get("passing_score", 70),
                    show_explanations=queued_job.quiz_config.get("show_explanations", True),
                )

            # Get RAG context if document_ids provided
            rag_context = None
            if queued_job.document_ids:
                try:
                    rag_service = RAGService()
                    rag_context = await rag_service.get_context_for_course_generation(
                        topic=queued_job.topic,
                        description=None,
                        document_ids=queued_job.document_ids,
                        user_id=queued_job.user_id
                    )
                    print(f"[WORKER] RAG context loaded: {len(rag_context)} chars", flush=True)
                except Exception as e:
                    print(f"[WORKER] RAG context error (continuing): {e}", flush=True)

            # Parse difficulty levels
            difficulty_start = DifficultyLevel(queued_job.difficulty_start)
            difficulty_end = DifficultyLevel(queued_job.difficulty_end)

            # Parse category
            try:
                profile_category = ProfileCategory(queued_job.category)
            except ValueError:
                profile_category = ProfileCategory.EDUCATION

            # Build course structure
            structure = CourseStructureConfig(
                number_of_sections=queued_job.num_sections,
                lectures_per_section=queued_job.lectures_per_section
            )

            # Build course context
            context = CourseContext(
                category=profile_category,
                profile_niche=queued_job.domain or queued_job.topic,
                profile_audience_level=queued_job.target_audience or "intermediate",
                specific_tools=queued_job.domain
            )

            # Build the generation request
            generate_request = GenerateCourseRequest(
                profile_id=queued_job.user_id,
                topic=queued_job.topic,
                difficulty_start=difficulty_start,
                difficulty_end=difficulty_end,
                structure=structure,
                context=context,
                language=queued_job.language,
                quiz_config=quiz_config,
                document_ids=queued_job.document_ids,
                rag_context=rag_context
            )

            # Build the preview request for outline generation
            preview_request = PreviewOutlineRequest(
                profile_id=queued_job.user_id,
                topic=queued_job.topic,
                difficulty_start=difficulty_start,
                difficulty_end=difficulty_end,
                structure=structure,
                context=context,
                document_ids=queued_job.document_ids,
                rag_context=rag_context
            )

            # Create the CourseJob object
            internal_job = CourseJob(
                job_id=job_id,
                request=generate_request
            )

            # Generate outline
            print(f"[WORKER] Generating outline for {job_id}...", flush=True)
            outline = await self.planner.generate_outline(preview_request)

            internal_job.outline = outline
            internal_job.lectures_total = outline.total_lectures
            internal_job.update_progress(CourseStage.GENERATING_LECTURES, 15, "Generating lectures...")

            # Serialize outline for Redis storage (frontend polling)
            # FIX: ERR-001 - Add logging to track outline serialization
            try:
                import json
                outline_dict = outline.model_dump(mode='json')
                outline_json_size = len(json.dumps(outline_dict))
                sections_count = len(outline_dict.get("sections", []))
                print(f"[WORKER] Outline serialized: {sections_count} sections, ~{outline_json_size} bytes", flush=True)
            except Exception as ser_err:
                print(f"[WORKER] ERROR: Outline serialization failed: {ser_err}", flush=True)
                raise

            await self._update_job_status(
                job_id,
                CourseJobStatus.GENERATING_LECTURES,
                progress=15,
                outline=outline_dict
            )
            print(f"[WORKER] Outline stored in Redis for {job_id}", flush=True)

            # Generate all lectures
            print(f"[WORKER] Generating {outline.total_lectures} lectures...", flush=True)

            # Progress callback to update Redis
            async def progress_callback(completed: int, total: int, current_title: str = None):
                progress = 15 + (completed / total * 75) if total > 0 else 15
                await self._update_job_status(
                    job_id,
                    CourseJobStatus.GENERATING_LECTURES,
                    progress=progress
                )

            # Use compositor to generate lectures
            await self.compositor.generate_all_lectures(
                job=internal_job,
                request=generate_request,
                progress_callback=lambda c, t, title: asyncio.create_task(progress_callback(c, t, title))
            )

            # Collect output URLs from generated lectures
            for section in internal_job.outline.sections:
                for lecture in section.lectures:
                    if lecture.video_url:
                        internal_job.output_urls.append(lecture.video_url)

            await self._update_job_status(
                job_id,
                CourseJobStatus.CREATING_PACKAGE,
                progress=90
            )

            # Create ZIP package
            print(f"[WORKER] Creating course package for {job_id}...", flush=True)
            zip_path = await self.compositor.create_course_zip(internal_job)

            internal_job.zip_url = zip_path
            internal_job.update_progress(CourseStage.COMPLETED, 100, "Course generation complete!")

            await self._update_job_status(
                job_id,
                CourseJobStatus.COMPLETED,
                progress=100,
                output_urls={"videos": internal_job.output_urls, "zip": zip_path}
            )

            print(f"[WORKER] Job {job_id} completed successfully!", flush=True)

        except Exception as e:
            print(f"[WORKER] Job {job_id} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

            await self._update_job_status(
                job_id,
                CourseJobStatus.FAILED,
                error=str(e)
            )

            raise  # Re-raise to trigger DLQ

        finally:
            if job_id in self._current_jobs:
                del self._current_jobs[job_id]

    async def start(self) -> None:
        """Start the worker"""
        self._running = True
        print(f"[WORKER] Starting course worker...", flush=True)
        print(f"[WORKER] Max concurrent jobs: {self.max_concurrent_jobs}", flush=True)

        # Connect to queue
        await self.queue_service.connect()

        # Start consuming
        await self.queue_service.consume(
            callback=self.process_job,
            max_concurrent=self.max_concurrent_jobs
        )

    def stop(self) -> None:
        """Stop the worker gracefully"""
        print(f"[WORKER] Stopping worker...", flush=True)
        self._running = False
        self.queue_service.stop_consuming()

    def get_current_jobs(self) -> dict:
        """Get currently processing jobs"""
        return self._current_jobs.copy()


# Global worker instance
_worker: Optional[CourseWorker] = None


def get_worker() -> CourseWorker:
    """Get the singleton worker instance"""
    global _worker
    if _worker is None:
        _worker = CourseWorker()
    return _worker


async def run_worker():
    """
    Entry point for running the worker as a standalone process.
    """
    worker = get_worker()

    # Handle shutdown signals (Unix only, skip on Windows)
    import sys
    if sys.platform != 'win32':
        loop = asyncio.get_event_loop()

        def shutdown_handler():
            print(f"[WORKER] Received shutdown signal", flush=True)
            worker.stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown_handler)

    try:
        await worker.start()
    except KeyboardInterrupt:
        print(f"[WORKER] Interrupted by user", flush=True)
        worker.stop()
    finally:
        await worker.queue_service.disconnect()


if __name__ == "__main__":
    # Run as standalone worker
    asyncio.run(run_worker())
