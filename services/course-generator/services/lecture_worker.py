"""
Lecture Worker Service

Phase 2 of the distributed course generation pipeline.
Processes individual lecture jobs from the lecture queue.

Flow:
1. Consume lecture job from Redis Stream
2. Generate lecture content (script, slides, video)
3. Save result to Redis
4. Update course progress
5. Trigger finalization when all lectures complete
"""
import asyncio
import os
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

from models.queue_models import (
    QueuedLectureJob,
    LectureResult,
    LectureJobStatus,
    CourseProgress,
    CourseJobStatus,
    QueuedFinalizationJob,
)
from services.lecture_queue import (
    LectureQueueService,
    CourseProgressService,
    get_lecture_queue,
    get_progress_service,
)
from services.finalization_queue import (
    FinalizationQueueService,
    get_finalization_queue,
)


# Max parallel lectures per worker
MAX_PARALLEL_LECTURES = int(os.getenv("MAX_PARALLEL_LECTURES", "3"))


class LectureWorker:
    """
    Worker that processes individual lecture generation jobs.

    Responsibilities:
    - Generate lecture content (script, slides, voiceover, video)
    - Track progress in Redis
    - Handle failures with retry
    - Trigger finalization when all lectures complete
    """

    def __init__(
        self,
        lecture_queue: LectureQueueService = None,
        progress_service: CourseProgressService = None,
        finalization_queue: FinalizationQueueService = None,
    ):
        self._lecture_queue = lecture_queue
        self._progress_service = progress_service
        self._finalization_queue = finalization_queue
        self._is_running = False
        self._semaphore = asyncio.Semaphore(MAX_PARALLEL_LECTURES)

    async def _get_lecture_queue(self) -> LectureQueueService:
        """Get or initialize lecture queue"""
        if self._lecture_queue is None:
            self._lecture_queue = await get_lecture_queue()
        return self._lecture_queue

    async def _get_progress_service(self) -> CourseProgressService:
        """Get or initialize progress service"""
        if self._progress_service is None:
            self._progress_service = await get_progress_service()
        return self._progress_service

    async def _get_finalization_queue(self) -> FinalizationQueueService:
        """Get or initialize finalization queue"""
        if self._finalization_queue is None:
            self._finalization_queue = await get_finalization_queue()
        return self._finalization_queue

    async def start(self, consumer_name: str = None) -> None:
        """Start consuming lecture jobs"""
        self._is_running = True
        lecture_queue = await self._get_lecture_queue()

        print(f"[LECTURE_WORKER] Starting worker (max parallel: {MAX_PARALLEL_LECTURES})...", flush=True)

        await lecture_queue.consume(
            handler=self.process_lecture_job,
            consumer_name=consumer_name,
            batch_size=MAX_PARALLEL_LECTURES,
        )

    def stop(self) -> None:
        """Stop the worker"""
        self._is_running = False

    async def process_lecture_job(self, job: QueuedLectureJob) -> None:
        """
        Process a single lecture job.

        Args:
            job: The lecture job to process
        """
        async with self._semaphore:
            await self._process_lecture_internal(job)

    async def _process_lecture_internal(self, job: QueuedLectureJob) -> None:
        """Internal lecture processing with full error handling"""

        progress_service = await self._get_progress_service()

        print(f"[LECTURE_WORKER] Processing lecture {job.lecture_id} ({job.lecture_title})", flush=True)

        # Create initial result
        result = LectureResult(
            lecture_id=job.lecture_id,
            status=LectureJobStatus.PROCESSING,
            started_at=datetime.utcnow().isoformat(),
        )

        try:
            # 1. Generate the lecture
            generation_result = await self._generate_lecture(job)

            # 2. Update result with success
            result.status = LectureJobStatus.COMPLETED
            result.video_url = generation_result.get("video_url")
            result.presentation_url = generation_result.get("presentation_url")
            result.presentation_job_id = generation_result.get("presentation_job_id")
            result.duration_seconds = generation_result.get("duration_seconds")
            result.slides_count = generation_result.get("slides_count")
            result.completed_at = datetime.utcnow().isoformat()

            # 3. Save result to Redis
            await progress_service.save_lecture_result(job.course_job_id, result)

            # 4. Increment completed counter
            completed = await progress_service.increment_completed_lectures(job.course_job_id)

            print(f"[LECTURE_WORKER] Completed lecture {job.lecture_id} ({completed} total)", flush=True)

            # 5. Check if all lectures are complete
            await self._check_and_trigger_finalization(job.course_job_id)

        except Exception as e:
            error_msg = str(e)
            error_tb = traceback.format_exc()

            print(f"[LECTURE_WORKER] Failed lecture {job.lecture_id}: {error_msg}", flush=True)

            # Update result with failure
            result.status = LectureJobStatus.FAILED
            result.error = error_msg
            result.error_traceback = error_tb
            result.retry_count = job.retry_count
            result.completed_at = datetime.utcnow().isoformat()

            # Save failed result
            await progress_service.save_lecture_result(job.course_job_id, result)

            # Increment failed counter
            await progress_service.increment_failed_lectures(
                job.course_job_id,
                job.lecture_id,
                error_msg
            )

            # Check if all lectures are complete (including failed)
            await self._check_and_trigger_finalization(job.course_job_id)

            # Re-raise for retry handling by queue
            raise

    async def _generate_lecture(self, job: QueuedLectureJob) -> Dict[str, Any]:
        """
        Generate a single lecture.

        This calls the presentation-generator service to create:
        - Script (voiceover text)
        - Slides (visual content)
        - Audio (TTS voiceover)
        - Video (final composition)

        Returns:
            Dict with video_url, presentation_url, etc.
        """
        import httpx

        presentation_url = os.getenv(
            "PRESENTATION_GENERATOR_URL",
            "http://presentation-generator:8006"
        )

        # Build presentation request
        request_data = {
            "topic": job.lecture_title,
            "description": job.lecture_description,
            "duration": job.duration_seconds,
            "language": job.language,
            "target_audience": job.target_audience or "general",
            "difficulty": job.difficulty,
            "style": {
                "theme": "dark",
                "typography": "modern",
            },
            # Pass RAG context if available
            "rag_context": job.rag_context,
            # Pass selected elements
            "selected_elements": job.selected_elements,
            # Course context for cross-lecture coherence
            "course_context": {
                "course_topic": job.course_topic,
                "section_title": job.section_title,
                "section_index": job.section_index,
                "lecture_index": job.lecture_index,
            },
        }

        print(f"[LECTURE_WORKER] Calling presentation-generator for {job.lecture_id}", flush=True)

        async with httpx.AsyncClient(timeout=600.0) as client:
            # Start generation job
            response = await client.post(
                f"{presentation_url}/api/v1/presentations/generate/v3",
                json=request_data
            )
            response.raise_for_status()
            job_data = response.json()

            presentation_job_id = job_data.get("job_id")
            print(f"[LECTURE_WORKER] Started presentation job {presentation_job_id}", flush=True)

            # Poll for completion
            result = await self._wait_for_presentation(
                client,
                presentation_url,
                presentation_job_id
            )

            return {
                "video_url": result.get("video_url"),
                "presentation_url": result.get("presentation_url"),
                "presentation_job_id": presentation_job_id,
                "duration_seconds": result.get("duration_seconds"),
                "slides_count": result.get("slides_count"),
            }

    async def _wait_for_presentation(
        self,
        client: "httpx.AsyncClient",
        base_url: str,
        job_id: str,
        timeout_seconds: int = 600,
        poll_interval: int = 5,
    ) -> Dict[str, Any]:
        """Poll for presentation job completion"""

        start_time = datetime.utcnow()

        while True:
            # Check timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Presentation job {job_id} timed out after {timeout_seconds}s")

            # Get job status
            response = await client.get(f"{base_url}/api/v1/presentations/jobs/v3/{job_id}")
            response.raise_for_status()
            status_data = response.json()

            status = status_data.get("status")

            if status == "completed":
                return status_data

            if status == "failed":
                error = status_data.get("error", "Unknown error")
                raise RuntimeError(f"Presentation job failed: {error}")

            # Still processing, wait and retry
            await asyncio.sleep(poll_interval)

    async def _check_and_trigger_finalization(self, course_job_id: str) -> None:
        """Check if all lectures are complete and trigger finalization"""

        progress_service = await self._get_progress_service()
        progress = await progress_service.get_course_progress(course_job_id)

        if not progress:
            return

        if not progress.is_complete:
            return

        # All lectures done (success or failure)
        print(f"[LECTURE_WORKER] All lectures complete for course {course_job_id}", flush=True)
        print(f"[LECTURE_WORKER]   Completed: {progress.completed_lectures}", flush=True)
        print(f"[LECTURE_WORKER]   Failed: {progress.failed_lectures}", flush=True)

        # Determine if we should finalize
        if progress.completed_lectures == 0:
            # All lectures failed, mark course as failed
            await progress_service.update_course_status(
                course_job_id,
                CourseJobStatus.FAILED,
                error="All lectures failed to generate"
            )
            return

        # Publish finalization job
        finalization_queue = await self._get_finalization_queue()

        finalization_job = QueuedFinalizationJob(
            course_job_id=course_job_id,
            user_id="",  # Will be fetched from progress
            force_finalization=progress.failed_lectures > 0,  # Force if partial success
            generate_quizzes=True,
            create_zip=True,
            created_at=datetime.utcnow().isoformat(),
        )

        await finalization_queue.publish(finalization_job)
        print(f"[LECTURE_WORKER] Triggered finalization for course {course_job_id}", flush=True)


# Convenience function for running the worker
async def run_lecture_worker(consumer_name: str = None):
    """Run the lecture worker (entry point for Docker container)"""
    worker = LectureWorker()
    await worker.start(consumer_name)


# For direct usage without queue
async def generate_lecture_direct(
    lecture_title: str,
    lecture_description: str,
    course_topic: str,
    section_title: str = "",
    language: str = "en",
    difficulty: str = "intermediate",
    target_audience: str = "general",
    duration_seconds: int = 300,
    rag_context: str = None,
    selected_elements: list = None,
) -> Dict[str, Any]:
    """
    Generate a single lecture without using the queue.

    Useful for testing or single-lecture generation.
    """
    import uuid

    job = QueuedLectureJob(
        job_id=str(uuid.uuid4()),
        course_job_id="direct",
        section_index=0,
        lecture_index=0,
        lecture_id=str(uuid.uuid4())[:8],
        lecture_title=lecture_title,
        lecture_description=lecture_description,
        section_title=section_title,
        course_topic=course_topic,
        difficulty=difficulty,
        language=language,
        target_audience=target_audience,
        duration_seconds=duration_seconds,
        rag_context=rag_context,
        selected_elements=selected_elements,
    )

    worker = LectureWorker()
    return await worker._generate_lecture(job)


# Entry point for running as module: python -m services.lecture_worker
if __name__ == "__main__":
    import socket
    consumer_name = os.getenv("CONSUMER_NAME", f"worker-{socket.gethostname()}")
    print(f"[LECTURE_WORKER] Starting as {consumer_name}...", flush=True)
    asyncio.run(run_lecture_worker(consumer_name))
