"""
Course Orchestrator Service

Phase 1 of the distributed course generation pipeline.
Generates the course outline and creates individual lecture jobs.

Flow:
1. Receive course generation request
2. Generate course outline using CoursePlanner
3. Save outline to Redis
4. Create QueuedLectureJob for each lecture
5. Publish all lecture jobs to the lecture queue
"""
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from models.course_models import (
    PreviewOutlineRequest,
    CourseOutline,
    CourseStructureConfig,
    Section,
    Lecture,
    DifficultyLevel,
)
from models.queue_models import (
    QueuedLectureJob,
    CourseProgress,
    CourseJobStatus,
    QueuedFinalizationJob,
)
from services.course_planner import CoursePlanner
from services.lecture_queue import (
    LectureQueueService,
    CourseProgressService,
    get_lecture_queue,
    get_progress_service,
)
from services.course_queue import QueuedCourseJob


class CourseOrchestrator:
    """
    Orchestrates the course generation pipeline.

    Responsibilities:
    - Generate course outline
    - Track progress in Redis
    - Create and publish lecture jobs
    - Handle orchestration errors
    """

    def __init__(
        self,
        planner: CoursePlanner = None,
        lecture_queue: LectureQueueService = None,
        progress_service: CourseProgressService = None,
    ):
        self.planner = planner or CoursePlanner()
        self._lecture_queue = lecture_queue
        self._progress_service = progress_service

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

    async def process_course_orchestration(
        self,
        job: QueuedCourseJob,
        rag_context: Optional[str] = None,
    ) -> CourseProgress:
        """
        Main orchestration entry point.

        Args:
            job: The queued course job from RabbitMQ
            rag_context: Optional RAG context from documents

        Returns:
            CourseProgress with status and lecture job IDs
        """
        print(f"[ORCHESTRATOR] Starting orchestration for job {job.job_id}", flush=True)

        progress_service = await self._get_progress_service()
        lecture_queue = await self._get_lecture_queue()

        # Initialize progress tracking
        progress = CourseProgress(
            course_job_id=job.job_id,
            status=CourseJobStatus.ORCHESTRATING,
            created_at=datetime.utcnow().isoformat(),
            started_at=datetime.utcnow().isoformat(),
        )
        await progress_service.create_course_progress(progress)

        try:
            # 1. Generate the course outline
            print(f"[ORCHESTRATOR] Generating outline for: {job.topic}", flush=True)
            outline = await self._generate_outline(job, rag_context)

            # Count total lectures
            total_lectures = sum(len(section.lectures) for section in outline.sections)
            print(f"[ORCHESTRATOR] Outline generated: {len(outline.sections)} sections, {total_lectures} lectures", flush=True)

            # 2. Update progress with outline
            progress.status = CourseJobStatus.GENERATING_LECTURES
            progress.total_lectures = total_lectures
            progress.outline_json = outline.model_dump_json()
            await progress_service.create_course_progress(progress)

            # Also save outline separately for quick access
            await progress_service.save_outline(job.job_id, outline.model_dump_json())

            # 3. Create lecture jobs
            lecture_jobs = self._create_lecture_jobs(job, outline, rag_context)

            # 4. Publish all lecture jobs to the queue
            await lecture_queue.publish_batch(lecture_jobs)

            print(f"[ORCHESTRATOR] Created {len(lecture_jobs)} lecture jobs for course {job.job_id}", flush=True)

            return progress

        except Exception as e:
            print(f"[ORCHESTRATOR] Failed to orchestrate course {job.job_id}: {e}", flush=True)

            # Update progress with error
            progress.status = CourseJobStatus.FAILED
            progress.error = str(e)
            await progress_service.create_course_progress(progress)

            raise

    async def _generate_outline(
        self,
        job: QueuedCourseJob,
        rag_context: Optional[str] = None,
    ) -> CourseOutline:
        """Generate course outline using the planner"""

        # Build request from job
        # NOTE: PreviewOutlineRequest uses 'structure' (CourseStructureConfig), not
        # 'num_sections'/'lectures_per_section' directly. Previous code passed unknown
        # fields that Pydantic silently ignored, causing default structure to be used.
        structure = CourseStructureConfig(
            number_of_sections=job.num_sections,
            lectures_per_section=job.lectures_per_section,
        )

        request = PreviewOutlineRequest(
            topic=job.topic,
            description=f"Course on {job.topic}",
            difficulty_start=DifficultyLevel(job.difficulty_start),
            difficulty_end=DifficultyLevel(job.difficulty_end),
            structure=structure,
            language=job.language,
            rag_context=rag_context,
        )

        # Generate outline
        outline = await self.planner.generate_outline(request)

        return outline

    def _create_lecture_jobs(
        self,
        job: QueuedCourseJob,
        outline: CourseOutline,
        rag_context: Optional[str] = None,
    ) -> List[QueuedLectureJob]:
        """Create QueuedLectureJob for each lecture in the outline"""

        lecture_jobs = []
        global_lecture_index = 0

        for section_idx, section in enumerate(outline.sections):
            for lecture_idx, lecture in enumerate(section.lectures):
                lecture_job = QueuedLectureJob(
                    job_id=f"{job.job_id}_L{section_idx:02d}_{lecture_idx:02d}",
                    course_job_id=job.job_id,
                    section_index=section_idx,
                    lecture_index=lecture_idx,
                    lecture_id=lecture.id,

                    # Lecture content
                    lecture_title=lecture.title,
                    lecture_description=lecture.description,
                    section_title=section.title,
                    course_topic=job.topic,

                    # Config
                    difficulty=lecture.difficulty.value if isinstance(lecture.difficulty, DifficultyLevel) else lecture.difficulty,
                    language=job.language,
                    target_audience=job.target_audience,
                    duration_seconds=lecture.duration_seconds,

                    # Elements
                    selected_elements=job.selected_elements or lecture.lesson_elements,
                    element_weights=lecture.element_weights,

                    # Quiz config
                    quiz_config=job.quiz_config,

                    # RAG context (can be truncated for large documents)
                    rag_context=self._truncate_rag_context(rag_context, 8000) if rag_context else None,

                    # Priority: earlier lectures have higher priority (lower number)
                    priority=min(10, 1 + (global_lecture_index // 3)),

                    # Metadata
                    created_at=datetime.utcnow().isoformat(),
                )

                lecture_jobs.append(lecture_job)
                global_lecture_index += 1

        return lecture_jobs

    def _truncate_rag_context(self, context: str, max_chars: int) -> str:
        """Truncate RAG context to fit within limits"""
        if not context or len(context) <= max_chars:
            return context

        # Truncate at a sentence boundary if possible
        truncated = context[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:
            truncated = truncated[:last_period + 1]

        return truncated + "\n\n[Context truncated for lecture-level processing]"

    async def get_orchestration_status(self, course_job_id: str) -> Optional[CourseProgress]:
        """Get the current orchestration status"""
        progress_service = await self._get_progress_service()
        return await progress_service.get_course_progress(course_job_id)


class OrchestrationWorker:
    """
    Worker that consumes course orchestration jobs from RabbitMQ.

    This runs as a separate process/container for scalability.
    """

    def __init__(self, orchestrator: CourseOrchestrator = None):
        self.orchestrator = orchestrator or CourseOrchestrator()
        self._is_running = False

    async def start(self) -> None:
        """Start the orchestration worker"""
        from services.course_queue import CourseQueueService

        self._is_running = True
        queue_service = CourseQueueService()
        await queue_service.connect()

        print("[ORCHESTRATION_WORKER] Starting worker...", flush=True)

        try:
            await queue_service.consume(self._handle_job)
        finally:
            await queue_service.disconnect()

    async def _handle_job(self, job: QueuedCourseJob) -> None:
        """Handle a single orchestration job"""
        print(f"[ORCHESTRATION_WORKER] Processing job: {job.job_id}", flush=True)

        try:
            # Get RAG context if source documents are provided
            rag_context = None
            if job.source_ids or job.document_ids:
                rag_context = await self._fetch_rag_context(job)

            # Run orchestration
            await self.orchestrator.process_course_orchestration(job, rag_context)

            print(f"[ORCHESTRATION_WORKER] Completed job: {job.job_id}", flush=True)

        except Exception as e:
            print(f"[ORCHESTRATION_WORKER] Failed job {job.job_id}: {e}", flush=True)
            raise

    async def _fetch_rag_context(self, job: QueuedCourseJob) -> Optional[str]:
        """Fetch RAG context from documents"""
        try:
            # Try SourceLibrary first
            if job.source_ids:
                from services.source_library import get_source_library
                library = await get_source_library()
                context = await library.get_context_for_generation(
                    source_ids=job.source_ids,
                    topic=job.topic,
                    max_chars=32000
                )
                if context:
                    return context

            # Fallback to RAGService for document_ids
            if job.document_ids:
                from services.retrieval_service import get_rag_service
                rag_service = await get_rag_service()
                context = await rag_service.get_context_for_course_generation(
                    topic=job.topic,
                    description=f"Course on {job.topic}",
                    document_ids=job.document_ids,
                    user_id=job.user_id,
                )
                return context

        except Exception as e:
            print(f"[ORCHESTRATION_WORKER] Failed to fetch RAG context: {e}", flush=True)

        return None

    def stop(self) -> None:
        """Stop the worker"""
        self._is_running = False


# Convenience function for running the worker
async def run_orchestration_worker():
    """Run the orchestration worker (entry point for Docker container)"""
    worker = OrchestrationWorker()
    await worker.start()


# For direct usage without queue
async def orchestrate_course_direct(
    topic: str,
    num_sections: int = 5,
    lectures_per_section: int = 3,
    user_id: str = "anonymous",
    language: str = "en",
    difficulty_start: str = "beginner",
    difficulty_end: str = "intermediate",
    target_audience: str = "general",
    source_ids: List[str] = None,
    document_ids: List[str] = None,
    selected_elements: List[str] = None,
    quiz_config: Dict = None,
) -> CourseProgress:
    """
    Orchestrate a course without using the queue.

    Useful for testing or single-server deployments.
    """
    import uuid

    job = QueuedCourseJob(
        job_id=str(uuid.uuid4()),
        topic=topic,
        num_sections=num_sections,
        lectures_per_section=lectures_per_section,
        user_id=user_id,
        language=language,
        difficulty_start=difficulty_start,
        difficulty_end=difficulty_end,
        target_audience=target_audience,
        source_ids=source_ids,
        document_ids=document_ids,
        selected_elements=selected_elements,
        quiz_config=quiz_config,
    )

    # Fetch RAG context
    rag_context = None
    if source_ids or document_ids:
        worker = OrchestrationWorker()
        rag_context = await worker._fetch_rag_context(job)

    # Run orchestration
    orchestrator = CourseOrchestrator()
    return await orchestrator.process_course_orchestration(job, rag_context)
