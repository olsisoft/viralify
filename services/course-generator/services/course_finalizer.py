"""
Course Finalizer Service

Phase 3 of the distributed course generation pipeline.
Assembles the final course from completed lectures.

Flow:
1. Receive finalization job from queue
2. Retrieve outline and all lecture results from Redis
3. Attach video URLs to outline
4. Generate quizzes for the course
5. Create course ZIP package
6. Mark course as completed
"""
import asyncio
import json
import os
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Any

from models.course_models import (
    CourseOutline,
    Section,
    Lecture,
)
from models.queue_models import (
    QueuedFinalizationJob,
    LectureResult,
    LectureJobStatus,
    CourseProgress,
    CourseJobStatus,
)
from services.lecture_queue import (
    CourseProgressService,
    get_progress_service,
)
from services.finalization_queue import (
    FinalizationQueueService,
    get_finalization_queue,
)


# Output directory for course packages
COURSE_OUTPUT_DIR = os.getenv("COURSE_OUTPUT_DIR", "/app/output/courses")


class CourseFinalizer:
    """
    Finalizes a course after all lectures are generated.

    Responsibilities:
    - Assemble course from lecture results
    - Generate quizzes
    - Create downloadable package
    - Update final status
    """

    def __init__(
        self,
        progress_service: CourseProgressService = None,
    ):
        self._progress_service = progress_service

    async def _get_progress_service(self) -> CourseProgressService:
        """Get or initialize progress service"""
        if self._progress_service is None:
            self._progress_service = await get_progress_service()
        return self._progress_service

    async def process_finalization(self, job: QueuedFinalizationJob) -> Dict[str, Any]:
        """
        Main finalization entry point.

        Args:
            job: The finalization job from the queue

        Returns:
            Dict with final URLs and status
        """
        print(f"[FINALIZER] Starting finalization for course {job.course_job_id}", flush=True)

        progress_service = await self._get_progress_service()

        # Update status to finalizing
        await progress_service.update_course_status(
            job.course_job_id,
            CourseJobStatus.FINALIZING
        )

        try:
            # 1. Retrieve outline
            outline_json = await progress_service.get_outline(job.course_job_id)
            if not outline_json:
                raise ValueError(f"No outline found for course {job.course_job_id}")

            outline = CourseOutline.model_validate_json(outline_json)
            print(f"[FINALIZER] Loaded outline: {outline.title}", flush=True)

            # 2. Retrieve all lecture results
            lecture_results = await progress_service.get_all_lecture_results(job.course_job_id)
            print(f"[FINALIZER] Loaded {len(lecture_results)} lecture results", flush=True)

            # 3. Attach video URLs to outline
            outline = self._attach_lecture_urls(outline, lecture_results)

            # 4. Generate quizzes if enabled
            quizzes = None
            if job.generate_quizzes:
                quizzes = await self._generate_quizzes(outline, job.quiz_config)

            # 5. Create course package
            package_result = await self._create_course_package(
                job.course_job_id,
                outline,
                lecture_results,
                quizzes,
                create_zip=job.create_zip,
            )

            # 6. Update final status
            final_status = CourseJobStatus.COMPLETED
            progress = await progress_service.get_course_progress(job.course_job_id)

            if progress and progress.is_partial_success:
                final_status = CourseJobStatus.PARTIAL_SUCCESS

            await progress_service.update_course_status(
                job.course_job_id,
                final_status
            )

            # 7. Set final URLs
            await progress_service.set_final_urls(
                job.course_job_id,
                zip_url=package_result.get("zip_url"),
                final_video_url=package_result.get("final_video_url"),
            )

            print(f"[FINALIZER] Completed finalization for course {job.course_job_id}", flush=True)

            return {
                "status": final_status.value,
                "zip_url": package_result.get("zip_url"),
                "final_video_url": package_result.get("final_video_url"),
                "outline": outline.model_dump(),
                "quizzes": quizzes,
            }

        except Exception as e:
            print(f"[FINALIZER] Failed finalization for {job.course_job_id}: {e}", flush=True)

            await progress_service.update_course_status(
                job.course_job_id,
                CourseJobStatus.FAILED,
                error=str(e)
            )

            raise

    def _attach_lecture_urls(
        self,
        outline: CourseOutline,
        lecture_results: Dict[str, LectureResult],
    ) -> CourseOutline:
        """Attach video URLs from results to outline lectures"""

        for section in outline.sections:
            for lecture in section.lectures:
                result = lecture_results.get(lecture.id)

                if result and result.status == LectureJobStatus.COMPLETED:
                    lecture.video_url = result.video_url
                    lecture.presentation_job_id = result.presentation_job_id
                    lecture.status = "completed"
                elif result and result.status == LectureJobStatus.FAILED:
                    lecture.status = "failed"
                    lecture.error = result.error

        return outline

    async def _generate_quizzes(
        self,
        outline: CourseOutline,
        quiz_config: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Generate quizzes for the course"""

        print(f"[FINALIZER] Generating quizzes...", flush=True)

        try:
            from services.quiz_generator import QuizGenerator

            generator = QuizGenerator()
            quizzes = []

            config = quiz_config or {}
            quiz_frequency = config.get("frequency", "per_section")

            if quiz_frequency == "per_lecture":
                # Generate quiz for each lecture
                for section in outline.sections:
                    for lecture in section.lectures:
                        if lecture.status == "completed":
                            quiz = await generator.generate_quiz(
                                topic=lecture.title,
                                objectives=lecture.objectives,
                                num_questions=config.get("questions_per_quiz", 5),
                            )
                            quizzes.append({
                                "lecture_id": lecture.id,
                                "section_id": section.id,
                                "type": "lecture_quiz",
                                "quiz": quiz,
                            })

            elif quiz_frequency == "per_section":
                # Generate quiz for each section
                for section in outline.sections:
                    completed_lectures = [l for l in section.lectures if l.status == "completed"]
                    if completed_lectures:
                        objectives = []
                        for lecture in completed_lectures:
                            objectives.extend(lecture.objectives)

                        quiz = await generator.generate_quiz(
                            topic=section.title,
                            objectives=objectives[:10],  # Limit objectives
                            num_questions=config.get("questions_per_quiz", 10),
                        )
                        quizzes.append({
                            "section_id": section.id,
                            "type": "section_quiz",
                            "quiz": quiz,
                        })

            else:  # end_of_course or default
                # Generate single quiz for entire course
                all_objectives = []
                for section in outline.sections:
                    for lecture in section.lectures:
                        if lecture.status == "completed":
                            all_objectives.extend(lecture.objectives)

                if all_objectives:
                    quiz = await generator.generate_quiz(
                        topic=outline.title,
                        objectives=all_objectives[:20],  # Limit objectives
                        num_questions=config.get("questions_per_quiz", 20),
                    )
                    quizzes.append({
                        "type": "final_quiz",
                        "quiz": quiz,
                    })

            print(f"[FINALIZER] Generated {len(quizzes)} quizzes", flush=True)
            return quizzes

        except ImportError:
            print(f"[FINALIZER] QuizGenerator not available, skipping quiz generation", flush=True)
            return []
        except Exception as e:
            print(f"[FINALIZER] Quiz generation failed: {e}", flush=True)
            return []

    async def _create_course_package(
        self,
        course_job_id: str,
        outline: CourseOutline,
        lecture_results: Dict[str, LectureResult],
        quizzes: List[Dict],
        create_zip: bool = True,
    ) -> Dict[str, Any]:
        """Create the final course package"""

        # Ensure output directory exists
        output_dir = Path(COURSE_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        course_dir = output_dir / course_job_id
        course_dir.mkdir(parents=True, exist_ok=True)

        # Save course manifest
        manifest = {
            "course_id": course_job_id,
            "title": outline.title,
            "description": outline.description,
            "language": outline.language,
            "total_sections": len(outline.sections),
            "total_lectures": sum(len(s.lectures) for s in outline.sections),
            "completed_lectures": sum(
                1 for r in lecture_results.values()
                if r.status == LectureJobStatus.COMPLETED
            ),
            "failed_lectures": sum(
                1 for r in lecture_results.values()
                if r.status == LectureJobStatus.FAILED
            ),
            "created_at": datetime.utcnow().isoformat(),
            "sections": [],
        }

        # Build section data
        for section in outline.sections:
            section_data = {
                "id": section.id,
                "title": section.title,
                "description": section.description,
                "lectures": [],
            }

            for lecture in section.lectures:
                result = lecture_results.get(lecture.id)
                lecture_data = {
                    "id": lecture.id,
                    "title": lecture.title,
                    "description": lecture.description,
                    "objectives": lecture.objectives,
                    "status": lecture.status,
                    "video_url": lecture.video_url,
                    "duration_seconds": result.duration_seconds if result else None,
                }
                section_data["lectures"].append(lecture_data)

            manifest["sections"].append(section_data)

        # Add quizzes to manifest
        if quizzes:
            manifest["quizzes"] = quizzes

        # Save manifest JSON
        manifest_path = course_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Save outline
        outline_path = course_dir / "outline.json"
        with open(outline_path, "w", encoding="utf-8") as f:
            f.write(outline.model_dump_json(indent=2))

        result = {
            "manifest_path": str(manifest_path),
            "outline_path": str(outline_path),
        }

        # Create ZIP if requested
        if create_zip:
            zip_path = output_dir / f"{course_job_id}.zip"
            await self._create_zip_package(course_dir, zip_path, lecture_results)
            result["zip_path"] = str(zip_path)
            result["zip_url"] = f"/output/courses/{course_job_id}.zip"

        return result

    async def _create_zip_package(
        self,
        course_dir: Path,
        zip_path: Path,
        lecture_results: Dict[str, LectureResult],
    ) -> None:
        """Create ZIP package with all course files"""

        print(f"[FINALIZER] Creating ZIP package at {zip_path}", flush=True)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add manifest and outline
            for file_path in course_dir.glob("*.json"):
                zipf.write(file_path, file_path.name)

            # Add video references (not the actual files, just a list)
            video_list = []
            for lecture_id, result in lecture_results.items():
                if result.video_url:
                    video_list.append({
                        "lecture_id": lecture_id,
                        "video_url": result.video_url,
                        "duration_seconds": result.duration_seconds,
                    })

            videos_content = json.dumps(video_list, indent=2)
            zipf.writestr("videos.json", videos_content)

        print(f"[FINALIZER] ZIP package created", flush=True)


class FinalizationWorker:
    """
    Worker that consumes finalization jobs from the queue.
    """

    def __init__(self, finalizer: CourseFinalizer = None):
        self.finalizer = finalizer or CourseFinalizer()
        self._is_running = False

    async def start(self, consumer_name: str = None) -> None:
        """Start consuming finalization jobs"""
        self._is_running = True
        finalization_queue = await get_finalization_queue()

        print("[FINALIZATION_WORKER] Starting worker...", flush=True)

        await finalization_queue.consume(
            handler=self._handle_job,
            consumer_name=consumer_name,
        )

    async def _handle_job(self, job: QueuedFinalizationJob) -> None:
        """Handle a single finalization job"""
        print(f"[FINALIZATION_WORKER] Processing finalization: {job.course_job_id}", flush=True)

        try:
            await self.finalizer.process_finalization(job)
            print(f"[FINALIZATION_WORKER] Completed finalization: {job.course_job_id}", flush=True)

        except Exception as e:
            print(f"[FINALIZATION_WORKER] Failed finalization {job.course_job_id}: {e}", flush=True)
            raise

    def stop(self) -> None:
        """Stop the worker"""
        self._is_running = False


# Convenience function for running the worker
async def run_finalization_worker(consumer_name: str = None):
    """Run the finalization worker (entry point for Docker container)"""
    worker = FinalizationWorker()
    await worker.start(consumer_name)


# For direct usage without queue
async def finalize_course_direct(course_job_id: str) -> Dict[str, Any]:
    """
    Finalize a course without using the queue.

    Useful for testing or manual finalization.
    """
    job = QueuedFinalizationJob(
        course_job_id=course_job_id,
        user_id="",
        generate_quizzes=True,
        create_zip=True,
    )

    finalizer = CourseFinalizer()
    return await finalizer.process_finalization(job)


# Entry point for running as module: python -m services.course_finalizer
if __name__ == "__main__":
    import socket
    import uuid

    # Generate unique consumer name:
    # 1. Use CONSUMER_NAME env var if set
    # 2. Otherwise use WORKER_ID (set per server) + hostname
    # 3. Fallback to hostname + short UUID for uniqueness
    worker_id = os.getenv("WORKER_ID", "")
    hostname = socket.gethostname()

    if os.getenv("CONSUMER_NAME"):
        consumer_name = os.getenv("CONSUMER_NAME")
    elif worker_id:
        consumer_name = f"finalizer-{worker_id}-{hostname}"
    else:
        # Add short UUID to ensure uniqueness across servers
        short_uuid = str(uuid.uuid4())[:8]
        consumer_name = f"finalizer-{hostname}-{short_uuid}"

    print(f"[FINALIZATION_WORKER] Starting as {consumer_name}...", flush=True)
    asyncio.run(run_finalization_worker(consumer_name))
