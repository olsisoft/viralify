"""
Course Compositor Service

Orchestrates parallel lecture generation via presentation-generator service.
"""
import asyncio
import os
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import httpx

from models.course_models import (
    CourseJob,
    CourseStage,
    GenerateCourseRequest,
    Lecture,
    Section,
)
from services.course_planner import CoursePlanner
from services.http_client import ResilientHTTPClient, RetryConfig
from services.lecture_editor import LectureEditorService


class CourseCompositor:
    """Service for orchestrating parallel lecture generation"""

    # Configuration constants - Tuned for performance + reliability
    MAX_RETRIES = 3  # Retries for failed lectures
    POLL_INTERVAL_MIN = 2.0  # Adaptive polling: minimum interval
    POLL_INTERVAL_MAX = 15.0  # Adaptive polling: maximum interval
    MAX_WAIT_PER_LECTURE = 1200.0  # 20 minutes per lecture
    HTTP_TIMEOUT = 90.0  # seconds for HTTP requests
    HEALTH_CHECK_INTERVAL = 30.0  # Check service health every 30s
    POLL_REQUEST_TIMEOUT = 60.0  # seconds for individual poll requests
    MAX_CONSECUTIVE_ERRORS = 15  # More tolerant of transient errors
    RETRY_BACKOFF_BASE = 5.0  # Base seconds for exponential backoff
    CONNECTION_RETRY_DELAY = 10.0  # Delay before retrying after connection loss
    MAX_PARALLEL_DOWNLOADS = 5  # For ZIP video downloads

    def __init__(
        self,
        presentation_generator_url: str = None,
        media_generator_url: str = None,
        max_parallel_lectures: int = 3  # Balance between speed and stability
    ):
        # Use Docker hostnames by default for container communication
        # In development, set env vars to override
        if presentation_generator_url is None:
            presentation_generator_url = os.getenv("PRESENTATION_GENERATOR_URL", "http://presentation-generator:8006")
        if media_generator_url is None:
            media_generator_url = os.getenv("MEDIA_GENERATOR_URL", "http://media-generator:8004")
        self.presentation_generator_url = presentation_generator_url
        self.media_generator_url = media_generator_url
        self.max_parallel_lectures = max_parallel_lectures
        self.semaphore = asyncio.Semaphore(max_parallel_lectures)
        self.output_dir = Path("/app/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Job cancellation tracking
        self._cancelled_jobs: set = set()

        # Initialize course planner for generating lecture prompts
        self.course_planner = CoursePlanner()

        # Resilient HTTP clients with retry logic for Docker network issues
        retry_config = RetryConfig(
            max_retries=5,
            initial_delay=2.0,
            max_delay=30.0,
            exponential_base=2.0,
        )
        self.presentation_client = ResilientHTTPClient(
            presentation_generator_url,
            timeout=self.HTTP_TIMEOUT,
            retry_config=retry_config,
        )
        self.media_client = ResilientHTTPClient(
            media_generator_url,
            timeout=120.0,
            retry_config=retry_config,
        )

        # Lecture editor service for storing components after generation
        self.lecture_editor = LectureEditorService(
            presentation_generator_url=presentation_generator_url,
            media_generator_url=media_generator_url
        )

    def cancel_job(self, job_id: str) -> bool:
        """Mark a job for cancellation"""
        self._cancelled_jobs.add(job_id)
        print(f"[COMPOSITOR] Job {job_id} marked for cancellation", flush=True)
        return True

    def is_job_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled"""
        return job_id in self._cancelled_jobs

    async def _check_service_health(self, service_url: str) -> bool:
        """Check if a service is healthy"""
        try:
            if "presentation" in service_url:
                response = await self.presentation_client.get("/health")
            else:
                response = await self.media_client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _extract_source_topics(rag_context: str, top_n: int = 25) -> List[str]:
        """
        Extract main topics from RAG context for topic lock.
        Cached at course level to avoid re-extraction per lesson.
        """
        if not rag_context:
            return []

        # Clean and tokenize
        text = rag_context.lower()
        # Remove URLs and special chars
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-ZàâäéèêëïîôùûüÿçœæÀÂÄÉÈÊËÏÎÔÙÛÜŸÇŒÆ\s-]', ' ', text)

        # Split into words
        words = text.split()

        # Filter: 3+ chars, not common stop words
        stop_words = {
            'the', 'and', 'for', 'that', 'this', 'with', 'are', 'from', 'can', 'will',
            'have', 'has', 'was', 'been', 'being', 'which', 'what', 'when', 'where',
            'les', 'des', 'une', 'pour', 'dans', 'que', 'qui', 'sur', 'par', 'avec',
            'est', 'sont', 'nous', 'vous', 'leur', 'cette', 'ces', 'mais', 'comme',
        }
        filtered = [w for w in words if len(w) >= 3 and w not in stop_words]

        # Count and return top topics
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(top_n)]

    async def generate_all_lectures(
        self,
        job: CourseJob,
        request: GenerateCourseRequest,
        progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None
    ):
        """
        Generate all lectures in parallel (max 3 at a time).

        Args:
            job: The course job containing outline
            request: Original generation request
            progress_callback: Callback for progress updates (completed, total, current_title)
        """
        if not job.outline:
            raise ValueError("No outline available for lecture generation")

        print(f"[COMPOSITOR] Starting lecture generation for {job.outline.total_lectures} lectures", flush=True)
        print(f"[COMPOSITOR] Max parallel: {self.max_parallel_lectures}", flush=True)

        # Extract source topics ONCE for all lectures (cache for presentation-generator)
        rag_context = getattr(request, 'rag_context', None)
        if rag_context and not getattr(request, 'source_topics', None):
            source_topics = self._extract_source_topics(rag_context, top_n=25)
            setattr(request, 'source_topics', source_topics)
            print(f"[COMPOSITOR] Extracted {len(source_topics)} topics from RAG (cached for all lectures)", flush=True)

        # Flatten lectures with their context
        lecture_tasks = []
        position = 0

        for section in job.outline.sections:
            for lecture in section.lectures:
                position += 1
                lecture_tasks.append({
                    "lecture": lecture,
                    "section": section,
                    "position": position,
                    "total": job.outline.total_lectures
                })

        total = len(lecture_tasks)
        completed = 0

        # Create tasks for parallel execution with retry support
        async def generate_with_semaphore(task_info: dict):
            nonlocal completed
            async with self.semaphore:
                lecture = task_info["lecture"]
                section = task_info["section"]

                # Check for cancellation
                if self.is_job_cancelled(job.job_id):
                    lecture.status = "cancelled"
                    lecture.error = "Job was cancelled"
                    completed += 1
                    print(f"[COMPOSITOR] Skipping cancelled lecture: {lecture.title}", flush=True)
                    return

                # Retry loop
                while lecture.retry_count <= self.MAX_RETRIES:
                    # Check cancellation before each attempt
                    if self.is_job_cancelled(job.job_id):
                        lecture.status = "cancelled"
                        lecture.error = "Job was cancelled"
                        break

                    attempt = lecture.retry_count + 1
                    retry_info = f" (attempt {attempt}/{self.MAX_RETRIES + 1})" if lecture.retry_count > 0 else ""

                    print(f"[COMPOSITOR] Starting lecture: {lecture.title}{retry_info}", flush=True)
                    lecture.status = "generating" if lecture.retry_count == 0 else "retrying"
                    lecture.error = None
                    lecture.progress_percent = 0.0
                    lecture.current_stage = "starting"

                    if progress_callback:
                        progress_callback(completed, total, lecture.title)

                    try:
                        # Use asyncio.wait_for to enforce timeout
                        video_url = await asyncio.wait_for(
                            self._generate_single_lecture(
                                lecture=lecture,
                                section=section,
                                outline=job.outline,
                                request=request,
                                position=task_info["position"],
                                total=task_info["total"],
                                job_id=job.job_id  # Pass job_id for cancellation check
                            ),
                            timeout=self.MAX_WAIT_PER_LECTURE
                        )

                        lecture.video_url = video_url
                        lecture.status = "completed"
                        lecture.progress_percent = 100.0
                        lecture.current_stage = "completed"
                        print(f"[COMPOSITOR] Completed: {lecture.title}", flush=True)

                        # Store components for editing (async, non-blocking)
                        if lecture.presentation_job_id:
                            try:
                                components_id = await self.lecture_editor.store_components_from_presentation_job(
                                    presentation_job_id=lecture.presentation_job_id,
                                    lecture_id=lecture.id,
                                    job_id=job.job_id,
                                    generation_params={
                                        "topic": lecture.title,
                                        "duration": lecture.duration_seconds,
                                        "style": request.style,
                                        "voice_id": request.voice_id,
                                        "typing_speed": request.typing_speed,
                                        "include_avatar": request.include_avatar,
                                    }
                                )
                                if components_id:
                                    lecture.components_id = components_id
                                    lecture.has_components = True
                                    print(f"[COMPOSITOR] Components stored for {lecture.title}: {components_id}", flush=True)
                            except Exception as store_error:
                                # Non-blocking - log but don't fail the lecture
                                print(f"[COMPOSITOR] Warning: Failed to store components for {lecture.title}: {str(store_error)}", flush=True)

                        break  # Success, exit retry loop

                    except asyncio.TimeoutError:
                        lecture.retry_count += 1
                        error_msg = f"Lecture generation timed out after {self.MAX_WAIT_PER_LECTURE/60:.0f} minutes"
                        print(f"[COMPOSITOR] TIMEOUT: {lecture.title} - {error_msg}", flush=True)

                        if lecture.retry_count <= self.MAX_RETRIES:
                            lecture.error = f"Attempt {attempt} timed out"
                            # Exponential backoff before retry
                            retry_delay = self.RETRY_BACKOFF_BASE * (2 ** (lecture.retry_count - 1))
                            print(f"[COMPOSITOR] Waiting {retry_delay:.0f}s before retry...", flush=True)
                            await asyncio.sleep(retry_delay)
                        else:
                            lecture.status = "failed"
                            lecture.error = error_msg
                            lecture.current_stage = "failed"

                    except asyncio.CancelledError:
                        lecture.status = "cancelled"
                        lecture.error = "Generation was cancelled"
                        lecture.current_stage = "cancelled"
                        print(f"[COMPOSITOR] Cancelled: {lecture.title}", flush=True)
                        break

                    except Exception as e:
                        lecture.retry_count += 1
                        error_msg = str(e)

                        if lecture.retry_count <= self.MAX_RETRIES:
                            # Will retry with exponential backoff
                            retry_delay = self.RETRY_BACKOFF_BASE * (2 ** (lecture.retry_count - 1))
                            print(f"[COMPOSITOR] Failed (will retry in {retry_delay:.0f}s): {lecture.title} - {error_msg}", flush=True)
                            lecture.error = f"Attempt {attempt} failed: {error_msg}"
                            await asyncio.sleep(retry_delay)
                        else:
                            # All retries exhausted
                            lecture.status = "failed"
                            lecture.error = f"Failed after {self.MAX_RETRIES + 1} attempts. Last error: {error_msg}"
                            lecture.current_stage = "failed"
                            lecture.can_regenerate = True  # Allow manual regeneration
                            # Track failed lecture in job
                            job.add_failed_lecture(lecture.id, lecture.error)
                            print(f"[COMPOSITOR] Failed permanently: {lecture.title} - {error_msg}", flush=True)

                completed += 1
                if progress_callback:
                    # Include failed count in progress callback
                    failed_count = len(job.failed_lecture_ids)
                    progress_callback(completed - failed_count, total, None if completed == total else lecture.title)

        # Run all lectures with semaphore limiting
        await asyncio.gather(
            *[generate_with_semaphore(task) for task in lecture_tasks],
            return_exceptions=True
        )

        # Count results
        completed_count = 0
        failed_count = 0
        failed_lectures = []

        for section in job.outline.sections:
            for lecture in section.lectures:
                if lecture.status == "completed":
                    completed_count += 1
                elif lecture.status == "failed":
                    failed_count += 1
                    failed_lectures.append(lecture.title)

        # Update job with final counts
        job.lectures_completed = completed_count
        job.lectures_failed = failed_count

        if failed_lectures:
            print(f"[COMPOSITOR] {len(failed_lectures)} lectures failed: {failed_lectures}", flush=True)

        # Determine final status
        final_stage = job.get_final_stage()
        if final_stage == CourseStage.PARTIAL_SUCCESS:
            print(f"[COMPOSITOR] PARTIAL SUCCESS: {completed_count}/{total} lectures generated, {failed_count} failed", flush=True)
            print(f"[COMPOSITOR] Failed lectures can be edited and regenerated individually", flush=True)
        elif final_stage == CourseStage.COMPLETED:
            print(f"[COMPOSITOR] Generation complete: {completed_count}/{total} lectures", flush=True)
        else:
            print(f"[COMPOSITOR] FAILED: All {total} lectures failed", flush=True)

        return final_stage

    async def _generate_single_lecture(
        self,
        lecture: Lecture,
        section: Section,
        outline,
        request: GenerateCourseRequest,
        position: int,
        total: int,
        job_id: str = None
    ) -> str:
        """
        Generate a single lecture via presentation-generator.

        Returns the video URL.
        """
        # Build the presentation topic/prompt from lecture context
        lesson_elements = {
            "concept_intro": request.lesson_elements.concept_intro,
            "diagram_schema": request.lesson_elements.diagram_schema,
            "code_typing": request.lesson_elements.code_typing,
            "code_execution": request.lesson_elements.code_execution,
            "voiceover_explanation": request.lesson_elements.voiceover_explanation,
            "curriculum_slide": request.lesson_elements.curriculum_slide,
        }

        # Extract programming language from context (do this BEFORE generating prompt)
        programming_language = None
        if request.context:
            # Try specific_tools field first
            if request.context.specific_tools:
                programming_language = request.context.specific_tools
            # Fall back to context_answers if available
            elif request.context.context_answers and request.context.context_answers.get("specific_tools"):
                programming_language = request.context.context_answers.get("specific_tools")

        # Get RAG context and pre-extracted topics from request
        rag_context = getattr(request, 'rag_context', None)
        source_topics = getattr(request, 'source_topics', None)

        # Generate the detailed prompt for this lecture (now with programming_language and RAG)
        topic_prompt = await self.course_planner.generate_lecture_prompt(
            lecture=lecture,
            section=section,
            outline=outline,
            lesson_elements=lesson_elements,
            position=position,
            total=total,
            rag_context=rag_context,  # Pass RAG context for deep document integration
            programming_language=programming_language
        )

        # Prepare presentation request
        # IMPORTANT: Distinguish between programming language and content language
        # - language: programming language for code syntax highlighting (e.g., "python", "javascript")
        # - content_language: human language for voiceover, titles, text (e.g., "en", "fr", "es")
        actual_programming_language = programming_language or "python"  # Default to Python if not specified

        # Extract practical_focus from context
        practical_focus = None
        if request.context:
            practical_focus = request.context.practical_focus
            if not practical_focus and request.context.context_answers:
                practical_focus = request.context.context_answers.get("practical_focus")

        # DEBUG: Log the duration being sent
        print(f"[COMPOSITOR] Sending duration={lecture.duration_seconds}s for lecture '{lecture.title}'", flush=True)

        presentation_request = {
            "topic": topic_prompt,
            "language": actual_programming_language,  # Programming language for code
            "content_language": outline.language,  # Content language (en, fr, es, etc.)
            "duration": lecture.duration_seconds,
            "style": request.style,
            "include_avatar": request.include_avatar,
            "avatar_id": request.avatar_id,
            "voice_id": request.voice_id,
            "execute_code": request.lesson_elements.code_execution,
            "show_typing_animation": request.lesson_elements.code_typing,
            "typing_speed": request.typing_speed,
            "title_style": getattr(request, 'title_style', 'engaging'),  # Title style for slides
            "target_audience": outline.target_audience,
            # Enable visuals if diagram_schema is enabled
            "enable_visuals": request.lesson_elements.diagram_schema,
            "visual_style": request.style or "dark",
            # Pass RAG context and pre-extracted topics (cached at course level)
            "rag_context": rag_context if rag_context else None,
            "source_topics": source_topics if source_topics else None,
            # Pass practical focus level for slide ratio adjustment
            "practical_focus": practical_focus,
        }

        # Start presentation generation using resilient client
        lecture.current_stage = "starting"
        response = await self.presentation_client.post(
            "/api/v1/presentations/generate",
            json=presentation_request
        )

        if response.status_code != 200:
            raise Exception(f"Failed to start generation: {response.text}")

        job_data = response.json()
        presentation_job_id = job_data.get("job_id")
        lecture.presentation_job_id = presentation_job_id

        print(f"[COMPOSITOR] Presentation job started: {presentation_job_id} for {lecture.title}", flush=True)

        # Poll for completion with progress tracking and cancellation support
        video_url = await self._poll_presentation_job(
            presentation_job_id, lecture, job_id=job_id
        )

        return video_url

    def _get_adaptive_poll_interval(self, elapsed: float, progress: float) -> float:
        """
        Calculate adaptive polling interval based on elapsed time and progress.

        - Start with frequent polling (2s) to catch fast completions
        - Gradually increase as time passes
        - Decrease if progress is high (near completion)
        """
        # Base interval increases with time (logarithmic growth)
        time_factor = min(elapsed / 60.0, 5.0)  # Cap at 5 minutes
        base_interval = self.POLL_INTERVAL_MIN + (time_factor * 2.0)

        # If progress is high (>80%), poll more frequently
        if progress > 80:
            base_interval = self.POLL_INTERVAL_MIN
        elif progress > 50:
            base_interval = min(base_interval, 5.0)

        return min(base_interval, self.POLL_INTERVAL_MAX)

    async def _poll_presentation_job(
        self,
        pres_job_id: str,
        lecture: Lecture,
        max_wait: float = None,
        job_id: str = None  # Course job ID for cancellation check
    ) -> str:
        """Poll presentation-generator until job completes, updating lecture progress.

        Uses ADAPTIVE polling intervals (2s → 15s) based on elapsed time and progress.
        Exponential backoff on errors. Uses resilient HTTP client with automatic retries.
        """
        max_wait = max_wait or self.MAX_WAIT_PER_LECTURE

        start_time = asyncio.get_event_loop().time()
        consecutive_errors = 0
        last_successful_poll = start_time
        last_progress_log = 0
        current_progress = 0.0

        while True:
            # Check for job cancellation
            if job_id and self.is_job_cancelled(job_id):
                raise asyncio.CancelledError("Job was cancelled")

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Presentation generation timed out after {max_wait}s ({max_wait/60:.0f} minutes)")

            try:
                # Use resilient client with built-in retry for transient network failures
                response = await self.presentation_client.get(
                    f"/api/v1/presentations/jobs/{pres_job_id}"
                )

                if response.status_code != 200:
                    consecutive_errors += 1
                    # Calculate backoff delay with exponential increase
                    backoff_delay = min(
                        self.RETRY_BACKOFF_BASE * (2 ** min(consecutive_errors - 1, 4)),  # Cap at 80s
                        60.0
                    )

                    if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        raise Exception(f"Too many errors polling job ({consecutive_errors}): {response.text}")

                    print(f"[COMPOSITOR] Poll error {consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS} for {pres_job_id}, backing off {backoff_delay:.1f}s", flush=True)
                    await asyncio.sleep(backoff_delay)
                    continue

                # Success - reset error count and update last successful time
                consecutive_errors = 0
                last_successful_poll = asyncio.get_event_loop().time()

                job_data = response.json()
                status = job_data.get("status")
                current_stage = job_data.get("current_stage", "unknown")
                current_progress = float(job_data.get("progress", 0))

                # Update lecture progress
                lecture.current_stage = current_stage
                lecture.progress_percent = current_progress

                if status == "completed":
                    video_url = job_data.get("output_url")
                    if not video_url:
                        # Sometimes output_url takes a moment to populate - retry a few times
                        if consecutive_errors < 3:
                            consecutive_errors += 1
                            print(f"[COMPOSITOR] Job {pres_job_id} completed but no output URL yet, retrying...", flush=True)
                            await asyncio.sleep(5.0)
                            continue
                        raise Exception("Job completed but no output URL after retries")
                    lecture.progress_percent = 100.0
                    lecture.current_stage = "completed"
                    return video_url

                if status == "failed":
                    error = job_data.get("error", "Unknown error")
                    lecture.current_stage = "failed"
                    raise Exception(f"Presentation generation failed: {error}")

                # Log progress periodically (every 30 seconds)
                current_time = int(elapsed)
                if current_time - last_progress_log >= 30:
                    last_progress_log = current_time
                    remaining = max_wait - elapsed
                    print(f"[COMPOSITOR] Job {pres_job_id}: {current_stage} ({current_progress:.0f}%) - {remaining/60:.1f}min remaining", flush=True)

            except httpx.TimeoutException as e:
                # Timeout on poll request - more tolerant
                consecutive_errors += 1
                backoff_delay = min(self.RETRY_BACKOFF_BASE * (2 ** min(consecutive_errors - 1, 3)), 30.0)

                print(f"[COMPOSITOR] Poll timeout {consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS} for {pres_job_id}: {str(e)}", flush=True)

                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    raise Exception(f"Presentation-generator not responding after {self.MAX_CONSECUTIVE_ERRORS} timeouts")

                await asyncio.sleep(backoff_delay)
                continue

            except httpx.RequestError as e:
                # Network/connection error - use longer delay and be more tolerant
                consecutive_errors += 1

                # Use exponential backoff with longer delays for connection errors
                backoff_delay = min(
                    self.CONNECTION_RETRY_DELAY * (1.5 ** min(consecutive_errors - 1, 5)),
                    60.0
                )

                print(f"[COMPOSITOR] Network error {consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS} polling {pres_job_id}: {str(e)}", flush=True)

                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    # Before giving up, check how long since last success
                    time_since_success = asyncio.get_event_loop().time() - last_successful_poll
                    if time_since_success < 120:  # If we had success within 2 minutes, keep trying
                        print(f"[COMPOSITOR] Continuing despite errors - last success was {time_since_success:.0f}s ago", flush=True)
                        consecutive_errors = self.MAX_CONSECUTIVE_ERRORS - 3  # Reset partially
                    else:
                        raise Exception(f"Lost connection to presentation-generator after {self.MAX_CONSECUTIVE_ERRORS} attempts")

                await asyncio.sleep(backoff_delay)
                continue

            except asyncio.CancelledError:
                raise  # Re-raise cancellation

            except Exception as e:
                # Unexpected error - log but continue polling
                consecutive_errors += 1
                print(f"[COMPOSITOR] Unexpected error polling {pres_job_id}: {str(e)}", flush=True)

                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    raise

                # Use adaptive interval on error
                adaptive_interval = self._get_adaptive_poll_interval(elapsed, current_progress)
                await asyncio.sleep(adaptive_interval)
                continue

            # OPTIMIZED: Use adaptive polling interval based on elapsed time and progress
            adaptive_interval = self._get_adaptive_poll_interval(elapsed, current_progress)
            await asyncio.sleep(adaptive_interval)

    async def create_course_zip(self, job: CourseJob) -> Optional[str]:
        """
        Create a ZIP file containing all course videos and metadata.

        OPTIMIZED: Downloads videos in parallel (up to MAX_PARALLEL_DOWNLOADS).

        Returns the ZIP file path.
        """
        import json

        if not job.outline or not job.output_urls:
            return None

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"course_{job.job_id}_{timestamp}.zip"
        zip_path = self.output_dir / zip_filename

        print(f"[COMPOSITOR] Creating ZIP: {zip_path}", flush=True)

        try:
            # Build metadata and collect video download tasks
            metadata = {
                "title": job.outline.title,
                "description": job.outline.description,
                "target_audience": job.outline.target_audience,
                "language": job.outline.language,
                "difficulty_range": f"{job.outline.difficulty_start.value} to {job.outline.difficulty_end.value}",
                "total_duration_minutes": job.outline.total_duration_minutes,
                "sections": []
            }

            # Collect all video download tasks
            video_tasks = []

            for section in job.outline.sections:
                section_data = {
                    "title": section.title,
                    "description": section.description,
                    "lectures": []
                }

                for lecture in section.lectures:
                    video_path = f"videos/section_{section.order + 1:02d}_lecture_{lecture.order + 1:02d}.mp4"
                    lecture_data = {
                        "title": lecture.title,
                        "description": lecture.description,
                        "objectives": lecture.objectives,
                        "difficulty": lecture.difficulty.value,
                        "duration_seconds": lecture.duration_seconds,
                        "video_file": video_path if lecture.video_url else None
                    }
                    section_data["lectures"].append(lecture_data)

                    # Collect video download task
                    if lecture.video_url:
                        video_tasks.append({
                            "url": lecture.video_url,
                            "path": video_path,
                            "title": lecture.title
                        })

                metadata["sections"].append(section_data)

            # OPTIMIZED: Download all videos in parallel with semaphore limiting
            download_semaphore = asyncio.Semaphore(self.MAX_PARALLEL_DOWNLOADS)
            downloaded_videos = {}
            download_errors = []  # Track errors for reporting

            async def download_with_semaphore(task: dict):
                async with download_semaphore:
                    try:
                        print(f"[COMPOSITOR] Downloading: {task['title']}", flush=True)
                        video_data = await self._download_video(task["url"])
                        downloaded_videos[task["path"]] = video_data
                        print(f"[COMPOSITOR] Downloaded: {task['title']} ({len(video_data) / 1024 / 1024:.1f} MB)", flush=True)
                    except Exception as e:
                        error_msg = f"{task['title']}: {str(e)}"
                        download_errors.append(error_msg)
                        print(f"[COMPOSITOR] Failed to download {error_msg}", flush=True)

            print(f"[COMPOSITOR] Downloading {len(video_tasks)} videos in parallel (max {self.MAX_PARALLEL_DOWNLOADS})...", flush=True)
            await asyncio.gather(
                *[download_with_semaphore(task) for task in video_tasks],
                return_exceptions=True
            )

            # Check if we have any videos - don't create empty ZIP
            if not downloaded_videos:
                print(f"[COMPOSITOR] No videos downloaded successfully. Errors: {download_errors}", flush=True)
                return None

            # Log partial failures
            if download_errors:
                print(f"[COMPOSITOR] Warning: {len(download_errors)}/{len(video_tasks)} downloads failed", flush=True)

            # Create ZIP with downloaded videos
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add videos
                for video_path, video_data in downloaded_videos.items():
                    zipf.writestr(video_path, video_data)

                # Add metadata JSON
                zipf.writestr("course_metadata.json", json.dumps(metadata, indent=2))

                # Add README
                readme = self._generate_readme(metadata)
                zipf.writestr("README.md", readme)

            print(f"[COMPOSITOR] ZIP created: {zip_path} ({len(downloaded_videos)}/{len(video_tasks)} videos)", flush=True)
            return str(zip_path)

        except Exception as e:
            print(f"[COMPOSITOR] Failed to create ZIP: {str(e)}", flush=True)
            return None

    async def _download_video(self, url: str) -> bytes:
        """Download video from URL or internal path.

        Handles:
        - Full HTTP URLs (http://... or https://...)
        - Internal container paths (/tmp/viralify/videos/...)
        - Presentation-generator paths (http://presentation-generator:8006/files/...)

        Uses resilient HTTP client with automatic retry for network issues.
        """
        from urllib.parse import urlparse

        # Convert internal paths to proper service URLs (normalized)
        download_url = self._resolve_video_url(url)

        print(f"[COMPOSITOR] Downloading video: {download_url}", flush=True)

        # Parse URL to extract path properly
        parsed = urlparse(download_url)
        path = parsed.path

        # Use appropriate resilient client based on normalized URL
        if self.presentation_generator_url in download_url:
            response = await self.presentation_client.get(path)
        elif self.media_generator_url in download_url:
            response = await self.media_client.get(path)
        else:
            # External URL - use httpx directly with retry
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(download_url)

        response.raise_for_status()
        return response.content

    def _resolve_video_url(self, url: str) -> str:
        """
        Convert any URL format to an internal Docker-accessible URL.

        Handles normalization of URL variants:
        - localhost:8004 -> configured media_generator_url
        - 127.0.0.1:8004 -> configured media_generator_url
        - media-generator:8004 -> configured media_generator_url
        - Internal file paths -> HTTP URLs
        """
        if not url:
            return url

        # Already a full HTTP URL - normalize to configured URLs
        if url.startswith("http://") or url.startswith("https://"):
            # Normalize all variants of media-generator URL
            for old_host in [
                "http://media-generator:8004",
                "http://localhost:8004",
                "http://127.0.0.1:8004"
            ]:
                if old_host in url:
                    url = url.replace(old_host, self.media_generator_url)
                    break

            # Normalize all variants of presentation-generator URL
            for old_host in [
                "http://presentation-generator:8006",
                "http://localhost:8006",
                "http://127.0.0.1:8006"
            ]:
                if old_host in url:
                    url = url.replace(old_host, self.presentation_generator_url)
                    break

            return url

        # Internal path from media-generator: /tmp/viralify/videos/xxx.mp4
        if url.startswith("/tmp/viralify/videos/"):
            filename = url.split("/")[-1]
            return f"{self.media_generator_url}/files/videos/{filename}"

        # Internal path from presentation-generator: /tmp/presentations/...
        if url.startswith("/tmp/presentations/"):
            # Extract the relative path after /tmp/presentations/
            relative_path = url.replace("/tmp/presentations/", "")
            return f"{self.presentation_generator_url}/files/presentations/{relative_path}"

        # Generic local path - try media-generator
        if url.startswith("/"):
            filename = url.split("/")[-1]
            return f"{self.media_generator_url}/files/videos/{filename}"

        # Return as-is if none of the above
        return url

    def _generate_readme(self, metadata: dict) -> str:
        """Generate a README.md for the course ZIP"""
        sections_text = ""
        for idx, section in enumerate(metadata["sections"], 1):
            sections_text += f"\n### Section {idx}: {section['title']}\n"
            sections_text += f"{section['description']}\n\n"
            for lec_idx, lecture in enumerate(section["lectures"], 1):
                sections_text += f"- **Lecture {idx}.{lec_idx}**: {lecture['title']}\n"
                sections_text += f"  - Duration: {lecture['duration_seconds'] // 60} minutes\n"
                sections_text += f"  - Difficulty: {lecture['difficulty']}\n"

        return f"""# {metadata['title']}

{metadata['description']}

## Course Information

- **Target Audience**: {metadata['target_audience']}
- **Programming Language**: {metadata['language']}
- **Difficulty Range**: {metadata['difficulty_range']}
- **Total Duration**: {metadata['total_duration_minutes']} minutes

## Course Structure

{sections_text}

---

Generated by Viralify Course Generator
"""
