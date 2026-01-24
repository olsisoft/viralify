"""
Course Generator Service

Main FastAPI application for generating educational courses
by orchestrating presentation-generator for individual lectures.
"""
import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from models.course_models import (
    GenerateCourseRequest,
    PreviewOutlineRequest,
    PreviewOutlineResponse,
    CourseJob,
    CourseJobResponse,
    CourseStage,
    CourseOutline,
    ReorderRequest,
    ContextQuestionsRequest,
    ContextQuestionsResponse,
    ProfileCategory,
    CourseContext,
)
from models.lesson_elements import (
    LessonElementType,
    LessonElement,
    COMMON_ELEMENTS,
    CATEGORY_ELEMENTS,
    QuizFrequency,
    QuizQuestionType,
    QuizConfig,
    get_elements_for_category,
    get_default_elements_for_category,
)
from services.course_planner import CoursePlanner
from services.course_compositor import CourseCompositor
from services.context_questions import CourseContextBuilder
from services.element_suggester import ElementSuggester
from services.quiz_generator import QuizGenerator, generate_quizzes_for_course
from services.retrieval_service import RAGService
from models.document_models import (
    Document,
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentListResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    DocumentStatus,
)
from models.source_models import (
    Source,
    SourceType,
    SourceStatus,
    CourseSource,
    SourceResponse,
    SourceSuggestion,
    CreateSourceRequest,
    BulkCreateSourceRequest,
    UpdateSourceRequest,
    LinkSourceToCourseRequest,
    BulkLinkSourcesRequest,
    SuggestSourcesRequest,
    SourceListResponse,
    SuggestSourcesResponse,
    CourseSourceResponse,
    CourseSourcesResponse,
)
from models.lecture_components import (
    LectureComponents,
    LectureComponentsResponse,
    SlideComponent,
    SlideComponentResponse,
    UpdateSlideRequest,
    RegenerateSlideRequest,
    RegenerateLectureRequest,
    RegenerateVoiceoverRequest,
    RecomposeVideoRequest,
    RegenerateResponse,
)
from services.source_library import SourceLibraryService
from services.lecture_editor import LectureEditorService
from services.course_queue import CourseQueueService, QueuedCourseJob, get_queue_service

# Redis for queue job status (when USE_QUEUE=true)
import redis.asyncio as aioredis

# Import Multi-Agent System (Legacy)
try:
    from agents.integration import (
        get_multi_agent_orchestrator,
        validate_course_config,
        generate_quality_code,
        MultiAgentOrchestrator,
    )
    MULTI_AGENT_AVAILABLE = True
    print("[STARTUP] Multi-Agent System (legacy) loaded successfully", flush=True)
except ImportError as e:
    print(f"[STARTUP] Multi-Agent System (legacy) not available: {e}", flush=True)
    MULTI_AGENT_AVAILABLE = False
    get_multi_agent_orchestrator = None
    validate_course_config = None
    generate_quality_code = None
    MultiAgentOrchestrator = None

# Import NEW Hierarchical LangGraph Orchestrator
try:
    from agents.orchestrator_graph import (
        CourseOrchestrator,
        get_course_orchestrator,
        create_course_orchestrator,
    )
    from agents.state import (
        OrchestratorState,
        create_orchestrator_state,
        ProductionStatus,
        PlanningStatus,
    )
    NEW_ORCHESTRATOR_AVAILABLE = True
    print("[STARTUP] NEW Hierarchical LangGraph Orchestrator loaded successfully", flush=True)
except ImportError as e:
    print(f"[STARTUP] NEW Hierarchical Orchestrator not available: {e}", flush=True)
    NEW_ORCHESTRATOR_AVAILABLE = False
    CourseOrchestrator = None
    get_course_orchestrator = None
    create_course_orchestrator = None
    OrchestratorState = None
    create_orchestrator_state = None
    ProductionStatus = None
    PlanningStatus = None

# Import CurriculumEnforcer module (Phase 6)
import sys
# Try Docker mount path first, then local development path
for curriculum_path in ['/app/curriculum-enforcer', '../curriculum-enforcer']:
    if curriculum_path not in sys.path:
        sys.path.insert(0, curriculum_path)
try:
    from curriculum_enforcer import (
        CurriculumEnforcerService,
        ContextType as CurriculumContextType,
        LessonContent,
        EnforcementRequest,
    )
    CURRICULUM_ENFORCER_AVAILABLE = True
    print("[STARTUP] CurriculumEnforcer module loaded successfully", flush=True)
except ImportError as e:
    print(f"[STARTUP] CurriculumEnforcer module not available: {e}", flush=True)
    CURRICULUM_ENFORCER_AVAILABLE = False
    CurriculumEnforcerService = None
    CurriculumContextType = None
    LessonContent = None
    EnforcementRequest = None


# In-memory job storage (use Redis in production)
jobs: Dict[str, CourseJob] = {}

# Export for worker access
course_jobs_db = jobs

# Service instances
course_planner: Optional[CoursePlanner] = None
course_compositor: Optional[CourseCompositor] = None
context_builder: Optional[CourseContextBuilder] = None
element_suggester: Optional[ElementSuggester] = None
rag_service: Optional[RAGService] = None
source_library: Optional[SourceLibraryService] = None
curriculum_enforcer: Optional[CurriculumEnforcerService] = None
queue_service: Optional[CourseQueueService] = None
lecture_editor: Optional[LectureEditorService] = None
multi_agent_orchestrator: Optional[MultiAgentOrchestrator] = None
course_orchestrator: Optional[CourseOrchestrator] = None  # NEW hierarchical orchestrator
redis_client: Optional[aioredis.Redis] = None

# Mode flags
USE_QUEUE = os.getenv("USE_QUEUE", "false").lower() == "true"
USE_MULTI_AGENT = os.getenv("USE_MULTI_AGENT", "true").lower() == "true"
USE_NEW_ORCHESTRATOR = os.getenv("USE_NEW_ORCHESTRATOR", "true").lower() == "true"  # Enable new LangGraph orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global course_planner, course_compositor, context_builder, element_suggester, rag_service, source_library, curriculum_enforcer, queue_service, lecture_editor, multi_agent_orchestrator, course_orchestrator, redis_client

    print("[STARTUP] Initializing Course Generator Service...", flush=True)

    # Initialize services
    openai_api_key = os.getenv("OPENAI_API_KEY")
    presentation_generator_url = os.getenv("PRESENTATION_GENERATOR_URL", "http://127.0.0.1:8006")
    media_generator_url = os.getenv("MEDIA_GENERATOR_URL", "http://127.0.0.1:8004")

    course_planner = CoursePlanner(openai_api_key=openai_api_key)
    course_compositor = CourseCompositor(
        presentation_generator_url=presentation_generator_url,
        media_generator_url=media_generator_url,
        max_parallel_lectures=3
    )
    context_builder = CourseContextBuilder()
    element_suggester = ElementSuggester(openai_api_key=openai_api_key)

    # Initialize RAG service (Phase 2)
    vector_backend = os.getenv("VECTOR_BACKEND", "memory")
    document_storage_path = os.getenv("DOCUMENT_STORAGE_PATH", "/tmp/viralify/documents")
    rag_service = RAGService(
        vector_backend=vector_backend,
        storage_path=document_storage_path,
    )

    # Initialize Source Library service
    database_url = os.getenv("DATABASE_URL")
    source_storage_path = os.getenv("SOURCE_STORAGE_PATH", "/tmp/viralify/sources")
    source_library = SourceLibraryService(
        vector_backend=vector_backend,
        storage_path=source_storage_path,
        database_url=database_url,
    )
    await source_library.initialize()

    # Initialize Curriculum Enforcer (Phase 6)
    if CURRICULUM_ENFORCER_AVAILABLE:
        curriculum_enforcer = CurriculumEnforcerService(openai_api_key=openai_api_key)
        print("[STARTUP] Curriculum Enforcer initialized", flush=True)

    # Initialize Lecture Editor Service
    lecture_editor = LectureEditorService(
        presentation_generator_url=presentation_generator_url,
        media_generator_url=media_generator_url
    )
    print("[STARTUP] Lecture Editor initialized", flush=True)

    # Initialize Multi-Agent Orchestrator (legacy - if enabled)
    if USE_MULTI_AGENT and MULTI_AGENT_AVAILABLE and not USE_NEW_ORCHESTRATOR:
        try:
            multi_agent_orchestrator = get_multi_agent_orchestrator()
            print("[STARTUP] Multi-Agent Orchestrator (legacy) initialized", flush=True)
        except Exception as e:
            print(f"[STARTUP] Multi-Agent Orchestrator init failed: {e}", flush=True)
            multi_agent_orchestrator = None
    else:
        print(f"[STARTUP] Legacy Multi-Agent mode: {'disabled' if not USE_MULTI_AGENT else 'superseded by new orchestrator' if USE_NEW_ORCHESTRATOR else 'not available'}", flush=True)

    # Initialize NEW Hierarchical LangGraph Orchestrator
    if USE_NEW_ORCHESTRATOR and NEW_ORCHESTRATOR_AVAILABLE:
        try:
            course_orchestrator = get_course_orchestrator()
            print("[STARTUP] NEW Hierarchical LangGraph Orchestrator initialized", flush=True)
        except Exception as e:
            print(f"[STARTUP] NEW Orchestrator init failed: {e}", flush=True)
            course_orchestrator = None
    else:
        print(f"[STARTUP] New Orchestrator mode: {'disabled' if not USE_NEW_ORCHESTRATOR else 'not available'}", flush=True)

    # Initialize RabbitMQ Queue (if enabled)
    if USE_QUEUE:
        try:
            queue_service = get_queue_service()
            await queue_service.connect()
            print("[STARTUP] RabbitMQ Queue connected", flush=True)
        except Exception as e:
            print(f"[STARTUP] RabbitMQ Queue connection failed: {e}", flush=True)
            print("[STARTUP] Falling back to in-process background tasks", flush=True)
            queue_service = None
    else:
        print("[STARTUP] Queue mode disabled, using in-process background tasks", flush=True)

    # Initialize Redis client for reading worker job status (when queue mode is enabled)
    if USE_QUEUE:
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/7")
            redis_client = aioredis.from_url(redis_url)
            await redis_client.ping()
            print(f"[STARTUP] Redis connected for job status sync", flush=True)
        except Exception as e:
            print(f"[STARTUP] Redis connection failed: {e}", flush=True)
            redis_client = None

    print(f"[STARTUP] Presentation Generator URL: {presentation_generator_url}", flush=True)
    print(f"[STARTUP] Media Generator URL: {media_generator_url}", flush=True)
    print(f"[STARTUP] RAG Service initialized (backend: {vector_backend})", flush=True)
    print(f"[STARTUP] Source Library initialized (DB: {'PostgreSQL' if database_url else 'in-memory'})", flush=True)
    print(f"[STARTUP] Queue Mode: {'enabled' if USE_QUEUE and queue_service else 'disabled'}", flush=True)
    print(f"[STARTUP] Legacy Multi-Agent Mode: {'enabled' if USE_MULTI_AGENT and multi_agent_orchestrator and not USE_NEW_ORCHESTRATOR else 'disabled'}", flush=True)
    print(f"[STARTUP] NEW Hierarchical Orchestrator: {'ENABLED' if USE_NEW_ORCHESTRATOR and course_orchestrator else 'disabled'}", flush=True)
    print("[STARTUP] Course Generator Service ready!", flush=True)

    yield

    # Cleanup
    if source_library:
        await source_library.close()

    if lecture_editor:
        await lecture_editor.close()

    if queue_service:
        await queue_service.disconnect()

    if redis_client:
        await redis_client.close()

    print("[SHUTDOWN] Course Generator Service shutting down...", flush=True)


app = FastAPI(
    title="Course Generator Service",
    description="Generate educational courses with multiple video lectures",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# URL CONVERSION HELPERS
# =============================================================================

# External URLs for browser access (configurable via env vars)
EXTERNAL_MEDIA_URL = os.getenv("EXTERNAL_MEDIA_URL", "http://localhost:8004")
EXTERNAL_PRESENTATION_URL = os.getenv("EXTERNAL_PRESENTATION_URL", "http://localhost:8006")

# Log configured external URLs at startup
print(f"[CONFIG] EXTERNAL_MEDIA_URL = {EXTERNAL_MEDIA_URL}", flush=True)
print(f"[CONFIG] EXTERNAL_PRESENTATION_URL = {EXTERNAL_PRESENTATION_URL}", flush=True)


def convert_internal_url_to_external(url: str) -> str:
    """
    Convert Docker internal URLs and local file paths to external URLs accessible from browser.

    Internal URLs like:
        http://media-generator:8004/files/videos/xxx.mp4
        http://localhost:8004/files/videos/xxx.mp4
        http://presentation-generator:8006/files/presentations/xxx.mp4

    Local file paths like:
        /tmp/viralify/videos/xxx.mp4
        /tmp/presentations/xxx.mp4

    Are converted to external URLs based on EXTERNAL_MEDIA_URL and EXTERNAL_PRESENTATION_URL.
    """
    if not url:
        return url

    # Replace all variants of media-generator URLs
    for old_url in [
        "http://media-generator:8004",
        "http://localhost:8004",
        "http://127.0.0.1:8004",
    ]:
        if old_url in url:
            url = url.replace(old_url, EXTERNAL_MEDIA_URL)
            break

    # Replace all variants of presentation-generator URLs
    for old_url in [
        "http://presentation-generator:8006",
        "http://localhost:8006",
        "http://127.0.0.1:8006",
    ]:
        if old_url in url:
            url = url.replace(old_url, EXTERNAL_PRESENTATION_URL)
            break

    # Handle local file paths (convert to HTTP URLs)
    if url.startswith("/tmp/viralify/videos/"):
        filename = url.replace("/tmp/viralify/videos/", "")
        url = f"{EXTERNAL_MEDIA_URL}/files/videos/{filename}"
    elif url.startswith("/tmp/presentations/"):
        filename = url.replace("/tmp/presentations/", "")
        url = f"{EXTERNAL_PRESENTATION_URL}/files/presentations/{filename}"

    return url


def convert_job_urls_for_response(job: CourseJob) -> CourseJob:
    """Convert all internal URLs in a job to external URLs for the frontend."""
    # Convert output_urls
    job.output_urls = [convert_internal_url_to_external(url) for url in job.output_urls]

    # Convert lecture video_urls in outline
    if job.outline and job.outline.sections:
        for section in job.outline.sections:
            for lecture in section.lectures:
                if lecture.video_url:
                    lecture.video_url = convert_internal_url_to_external(lecture.video_url)

    return job


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "course-generator",
        "timestamp": datetime.utcnow().isoformat()
    }


# =============================================================================
# CONTEXT QUESTIONS ENDPOINT
# =============================================================================

@app.post("/api/v1/courses/context-questions", response_model=ContextQuestionsResponse)
async def get_context_questions(request: ContextQuestionsRequest):
    """
    Get contextual questions based on profile category.
    Optionally generates AI-powered questions specific to the topic.
    """
    if not context_builder:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        print(f"[CONTEXT] Getting questions for category: {request.category.value}", flush=True)

        # Get base questions for category
        base_questions = context_builder.get_base_questions(request.category)

        # Optionally generate AI questions
        ai_questions = []
        if request.generate_ai_questions and request.topic:
            print(f"[CONTEXT] Generating AI questions for topic: {request.topic}", flush=True)
            ai_questions = await context_builder.generate_topic_questions(
                topic=request.topic,
                category=request.category,
                existing_questions=base_questions
            )

        return ContextQuestionsResponse(
            category=request.category,
            base_questions=base_questions,
            ai_questions=ai_questions
        )

    except Exception as e:
        print(f"[CONTEXT] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/courses/context-questions/{niche}")
async def get_questions_by_niche(niche: str, topic: Optional[str] = None, generate_ai: bool = False):
    """
    Get contextual questions by detecting category from niche name.
    Convenience endpoint that auto-detects the category.
    """
    if not context_builder:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Detect category from niche
        category = context_builder.get_category_from_niche(niche)
        print(f"[CONTEXT] Detected category '{category.value}' from niche '{niche}'", flush=True)

        # Get base questions
        base_questions = context_builder.get_base_questions(category)

        # Optionally generate AI questions
        ai_questions = []
        if generate_ai and topic:
            ai_questions = await context_builder.generate_topic_questions(
                topic=topic,
                category=category,
                existing_questions=base_questions
            )

        return ContextQuestionsResponse(
            category=category,
            base_questions=base_questions,
            ai_questions=ai_questions
        )

    except Exception as e:
        print(f"[CONTEXT] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# COURSE GENERATION ENDPOINTS
# =============================================================================

@app.post("/api/v1/courses/preview-outline", response_model=PreviewOutlineResponse)
async def preview_outline(request: PreviewOutlineRequest):
    """
    Preview course outline/curriculum before generation.
    Returns a structured outline that can be modified via drag & drop.
    If document_ids are provided, RAG context will be fetched and returned
    to avoid double-fetching during generation (optimization).
    """
    if not course_planner:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        print(f"[PREVIEW] Generating outline for: {request.topic}", flush=True)

        # Fetch RAG context if documents are provided
        rag_context = None
        if request.document_ids and rag_service:
            print(f"[PREVIEW] Fetching RAG context from {len(request.document_ids)} documents", flush=True)
            user_id = request.profile_id or "anonymous"
            rag_context = await rag_service.get_context_for_course_generation(
                topic=request.topic,
                description=request.description,
                document_ids=request.document_ids,
                user_id=user_id,
                max_tokens=6000,  # More context for outline generation
            )
            request.rag_context = rag_context
            print(f"[PREVIEW] RAG context fetched: {len(rag_context)} chars", flush=True)

        outline = await course_planner.generate_outline(request)
        print(f"[PREVIEW] Generated outline: {outline.section_count} sections, {outline.total_lectures} lectures", flush=True)

        # OPTIMIZED: Return outline with RAG context for reuse in generate
        return PreviewOutlineResponse(outline=outline, rag_context=rag_context)
    except Exception as e:
        print(f"[PREVIEW] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/courses/generate", response_model=CourseJobResponse)
async def generate_course(
    request: GenerateCourseRequest,
    background_tasks: BackgroundTasks,
    curriculum_context: Optional[str] = None
):
    """
    Start course generation.
    Returns job ID for tracking progress.

    Parameters:
    - curriculum_context: Optional curriculum context type for structure enforcement.
      Options: "education" (default), "enterprise", "bootcamp", "tutorial", "workshop", "certification"
    """
    if not course_planner or not course_compositor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Debug: Log incoming document_ids
    print(f"[GENERATE] Received document_ids: {request.document_ids}", flush=True)

    # Fetch RAG context if documents are provided (same as preview_outline)
    if request.document_ids and rag_service:
        print(f"[GENERATE] Fetching RAG context from {len(request.document_ids)} documents", flush=True)
        user_id = request.profile_id or "anonymous"
        rag_context = await rag_service.get_context_for_course_generation(
            topic=request.topic,
            description=request.description,
            document_ids=request.document_ids,
            user_id=user_id,
            max_tokens=6000,
        )
        request.rag_context = rag_context
        print(f"[GENERATE] RAG context fetched: {len(rag_context) if rag_context else 0} chars", flush=True)

    # Create job
    job = CourseJob(request=request)
    jobs[job.job_id] = job

    # Store curriculum context for later use
    job.curriculum_context = curriculum_context

    print(f"[GENERATE] Starting course generation job: {job.job_id}", flush=True)
    print(f"[GENERATE] Topic: {request.topic}", flush=True)
    print(f"[GENERATE] Structure: {request.structure.number_of_sections} sections x {request.structure.lectures_per_section} lectures", flush=True)
    if curriculum_context:
        print(f"[GENERATE] Curriculum context: {curriculum_context}", flush=True)

    # Use queue if available, otherwise fall back to background tasks
    if USE_QUEUE and queue_service:
        try:
            # Create queued job
            queued_job = QueuedCourseJob(
                job_id=job.job_id,
                topic=request.topic,
                num_sections=request.structure.number_of_sections,
                lectures_per_section=request.structure.lectures_per_section,
                user_id=request.profile_id or "anonymous",
                difficulty_start=request.difficulty_start.value if request.difficulty_start else "beginner",
                difficulty_end=request.difficulty_end.value if request.difficulty_end else "intermediate",
                target_audience=(
                    request.context.profile_audience_description
                    if request.context and request.context.profile_audience_description
                    else f"{request.context.profile_audience_level} learners" if request.context and request.context.profile_audience_level
                    else "general"
                ),
                language=request.language or "en",
                category=request.context.category.value if request.context and request.context.category else "education",
                domain=request.context.specific_tools if request.context else None,
                quiz_config=request.quiz_config.model_dump() if request.quiz_config else None,
                document_ids=request.document_ids,
                priority=5,
            )

            success = await queue_service.publish(queued_job)
            if success:
                print(f"[GENERATE] Job {job.job_id} queued successfully", flush=True)
            else:
                # Fall back to background tasks if queue publish fails
                print(f"[GENERATE] Queue publish failed, using background task", flush=True)
                background_tasks.add_task(run_course_generation, job.job_id)
        except Exception as e:
            print(f"[GENERATE] Queue error: {e}, using background task", flush=True)
            background_tasks.add_task(run_course_generation, job.job_id)
    else:
        # Use in-process background task
        background_tasks.add_task(run_course_generation, job.job_id)

    return CourseJobResponse(
        job_id=job.job_id,
        status=job.status,
        current_stage=job.current_stage,
        progress=job.progress,
        message="Course generation started" + (" (queued)" if USE_QUEUE and queue_service else ""),
        created_at=job.created_at,
        updated_at=job.updated_at
    )


async def run_course_generation(job_id: str):
    """
    Background task to run course generation.

    Uses the NEW Hierarchical LangGraph Orchestrator if enabled,
    otherwise falls back to the legacy sequential pipeline.
    """
    job = jobs.get(job_id)
    if not job:
        return

    # Check if we should use the new orchestrator
    if USE_NEW_ORCHESTRATOR and course_orchestrator:
        await run_course_generation_with_new_orchestrator(job_id, job)
    else:
        await run_course_generation_legacy(job_id, job)


async def run_course_generation_with_new_orchestrator(job_id: str, job: CourseJob):
    """
    Run course generation using the NEW Hierarchical LangGraph Orchestrator.

    Architecture:
        OrchestratorGraph
            ├── PlanningSubgraph (curriculum planning)
            └── ProductionSubgraph ×N (per lecture with recovery)
    """
    print(f"[JOB:{job_id}] Using NEW Hierarchical LangGraph Orchestrator", flush=True)

    try:
        # Build orchestrator parameters from job request
        params = _extract_orchestrator_params(job)

        # Progress callback to update job status
        def update_progress(stage: str, completed: int, total: int, errors: list,
                           in_progress: int = 0, current_lectures: list = None,
                           lecture_update: dict = None, outline_data: dict = None):
            stage_map = {
                "validating": (CourseStage.PLANNING, 2),
                "planning": (CourseStage.PLANNING, 5),
                "outline_ready": (CourseStage.GENERATING_LECTURES, 10),
                "producing": (CourseStage.GENERATING_LECTURES, 10 + int(80 * completed / max(total, 1))),
                "packaging": (CourseStage.COMPILING, 92),
                "done": (CourseStage.COMPLETED, 100),
            }
            course_stage, progress = stage_map.get(stage, (CourseStage.PLANNING, 0))

            # Handle outline_ready stage - set job.outline BEFORE production starts
            if stage == "outline_ready" and outline_data:
                try:
                    if isinstance(outline_data, dict):
                        job.outline = CourseOutline(**outline_data)
                    else:
                        job.outline = outline_data
                    job.lectures_total = total
                    print(f"[PROGRESS] Outline set on job with {total} lectures", flush=True)
                except Exception as e:
                    print(f"[PROGRESS] Failed to set outline: {e}", flush=True)
                message = f"Outline ready, starting lecture generation..."
                job.update_progress(course_stage, progress, message)
                return

            if stage == "producing" and total > 0:
                job.lectures_completed = completed
                job.lectures_in_progress = in_progress
                job.lectures_total = total
                job.current_lectures = current_lectures or []

                # Update individual lecture status in outline
                if lecture_update and job.outline:
                    lecture_id = lecture_update.get("lecture_id")
                    for section in job.outline.sections:
                        for lecture in section.lectures:
                            if lecture.id == lecture_id:
                                lecture.status = lecture_update.get("status", lecture.status)
                                lecture.current_stage = lecture_update.get("current_stage", lecture.current_stage)
                                lecture.progress_percent = lecture_update.get("progress_percent", lecture.progress_percent)
                                if lecture_update.get("video_url"):
                                    lecture.video_url = lecture_update.get("video_url")
                                if lecture_update.get("error"):
                                    lecture.error = lecture_update.get("error")
                                print(f"[PROGRESS] Updated lecture {lecture_id}: status={lecture.status}", flush=True)
                                break

                if in_progress > 0 and completed == 0:
                    message = f"Generating {in_progress} lecture{'s' if in_progress > 1 else ''} in parallel..."
                elif completed > 0:
                    message = f"Generating lectures... {completed}/{total} completed"
                else:
                    message = f"Starting lecture generation..."
            elif stage == "packaging":
                message = "Preparing course package..."
            elif stage == "done":
                message = "Course generation complete!"
            else:
                message = f"Stage: {stage}"

            job.update_progress(course_stage, progress, message)

        # Run the orchestrator
        result = await course_orchestrator.run(
            job_id=job_id,
            progress_callback=update_progress,
            **params
        )

        # Process the result
        await _process_orchestrator_result(job_id, job, result)

    except Exception as e:
        print(f"[JOB:{job_id}] Orchestrator error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        job.error = str(e)
        job.update_progress(CourseStage.FAILED, job.progress, f"Error: {str(e)}")


def _extract_orchestrator_params(job: CourseJob) -> dict:
    """Extract parameters for the orchestrator from job request."""
    request = job.request

    # Extract lesson elements
    lesson_elements_enabled = {}
    if request.context and hasattr(request.context, 'lesson_elements'):
        lesson_elements_enabled = request.context.lesson_elements or {}

    # Extract quiz config
    quiz_enabled = False
    quiz_frequency = "per_section"
    if request.quiz_config:
        quiz_enabled = request.quiz_config.enabled
        quiz_frequency = request.quiz_config.frequency.value if request.quiz_config.frequency else "per_section"

    return {
        "topic": request.topic,
        "description": request.description,
        "profile_category": request.context.category.value if request.context and request.context.category else "education",
        "difficulty_start": request.difficulty_start.value if request.difficulty_start else "beginner",
        "difficulty_end": request.difficulty_end.value if request.difficulty_end else "intermediate",
        "content_language": request.language or "en",
        "programming_language": request.context.specific_tools if request.context and request.context.specific_tools else None,
        "target_audience": (
            request.context.profile_audience_description
            if request.context and request.context.profile_audience_description
            else f"{request.context.profile_audience_level} learners" if request.context and request.context.profile_audience_level
            else "general learners"
        ),
        "structure": {
            "total_duration_minutes": request.structure.total_duration_minutes if request.structure else 60,
            "number_of_sections": request.structure.number_of_sections if request.structure else 4,
            "lectures_per_section": request.structure.lectures_per_section if request.structure else 3,
            "random_structure": False,
        },
        "lesson_elements": lesson_elements_enabled,  # Used by validator and course_graph
        "quiz_enabled": quiz_enabled,
        "quiz_frequency": quiz_frequency,
        "rag_context": request.rag_context,
        "document_ids": request.document_ids or [],
        "voice_id": request.voice_id if hasattr(request, 'voice_id') else "default",
        "style": request.style if hasattr(request, 'style') else "modern",
        "typing_speed": (request.typing_speed.value if hasattr(request.typing_speed, 'value') else request.typing_speed) if hasattr(request, 'typing_speed') and request.typing_speed else "natural",
        "include_avatar": request.include_avatar if hasattr(request, 'include_avatar') else False,
        "avatar_id": request.avatar_id if hasattr(request, 'avatar_id') else None,
    }


async def _process_orchestrator_result(job_id: str, job: CourseJob, result: dict):
    """Process the result from the orchestrator and update job."""
    final_status = result.get("final_status", "failed")
    video_urls = result.get("video_urls", {})
    outline = result.get("outline")
    lectures_completed = result.get("lectures_completed", [])
    lectures_failed = result.get("lectures_failed", [])
    lectures_skipped = result.get("lectures_skipped", [])
    total_lectures = result.get("total_lectures", 0)
    zip_url = result.get("output_zip_url")
    errors = result.get("errors", [])

    # Update job with outline if available
    if outline:
        # Convert dict outline to CourseOutline model if needed
        try:
            from models.course_models import CourseOutline
            if isinstance(outline, dict):
                job.outline = CourseOutline(**outline)
            else:
                job.outline = outline
        except Exception as e:
            print(f"[JOB:{job_id}] Warning: Could not parse outline: {e}", flush=True)

    # Update job metrics
    job.lectures_completed = len(lectures_completed)
    job.lectures_total = total_lectures
    job.failed_lecture_ids = [f.get("lecture_id") for f in lectures_failed]

    # Collect output URLs
    job.output_urls = list(video_urls.values())

    # Set ZIP URL
    if zip_url:
        job.zip_url = zip_url

    # Determine final stage
    if final_status == "success":
        job.update_progress(CourseStage.COMPLETED, 100, "Course generation complete!")
        print(f"[JOB:{job_id}] Course completed: {len(job.output_urls)} videos", flush=True)

    elif final_status == "partial":
        success_count = len(lectures_completed)
        failed_count = len(lectures_failed)
        skipped_count = len(lectures_skipped)
        job.update_progress(
            CourseStage.PARTIAL_SUCCESS,
            100,
            f"Course partially complete: {success_count}/{total_lectures} lectures. "
            f"{failed_count} failed, {skipped_count} skipped."
        )
        print(f"[JOB:{job_id}] PARTIAL SUCCESS: {success_count}/{total_lectures} videos", flush=True)

    else:
        error_msg = errors[0] if errors else "Unknown error"
        job.error = error_msg
        job.update_progress(CourseStage.FAILED, 100, f"Course generation failed: {error_msg}")
        print(f"[JOB:{job_id}] FAILED: {error_msg}", flush=True)


async def run_course_generation_legacy(job_id: str, job: CourseJob):
    """
    Legacy course generation pipeline (sequential).

    Kept for backward compatibility when USE_NEW_ORCHESTRATOR=false.
    """
    print(f"[JOB:{job_id}] Using LEGACY sequential pipeline", flush=True)

    try:
        # Stage 0: Multi-Agent Validation & Enrichment (if enabled)
        enrichment_result = None
        if USE_MULTI_AGENT and multi_agent_orchestrator:
            job.update_progress(CourseStage.PLANNING, 2, "Validating configuration...")
            print(f"[JOB:{job_id}] Running multi-agent validation...", flush=True)

            try:
                # Extract lesson elements from request
                lesson_elements = {}
                if job.request.context and hasattr(job.request.context, 'lesson_elements'):
                    lesson_elements = job.request.context.lesson_elements or {}

                # Extract quiz config
                quiz_config = None
                if job.request.quiz_config:
                    quiz_config = {
                        "enabled": job.request.quiz_config.enabled,
                        "frequency": job.request.quiz_config.frequency.value if job.request.quiz_config.frequency else "per_section",
                        "question_types": [qt.value for qt in (job.request.quiz_config.question_types or [])],
                    }

                enrichment_result = await multi_agent_orchestrator.validate_and_enrich(
                    job_id=job_id,
                    topic=job.request.topic,
                    description=job.request.description,
                    profile_category=job.request.context.category.value if job.request.context and job.request.context.category else "education",
                    difficulty_start=job.request.difficulty_start.value if job.request.difficulty_start else "beginner",
                    difficulty_end=job.request.difficulty_end.value if job.request.difficulty_end else "intermediate",
                    content_language=job.request.language or "en",
                    programming_language=job.request.context.specific_tools if job.request.context and job.request.context.specific_tools else "python",
                    target_audience=(
                        job.request.context.profile_audience_description
                        if job.request.context and job.request.context.profile_audience_description
                        else None
                    ),
                    structure={
                        "number_of_sections": job.request.structure.number_of_sections,
                        "lectures_per_section": job.request.structure.lectures_per_section,
                        "total_duration_minutes": job.request.structure.total_duration_minutes,
                    } if job.request.structure else None,
                    lesson_elements=lesson_elements,
                    quiz_config=quiz_config,
                    document_ids=job.request.document_ids,
                    rag_context=job.request.rag_context,
                )

                if not enrichment_result.get("validated"):
                    errors = enrichment_result.get("validation_errors", [])
                    print(f"[JOB:{job_id}] Multi-agent validation warnings: {len(errors)} issues", flush=True)
                    for err in errors[:5]:
                        print(f"[JOB:{job_id}]   - {err.get('field', 'unknown')}: {err.get('message', 'error')}", flush=True)
                else:
                    print(f"[JOB:{job_id}] Multi-agent validation PASSED", flush=True)

                if enrichment_result.get("code_expert_prompt"):
                    job.code_expert_prompt = enrichment_result["code_expert_prompt"]
                    print(f"[JOB:{job_id}] Code expert prompt enriched", flush=True)

                suggestions = enrichment_result.get("suggestions", [])
                if suggestions:
                    print(f"[JOB:{job_id}] Configuration suggestions:", flush=True)
                    for sug in suggestions[:3]:
                        print(f"[JOB:{job_id}]   - {sug}", flush=True)

            except Exception as e:
                print(f"[JOB:{job_id}] Multi-agent validation error (non-blocking): {e}", flush=True)

        # Stage 1: Planning (0-10%)
        job.update_progress(CourseStage.PLANNING, 5, "Generating course curriculum...")

        rag_context = job.request.rag_context
        if rag_context:
            print(f"[JOB:{job_id}] Using pre-fetched RAG context: {len(rag_context)} chars", flush=True)
        elif job.request.document_ids and rag_service:
            print(f"[JOB:{job_id}] Fetching RAG context from {len(job.request.document_ids)} documents", flush=True)
            rag_context = await rag_service.get_context_for_course_generation(
                topic=job.request.topic,
                description=job.request.description,
                document_ids=job.request.document_ids,
                user_id=job.request.profile_id,
                max_tokens=6000,
            )
            print(f"[JOB:{job_id}] RAG context fetched: {len(rag_context)} chars", flush=True)
            job.request.rag_context = rag_context

        if job.request.approved_outline:
            outline = job.request.approved_outline
            print(f"[JOB:{job_id}] Using pre-approved outline", flush=True)
        else:
            preview_request = PreviewOutlineRequest(
                profile_id=job.request.profile_id,
                topic=job.request.topic,
                description=job.request.description,
                difficulty_start=job.request.difficulty_start,
                difficulty_end=job.request.difficulty_end,
                structure=job.request.structure,
                context=job.request.context,
                language=job.request.language or "en",
                document_ids=job.request.document_ids,
                rag_context=rag_context,
            )
            outline = await course_planner.generate_outline(preview_request)

        # Apply Curriculum Enforcer if available
        curriculum_ctx = getattr(job, 'curriculum_context', None)
        if CURRICULUM_ENFORCER_AVAILABLE and curriculum_enforcer and curriculum_ctx:
            print(f"[JOB:{job_id}] Applying curriculum enforcement: {curriculum_ctx}", flush=True)
            try:
                ctx = CurriculumContextType(curriculum_ctx)
                for section in outline.sections:
                    for lecture in section.lectures:
                        lesson_content = LessonContent(
                            lesson_id=lecture.id,
                            title=lecture.title,
                            slides=lecture.slides if hasattr(lecture, 'slides') else [],
                            lesson_type=lecture.lecture_type if hasattr(lecture, 'lecture_type') else None,
                        )
                        result = await curriculum_enforcer.enforce(
                            EnforcementRequest(
                                content=lesson_content,
                                context_type=ctx,
                                auto_fix=True,
                                preserve_content=True,
                            )
                        )
                        if result.restructured_content and result.changes_made:
                            if hasattr(lecture, 'slides'):
                                lecture.slides = result.restructured_content.slides
                            print(f"[JOB:{job_id}] Restructured lecture '{lecture.title}': {result.changes_made}", flush=True)
                print(f"[JOB:{job_id}] Curriculum enforcement complete", flush=True)
            except Exception as e:
                print(f"[JOB:{job_id}] Curriculum enforcement warning: {str(e)}", flush=True)

        job.outline = outline
        job.lectures_total = outline.total_lectures
        job.update_progress(CourseStage.PLANNING, 10, f"Curriculum ready: {outline.total_lectures} lectures")
        print(f"[JOB:{job_id}] Outline ready: {outline.section_count} sections, {outline.total_lectures} lectures", flush=True)

        # Stage 2: Generate lectures (10-90%)
        job.update_progress(CourseStage.GENERATING_LECTURES, 10, "Generating lectures...")

        final_stage = await course_compositor.generate_all_lectures(
            job=job,
            request=job.request,
            progress_callback=lambda completed, total, title: job.update_lecture_progress(completed, total, title)
        )

        if final_stage == CourseStage.FAILED:
            job.update_progress(CourseStage.FAILED, job.progress, "All lectures failed to generate")
            print(f"[JOB:{job_id}] All lectures failed - course generation aborted", flush=True)
            return

        for section in job.outline.sections:
            for lecture in section.lectures:
                if lecture.video_url:
                    job.output_urls.append(lecture.video_url)

        # Stage 2.5: Generate quizzes if enabled
        if job.request.quiz_config and job.request.quiz_config.enabled:
            job.update_progress(CourseStage.GENERATING_LECTURES, 88, "Generating quizzes...")
            print(f"[JOB:{job_id}] Generating quizzes (frequency: {job.request.quiz_config.frequency.value})...", flush=True)
            try:
                quizzes = await generate_quizzes_for_course(
                    outline=job.outline,
                    config=job.request.quiz_config,
                )
                for lecture_id, quiz in quizzes.get("lecture_quizzes", {}).items():
                    for section in job.outline.sections:
                        for lecture in section.lectures:
                            if lecture.id == lecture_id:
                                lecture.quiz = quiz
                                print(f"[JOB:{job_id}] Quiz attached to lecture: {lecture.title}", flush=True)
                for section_id, quiz in quizzes.get("section_quizzes", {}).items():
                    for section in job.outline.sections:
                        if section.id == section_id:
                            section.quiz = quiz
                            print(f"[JOB:{job_id}] Quiz attached to section: {section.title}", flush=True)
                if quizzes.get("final_quiz"):
                    job.outline.final_quiz = quizzes["final_quiz"]
                    print(f"[JOB:{job_id}] Final course quiz generated", flush=True)
                print(f"[JOB:{job_id}] Quiz generation complete", flush=True)
            except Exception as e:
                print(f"[JOB:{job_id}] Quiz generation warning: {str(e)}", flush=True)

        # Stage 3: Compiling (90-95%)
        job.update_progress(CourseStage.COMPILING, 92, "Preparing course package...")

        if job.output_urls:
            zip_url = await course_compositor.create_course_zip(job)
            job.zip_url = zip_url

        # Stage 4: Determine final status
        if final_stage == CourseStage.PARTIAL_SUCCESS:
            failed_count = len(job.failed_lecture_ids)
            total_count = job.lectures_total
            success_count = job.lectures_completed
            job.update_progress(
                CourseStage.PARTIAL_SUCCESS,
                100,
                f"Course partially complete: {success_count}/{total_count} lectures generated. {failed_count} lectures can be regenerated."
            )
            print(f"[JOB:{job_id}] PARTIAL SUCCESS: {success_count}/{total_count} videos, {failed_count} failed", flush=True)
        else:
            job.update_progress(CourseStage.COMPLETED, 100, "Course generation complete!")
            print(f"[JOB:{job_id}] Course completed: {len(job.output_urls)} videos", flush=True)

    except Exception as e:
        print(f"[JOB:{job_id}] Error: {str(e)}", flush=True)
        job.error = str(e)
        job.update_progress(CourseStage.FAILED, job.progress, f"Error: {str(e)}")


@app.get("/api/v1/courses/jobs/{job_id}", response_model=CourseJobResponse)
async def get_job_status(job_id: str):
    """Get status of a course generation job"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Sync job status from Redis when queue mode is enabled
    # The worker updates Redis, so we need to read from there
    if USE_QUEUE and redis_client:
        try:
            redis_data = await redis_client.hgetall(f"course_job:{job_id}")
            if redis_data:
                # Decode bytes to string
                redis_data = {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v for k, v in redis_data.items()}

                # Map Redis status to CourseStage
                status_map = {
                    "queued": CourseStage.PLANNING,
                    "generating_outline": CourseStage.PLANNING,
                    "generating_lectures": CourseStage.GENERATING_LECTURES,
                    "creating_package": CourseStage.COMPILING,
                    "completed": CourseStage.COMPLETED,
                    "failed": CourseStage.FAILED,
                }

                redis_status = redis_data.get("status", "")
                if redis_status in status_map:
                    new_stage = status_map[redis_status]
                    progress = float(redis_data.get("progress", 0))

                    # Update in-memory job with Redis state
                    job.current_stage = new_stage
                    job.progress = progress
                    job.updated_at = datetime.fromisoformat(redis_data.get("updated_at", datetime.utcnow().isoformat()))

                    # Handle completion
                    if redis_status == "completed":
                        output_urls_str = redis_data.get("output_urls", "{}")
                        if output_urls_str and output_urls_str != "{}":
                            import json
                            try:
                                output_data = json.loads(output_urls_str.replace("'", '"'))
                                if isinstance(output_data, dict):
                                    job.output_urls = output_data.get("videos", [])
                                    job.zip_url = output_data.get("zip")
                            except:
                                pass
                        job.message = "Course generation complete!"

                    # Handle error
                    if redis_data.get("error"):
                        job.error = redis_data.get("error")

                    # Build progress message
                    if redis_status == "generating_lectures":
                        job.message = f"Generating lectures... ({progress:.0f}%)"
                    elif redis_status == "creating_package":
                        job.message = "Creating course package..."

        except Exception as e:
            print(f"[STATUS] Redis sync error for {job_id}: {e}", flush=True)
            # Continue with in-memory state if Redis fails

    return job_to_response(job)


@app.post("/api/v1/courses/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a running course generation job.

    Marks the job for cancellation. Running lectures will be stopped
    at the next checkpoint.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.current_stage in [CourseStage.COMPLETED, CourseStage.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in {job.current_stage.value} state"
        )

    # Mark job for cancellation in compositor
    if course_compositor:
        course_compositor.cancel_job(job_id)

    # Update job status
    job.update_progress(CourseStage.FAILED, job.progress, "Job cancelled by user")
    job.error = "Cancelled by user"

    print(f"[CANCEL] Job {job_id} cancelled by user", flush=True)

    return {
        "message": "Job cancellation requested",
        "job_id": job_id,
        "status": "cancelling"
    }


def job_to_response(job: CourseJob) -> CourseJobResponse:
    """Convert a CourseJob to CourseJobResponse with external URLs."""
    import copy

    # Convert internal Docker URLs to external URLs for browser access
    external_output_urls = [convert_internal_url_to_external(url) for url in job.output_urls]

    # Deep copy outline and convert lecture video URLs
    outline_copy = None
    if job.outline:
        outline_copy = copy.deepcopy(job.outline)
        for section in outline_copy.sections:
            for lecture in section.lectures:
                if lecture.video_url:
                    lecture.video_url = convert_internal_url_to_external(lecture.video_url)

    return CourseJobResponse(
        job_id=job.job_id,
        status=job.status,
        current_stage=job.current_stage,
        progress=job.progress,
        message=job.message,
        outline=outline_copy,
        lectures_total=job.lectures_total,
        lectures_completed=job.lectures_completed,
        lectures_in_progress=job.lectures_in_progress,
        lectures_failed=job.lectures_failed,
        current_lecture_title=job.current_lecture_title,
        current_lectures=job.current_lectures,
        output_urls=external_output_urls,
        zip_url=job.zip_url,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        error=job.error,
        failed_lecture_ids=job.failed_lecture_ids,
        failed_lecture_errors=job.failed_lecture_errors,
        is_partial_success=job.is_partial_success(),
        can_download_partial=job.lectures_completed > 0 and job.lectures_failed > 0
    )


@app.get("/api/v1/courses/jobs", response_model=List[CourseJobResponse])
async def list_jobs(limit: int = 20, offset: int = 0):
    """List all course generation jobs"""
    all_jobs = list(jobs.values())
    # Sort by created_at descending
    all_jobs.sort(key=lambda j: j.created_at, reverse=True)
    # Paginate
    paginated = all_jobs[offset:offset + limit]

    return [job_to_response(job) for job in paginated]


@app.delete("/api/v1/courses/jobs")
async def clear_job_history(keep_active: bool = True):
    """
    Clear job history.

    Parameters:
    - keep_active: If True (default), only removes completed/failed jobs.
                   If False, removes all jobs (use with caution).
    """
    global jobs

    if keep_active:
        # Only remove completed/failed jobs
        completed_stages = [CourseStage.COMPLETED, CourseStage.FAILED]
        jobs_to_remove = [
            job_id for job_id, job in jobs.items()
            if job.current_stage in completed_stages
        ]
    else:
        # Remove all jobs
        jobs_to_remove = list(jobs.keys())

    removed_count = len(jobs_to_remove)
    for job_id in jobs_to_remove:
        del jobs[job_id]

    print(f"[CLEANUP] Removed {removed_count} jobs from history", flush=True)

    return {
        "message": f"Cleared {removed_count} jobs from history",
        "removed_count": removed_count,
        "remaining_count": len(jobs)
    }


@app.get("/api/v1/courses/queue/stats")
async def get_queue_stats():
    """
    Get queue statistics.

    Returns pending jobs, active consumers, and failed jobs count.
    """
    if not USE_QUEUE or not queue_service:
        return {
            "queue_enabled": False,
            "message": "Queue mode is disabled. Jobs are processed in-process."
        }

    try:
        stats = await queue_service.get_queue_stats()
        stats["queue_enabled"] = True
        return stats
    except Exception as e:
        return {
            "queue_enabled": True,
            "error": str(e),
            "message": "Failed to fetch queue stats"
        }


@app.post("/api/v1/courses/queue/retry/{job_id}")
async def retry_failed_job(job_id: str):
    """
    Retry a failed job by moving it from the dead letter queue back to the main queue.
    """
    if not USE_QUEUE or not queue_service:
        raise HTTPException(
            status_code=400,
            detail="Queue mode is disabled"
        )

    try:
        success = await queue_service.requeue_failed_job(job_id)
        if success:
            return {"message": f"Job {job_id} requeued for retry"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found in dead letter queue"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/courses/{job_id}/reorder")
async def reorder_outline(job_id: str, request: ReorderRequest):
    """
    Reorder sections and lectures via drag & drop.
    Only works for jobs in PLANNING stage or before generation starts.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.current_stage not in [CourseStage.QUEUED, CourseStage.PLANNING]:
        raise HTTPException(
            status_code=400,
            detail="Cannot reorder after generation has started"
        )

    if not job.outline:
        raise HTTPException(status_code=400, detail="No outline to reorder")

    # Rebuild sections from reorder request
    try:
        section_map = {s.id: s for s in job.outline.sections}
        lecture_map = {}
        for section in job.outline.sections:
            for lecture in section.lectures:
                lecture_map[lecture.id] = lecture

        new_sections = []
        for idx, section_data in enumerate(request.sections):
            section_id = section_data.get("id")
            lecture_ids = section_data.get("lectures", [])

            if section_id in section_map:
                section = section_map[section_id]
                section.order = idx
                section.lectures = []
                for lec_idx, lec_id in enumerate(lecture_ids):
                    if lec_id in lecture_map:
                        lecture = lecture_map[lec_id]
                        lecture.order = lec_idx
                        section.lectures.append(lecture)
                new_sections.append(section)

        job.outline.sections = new_sections
        job.updated_at = datetime.utcnow()

        return {"message": "Outline reordered successfully", "outline": job.outline}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/courses/{job_id}/download")
async def download_course(job_id: str):
    """Download course as ZIP file"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Allow download for both COMPLETED and PARTIAL_SUCCESS
    if job.current_stage not in [CourseStage.COMPLETED, CourseStage.PARTIAL_SUCCESS]:
        raise HTTPException(status_code=400, detail="Course not yet completed")

    if not job.zip_url:
        raise HTTPException(status_code=404, detail="ZIP not available")

    # If zip_url is a local file path
    if job.zip_url.startswith("/"):
        return FileResponse(
            path=job.zip_url,
            media_type="application/zip",
            filename=f"course_{job_id}.zip"
        )

    # Otherwise return the URL
    return {"download_url": job.zip_url}


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@app.get("/api/v1/courses/config/difficulty-levels")
async def get_difficulty_levels():
    """Get available difficulty levels"""
    return [
        {"id": "beginner", "name": "Beginner", "description": "No prior experience needed"},
        {"id": "intermediate", "name": "Intermediate", "description": "Some experience required"},
        {"id": "advanced", "name": "Advanced", "description": "Solid understanding required"},
        {"id": "very_advanced", "name": "Very Advanced", "description": "Expert knowledge needed"},
        {"id": "expert", "name": "Expert", "description": "Mastery level content"}
    ]


@app.get("/api/v1/courses/config/lesson-elements")
async def get_lesson_elements():
    """Get available lesson element options (legacy endpoint)"""
    return [
        {"id": "concept_intro", "name": "Concept Introduction", "description": "Start with theory explanation", "default": True},
        {"id": "diagram_schema", "name": "Diagram/Schema", "description": "Visual diagrams and flowcharts", "default": True},
        {"id": "code_typing", "name": "Code Typing Animation", "description": "Show code being typed", "default": True},
        {"id": "code_execution", "name": "Code Execution", "description": "Execute and show output", "default": False},
        {"id": "voiceover_explanation", "name": "Voiceover During Code", "description": "Narration while typing", "default": True},
        {"id": "curriculum_slide", "name": "Curriculum Slide", "description": "Show position in course", "default": True, "readonly": True}
    ]


# =============================================================================
# ADAPTIVE LESSON ELEMENTS ENDPOINTS (Phase 1)
# =============================================================================

@app.get("/api/v1/courses/config/categories")
async def get_categories():
    """Get all available profile categories with their info"""
    return [
        {"id": "tech", "name": "Technique", "icon": "💻", "description": "Programmation, développement, IA, data science"},
        {"id": "business", "name": "Business", "icon": "💼", "description": "Entrepreneuriat, marketing, vente, management"},
        {"id": "health", "name": "Santé/Fitness", "icon": "🏃", "description": "Fitness, nutrition, yoga, bien-être"},
        {"id": "creative", "name": "Créatif", "icon": "🎨", "description": "Design, illustration, vidéo, photo, musique"},
        {"id": "education", "name": "Éducation", "icon": "📚", "description": "Enseignement, langues, sciences, examens"},
        {"id": "lifestyle", "name": "Lifestyle", "icon": "✨", "description": "Productivité, développement personnel, relations"},
    ]


@app.get("/api/v1/courses/config/elements/{category}")
async def get_elements_by_category(category: str):
    """
    Get lesson elements available for a specific category.
    Returns both common elements and category-specific elements.
    """
    try:
        cat = ProfileCategory(category)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

    elements = get_elements_for_category(cat)
    defaults = get_default_elements_for_category(cat)

    return {
        "category": category,
        "common_elements": [
            {
                "id": el.id.value,
                "name": el.name,
                "description": el.description,
                "icon": el.icon,
                "is_required": el.is_required,
                "enabled": defaults.get(el.id, True),
            }
            for el in COMMON_ELEMENTS
        ],
        "category_elements": [
            {
                "id": el.id.value,
                "name": el.name,
                "description": el.description,
                "icon": el.icon,
                "is_required": el.is_required,
                "enabled": defaults.get(el.id, False),
                "presentation_type": el.presentation_type,
            }
            for el in CATEGORY_ELEMENTS.get(cat, [])
        ],
    }


@app.post("/api/v1/courses/config/suggest-elements")
async def suggest_elements(
    topic: str,
    description: Optional[str] = None,
    category: Optional[str] = None,
):
    """
    AI-powered element suggestion based on topic (Option C).
    Analyzes the topic and suggests the most relevant elements.
    """
    if not element_suggester:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        cat = None
        if category:
            try:
                cat = ProfileCategory(category)
            except ValueError:
                pass

        result = await element_suggester.get_smart_defaults(
            topic=topic,
            description=description,
            category=cat,
        )

        return result

    except Exception as e:
        print(f"[SUGGEST] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/courses/config/detect-category")
async def detect_category(topic: str, description: Optional[str] = None):
    """
    Auto-detect the best category, domain, and keywords for a topic using AI.
    Returns comprehensive analysis for course configuration.
    """
    if not element_suggester:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        category, confidence = await element_suggester.detect_category(topic, description)

        # Also detect domain and keywords
        domain_info = await element_suggester.detect_domain_and_keywords(
            topic, description, category
        )

        return {
            "category": category.value,
            "confidence": confidence,
            "domain": domain_info.get("domain"),
            "domain_options": domain_info.get("domain_options", []),
            "keywords": domain_info.get("keywords", []),
        }
    except Exception as e:
        print(f"[DETECT] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# QUIZ CONFIGURATION ENDPOINTS (Phase 1)
# =============================================================================

@app.get("/api/v1/courses/config/quiz-options")
async def get_quiz_options():
    """Get available quiz configuration options"""
    return {
        "frequencies": [
            {"id": "per_lecture", "name": "Par lecture", "description": "Quiz à la fin de chaque lecture"},
            {"id": "per_section", "name": "Par section", "description": "Quiz à la fin de chaque section"},
            {"id": "end_of_course", "name": "Fin de cours", "description": "Un seul quiz final"},
            {"id": "custom", "name": "Personnalisé", "description": "Toutes les N lectures"},
        ],
        "question_types": [
            {"id": "multiple_choice", "name": "QCM", "description": "Une seule bonne réponse"},
            {"id": "multi_select", "name": "Choix multiples", "description": "Plusieurs bonnes réponses"},
            {"id": "true_false", "name": "Vrai/Faux", "description": "Vrai ou faux"},
            {"id": "fill_blank", "name": "Texte à trous", "description": "Compléter le texte"},
            {"id": "matching", "name": "Association", "description": "Associer des éléments"},
        ],
        "defaults": {
            "enabled": True,
            "frequency": "per_section",
            "questions_per_quiz": 5,
            "passing_score": 70,
            "show_explanations": True,
            "question_types": ["multiple_choice", "true_false"],
        },
    }


# =============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS (Phase 2 - RAG)
# =============================================================================

@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    course_id: Optional[str] = Form(None),
):
    """
    Upload a document for RAG processing.

    Supported formats: PDF, DOCX, DOC, PPTX, PPT, TXT, MD, XLSX, XLS, CSV
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        # Read file content
        content = await file.read()

        print(f"[UPLOAD] Receiving file: {file.filename} ({len(content)} bytes) for user {user_id}", flush=True)

        # Process document
        document = await rag_service.upload_document(
            file_content=content,
            filename=file.filename,
            user_id=user_id,
            course_id=course_id,
        )

        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            document_type=document.document_type,
            status=document.status,
            message=f"Document uploaded and processed: {document.chunk_count} chunks created",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[UPLOAD] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/upload-url", response_model=DocumentUploadResponse)
async def upload_from_url(
    url: str = Form(...),
    user_id: str = Form(...),
    course_id: Optional[str] = Form(None),
):
    """
    Upload a document from URL (web page or YouTube video).
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        print(f"[UPLOAD] Fetching URL: {url} for user {user_id}", flush=True)

        document = await rag_service.upload_from_url(
            url=url,
            user_id=user_id,
            course_id=course_id,
        )

        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            document_type=document.document_type,
            status=document.status,
            message=f"URL content processed: {document.chunk_count} chunks created",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[UPLOAD] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents", response_model=DocumentListResponse)
async def list_documents(
    user_id: str,
    course_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """
    List documents for a user, optionally filtered by course.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        documents = await rag_service.list_documents(user_id, course_id)

        # Paginate
        total = len(documents)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = documents[start:end]

        # Remove chunks from response (too large)
        for doc in paginated:
            doc.chunks = []
            doc.raw_content = None  # Also don't return full content

        return DocumentListResponse(
            documents=paginated,
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        print(f"[LIST] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{document_id}")
async def get_document(document_id: str, user_id: str):
    """
    Get a specific document by ID.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        document = await rag_service.get_document(document_id, user_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Remove chunks from response
        document.chunks = []

        return document

    except HTTPException:
        raise
    except Exception as e:
        print(f"[GET] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str, user_id: str):
    """
    Delete a document and its vectors.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        success = await rag_service.delete_document(document_id, user_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found or access denied")

        return {"message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[DELETE] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/query", response_model=RAGQueryResponse)
async def query_documents(request: RAGQueryRequest):
    """
    Query documents using RAG (semantic search).
    Returns relevant chunks and combined context for LLM.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        response = await rag_service.query(request)
        return response

    except Exception as e:
        print(f"[QUERY] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/context/{course_id}")
async def get_course_context(
    course_id: str,
    user_id: str,
    topic: str,
    description: Optional[str] = None,
    max_tokens: int = 4000,
):
    """
    Get RAG context for course generation.
    This is used internally by the course generator.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        # Get documents for course
        documents = await rag_service.list_documents(user_id, course_id)
        document_ids = [d.id for d in documents if d.status == DocumentStatus.READY]

        if not document_ids:
            return {"context": "", "document_count": 0}

        context = await rag_service.get_context_for_course_generation(
            topic=topic,
            description=description,
            document_ids=document_ids,
            user_id=user_id,
            max_tokens=max_tokens,
        )

        return {
            "context": context,
            "document_count": len(document_ids),
        }

    except Exception as e:
        print(f"[CONTEXT] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SOURCE LIBRARY ENDPOINTS (Multi-Source RAG)
# =============================================================================

@app.get("/api/v1/sources", response_model=SourceListResponse)
async def list_sources(
    user_id: str,
    source_type: Optional[SourceType] = None,
    status: Optional[SourceStatus] = None,
    tags: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
):
    """
    List user's sources from the library with filtering and pagination.
    """
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        # Parse tags from comma-separated string
        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        sources, total = await source_library.list_sources(
            user_id=user_id,
            source_type=source_type,
            status=status,
            tags=tag_list,
            search=search,
            page=page,
            page_size=page_size,
        )

        return SourceListResponse(
            sources=[SourceResponse.from_source(s) for s in sources],
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        print(f"[SOURCES] List error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sources/upload", response_model=SourceResponse)
async def upload_source_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    name: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
):
    """
    Upload a file to the source library.
    Supported formats: PDF, DOCX, DOC, PPTX, PPT, TXT, MD, XLSX, XLS, CSV
    """
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        content = await file.read()
        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        print(f"[SOURCES] Uploading file: {file.filename} for user {user_id}", flush=True)

        source = await source_library.create_source_from_file(
            file_content=content,
            filename=file.filename,
            user_id=user_id,
            name=name,
            tags=tag_list,
        )

        return SourceResponse.from_source(source)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[SOURCES] Upload error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sources/url", response_model=SourceResponse)
async def create_source_from_url(
    url: str = Form(...),
    user_id: str = Form(...),
    name: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
):
    """
    Create a source from URL (web page or YouTube video).
    """
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        print(f"[SOURCES] Creating URL source: {url} for user {user_id}", flush=True)

        source = await source_library.create_source_from_url(
            url=url,
            user_id=user_id,
            name=name,
            tags=tag_list,
        )

        return SourceResponse.from_source(source)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[SOURCES] URL error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sources/note", response_model=SourceResponse)
async def create_note_source(
    content: str = Form(...),
    user_id: str = Form(...),
    name: str = Form(...),
    tags: Optional[str] = Form(None),
):
    """
    Create a source from user notes/text.
    """
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        print(f"[SOURCES] Creating note source: {name} for user {user_id}", flush=True)

        source = await source_library.create_note_source(
            content=content,
            user_id=user_id,
            name=name,
            tags=tag_list,
        )

        return SourceResponse.from_source(source)

    except Exception as e:
        print(f"[SOURCES] Note error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sources/bulk")
async def create_sources_bulk(request: BulkCreateSourceRequest):
    """
    Create multiple sources at once (URLs and notes only).
    """
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    results = []
    errors = []

    for source_req in request.sources:
        try:
            if source_req.source_type == SourceType.URL:
                source = await source_library.create_source_from_url(
                    url=source_req.source_url,
                    user_id=request.user_id,
                    name=source_req.name,
                    tags=source_req.tags,
                )
            elif source_req.source_type == SourceType.YOUTUBE:
                source = await source_library.create_source_from_url(
                    url=source_req.source_url,
                    user_id=request.user_id,
                    name=source_req.name,
                    tags=source_req.tags,
                )
            elif source_req.source_type == SourceType.NOTE:
                source = await source_library.create_note_source(
                    content=source_req.note_content or "",
                    user_id=request.user_id,
                    name=source_req.name,
                    tags=source_req.tags,
                )
            else:
                errors.append({"name": source_req.name, "error": "File sources must be uploaded individually"})
                continue

            results.append(SourceResponse.from_source(source))

        except Exception as e:
            errors.append({"name": source_req.name, "error": str(e)})

    return {
        "created": [r.model_dump() for r in results],
        "errors": errors,
        "total_created": len(results),
        "total_errors": len(errors),
    }


@app.get("/api/v1/sources/{source_id}", response_model=SourceResponse)
async def get_source(source_id: str, user_id: str):
    """Get a specific source by ID."""
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        source = await source_library.get_source(source_id, user_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        return SourceResponse.from_source(source)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[SOURCES] Get error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/v1/sources/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: str,
    user_id: str,
    request: UpdateSourceRequest,
):
    """Update source metadata (name, tags, note content)."""
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        source = await source_library.update_source(source_id, user_id, request)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        return SourceResponse.from_source(source)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[SOURCES] Update error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/sources/{source_id}")
async def delete_source(source_id: str, user_id: str):
    """Delete a source from the library."""
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        success = await source_library.delete_source(source_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Source not found or access denied")

        return {"message": "Source deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[SOURCES] Delete error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# COURSE-SOURCE LINKING ENDPOINTS
# =============================================================================

@app.post("/api/v1/courses/{course_id}/sources", response_model=CourseSourceResponse)
async def link_source_to_course(
    course_id: str,
    source_id: str,
    user_id: str,
    is_primary: bool = False,
):
    """Link a source from the library to a course."""
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        link = await source_library.link_to_course(
            course_id=course_id,
            source_id=source_id,
            user_id=user_id,
            is_primary=is_primary,
        )

        if not link:
            raise HTTPException(status_code=400, detail="Source not ready or not found")

        source = await source_library.get_source(source_id, user_id)

        return CourseSourceResponse(
            id=link.id,
            course_id=link.course_id,
            source_id=link.source_id,
            source=SourceResponse.from_source(source),
            relevance_score=link.relevance_score,
            is_primary=link.is_primary,
            added_at=link.added_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[SOURCES] Link error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/courses/{course_id}/sources/bulk")
async def link_sources_bulk(course_id: str, request: BulkLinkSourcesRequest):
    """Link multiple sources to a course at once."""
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    results = []
    errors = []

    for source_id in request.source_ids:
        try:
            link = await source_library.link_to_course(
                course_id=course_id,
                source_id=source_id,
                user_id=request.user_id,
            )

            if link:
                source = await source_library.get_source(source_id, request.user_id)
                results.append({
                    "source_id": source_id,
                    "source_name": source.name if source else "Unknown",
                    "linked": True,
                })
            else:
                errors.append({"source_id": source_id, "error": "Source not ready or not found"})

        except Exception as e:
            errors.append({"source_id": source_id, "error": str(e)})

    return {
        "linked": results,
        "errors": errors,
        "total_linked": len(results),
        "total_errors": len(errors),
    }


@app.get("/api/v1/courses/{course_id}/sources", response_model=CourseSourcesResponse)
async def get_course_sources(course_id: str, user_id: str):
    """Get all sources linked to a course."""
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        links = await source_library.get_course_sources(course_id, user_id)

        responses = []
        for link, source in links:
            responses.append(CourseSourceResponse(
                id=link.id,
                course_id=link.course_id,
                source_id=link.source_id,
                source=SourceResponse.from_source(source),
                relevance_score=link.relevance_score,
                is_primary=link.is_primary,
                added_at=link.added_at,
            ))

        return CourseSourcesResponse(
            course_id=course_id,
            sources=responses,
            total=len(responses),
        )

    except Exception as e:
        print(f"[SOURCES] Get course sources error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/courses/{course_id}/sources/{source_id}")
async def unlink_source_from_course(course_id: str, source_id: str, user_id: str):
    """Remove a source from a course (doesn't delete from library)."""
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        success = await source_library.unlink_from_course(course_id, source_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Link not found")

        return {"message": "Source unlinked from course"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[SOURCES] Unlink error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/courses/{course_id}/sources/context")
async def get_course_sources_context(
    course_id: str,
    user_id: str,
    topic: str,
    max_tokens: int = 4000,
):
    """Get RAG context from course sources for generation."""
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        context = await source_library.get_context_for_course(
            course_id=course_id,
            topic=topic,
            user_id=user_id,
            max_tokens=max_tokens,
        )

        links = await source_library.get_course_sources(course_id, user_id)

        return {
            "context": context,
            "source_count": len(links),
            "token_estimate": len(context.split()) * 1.3,  # Rough estimate
        }

    except Exception as e:
        print(f"[SOURCES] Context error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AI SOURCE SUGGESTIONS ENDPOINT
# =============================================================================

@app.post("/api/v1/sources/suggest", response_model=SuggestSourcesResponse)
async def suggest_sources(request: SuggestSourcesRequest):
    """
    Get AI suggestions for sources based on course topic.
    Also returns existing relevant sources from user's library.
    """
    if not source_library:
        raise HTTPException(status_code=503, detail="Source library not initialized")

    try:
        print(f"[SOURCES] Suggesting sources for: {request.topic}", flush=True)

        suggestions, relevant_existing = await source_library.suggest_sources(
            topic=request.topic,
            description=request.description,
            user_id=request.user_id,
            language=request.language,
            max_suggestions=request.max_suggestions,
        )

        return SuggestSourcesResponse(
            topic=request.topic,
            suggestions=suggestions,
            existing_relevant_sources=[SourceResponse.from_source(s) for s in relevant_existing],
        )

    except Exception as e:
        print(f"[SOURCES] Suggest error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/sources/types")
async def get_source_types():
    """Get available source types."""
    return {
        "types": [
            {"id": "file", "name": "Fichier", "icon": "📄", "description": "PDF, Word, PowerPoint, Excel, etc."},
            {"id": "url", "name": "Page Web", "icon": "🌐", "description": "Article ou documentation en ligne"},
            {"id": "youtube", "name": "YouTube", "icon": "🎬", "description": "Vidéo YouTube (transcription)"},
            {"id": "note", "name": "Note", "icon": "📝", "description": "Texte personnel ou notes"},
        ]
    }


# =============================================================================
# ANALYTICS ENDPOINTS (Phase 5A)
# =============================================================================

from models.analytics_models import (
    TimeRange,
    APIProvider,
    DashboardSummary,
    UserAnalyticsSummary,
    APIUsageReport,
    UsageQuota,
    TrackEventRequest,
    MetricType,
)
from services.analytics_service import get_analytics_service


@app.get("/api/v1/analytics/dashboard", response_model=DashboardSummary)
async def get_analytics_dashboard(
    user_id: Optional[str] = None,
    time_range: TimeRange = TimeRange.MONTH,
    include_trends: bool = True,
):
    """
    Get analytics dashboard with course metrics, API usage, and engagement.
    Admin view if user_id is not provided.
    """
    try:
        service = get_analytics_service()
        dashboard = await service.get_dashboard(
            user_id=user_id,
            time_range=time_range,
            include_trends=include_trends,
        )
        return dashboard
    except Exception as e:
        print(f"[ANALYTICS] Dashboard error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/user/{user_id}", response_model=UserAnalyticsSummary)
async def get_user_analytics(
    user_id: str,
    time_range: TimeRange = TimeRange.MONTH,
):
    """Get analytics summary for a specific user."""
    try:
        service = get_analytics_service()
        summary = await service.get_user_summary(user_id, time_range)
        return summary
    except Exception as e:
        print(f"[ANALYTICS] User analytics error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/api-usage", response_model=APIUsageReport)
async def get_api_usage_report(
    user_id: Optional[str] = None,
    provider: Optional[APIProvider] = None,
    time_range: TimeRange = TimeRange.MONTH,
):
    """Get detailed API usage report with costs and projections."""
    try:
        service = get_analytics_service()
        report = await service.get_api_usage_report(
            user_id=user_id,
            provider=provider,
            time_range=time_range,
        )
        return report
    except Exception as e:
        print(f"[ANALYTICS] API usage error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/quota/{user_id}", response_model=UsageQuota)
async def get_user_quota(user_id: str):
    """Get user quota and current usage status."""
    try:
        service = get_analytics_service()
        quota = await service.get_user_quota(user_id)
        return quota
    except Exception as e:
        print(f"[ANALYTICS] Quota error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analytics/track")
async def track_analytics_event(request: TrackEventRequest):
    """Track a custom analytics event."""
    try:
        service = get_analytics_service()

        if request.event_type == MetricType.COURSE_CREATED:
            await service.track_course_created(
                user_id=request.user_id,
                course_id=request.course_id or "",
                title=request.metadata.get("title", ""),
                category=request.metadata.get("category"),
                lecture_count=request.metadata.get("lecture_count", 0),
            )
        elif request.event_type == MetricType.VIEW:
            await service.track_view(
                course_id=request.course_id or "",
                owner_user_id=request.user_id,
                viewer_id=request.metadata.get("viewer_id"),
                watch_duration_seconds=request.metadata.get("duration", 0),
                completed=request.metadata.get("completed", False),
            )
        else:
            # Generic event tracking
            from models.analytics_models import AnalyticsEvent
            event = AnalyticsEvent(
                user_id=request.user_id,
                event_type=request.event_type,
                metadata=request.metadata,
            )
            await service.repository.save_event(event)

        return {"status": "tracked", "event_type": request.event_type.value}

    except Exception as e:
        print(f"[ANALYTICS] Track error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/time-ranges")
async def get_time_ranges():
    """Get available time ranges for analytics queries."""
    return {
        "ranges": [
            {"value": "today", "label": "Today"},
            {"value": "week", "label": "Last 7 days"},
            {"value": "month", "label": "Last 30 days"},
            {"value": "quarter", "label": "Last 90 days"},
            {"value": "year", "label": "Last year"},
            {"value": "all_time", "label": "All time"},
        ]
    }


@app.get("/api/v1/analytics/providers")
async def get_api_providers():
    """Get list of tracked API providers."""
    return {
        "providers": [
            {"value": "openai", "label": "OpenAI", "tracks": ["tokens", "cost"]},
            {"value": "elevenlabs", "label": "ElevenLabs", "tracks": ["characters", "cost"]},
            {"value": "d-id", "label": "D-ID", "tracks": ["minutes", "cost"]},
            {"value": "replicate", "label": "Replicate", "tracks": ["seconds", "cost"]},
            {"value": "pexels", "label": "Pexels", "tracks": ["requests"]},
            {"value": "cloudinary", "label": "Cloudinary", "tracks": ["storage", "bandwidth"]},
        ]
    }


# =============================================================================
# TRANSLATION ENDPOINTS (Phase 5B)
# =============================================================================

from models.translation_models import (
    SupportedLanguage,
    TranslateTextRequest,
    TranslateTextResponse,
    DetectLanguageRequest,
    DetectLanguageResponse,
    SupportedLanguagesResponse,
    LanguageInfo,
    LANGUAGE_INFO,
)
from services.translation_service import get_translation_service


@app.get("/api/v1/translation/languages", response_model=SupportedLanguagesResponse)
async def get_supported_languages():
    """Get list of supported languages for translation."""
    service = get_translation_service()
    languages = service.get_supported_languages()
    return SupportedLanguagesResponse(languages=languages)


@app.post("/api/v1/translation/translate", response_model=TranslateTextResponse)
async def translate_text(request: TranslateTextRequest):
    """Translate a single text to target language."""
    try:
        service = get_translation_service()
        translated = await service.translate_text(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            context=request.context,
        )
        return TranslateTextResponse(
            original=request.text,
            translated=translated,
            source_language=request.source_language,
            target_language=request.target_language,
        )
    except Exception as e:
        print(f"[TRANSLATION] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/translation/translate-batch")
async def translate_texts_batch(
    texts: List[str],
    source_language: SupportedLanguage = SupportedLanguage.ENGLISH,
    target_language: SupportedLanguage = SupportedLanguage.FRENCH,
    context: Optional[str] = None,
):
    """Translate multiple texts in a single request."""
    try:
        service = get_translation_service()
        translated = await service.translate_texts_batch(
            texts=texts,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )
        return {
            "translations": [
                {"original": orig, "translated": trans}
                for orig, trans in zip(texts, translated)
            ],
            "source_language": source_language.value,
            "target_language": target_language.value,
        }
    except Exception as e:
        print(f"[TRANSLATION] Batch error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/translation/detect", response_model=DetectLanguageResponse)
async def detect_language(request: DetectLanguageRequest):
    """Detect the language of a text."""
    try:
        service = get_translation_service()
        result = await service.detect_language(request.text)
        return result
    except Exception as e:
        print(f"[TRANSLATION] Detection error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/translation/course/{course_id}")
async def translate_course(
    course_id: str,
    target_languages: List[SupportedLanguage],
    source_language: SupportedLanguage = SupportedLanguage.ENGLISH,
    background_tasks: BackgroundTasks = None,
):
    """
    Translate an entire course to one or more languages.
    Returns immediately with job ID, translation runs in background.
    """
    # Get course data
    job = jobs.get(course_id)
    if not job or not job.outline:
        raise HTTPException(status_code=404, detail="Course not found")

    # For now, return synchronously (would be async in production)
    try:
        service = get_translation_service()

        course_data = {
            "id": course_id,
            "title": job.outline.course_title,
            "description": job.request.description or "",
            "objectives": job.outline.learning_objectives,
            "lectures": [
                {
                    "id": str(i),
                    "title": lecture.title,
                    "description": lecture.description,
                    "script": lecture.script if hasattr(lecture, 'script') else "",
                    "key_points": lecture.key_points,
                }
                for i, lecture in enumerate(job.outline.lectures)
            ],
        }

        translations = []
        for target_lang in target_languages:
            translation = await service.translate_course(
                course_data=course_data,
                source_language=source_language,
                target_language=target_lang,
            )
            translations.append(translation)

        return {
            "course_id": course_id,
            "source_language": source_language.value,
            "translations": [t.model_dump() for t in translations],
        }

    except Exception as e:
        print(f"[TRANSLATION] Course translation error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BILLING ENDPOINTS (Phase 5C)
# =============================================================================

from fastapi import Request, Header
from models.billing_models import (
    PaymentProvider,
    SubscriptionPlan,
    BillingInterval,
    PlanInfo,
    CreateCheckoutRequest,
    CheckoutSessionResponse,
    CreatePortalRequest,
    PortalSessionResponse,
    CancelSubscriptionRequest,
    SubscriptionResponse,
)
from services.billing_service import get_billing_service


@app.get("/api/v1/billing/plans")
async def get_subscription_plans():
    """Get all available subscription plans with pricing."""
    service = get_billing_service()
    plans = service.get_all_plans()
    return {"plans": [p.model_dump() for p in plans]}


@app.get("/api/v1/billing/plans/{plan_id}", response_model=PlanInfo)
async def get_plan_details(plan_id: SubscriptionPlan):
    """Get details for a specific plan."""
    service = get_billing_service()
    return service.get_plan_info(plan_id)


@app.post("/api/v1/billing/checkout", response_model=CheckoutSessionResponse)
async def create_checkout_session(request: CreateCheckoutRequest):
    """Create a checkout session for subscription purchase."""
    try:
        service = get_billing_service()
        session = await service.create_checkout_session(
            user_id=request.user_id,
            email="user@example.com",  # Would come from auth context
            plan=request.plan,
            interval=request.billing_interval,
            provider=request.provider,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
        )
        return session
    except Exception as e:
        print(f"[BILLING] Checkout error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/billing/subscription/{user_id}", response_model=SubscriptionResponse)
async def get_user_subscription(user_id: str):
    """Get user's current subscription."""
    try:
        service = get_billing_service()
        subscription = await service.get_subscription(user_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No subscription found")
        return subscription
    except HTTPException:
        raise
    except Exception as e:
        print(f"[BILLING] Subscription error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/billing/cancel")
async def cancel_subscription(request: CancelSubscriptionRequest):
    """Cancel user's subscription."""
    try:
        service = get_billing_service()
        subscription = await service.cancel_subscription(
            user_id=request.user_id,
            reason=request.reason,
            cancel_immediately=request.cancel_immediately,
        )
        return {
            "status": "canceled",
            "subscription_id": subscription.id,
            "canceled_at": subscription.canceled_at.isoformat() if subscription.canceled_at else None,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[BILLING] Cancel error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/billing/portal", response_model=PortalSessionResponse)
async def create_billing_portal(request: CreatePortalRequest):
    """Create a billing portal session for managing subscription."""
    try:
        service = get_billing_service()
        portal = await service.create_portal_session(
            user_id=request.user_id,
            return_url=request.return_url,
        )
        return portal
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[BILLING] Portal error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/billing/webhooks/stripe")
async def handle_stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
):
    """Handle Stripe webhook events."""
    try:
        payload = await request.body()
        service = get_billing_service()
        result = await service.handle_webhook(
            provider=PaymentProvider.STRIPE,
            payload=payload,
            signature=stripe_signature or "",
        )
        return result
    except Exception as e:
        print(f"[BILLING] Webhook error: {str(e)}", flush=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/billing/webhooks/paypal")
async def handle_paypal_webhook(request: Request):
    """Handle PayPal webhook events."""
    try:
        payload = await request.body()
        service = get_billing_service()
        result = await service.handle_webhook(
            provider=PaymentProvider.PAYPAL,
            payload=payload,
            signature="",
        )
        return result
    except Exception as e:
        print(f"[BILLING] PayPal webhook error: {str(e)}", flush=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/billing/invoices/{user_id}")
async def get_user_invoices(user_id: str):
    """Get user's invoice history."""
    try:
        service = get_billing_service()
        invoices = await service.get_user_invoices(user_id)
        return {"invoices": [i.model_dump() for i in invoices]}
    except Exception as e:
        print(f"[BILLING] Invoices error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# COLLABORATION ENDPOINTS (Phase 5D)
# =============================================================================

from models.collaboration_models import (
    TeamRole,
    SharePermission,
    Workspace,
    TeamInvitation,
    CourseShare,
    ActivityLog,
    CreateWorkspaceRequest,
    UpdateWorkspaceRequest,
    InviteMemberRequest,
    UpdateMemberRoleRequest,
    ShareCourseRequest,
    AcceptInvitationRequest,
    WorkspaceResponse,
    WorkspaceListResponse,
    MemberListResponse,
    ActivityLogResponse,
)
from services.collaboration_service import get_collaboration_service


@app.post("/api/v1/workspaces", response_model=Workspace)
async def create_workspace(request: CreateWorkspaceRequest):
    """Create a new team workspace."""
    try:
        service = get_collaboration_service()
        workspace = await service.create_workspace(
            name=request.name,
            owner_id=request.owner_id,
            owner_name="User",  # Would come from auth context
            owner_email="user@example.com",
            description=request.description,
        )
        return workspace
    except Exception as e:
        print(f"[COLLABORATION] Create workspace error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/workspaces", response_model=WorkspaceListResponse)
async def list_user_workspaces(user_id: str):
    """List all workspaces user is a member of."""
    try:
        service = get_collaboration_service()
        workspaces = await service.get_user_workspaces(user_id)
        return WorkspaceListResponse(workspaces=workspaces, total=len(workspaces))
    except Exception as e:
        print(f"[COLLABORATION] List workspaces error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/workspaces/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(workspace_id: str, user_id: str):
    """Get workspace details with user's role and permissions."""
    try:
        service = get_collaboration_service()
        response = await service.get_workspace(workspace_id, user_id)
        if not response:
            raise HTTPException(status_code=404, detail="Workspace not found or access denied")
        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"[COLLABORATION] Get workspace error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/v1/workspaces/{workspace_id}")
async def update_workspace(
    workspace_id: str,
    request: UpdateWorkspaceRequest,
    user_id: str,
):
    """Update workspace settings."""
    try:
        service = get_collaboration_service()
        workspace = await service.update_workspace(
            workspace_id=workspace_id,
            user_id=user_id,
            name=request.name,
            description=request.description,
            logo_url=request.logo_url,
            default_member_role=request.default_member_role,
            allow_external_sharing=request.allow_external_sharing,
        )
        return workspace
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[COLLABORATION] Update workspace error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/workspaces/{workspace_id}/invite", response_model=TeamInvitation)
async def invite_member(
    workspace_id: str,
    request: InviteMemberRequest,
    inviter_id: str,
    inviter_name: str = "User",
):
    """Invite a new member to the workspace."""
    try:
        service = get_collaboration_service()
        invitation = await service.invite_member(
            workspace_id=workspace_id,
            inviter_id=inviter_id,
            inviter_name=inviter_name,
            email=request.email,
            role=request.role,
        )
        return invitation
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[COLLABORATION] Invite error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/workspaces/accept-invitation", response_model=Workspace)
async def accept_invitation(request: AcceptInvitationRequest):
    """Accept a team invitation."""
    try:
        service = get_collaboration_service()
        workspace = await service.accept_invitation(
            invite_token=request.invite_token,
            user_id=request.user_id,
            user_name=request.user_name,
            user_email=request.user_email,
        )
        return workspace
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[COLLABORATION] Accept invitation error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/workspaces/{workspace_id}/members/{member_id}")
async def remove_member(
    workspace_id: str,
    member_id: str,
    remover_id: str,
    remover_name: str = "User",
):
    """Remove a member from the workspace."""
    try:
        service = get_collaboration_service()
        workspace = await service.remove_member(
            workspace_id=workspace_id,
            remover_id=remover_id,
            remover_name=remover_name,
            member_user_id=member_id,
        )
        return {"status": "removed", "workspace_id": workspace.id}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[COLLABORATION] Remove member error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/v1/workspaces/{workspace_id}/members/{member_id}/role")
async def update_member_role(
    workspace_id: str,
    member_id: str,
    request: UpdateMemberRoleRequest,
    updater_id: str,
):
    """Update a member's role."""
    try:
        service = get_collaboration_service()
        member = await service.update_member_role(
            workspace_id=workspace_id,
            updater_id=updater_id,
            member_user_id=member_id,
            new_role=request.new_role,
        )
        return member
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[COLLABORATION] Update role error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/workspaces/{workspace_id}/leave")
async def leave_workspace(workspace_id: str, user_id: str):
    """Leave a workspace."""
    try:
        service = get_collaboration_service()
        await service.leave_workspace(workspace_id, user_id)
        return {"status": "left", "workspace_id": workspace_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[COLLABORATION] Leave error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/courses/{course_id}/share", response_model=CourseShare)
async def share_course(course_id: str, request: ShareCourseRequest, sharer_id: str):
    """Share a course with users, workspaces, or create public link."""
    try:
        service = get_collaboration_service()
        share = await service.share_course(
            course_id=course_id,
            sharer_id=sharer_id,
            permission=request.permission,
            share_with_user_id=request.share_with_user_id,
            share_with_workspace_id=request.share_with_workspace_id,
            share_with_email=request.share_with_email,
            create_public_link=request.create_public_link,
        )
        return share
    except Exception as e:
        print(f"[COLLABORATION] Share error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/courses/{course_id}/shares")
async def get_course_shares(course_id: str):
    """Get all shares for a course."""
    try:
        service = get_collaboration_service()
        shares = await service.get_course_shares(course_id)
        return {"shares": [s.model_dump() for s in shares]}
    except Exception as e:
        print(f"[COLLABORATION] Get shares error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/workspaces/{workspace_id}/activity", response_model=ActivityLogResponse)
async def get_activity_log(
    workspace_id: str,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
):
    """Get workspace activity log."""
    try:
        service = get_collaboration_service()
        activities = await service.get_activity_log(
            workspace_id=workspace_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
        return ActivityLogResponse(
            activities=activities,
            total=len(activities),
            has_more=len(activities) == limit,
        )
    except Exception as e:
        print(f"[COLLABORATION] Activity log error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/collaboration/roles")
async def get_available_roles():
    """Get available team roles and their permissions."""
    from models.collaboration_models import ROLE_PERMISSIONS
    return {
        "roles": [
            {
                "id": role,
                "name": role.replace("_", " ").title(),
                "permissions": perms,
            }
            for role, perms in ROLE_PERMISSIONS.items()
        ]
    }


# =============================================================================
# CURRICULUM ENFORCER ENDPOINTS (Phase 6)
# =============================================================================

@app.get("/api/v1/curriculum/status")
async def get_curriculum_enforcer_status():
    """Check if Curriculum Enforcer module is available."""
    return {
        "available": CURRICULUM_ENFORCER_AVAILABLE,
        "initialized": curriculum_enforcer is not None,
    }


@app.get("/api/v1/curriculum/templates")
async def get_curriculum_templates():
    """Get all available curriculum templates."""
    if not CURRICULUM_ENFORCER_AVAILABLE or not curriculum_enforcer:
        raise HTTPException(status_code=503, detail="Curriculum Enforcer not available")

    try:
        templates = curriculum_enforcer.list_templates()
        return {"templates": templates}
    except Exception as e:
        print(f"[CURRICULUM] Templates error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/curriculum/templates/{context_type}")
async def get_curriculum_template(context_type: str):
    """Get details for a specific curriculum template."""
    if not CURRICULUM_ENFORCER_AVAILABLE or not curriculum_enforcer:
        raise HTTPException(status_code=503, detail="Curriculum Enforcer not available")

    try:
        # Validate context type
        try:
            ctx = CurriculumContextType(context_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid context type: {context_type}. Valid options: education, enterprise, bootcamp, tutorial, workshop, certification"
            )

        phases = curriculum_enforcer.get_template_phases(ctx)
        return {
            "context_type": context_type,
            "phases": phases,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[CURRICULUM] Template error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/curriculum/context-types")
async def get_curriculum_context_types():
    """Get all available curriculum context types with descriptions."""
    return {
        "context_types": [
            {
                "id": "education",
                "name": "Éducation",
                "description": "Cours en ligne, tutoriels, contenu académique",
                "flow": "Hook → Concept → Theory → Visualization → Code Demo → Recap",
            },
            {
                "id": "enterprise",
                "name": "Entreprise",
                "description": "Formation corporate, développement professionnel",
                "flow": "Context → Concept → Use Case → ROI → Action Items",
            },
            {
                "id": "bootcamp",
                "name": "Bootcamp",
                "description": "Bootcamps intensifs, ateliers pratiques",
                "flow": "Concept → Code Demo → Exercise → Exercise → Challenge → Review",
            },
            {
                "id": "tutorial",
                "name": "Tutoriel",
                "description": "Guides pratiques rapides, documentation",
                "flow": "Objectives → Prerequisites → Steps → Recap → Next Steps",
            },
            {
                "id": "workshop",
                "name": "Workshop",
                "description": "Sessions collaboratives pratiques",
                "flow": "Context → Example → Exercise → Review → Exercise → Recap",
            },
            {
                "id": "certification",
                "name": "Certification",
                "description": "Préparation aux examens, cours certifiants",
                "flow": "Objectives → Theory → Examples → Practice → Quiz → Recap",
            },
        ]
    }


@app.post("/api/v1/curriculum/validate")
async def validate_lesson_structure(
    lesson_data: dict,
    context_type: str = "education",
):
    """
    Validate a lesson's structure against a curriculum template.
    Returns validation score and missing phases.
    """
    if not CURRICULUM_ENFORCER_AVAILABLE or not curriculum_enforcer:
        raise HTTPException(status_code=503, detail="Curriculum Enforcer not available")

    try:
        # Validate context type
        try:
            ctx = CurriculumContextType(context_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid context type: {context_type}")

        # Create LessonContent from request
        content = LessonContent(
            lesson_id=lesson_data.get("id", "unknown"),
            title=lesson_data.get("title", "Untitled"),
            slides=lesson_data.get("slides", []),
            lesson_type=lesson_data.get("type"),
            section_position=lesson_data.get("section_position", 0),
            total_lessons_in_section=lesson_data.get("total_lessons", 1),
        )

        # Validate only (no auto-fix)
        validation = await curriculum_enforcer.validate_only(content, ctx)

        return {
            "is_valid": validation.is_valid,
            "score": validation.score,
            "missing_required_phases": validation.missing_required_phases,
            "missing_optional_phases": validation.missing_optional_phases,
            "order_issues": validation.order_issues,
            "duration_issues": validation.duration_issues,
            "detected_phases": [
                {"phase": dp.phase.value, "slide_index": dp.slide_index, "confidence": dp.confidence}
                for dp in validation.detected_phases
            ],
            "recommendations": validation.recommendations,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[CURRICULUM] Validation error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/curriculum/enforce")
async def enforce_lesson_structure(
    lesson_data: dict,
    context_type: str = "education",
    auto_fix: bool = True,
    preserve_content: bool = True,
):
    """
    Enforce curriculum structure on a lesson.
    Can auto-fix missing phases if requested.
    """
    if not CURRICULUM_ENFORCER_AVAILABLE or not curriculum_enforcer:
        raise HTTPException(status_code=503, detail="Curriculum Enforcer not available")

    try:
        # Validate context type
        try:
            ctx = CurriculumContextType(context_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid context type: {context_type}")

        # Create LessonContent from request
        content = LessonContent(
            lesson_id=lesson_data.get("id", "unknown"),
            title=lesson_data.get("title", "Untitled"),
            slides=lesson_data.get("slides", []),
            lesson_type=lesson_data.get("type"),
            section_position=lesson_data.get("section_position", 0),
            total_lessons_in_section=lesson_data.get("total_lessons", 1),
        )

        # Create enforcement request
        request = EnforcementRequest(
            content=content,
            context_type=ctx,
            auto_fix=auto_fix,
            preserve_content=preserve_content,
        )

        # Enforce structure
        result = await curriculum_enforcer.enforce(request)

        response = {
            "success": result.success,
            "changes_made": result.changes_made,
            "original_validation": {
                "is_valid": result.original_validation.is_valid,
                "score": result.original_validation.score,
                "missing_required_phases": result.original_validation.missing_required_phases,
            },
        }

        if result.final_validation:
            response["final_validation"] = {
                "is_valid": result.final_validation.is_valid,
                "score": result.final_validation.score,
                "missing_required_phases": result.final_validation.missing_required_phases,
            }

        if result.restructured_content:
            response["restructured_slides"] = result.restructured_content.slides

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"[CURRICULUM] Enforcement error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Lecture Editor Endpoints
# =============================================================================

@app.get("/api/v1/courses/jobs/{job_id}/lectures/{lecture_id}/components", response_model=LectureComponentsResponse)
async def get_lecture_components(job_id: str, lecture_id: str):
    """
    Get editable components of a lecture.
    Returns slides, voiceover, and other editable elements.
    """
    if not lecture_editor:
        raise HTTPException(status_code=503, detail="Lecture editor service not available")

    # Verify job exists
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Find lecture in job
    lecture = None
    for section in job.outline.sections:
        for lec in section.lectures:
            if lec.id == lecture_id:
                lecture = lec
                break
        if lecture:
            break

    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    if not lecture.has_components:
        raise HTTPException(status_code=404, detail="Lecture components not available. Lecture may have failed or components were not stored.")

    # Get components from database
    components = await lecture_editor.get_components(lecture_id)
    if not components:
        raise HTTPException(status_code=404, detail="Components not found in database")

    return LectureComponentsResponse(
        lecture_id=components.lecture_id,
        job_id=components.job_id,
        status=components.status,
        slides=components.slides,
        voiceover=components.voiceover,
        total_duration=components.total_duration,
        video_url=convert_internal_url_to_external(components.video_url) if components.video_url else None,
        is_edited=components.is_edited,
        created_at=components.created_at,
        updated_at=components.updated_at,
        error=components.error
    )


@app.patch("/api/v1/courses/jobs/{job_id}/lectures/{lecture_id}/slides/{slide_id}", response_model=SlideComponentResponse)
async def update_slide(job_id: str, lecture_id: str, slide_id: str, updates: UpdateSlideRequest):
    """
    Update a slide's content (title, content, voiceover text, code, etc.).
    The slide will be marked as edited.
    """
    if not lecture_editor:
        raise HTTPException(status_code=503, detail="Lecture editor service not available")

    # Verify job and lecture exist
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        slide = await lecture_editor.update_slide(lecture_id, slide_id, updates)
        if not slide:
            raise HTTPException(status_code=404, detail="Slide not found")

        return SlideComponentResponse(
            slide=slide,
            lecture_id=lecture_id,
            message="Slide updated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/courses/jobs/{job_id}/lectures/{lecture_id}/slides/{slide_id}/regenerate", response_model=SlideComponentResponse)
async def regenerate_slide(job_id: str, lecture_id: str, slide_id: str, options: RegenerateSlideRequest):
    """
    Regenerate a single slide (image and/or animation).
    Use after editing slide content to update the visual.
    """
    if not lecture_editor:
        raise HTTPException(status_code=503, detail="Lecture editor service not available")

    # Verify job exists
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        slide = await lecture_editor.regenerate_slide(lecture_id, slide_id, options)
        if not slide:
            raise HTTPException(status_code=404, detail="Slide not found")

        return SlideComponentResponse(
            slide=slide,
            lecture_id=lecture_id,
            message="Slide regenerated successfully"
        )
    except Exception as e:
        print(f"[EDITOR] Slide regeneration failed: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/courses/jobs/{job_id}/lectures/{lecture_id}/regenerate-voiceover", response_model=RegenerateResponse)
async def regenerate_voiceover(job_id: str, lecture_id: str, options: RegenerateVoiceoverRequest):
    """
    Regenerate voiceover audio from slide texts.
    Use after editing voiceover texts to update the audio.
    """
    if not lecture_editor:
        raise HTTPException(status_code=503, detail="Lecture editor service not available")

    # Verify job exists
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        voiceover = await lecture_editor.regenerate_voiceover(lecture_id, options)
        if not voiceover:
            raise HTTPException(status_code=404, detail="Lecture not found")

        return RegenerateResponse(
            success=True,
            message="Voiceover regenerated successfully",
            result={
                "audio_url": convert_internal_url_to_external(voiceover.audio_url) if voiceover.audio_url else None,
                "duration_seconds": voiceover.duration_seconds
            }
        )
    except Exception as e:
        print(f"[EDITOR] Voiceover regeneration failed: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/courses/jobs/{job_id}/lectures/{lecture_id}/upload-audio")
async def upload_custom_audio(
    job_id: str,
    lecture_id: str,
    file: UploadFile = File(...),
):
    """
    Upload custom audio to replace generated voiceover.
    Supports MP3, WAV, M4A formats.
    """
    if not lecture_editor:
        raise HTTPException(status_code=503, detail="Lecture editor service not available")

    # Verify job exists
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Validate file type
    allowed_types = ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/m4a"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: MP3, WAV, M4A"
        )

    try:
        audio_data = await file.read()
        voiceover = await lecture_editor.upload_custom_audio(lecture_id, audio_data, file.filename)

        return {
            "success": True,
            "message": "Custom audio uploaded successfully",
            "audio_url": convert_internal_url_to_external(voiceover.audio_url) if voiceover.audio_url else None,
            "duration_seconds": voiceover.duration_seconds,
            "is_custom": True
        }
    except Exception as e:
        print(f"[EDITOR] Audio upload failed: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/courses/jobs/{job_id}/lectures/{lecture_id}/regenerate", response_model=RegenerateResponse)
async def regenerate_lecture(job_id: str, lecture_id: str, options: RegenerateLectureRequest):
    """
    Regenerate entire lecture.
    If use_edited_components=True, keeps edited slides and regenerates only non-edited ones.
    If use_edited_components=False, regenerates everything from scratch.
    """
    if not lecture_editor:
        raise HTTPException(status_code=503, detail="Lecture editor service not available")

    # Verify job and lecture exist
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Find lecture
    lecture = None
    for section in job.outline.sections:
        for lec in section.lectures:
            if lec.id == lecture_id:
                lecture = lec
                break
        if lecture:
            break

    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    if not lecture.can_regenerate:
        raise HTTPException(status_code=400, detail="This lecture cannot be regenerated")

    try:
        # Mark lecture as regenerating
        lecture.status = "generating"
        lecture.error = None

        video_url = await lecture_editor.regenerate_lecture(lecture_id, options, lecture)

        # Update lecture
        lecture.video_url = video_url
        lecture.status = "completed"
        lecture.is_edited = True

        # Update job output URLs if needed
        if video_url and video_url not in job.output_urls:
            job.output_urls.append(video_url)

        # Remove from failed list if it was there
        if lecture_id in job.failed_lecture_ids:
            job.failed_lecture_ids.remove(lecture_id)
            if lecture_id in job.failed_lecture_errors:
                del job.failed_lecture_errors[lecture_id]
            job.lectures_failed = len(job.failed_lecture_ids)
            job.lectures_completed += 1

            # Update status if no more failures
            if not job.failed_lecture_ids and job.current_stage == CourseStage.PARTIAL_SUCCESS:
                job.update_progress(CourseStage.COMPLETED, 100, "Course generation complete!")

        return RegenerateResponse(
            success=True,
            message="Lecture regenerated successfully",
            result={
                "video_url": convert_internal_url_to_external(video_url) if video_url else None,
                "lecture_id": lecture_id
            }
        )
    except Exception as e:
        lecture.status = "failed"
        lecture.error = str(e)
        print(f"[EDITOR] Lecture regeneration failed: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/courses/jobs/{job_id}/lectures/{lecture_id}/recompose", response_model=RegenerateResponse)
async def recompose_lecture_video(job_id: str, lecture_id: str, options: RecomposeVideoRequest):
    """
    Recompose lecture video from current components.
    Use after editing slides/voiceover to create new video without regenerating content.
    """
    if not lecture_editor:
        raise HTTPException(status_code=503, detail="Lecture editor service not available")

    # Verify job exists
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        video_url = await lecture_editor.recompose_video(lecture_id, options)

        # Update lecture in job
        for section in job.outline.sections:
            for lecture in section.lectures:
                if lecture.id == lecture_id:
                    lecture.video_url = video_url
                    lecture.is_edited = True
                    break

        return RegenerateResponse(
            success=True,
            message="Video recomposed successfully",
            result={
                "video_url": convert_internal_url_to_external(video_url) if video_url else None
            }
        )
    except Exception as e:
        print(f"[EDITOR] Video recomposition failed: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/courses/jobs/{job_id}/retry-failed", response_model=RegenerateResponse)
async def retry_all_failed_lectures(job_id: str, background_tasks: BackgroundTasks):
    """
    Retry generation for all failed lectures in a course.
    Runs in background and returns immediately.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.failed_lecture_ids:
        raise HTTPException(status_code=400, detail="No failed lectures to retry")

    # Start background regeneration
    async def retry_failed_lectures():
        for lecture_id in list(job.failed_lecture_ids):
            try:
                # Find lecture
                lecture = None
                for section in job.outline.sections:
                    for lec in section.lectures:
                        if lec.id == lecture_id:
                            lecture = lec
                            break
                    if lecture:
                        break

                if not lecture:
                    continue

                print(f"[RETRY] Retrying failed lecture: {lecture.title}", flush=True)

                # Regenerate using compositor (same as initial generation)
                lecture.status = "generating"
                lecture.retry_count = 0  # Reset retry count
                lecture.error = None

                video_url = await course_compositor._generate_single_lecture(
                    lecture=lecture,
                    section=section,
                    outline=job.outline,
                    request=job.request,
                    position=lecture.order,
                    total=job.lectures_total,
                    job_id=job_id
                )

                lecture.video_url = video_url
                lecture.status = "completed"

                # Store components
                if lecture.presentation_job_id:
                    components_id = await lecture_editor.store_components_from_presentation_job(
                        presentation_job_id=lecture.presentation_job_id,
                        lecture_id=lecture.id,
                        job_id=job_id
                    )
                    if components_id:
                        lecture.components_id = components_id
                        lecture.has_components = True

                # Update job
                job.output_urls.append(video_url)
                job.failed_lecture_ids.remove(lecture_id)
                if lecture_id in job.failed_lecture_errors:
                    del job.failed_lecture_errors[lecture_id]
                job.lectures_failed = len(job.failed_lecture_ids)
                job.lectures_completed += 1

                print(f"[RETRY] Successfully regenerated: {lecture.title}", flush=True)

            except Exception as e:
                print(f"[RETRY] Failed to regenerate lecture {lecture_id}: {str(e)}", flush=True)

        # Update job status
        if not job.failed_lecture_ids:
            job.update_progress(CourseStage.COMPLETED, 100, "All lectures regenerated successfully!")
        else:
            job.update_progress(
                CourseStage.PARTIAL_SUCCESS,
                100,
                f"Retry complete. {job.lectures_failed} lectures still failed."
            )

    background_tasks.add_task(retry_failed_lectures)

    return RegenerateResponse(
        success=True,
        message=f"Retrying {len(job.failed_lecture_ids)} failed lectures in background",
        job_id=job_id
    )


# =============================================================================
# MULTI-AGENT API ENDPOINTS
# =============================================================================

@app.get("/api/v1/multi-agent/status")
async def get_multi_agent_status():
    """Get the status of the multi-agent system"""
    return {
        "available": MULTI_AGENT_AVAILABLE,
        "enabled": USE_MULTI_AGENT,
        "initialized": multi_agent_orchestrator is not None,
        "agents": [
            "InputValidatorAgent",
            "TechnicalReviewerAgent",
            "PedagogicalAgent",
            "CodeExpertAgent",
            "CodeReviewerAgent",
        ] if MULTI_AGENT_AVAILABLE else [],
    }


@app.post("/api/v1/multi-agent/validate")
async def validate_course_configuration(
    topic: str,
    description: Optional[str] = None,
    profile_category: str = "education",
    difficulty_start: str = "beginner",
    difficulty_end: str = "intermediate",
    content_language: str = "en",
    programming_language: str = "python",
    target_audience: Optional[str] = None,
    number_of_sections: int = 4,
    lectures_per_section: int = 3,
    total_duration_minutes: int = 60,
):
    """
    Validate course configuration through the multi-agent system.

    This endpoint runs InputValidatorAgent and TechnicalReviewerAgent
    to validate all configuration and generate enriched prompts.
    """
    if not MULTI_AGENT_AVAILABLE or not multi_agent_orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Multi-agent system not available"
        )

    import uuid
    job_id = f"validate_{uuid.uuid4().hex[:8]}"

    result = await multi_agent_orchestrator.validate_and_enrich(
        job_id=job_id,
        topic=topic,
        description=description,
        profile_category=profile_category,
        difficulty_start=difficulty_start,
        difficulty_end=difficulty_end,
        content_language=content_language,
        programming_language=programming_language,
        target_audience=target_audience,
        structure={
            "number_of_sections": number_of_sections,
            "lectures_per_section": lectures_per_section,
            "total_duration_minutes": total_duration_minutes,
        },
    )

    return {
        "validated": result.get("validated"),
        "validation_errors": result.get("validation_errors", []),
        "warnings": result.get("warnings", []),
        "suggestions": result.get("suggestions", []),
        "prompt_enrichments": result.get("prompt_enrichments", {}),
    }


@app.post("/api/v1/multi-agent/generate-code")
async def generate_quality_code_block(
    concept: str,
    language: str = "python",
    persona_level: str = "intermediate",
    rag_context: Optional[str] = None,
    max_retries: int = 3,
):
    """
    Generate production-quality code through the multi-agent system.

    This runs CodeExpertAgent -> CodeReviewerAgent loop with refinement
    until the code is approved or max retries are reached.
    """
    if not MULTI_AGENT_AVAILABLE or not multi_agent_orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Multi-agent system not available"
        )

    result = await multi_agent_orchestrator.process_code_block(
        concept=concept,
        language=language,
        persona_level=persona_level,
        rag_context=rag_context,
        max_retries=max_retries,
    )

    return {
        "approved": result.get("approved"),
        "code": result.get("code"),
        "explanation": result.get("explanation"),
        "expected_output": result.get("expected_output"),
        "complexity_score": result.get("complexity_score"),
        "quality_score": result.get("quality_score"),
        "patterns_used": result.get("patterns_used", []),
        "rejection_reasons": result.get("rejection_reasons", []),
        "iterations": result.get("iterations"),
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8007))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "development") == "development"
    )
