"""
NEXUS Engine Service
Neural Execution & Understanding Synthesis

FastAPI microservice for pedagogical code generation.
Integrates with MAESTRO and presentation-generator.
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import uuid
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# NEXUS imports
from core.pipeline import NEXUSPipeline, NexusConfig, PipelineProgress, create_nexus_pipeline
from models.data_models import (
    NexusRequest, NexusResponse, TargetAudience, CodeVerbosity, ExecutionMode
)
from providers.llm_provider import LLMConfig, LLMProvider, create_llm_provider


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR API
# =============================================================================

class GenerateCodeRequest(BaseModel):
    """Request to generate pedagogical code"""
    project_description: str = Field(..., description="Description of the project to generate")
    lesson_context: str = Field(default="", description="Context of the lesson")
    skill_level: str = Field(default="intermediate", description="beginner, intermediate, advanced, expert")
    language: str = Field(default="python", description="Programming language")
    target_audience: str = Field(default="student", description="developer, architect, student, lead")
    verbosity: str = Field(default="standard", description="minimal, standard, verbose, production")
    allocated_time_seconds: int = Field(default=300, description="Time budget in seconds")
    max_segments: int = Field(default=10, description="Maximum code segments")
    show_mistakes: bool = Field(default=True, description="Include common mistakes")
    show_evolution: bool = Field(default=False, description="Show v1->v2->v3 progression")
    include_tests: bool = Field(default=False, description="Include test code")

    class Config:
        json_schema_extra = {
            "example": {
                "project_description": "une plateforme e-commerce avec panier et paiement",
                "lesson_context": "Module 3: Architecture backend",
                "skill_level": "intermediate",
                "language": "python",
                "target_audience": "student",
                "verbosity": "standard",
                "allocated_time_seconds": 300,
            }
        }


class DecomposeRequest(BaseModel):
    """Request for domain decomposition only"""
    project_description: str
    lesson_context: str = ""
    skill_level: str = "intermediate"
    language: str = "python"


class CodeSegmentResponse(BaseModel):
    """A generated code segment"""
    id: str
    filename: str
    code: str
    language: str
    component_type: str
    explanation: str
    key_concepts: List[str]
    common_mistakes: List[str]
    narration_script: str
    duration_seconds: int
    display_order: int


class GenerateCodeResponse(BaseModel):
    """Response with generated code"""
    request_id: str
    project_name: str
    language: str
    framework: str
    code_segments: List[CodeSegmentResponse]
    total_duration_seconds: int
    total_lines_of_code: int
    generation_time_ms: int
    sync_metadata: Dict[str, Any]


class JobStatusResponse(BaseModel):
    """Status of a generation job"""
    job_id: str
    status: str  # queued, processing, completed, failed
    progress: float
    stage: str
    message: str
    result: Optional[GenerateCodeResponse] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    llm_provider: str
    timestamp: str


# =============================================================================
# IN-MEMORY JOB STORAGE
# =============================================================================

jobs: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# NEXUS SERVICE
# =============================================================================

class NexusService:
    """Service wrapper for NEXUS pipeline"""

    def __init__(self):
        self.pipeline: Optional[NEXUSPipeline] = None
        self.config: Optional[NexusConfig] = None
        self.llm_provider_name: str = "unknown"

    def initialize(self):
        """Initialize the NEXUS pipeline"""
        # Get LLM configuration from environment
        provider = os.getenv("LLM_PROVIDER", "groq")
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

        if not api_key:
            logger.warning("No API key found. NEXUS will fail on generation requests.")
            return

        # Create LLM config
        if provider == "groq":
            llm_config = LLMConfig.for_groq(api_key, model)
        elif provider == "openai":
            llm_config = LLMConfig.for_openai(api_key, model or "gpt-4o")
        elif provider == "anthropic":
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", api_key)
            llm_config = LLMConfig.for_anthropic(anthropic_key, model or "claude-sonnet-4-20250514")
        else:
            llm_config = LLMConfig.for_groq(api_key, model)

        llm = create_llm_provider(llm_config)

        # Create NEXUS config
        self.config = NexusConfig(
            enable_reviewer=True,
            enable_executor=os.getenv("NEXUS_ENABLE_EXECUTOR", "false").lower() == "true",
            enable_narrator=True,
            max_feedback_iterations=3,
            sandbox_enabled=os.getenv("NEXUS_SANDBOX_ENABLED", "false").lower() == "true",
            verbose=True,
        )

        # Create pipeline
        self.pipeline = NEXUSPipeline(llm, self.config)
        self.llm_provider_name = provider

        logger.info(f"NEXUS Service initialized with {provider} provider")

    def generate(self, request: GenerateCodeRequest) -> NexusResponse:
        """Generate code using NEXUS pipeline"""
        if not self.pipeline:
            raise RuntimeError("NEXUS pipeline not initialized")

        # Map string values to enums
        audience_map = {
            "developer": TargetAudience.DEVELOPER,
            "architect": TargetAudience.ARCHITECT,
            "student": TargetAudience.STUDENT,
            "lead": TargetAudience.TECHNICAL_LEAD,
        }

        verbosity_map = {
            "minimal": CodeVerbosity.MINIMAL,
            "standard": CodeVerbosity.STANDARD,
            "verbose": CodeVerbosity.VERBOSE,
            "production": CodeVerbosity.PRODUCTION,
        }

        # Create NEXUS request
        nexus_request = NexusRequest(
            project_description=request.project_description,
            lesson_context=request.lesson_context,
            skill_level=request.skill_level,
            language=request.language,
            target_audience=audience_map.get(request.target_audience, TargetAudience.STUDENT),
            verbosity=verbosity_map.get(request.verbosity, CodeVerbosity.STANDARD),
            allocated_time_seconds=request.allocated_time_seconds,
            max_segments=request.max_segments,
            show_mistakes=request.show_mistakes,
            show_evolution=request.show_evolution,
            include_tests=request.include_tests,
        )

        # Generate
        return self.pipeline.generate(nexus_request)


# Global service instance
nexus_service = NexusService()


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("[STARTUP] Initializing NEXUS Engine Service...")

    nexus_service.initialize()

    logger.info("[STARTUP] NEXUS Engine Service ready!")

    yield

    logger.info("[SHUTDOWN] NEXUS Engine Service shutting down...")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="NEXUS Engine",
    description="Neural Execution & Understanding Synthesis - Pedagogical Code Generation",
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


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if nexus_service.pipeline else "degraded",
        service="nexus-engine",
        version="1.0.0",
        llm_provider=nexus_service.llm_provider_name,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/api/v1/nexus/generate", response_model=JobStatusResponse)
async def generate_code(request: GenerateCodeRequest, background_tasks: BackgroundTasks):
    """
    Start code generation job.
    Returns job ID for tracking progress.
    """
    if not nexus_service.pipeline:
        raise HTTPException(status_code=503, detail="NEXUS service not initialized")

    job_id = str(uuid.uuid4())

    # Create job entry
    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "stage": "queued",
        "message": "Job queued for processing",
        "result": None,
        "error": None,
        "created_at": datetime.utcnow(),
    }

    # Run in background
    background_tasks.add_task(run_generation_job, job_id, request)

    return JobStatusResponse(
        job_id=job_id,
        status="queued",
        progress=0.0,
        stage="queued",
        message="Job queued for processing",
    )


@app.post("/api/v1/nexus/generate-sync", response_model=GenerateCodeResponse)
async def generate_code_sync(request: GenerateCodeRequest):
    """
    Generate code synchronously (blocking).
    Use for smaller projects or when immediate response is needed.
    """
    if not nexus_service.pipeline:
        raise HTTPException(status_code=503, detail="NEXUS service not initialized")

    try:
        # Run generation in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, nexus_service.generate, request)

        return convert_nexus_response(response)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/nexus/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a generation job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        stage=job["stage"],
        message=job["message"],
        result=job.get("result"),
        error=job.get("error"),
    )


@app.get("/api/v1/nexus/config/audiences")
async def get_audiences():
    """Get available target audiences"""
    return {
        "audiences": [
            {"id": "developer", "name": "Developer", "description": "Production-ready code with error handling"},
            {"id": "architect", "name": "Architect", "description": "Focus on patterns and structure"},
            {"id": "student", "name": "Student", "description": "Pedagogical, well-commented, progressive"},
            {"id": "lead", "name": "Technical Lead", "description": "Balance between practical and architectural"},
        ]
    }


@app.get("/api/v1/nexus/config/verbosity-levels")
async def get_verbosity_levels():
    """Get available verbosity levels"""
    return {
        "levels": [
            {"id": "minimal", "name": "Minimal", "description": "Skeleton code, essential only"},
            {"id": "standard", "name": "Standard", "description": "Clean code with key comments"},
            {"id": "verbose", "name": "Verbose", "description": "Highly commented, every line explained"},
            {"id": "production", "name": "Production", "description": "Production-ready with error handling and logs"},
        ]
    }


@app.get("/api/v1/nexus/config/languages")
async def get_supported_languages():
    """Get supported programming languages"""
    return {
        "languages": [
            {"id": "python", "name": "Python", "frameworks": ["flask", "fastapi", "django"]},
            {"id": "javascript", "name": "JavaScript", "frameworks": ["express", "nestjs", "koa"]},
            {"id": "typescript", "name": "TypeScript", "frameworks": ["express", "nestjs"]},
            {"id": "java", "name": "Java", "frameworks": ["spring-boot"]},
            {"id": "go", "name": "Go", "frameworks": ["gin", "fiber", "echo"]},
            {"id": "rust", "name": "Rust", "frameworks": ["actix-web", "rocket"]},
        ]
    }


# =============================================================================
# BACKGROUND JOB RUNNER
# =============================================================================

async def run_generation_job(job_id: str, request: GenerateCodeRequest):
    """Run generation job in background"""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["stage"] = "decomposition"
        jobs[job_id]["message"] = "Analyzing project domain..."
        jobs[job_id]["progress"] = 10.0

        # Set up progress callback
        def on_progress(p: PipelineProgress):
            jobs[job_id]["stage"] = p.stage
            jobs[job_id]["message"] = p.message
            jobs[job_id]["progress"] = p.percent

        nexus_service.pipeline.set_progress_callback(on_progress)

        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, nexus_service.generate, request)

        # Convert response
        result = convert_nexus_response(response)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100.0
        jobs[job_id]["stage"] = "completed"
        jobs[job_id]["message"] = f"Generated {len(result.code_segments)} code segments"
        jobs[job_id]["result"] = result

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Generation failed: {str(e)}"


def convert_nexus_response(response: NexusResponse) -> GenerateCodeResponse:
    """Convert NEXUS response to API response"""
    segments = []
    for seg in response.get_segments_ordered():
        segments.append(CodeSegmentResponse(
            id=seg.id,
            filename=seg.filename,
            code=seg.code,
            language=seg.language,
            component_type=seg.component_type.value,
            explanation=seg.explanation,
            key_concepts=seg.key_concepts,
            common_mistakes=seg.common_mistakes,
            narration_script=seg.narration_script,
            duration_seconds=seg.duration_seconds,
            display_order=seg.display_order,
        ))

    return GenerateCodeResponse(
        request_id=response.request_id,
        project_name=response.architecture_dna.project_name,
        language=response.architecture_dna.language,
        framework=response.architecture_dna.framework,
        code_segments=segments,
        total_duration_seconds=response.total_duration_seconds,
        total_lines_of_code=response.total_lines_of_code,
        generation_time_ms=response.generation_time_ms,
        sync_metadata=response.sync_metadata,
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8009"))
    uvicorn.run(app, host="0.0.0.0", port=port)
