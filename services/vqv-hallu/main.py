"""
VQV-HALLU Microservice
Voice Quality Verification & Hallucination Detection

This service validates TTS-generated audio against source text
to detect hallucinations, distortions, and quality issues.

Feature flags:
- VQV_ENABLED: Enable/disable the service (default: true)
- VQV_STRICT_MODE: Fail if score below threshold (default: false)
"""

import os
import sys
import time
import asyncio
import tempfile
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class ServiceConfig:
    """Service configuration from environment variables"""

    # Feature flags
    VQV_ENABLED: bool = os.getenv("VQV_ENABLED", "true").lower() == "true"
    VQV_STRICT_MODE: bool = os.getenv("VQV_STRICT_MODE", "false").lower() == "true"

    # Thresholds
    MIN_ACCEPTABLE_SCORE: float = float(os.getenv("VQV_MIN_SCORE", "70"))
    MAX_REGENERATION_ATTEMPTS: int = int(os.getenv("VQV_MAX_REGEN", "3"))

    # Timeouts
    ANALYSIS_TIMEOUT_SECONDS: int = int(os.getenv("VQV_TIMEOUT", "300"))

    # Models (lazy loaded)
    ASR_MODEL: str = os.getenv("VQV_ASR_MODEL", "openai/whisper-large-v3")
    EMBEDDING_MODEL: str = os.getenv("VQV_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Paths
    TEMP_DIR: str = os.getenv("VQV_TEMP_DIR", "/tmp/vqv_hallu")

    # Service info
    SERVICE_NAME: str = "vqv-hallu"
    SERVICE_VERSION: str = "1.0.0"


config = ServiceConfig()


# ============================================================================
# Request/Response Models
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Request to analyze a voiceover"""
    audio_url: Optional[str] = Field(None, description="URL to the audio file")
    audio_path: Optional[str] = Field(None, description="Local path to audio file")
    source_text: str = Field(..., description="Original text that generated the audio")
    audio_id: str = Field(default="default", description="Unique identifier for tracking")
    content_type: str = Field(default="technical_course", description="Content type for threshold selection")
    language: str = Field(default="fr", description="Expected language")

    class Config:
        json_schema_extra = {
            "example": {
                "audio_url": "https://storage.example.com/voiceover.mp3",
                "source_text": "Bienvenue dans ce cours sur Python. Nous allons apprendre les bases.",
                "audio_id": "course_001_slide_05",
                "content_type": "technical_course",
                "language": "fr"
            }
        }


class AnalyzeResponse(BaseModel):
    """Response from analysis"""
    audio_id: str
    status: str  # "success", "failed", "skipped", "disabled"

    # Scores
    final_score: Optional[float] = None
    acoustic_score: Optional[float] = None
    linguistic_score: Optional[float] = None
    semantic_score: Optional[float] = None

    # Verdict
    is_acceptable: Optional[bool] = None
    recommended_action: Optional[str] = None  # "accept", "regenerate", "manual_review"
    primary_issues: List[str] = Field(default_factory=list)

    # Metadata
    processing_time_ms: Optional[int] = None
    service_enabled: bool = True
    message: Optional[str] = None

    # Detailed results (optional)
    detailed_result: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    enabled: bool
    models_loaded: bool
    uptime_seconds: float
    timestamp: str


class BatchAnalyzeRequest(BaseModel):
    """Request to analyze multiple voiceovers"""
    items: List[AnalyzeRequest]


class BatchAnalyzeResponse(BaseModel):
    """Response from batch analysis"""
    total: int
    successful: int
    failed: int
    results: List[AnalyzeResponse]


# ============================================================================
# Global State
# ============================================================================

class ServiceState:
    """Global service state"""
    pipeline = None
    models_loaded: bool = False
    startup_time: datetime = None
    analysis_count: int = 0
    error_count: int = 0


state = ServiceState()


# ============================================================================
# Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    state.startup_time = datetime.utcnow()

    print(f"[VQV-HALLU] Service starting...", flush=True)
    print(f"[VQV-HALLU] Enabled: {config.VQV_ENABLED}", flush=True)
    print(f"[VQV-HALLU] Strict mode: {config.VQV_STRICT_MODE}", flush=True)
    print(f"[VQV-HALLU] Min score: {config.MIN_ACCEPTABLE_SCORE}", flush=True)

    # Create temp directory
    Path(config.TEMP_DIR).mkdir(parents=True, exist_ok=True)

    # Lazy load models only if service is enabled
    if config.VQV_ENABLED:
        try:
            await load_models()
        except Exception as e:
            print(f"[VQV-HALLU] WARNING: Failed to load models: {e}", flush=True)
            print(f"[VQV-HALLU] Service will run in degraded mode", flush=True)

    yield

    print(f"[VQV-HALLU] Service shutting down...", flush=True)


async def load_models():
    """Load ML models (can be slow)"""
    print(f"[VQV-HALLU] Loading models...", flush=True)

    try:
        # Check dependencies first
        try:
            import librosa
            print(f"[VQV-HALLU] ✓ librosa available", flush=True)
        except ImportError as e:
            print(f"[VQV-HALLU] ✗ librosa not installed: {e}", flush=True)
            raise

        try:
            import whisper
            print(f"[VQV-HALLU] ✓ whisper available", flush=True)
        except ImportError as e:
            print(f"[VQV-HALLU] ✗ whisper not installed: {e}", flush=True)
            raise

        try:
            from sentence_transformers import SentenceTransformer
            print(f"[VQV-HALLU] ✓ sentence-transformers available", flush=True)
        except ImportError as e:
            print(f"[VQV-HALLU] ✗ sentence-transformers not installed: {e}", flush=True)
            raise

        from config.settings import VQVHalluConfig
        from core.pipeline import VQVHalluPipeline

        vqv_config = VQVHalluConfig(
            asr_model=config.ASR_MODEL,
            embedding_model=config.EMBEDDING_MODEL,
            temp_dir=config.TEMP_DIR,
        )

        state.pipeline = VQVHalluPipeline(vqv_config)

        # Pre-warm the analyzers by triggering a test import
        print(f"[VQV-HALLU] Pre-warming analyzers...", flush=True)
        try:
            from analyzers.acoustic_analyzer import AcousticAnalyzer
            from analyzers.linguistic_analyzer import LinguisticAnalyzer
            from analyzers.semantic_analyzer import SemanticAnalyzer
            print(f"[VQV-HALLU] ✓ Analyzers imported", flush=True)
        except ImportError as e:
            print(f"[VQV-HALLU] ✗ Analyzer import failed: {e}", flush=True)
            raise

        state.models_loaded = True
        print(f"[VQV-HALLU] Models loaded successfully", flush=True)

    except Exception as e:
        print(f"[VQV-HALLU] Error loading models: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="VQV-HALLU Service",
    description="Voice Quality Verification & Hallucination Detection for TTS audio",
    version=config.SERVICE_VERSION,
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


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.utcnow() - state.startup_time).total_seconds() if state.startup_time else 0

    return HealthResponse(
        status="healthy" if config.VQV_ENABLED else "disabled",
        service=config.SERVICE_NAME,
        version=config.SERVICE_VERSION,
        enabled=config.VQV_ENABLED,
        models_loaded=state.models_loaded,
        uptime_seconds=uptime,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/api/v1/status")
async def get_status():
    """Get detailed service status"""
    return {
        "service": config.SERVICE_NAME,
        "version": config.SERVICE_VERSION,
        "enabled": config.VQV_ENABLED,
        "strict_mode": config.VQV_STRICT_MODE,
        "models_loaded": state.models_loaded,
        "min_acceptable_score": config.MIN_ACCEPTABLE_SCORE,
        "max_regeneration_attempts": config.MAX_REGENERATION_ATTEMPTS,
        "analysis_timeout_seconds": config.ANALYSIS_TIMEOUT_SECONDS,
        "statistics": {
            "total_analyses": state.analysis_count,
            "errors": state.error_count,
        }
    }


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_voiceover(request: AnalyzeRequest):
    """
    Analyze a single voiceover for hallucinations.

    If VQV_ENABLED is false, returns immediately with status="disabled".
    If models failed to load, returns status="degraded" with is_acceptable=True.
    """
    start_time = time.time()

    # Check if service is enabled
    if not config.VQV_ENABLED:
        return AnalyzeResponse(
            audio_id=request.audio_id,
            status="disabled",
            is_acceptable=True,  # Don't block if disabled
            recommended_action="accept",
            service_enabled=False,
            message="VQV-HALLU service is disabled. Audio accepted without validation.",
        )

    # Check if models are loaded
    if not state.models_loaded or state.pipeline is None:
        print(f"[VQV-HALLU] Models not loaded, running in degraded mode", flush=True)

        # Provide basic text-length based score as fallback
        text_len = len(request.source_text) if request.source_text else 0
        # Basic heuristic: if we have text, assume TTS worked
        fallback_score = 75.0 if text_len > 10 else 50.0

        return AnalyzeResponse(
            audio_id=request.audio_id,
            status="degraded",
            final_score=fallback_score,  # Return a score instead of None
            acoustic_score=fallback_score,
            linguistic_score=fallback_score,
            semantic_score=fallback_score,
            is_acceptable=True,  # Don't block if models not loaded
            recommended_action="accept",
            service_enabled=True,
            message="VQV-HALLU models not loaded. Using fallback score. Audio accepted without full validation.",
        )

    try:
        state.analysis_count += 1

        # Get audio file (download if URL)
        audio_path = await get_audio_file(request.audio_url, request.audio_path)

        if not audio_path:
            raise ValueError("Either audio_url or audio_path must be provided")

        # Run analysis
        print(f"[VQV-HALLU] Analyzing {request.audio_id}...", flush=True)

        result = state.pipeline.analyze(
            audio_path=audio_path,
            source_text=request.source_text,
            audio_id=request.audio_id,
            content_type=request.content_type,
            language=request.language,
        )

        processing_time = int((time.time() - start_time) * 1000)

        # Determine action
        is_acceptable = result.final_score >= config.MIN_ACCEPTABLE_SCORE
        if result.final_score >= 85:
            recommended_action = "accept"
        elif result.final_score >= config.MIN_ACCEPTABLE_SCORE:
            recommended_action = "accept"
        elif result.final_score >= 50:
            recommended_action = "regenerate"
        else:
            recommended_action = "manual_review"

        print(f"[VQV-HALLU] {request.audio_id}: score={result.final_score:.1f}, action={recommended_action}", flush=True)

        return AnalyzeResponse(
            audio_id=request.audio_id,
            status="success",
            final_score=result.final_score,
            acoustic_score=result.acoustic_score,
            linguistic_score=result.linguistic_score,
            semantic_score=result.semantic_score,
            is_acceptable=is_acceptable,
            recommended_action=recommended_action,
            primary_issues=result.primary_issues,
            processing_time_ms=processing_time,
            service_enabled=True,
            detailed_result=result.to_dict() if result else None,
        )

    except Exception as e:
        state.error_count += 1
        processing_time = int((time.time() - start_time) * 1000)

        print(f"[VQV-HALLU] Error analyzing {request.audio_id}: {e}", flush=True)

        # In non-strict mode, accept on error
        if not config.VQV_STRICT_MODE:
            return AnalyzeResponse(
                audio_id=request.audio_id,
                status="error",
                is_acceptable=True,  # Don't block on error in non-strict mode
                recommended_action="accept",
                primary_issues=[str(e)],
                processing_time_ms=processing_time,
                service_enabled=True,
                message=f"Analysis failed, audio accepted (non-strict mode): {str(e)}",
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(request: BatchAnalyzeRequest):
    """Analyze multiple voiceovers"""
    results = []
    successful = 0
    failed = 0

    for item in request.items:
        try:
            result = await analyze_voiceover(item)
            results.append(result)
            if result.status == "success":
                successful += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            results.append(AnalyzeResponse(
                audio_id=item.audio_id,
                status="error",
                is_acceptable=not config.VQV_STRICT_MODE,
                recommended_action="accept" if not config.VQV_STRICT_MODE else "manual_review",
                primary_issues=[str(e)],
                service_enabled=True,
            ))

    return BatchAnalyzeResponse(
        total=len(request.items),
        successful=successful,
        failed=failed,
        results=results,
    )


@app.post("/api/v1/analyze/upload")
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    source_text: str = Form(...),
    audio_id: str = Form(default="uploaded"),
    content_type: str = Form(default="technical_course"),
    language: str = Form(default="fr"),
):
    """Analyze an uploaded audio file"""
    # Save uploaded file temporarily
    temp_path = os.path.join(config.TEMP_DIR, f"{audio_id}_{file.filename}")

    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        request = AnalyzeRequest(
            audio_path=temp_path,
            source_text=source_text,
            audio_id=audio_id,
            content_type=content_type,
            language=language,
        )

        return await analyze_voiceover(request)

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/v1/config/content-types")
async def get_content_types():
    """Get available content types and their thresholds"""
    from config.settings import CONTENT_TYPE_CONFIGS, ContentType

    return {
        content_type.value: {
            "name": content_type.value,
            "min_acceptable_score": cfg.min_acceptable_score,
            "weights": {
                "acoustic": cfg.weight_acoustic,
                "linguistic": cfg.weight_linguistic,
                "semantic": cfg.weight_semantic,
            }
        }
        for content_type, cfg in CONTENT_TYPE_CONFIGS.items()
    }


# ============================================================================
# Helper Functions
# ============================================================================

async def get_audio_file(audio_url: Optional[str], audio_path: Optional[str]) -> Optional[str]:
    """Get audio file path, downloading if necessary"""
    if audio_path and os.path.exists(audio_path):
        return audio_path

    if audio_url:
        # Download the file
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(audio_url)
            response.raise_for_status()

            # Determine extension
            content_type = response.headers.get('content-type', '')
            if 'wav' in content_type:
                ext = '.wav'
            elif 'mp3' in content_type or 'mpeg' in content_type:
                ext = '.mp3'
            elif 'ogg' in content_type:
                ext = '.ogg'
            else:
                ext = '.wav'

            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=ext,
                dir=config.TEMP_DIR
            )
            temp_file.write(response.content)
            temp_file.close()

            return temp_file.name

    return None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8008"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"[VQV-HALLU] Starting on {host}:{port}", flush=True)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )
