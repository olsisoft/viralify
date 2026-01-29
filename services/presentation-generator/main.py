"""
Presentation Generator Service

FastAPI service for generating code presentations and tutorials.
Transforms prompts into professional video presentations with slides,
syntax-highlighted code, and voiceover narration.
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from models.presentation_models import (
    GeneratePresentationRequest,
    PresentationJob,
    PresentationStyle,
    SlidePreviewRequest,
    SlidePreviewResponse,
    LanguageInfo,
    StyleInfo,
    Slide,
    SlideType,
)
from services.presentation_compositor import PresentationCompositorService
from services.slide_generator import SlideGeneratorService
from services.redis_job_store import job_store, RedisConnectionError
from services.voiceover_enforcer import enforce_voiceover_duration

# WeaveGraph imports for concept extraction (Phase 2 & 3)
try:
    from services.weave_graph import WeaveGraphBuilder, get_weave_graph_builder
    HAS_WEAVE_GRAPH = True
except ImportError:
    HAS_WEAVE_GRAPH = False
    WeaveGraphBuilder = None

# Import VisualGenerator module (Phase 6)
import sys
# Try Docker mount path first, then local development path
# NOTE: Docker mounts to /app/visual_generator (underscore), not /app/visual-generator
for visual_path in ['/app/visual_generator', '/app', '../visual-generator']:
    if visual_path not in sys.path:
        sys.path.insert(0, visual_path)
try:
    from visual_generator import (
        VisualGeneratorService,
        VisualGenerationRequest,
        DiagramStyle,
    )
    VISUAL_GENERATOR_AVAILABLE = True
    print("[STARTUP] VisualGenerator module loaded successfully", flush=True)
except ImportError as e:
    print(f"[STARTUP] VisualGenerator module not available: {e}", flush=True)
    VISUAL_GENERATOR_AVAILABLE = False
    VisualGeneratorService = None
    VisualGenerationRequest = None
    DiagramStyle = None

# Import NEXUS Adapter for pedagogical code generation (Phase 8B)
try:
    from services.nexus_adapter import (
        NexusAdapterService,
        NexusGenerationResult,
        get_nexus_adapter,
    )
    NEXUS_ADAPTER_AVAILABLE = True
    print("[STARTUP] NEXUS Adapter loaded successfully", flush=True)
except ImportError as e:
    print(f"[STARTUP] NEXUS Adapter not available: {e}", flush=True)
    NEXUS_ADAPTER_AVAILABLE = False
    NexusAdapterService = None
    NexusGenerationResult = None
    get_nexus_adapter = None


# Global services
compositor: PresentationCompositorService = None
slide_generator: SlideGeneratorService = None
visual_generator: Optional[VisualGeneratorService] = None
weave_graph_builder: Optional[WeaveGraphBuilder] = None
nexus_adapter: Optional[NexusAdapterService] = None

# Feature flags
USE_NEXUS = os.getenv("USE_NEXUS", "true").lower() == "true"


async def extract_concepts_background(rag_context: str, user_id: str, document_id: str):
    """
    Background task to extract concepts from RAG context and store in WeaveGraph.

    This populates the concept graph for:
    - Query expansion (Phase 2)
    - Resonance propagation (Phase 3)
    """
    global weave_graph_builder

    if not HAS_WEAVE_GRAPH or not rag_context:
        return

    try:
        # Get singleton builder (creates if needed)
        if weave_graph_builder is None:
            weave_graph_builder = get_weave_graph_builder()

        # Extract and store concepts from the RAG content (skips if already processed)
        new_concepts = await weave_graph_builder.add_document(
            document_id=document_id,
            content=rag_context,
            user_id=user_id
        )

        if new_concepts > 0:
            print(f"[WEAVE_GRAPH] Extracted {new_concepts} concepts from document {document_id[:8]}...", flush=True)

    except Exception as e:
        print(f"[WEAVE_GRAPH] Concept extraction failed: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global compositor, slide_generator, visual_generator, nexus_adapter

    print("[STARTUP] Initializing Presentation Generator Service...", flush=True)

    # Initialize services
    compositor = PresentationCompositorService()
    slide_generator = SlideGeneratorService()

    # Initialize Visual Generator (Phase 6)
    if VISUAL_GENERATOR_AVAILABLE:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        output_dir = os.getenv("VISUAL_OUTPUT_DIR", "/tmp/viralify/visuals")
        visual_generator = VisualGeneratorService(
            openai_api_key=openai_api_key,
            output_dir=output_dir,
        )
        print(f"[STARTUP] Visual Generator initialized (output: {output_dir})", flush=True)

    # Initialize NEXUS Adapter (Phase 8B - Pedagogical Code Generation)
    if USE_NEXUS and NEXUS_ADAPTER_AVAILABLE:
        nexus_url = os.getenv("NEXUS_ENGINE_URL", "http://nexus-engine:8009")
        nexus_adapter = get_nexus_adapter()
        # Check if NEXUS is available (non-blocking)
        try:
            is_available = await nexus_adapter.is_available()
            if is_available:
                print(f"[STARTUP] NEXUS Adapter initialized - engine available at {nexus_url}", flush=True)
            else:
                print(f"[STARTUP] NEXUS Adapter initialized - engine not reachable (will retry on use)", flush=True)
        except Exception as e:
            print(f"[STARTUP] NEXUS Adapter init warning: {e}", flush=True)
    else:
        print(f"[STARTUP] NEXUS mode: {'disabled' if not USE_NEXUS else 'adapter not available'}", flush=True)

    print("[STARTUP] Services initialized", flush=True)

    yield

    print("[SHUTDOWN] Cleaning up...", flush=True)
    # Close NEXUS adapter
    if nexus_adapter:
        await nexus_adapter.close()
    # Close Redis connection
    await job_store.close()


# Create FastAPI app
app = FastAPI(
    title="Presentation Generator Service",
    description="Generate professional code presentations and tutorials from prompts",
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


# ==============================================================================
# HEALTH CHECK
# ==============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "presentation-generator",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/nexus/status")
async def nexus_status():
    """
    Check NEXUS Engine status.

    Returns availability and configuration of the NEXUS pedagogical
    code generation service.
    """
    if not USE_NEXUS:
        return {
            "enabled": False,
            "available": False,
            "message": "NEXUS is disabled via USE_NEXUS=false",
        }

    if not NEXUS_ADAPTER_AVAILABLE:
        return {
            "enabled": True,
            "available": False,
            "message": "NEXUS adapter module not loaded",
        }

    if not nexus_adapter:
        return {
            "enabled": True,
            "available": False,
            "message": "NEXUS adapter not initialized",
        }

    try:
        is_available = await nexus_adapter.is_available()
        return {
            "enabled": True,
            "available": is_available,
            "message": "NEXUS engine ready" if is_available else "NEXUS engine not reachable",
            "url": os.getenv("NEXUS_ENGINE_URL", "http://nexus-engine:8009"),
        }
    except Exception as e:
        return {
            "enabled": True,
            "available": False,
            "message": f"Error checking NEXUS: {str(e)}",
        }


# ==============================================================================
# PRESENTATION GENERATION
# ==============================================================================

@app.post("/api/v1/presentations/generate", response_model=Dict)
async def generate_presentation(
    request: GeneratePresentationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a complete presentation from a topic prompt.

    This endpoint starts an async job that:
    1. Uses GPT-4 to plan the presentation structure
    2. Generates slide images with syntax highlighting
    3. Creates voiceover narration
    4. Composes the final video

    Returns a job_id to track progress.
    """
    print(f"[GENERATE] Starting presentation for: {request.topic[:50]}...", flush=True)
    # Debug: Check RAG context received from course-generator
    rag_ctx = getattr(request, 'rag_context', None)
    doc_ids = getattr(request, 'document_ids', []) or []
    source_ids = getattr(request, 'source_ids', []) or []
    print(f"[GENERATE] RAG context received: {len(rag_ctx) if rag_ctx else 0} chars, document_ids: {len(doc_ids)}, source_ids: {len(source_ids)}", flush=True)

    # Phase 2 & 3: Extract concepts from RAG context for WeaveGraph
    if rag_ctx and HAS_WEAVE_GRAPH:
        user_id = getattr(request, 'user_id', 'default')
        # Use first document/source ID or generate one
        doc_id = doc_ids[0] if doc_ids else (source_ids[0] if source_ids else f"rag_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
        background_tasks.add_task(extract_concepts_background, rag_ctx, user_id, doc_id)

    try:
        job = await compositor.generate_presentation(request)

        return {
            "success": True,
            "job_id": job.job_id,
            "status": job.status,
            "message": "Presentation generation started",
            "estimated_duration_seconds": request.duration
        }

    except Exception as e:
        print(f"[GENERATE] Error: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/presentations/generate/v2", response_model=Dict)
async def generate_presentation_v2(
    request: GeneratePresentationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a presentation using the LangGraph orchestrator (V2).

    This endpoint uses an improved pipeline with:
    - Visual-audio alignment validation
    - Automatic timing synchronization
    - Code execution with output capture
    - Feedback loops for content correction

    Returns a job_id to track progress.
    """
    from services.langgraph_orchestrator import LangGraphOrchestrator
    import uuid

    print(f"[GENERATE-V2] Starting LangGraph presentation for: {request.topic[:50]}...", flush=True)

    # Phase 2 & 3: Extract concepts from RAG context for WeaveGraph
    rag_ctx = getattr(request, 'rag_context', None)
    if rag_ctx and HAS_WEAVE_GRAPH:
        user_id = getattr(request, 'user_id', 'default')
        doc_ids = getattr(request, 'document_ids', []) or []
        source_ids = getattr(request, 'source_ids', []) or []
        doc_id = doc_ids[0] if doc_ids else (source_ids[0] if source_ids else f"rag_v2_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
        background_tasks.add_task(extract_concepts_background, rag_ctx, user_id, doc_id)

    job_id = str(uuid.uuid4())

    # Store initial job status in Redis
    initial_job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "phase": "initializing",
        "created_at": datetime.utcnow().isoformat(),
        "request": request.model_dump()
    }
    await job_store.save(job_id, initial_job, prefix="v2")

    # Run in background
    background_tasks.add_task(
        _run_langgraph_generation,
        job_id,
        request
    )

    return {
        "success": True,
        "job_id": job_id,
        "status": "pending",
        "message": "LangGraph presentation generation started (V2)",
        "estimated_duration_seconds": request.duration
    }


# LangGraph jobs now stored in Redis via job_store (prefix="v2")
# Legacy dict removed - use job_store.get(job_id, prefix="v2") instead


async def _run_langgraph_generation(job_id: str, request: GeneratePresentationRequest):
    """Background task to run LangGraph generation"""
    from services.langgraph_orchestrator import LangGraphOrchestrator
    from services.rag_client import get_rag_client

    try:
        await job_store.update_field(job_id, "status", "processing", prefix="v2")

        # Fetch RAG context if documents are provided
        document_ids = getattr(request, 'document_ids', [])
        if document_ids and not request.rag_context:
            await job_store.update_field(job_id, "phase", "Fetching document context...", prefix="v2")
            print(f"[GENERATE-V2] Fetching RAG context for {len(document_ids)} documents", flush=True)

            rag_client = get_rag_client()
            rag_context = await rag_client.get_context_for_presentation(
                document_ids=document_ids,
                topic=request.topic,
                max_chunks=40,  # Increased from 15 for better RAG coverage (90%+)
                include_diagrams=getattr(request, 'use_documents_for_diagrams', True)
            )

            if rag_context:
                request.rag_context = rag_context
                print(f"[GENERATE-V2] RAG context fetched: {len(rag_context)} chars", flush=True)
            else:
                print(f"[GENERATE-V2] No RAG context retrieved", flush=True)

        orchestrator = LangGraphOrchestrator()

        async def on_progress(percent: int, message: str):
            await job_store.update_fields(job_id, {
                "progress": percent,
                "phase": message
            }, prefix="v2")
            print(f"[GENERATE-V2] Job {job_id}: {percent}% - {message}", flush=True)

        result = await orchestrator.generate_video(request, job_id, on_progress)

        # Determine status: if video was generated, consider it successful even with warnings
        has_output = result.get("output_video_url") is not None
        has_critical_errors = any("failed" in str(e).lower() or "error" in str(e).lower()
                                   for e in result.get("errors", [])
                                   if "voiceover" not in str(e).lower())  # Voiceover failure is non-critical

        if has_output and not has_critical_errors:
            status = "completed"
        elif has_output:
            status = "completed_with_warnings"
        else:
            status = "failed"

        await job_store.update_fields(job_id, {
            "status": status,
            "progress": 100,
            "result": result,
            "completed_at": datetime.utcnow().isoformat()
        }, prefix="v2")

    except Exception as e:
        print(f"[GENERATE-V2] Error for job {job_id}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        await job_store.update_fields(job_id, {
            "status": "failed",
            "error": str(e)
        }, prefix="v2")


@app.get("/api/v1/presentations/jobs/v2/{job_id}")
async def get_langgraph_job_status(job_id: str):
    """Get the status of a LangGraph presentation generation job (V2)."""
    try:
        job = await job_store.get(job_id, prefix="v2")
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except RedisConnectionError as e:
        print(f"[JOB_STATUS] Redis unavailable for V2 job {job_id}: {e}", flush=True)
        raise HTTPException(
            status_code=503,
            detail="Job store temporarily unavailable. Please retry.",
            headers={"Retry-After": "5"}
        )


# ==============================================================================
# V3: MULTI-AGENT SYSTEM (Scene-by-Scene with Perfect Sync)
# ==============================================================================

# MultiAgent V3 jobs now stored in Redis via job_store (prefix="v3")
# Legacy dict removed - use job_store.get(job_id, prefix="v3") instead


@app.post("/api/v1/presentations/generate/v3", response_model=Dict)
async def generate_presentation_v3(
    request: GeneratePresentationRequest,
    background_tasks: BackgroundTasks,
    enable_visuals: bool = False,
    visual_style: str = "dark",
):
    """
    Generate a presentation using the Multi-Agent Scene-by-Scene architecture (V3).

    This is the most advanced pipeline featuring:
    - Scene-by-Scene processing with local sync validation
    - Word-level audio timestamps for precise sync
    - Visual elements aligned to exact audio moments
    - Automatic regeneration if sync fails
    - Parallel scene processing for speed
    - (NEW) AI-generated diagrams and visualizations

    Each scene goes through:
    1. Scene Planner Agent - Plans timing cues
    2. Audio Agent - Generates TTS with word timestamps
    3. Visual Sync Agent - Aligns visuals to audio
    4. Animation Agent - Creates timed animations
    5. (NEW) Visual Generator - Creates diagrams/charts if needed
    6. Scene Validator - Verifies sync, triggers regeneration if needed
    7. Compositor Agent - Assembles final video

    Parameters:
    - enable_visuals: Enable AI diagram/chart generation for slides
    - visual_style: Style for generated visuals (dark, light, colorful)

    Returns a job_id to track progress.
    """
    import uuid

    print(f"[GENERATE-V3] Starting Multi-Agent presentation for: {request.topic[:50]}...", flush=True)
    if enable_visuals:
        print(f"[GENERATE-V3] Visual generation enabled (style: {visual_style})", flush=True)

    # Phase 2 & 3: Extract concepts from RAG context for WeaveGraph
    rag_ctx = getattr(request, 'rag_context', None)
    if rag_ctx and HAS_WEAVE_GRAPH:
        user_id = getattr(request, 'user_id', 'default')
        doc_ids = getattr(request, 'document_ids', []) or []
        source_ids = getattr(request, 'source_ids', []) or []
        doc_id = doc_ids[0] if doc_ids else (source_ids[0] if source_ids else f"rag_v3_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
        background_tasks.add_task(extract_concepts_background, rag_ctx, user_id, doc_id)

    job_id = str(uuid.uuid4())

    # Store initial job status in Redis
    initial_job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "phase": "initializing",
        "created_at": datetime.utcnow().isoformat(),
        "request": request.model_dump(),
        "scene_statuses": [],
        "scene_videos": [],  # Progressive download: individual lessons available as they complete
        "enable_visuals": enable_visuals,
        "visual_style": visual_style,
    }
    await job_store.save(job_id, initial_job, prefix="v3")

    # Run in background
    background_tasks.add_task(
        _run_multiagent_generation,
        job_id,
        request,
        enable_visuals,
        visual_style,
    )

    return {
        "success": True,
        "job_id": job_id,
        "status": "pending",
        "message": "Multi-Agent presentation generation started (V3 - Scene-by-Scene Sync)" + (" with AI visuals" if enable_visuals else ""),
        "estimated_duration_seconds": request.duration,
        "visuals_enabled": enable_visuals,
    }


async def _run_multiagent_generation(
    job_id: str,
    request: GeneratePresentationRequest,
    enable_visuals: bool = False,
    visual_style: str = "dark",
):
    """Background task to run multi-agent generation"""
    from services.agents import generate_presentation_video
    from services.script_generator import ScriptGenerator
    from services.rag_client import get_rag_client

    try:
        await job_store.update_fields(job_id, {
            "status": "processing",
            "phase": "fetching_documents",
            "progress": 2
        }, prefix="v3")

        # Fetch RAG context if documents are provided
        document_ids = getattr(request, 'document_ids', [])
        if document_ids and not request.rag_context:
            print(f"[GENERATE-V3] Fetching RAG context for {len(document_ids)} documents", flush=True)

            rag_client = get_rag_client()
            rag_context = await rag_client.get_context_for_presentation(
                document_ids=document_ids,
                topic=request.topic,
                max_chunks=40,  # Increased from 15 for better RAG coverage (90%+)
                include_diagrams=getattr(request, 'use_documents_for_diagrams', True)
            )

            if rag_context:
                request.rag_context = rag_context
                print(f"[GENERATE-V3] RAG context fetched: {len(rag_context)} chars", flush=True)

        await job_store.update_fields(job_id, {
            "phase": "generating_script",
            "progress": 5
        }, prefix="v3")

        # Step 1: Generate script/slides using GPT-4
        script_generator = ScriptGenerator()
        script = await script_generator.generate_script(
            topic=request.topic,
            language=request.language,
            style=request.style.value,
            duration=request.duration,
            execute_code=request.execute_code,
            content_language=request.content_language
        )

        # Step 1.5: ENFORCE VOICEOVER DURATION - expand short voiceovers
        # Convert script to dict format for enforcer
        script_data_for_enforcer = {
            "slides": [
                {
                    "title": s.title,
                    "type": s.type.value,
                    "voiceover_text": s.voiceover_text,
                    "bullet_points": s.bullet_points or [],
                    "code_blocks": [{"code": cb.code, "language": cb.language} for cb in (s.code_blocks or [])],
                    "diagram_description": getattr(s, 'diagram_description', ''),
                }
                for s in script.slides
            ]
        }

        content_language = getattr(request, 'content_language', 'en') or 'en'
        enforced_script, enforcement_result = await enforce_voiceover_duration(
            script_data=script_data_for_enforcer,
            target_duration=request.duration,
            content_language=content_language
        )

        # Update script slides with enforced voiceovers
        if enforcement_result.slides_expanded > 0:
            print(f"[GENERATE-V3] ENFORCER: Expanded {enforcement_result.slides_expanded}/{enforcement_result.total_slides} slides", flush=True)
            print(f"[GENERATE-V3] ENFORCER: {enforcement_result.original_words} -> {enforcement_result.final_words} words ({enforcement_result.duration_ratio:.0%})", flush=True)

            # Apply expanded voiceovers back to script
            for i, enforced_slide in enumerate(enforced_script.get("slides", [])):
                if i < len(script.slides):
                    script.slides[i].voiceover_text = enforced_slide.get("voiceover_text", script.slides[i].voiceover_text)

        # Convert slides to format expected by multi-agent system
        slides = []
        for slide in script.slides:
            slide_data = {
                "id": getattr(slide, 'id', None) or str(uuid.uuid4())[:8],
                "title": slide.title,
                "type": slide.type.value,
                "voiceover_text": slide.voiceover_text,
                "duration": slide.duration
            }

            # Add code if present
            if slide.code_blocks:
                slide_data["code"] = slide.code_blocks[0].code
                slide_data["language"] = slide.code_blocks[0].language
                slide_data["code_blocks"] = [
                    {
                        "id": getattr(cb, 'id', None) or str(uuid.uuid4())[:8],
                        "code": cb.code,
                        "language": cb.language,
                        "filename": getattr(cb, 'filename', None),
                        "expected_output": cb.expected_output,
                    }
                    for cb in slide.code_blocks
                ]
                # Add expected output from code block
                if slide.code_blocks[0].expected_output:
                    slide_data["expected_output"] = slide.code_blocks[0].expected_output

            # Add bullet points
            if slide.bullet_points:
                slide_data["bullet_points"] = slide.bullet_points

            # Add other slide attributes for editing
            if hasattr(slide, 'subtitle') and slide.subtitle:
                slide_data["subtitle"] = slide.subtitle
            if hasattr(slide, 'content') and slide.content:
                slide_data["content"] = slide.content
            if hasattr(slide, 'diagram_type') and slide.diagram_type:
                slide_data["diagram_type"] = slide.diagram_type

            slides.append(slide_data)

        # Store slides in job for later retrieval by lecture editor
        await job_store.update_fields(job_id, {
            "phase": "processing_scenes",
            "progress": 20,
            "script": {
                "title": script.title,
                "slide_count": len(script.slides),
                "estimated_duration": script.total_duration,
                "slides": slides  # Store full slides for editing
            }
        }, prefix="v3")

        # Step 1.5: Generate visuals for slides if enabled (Phase 6)
        if enable_visuals and VISUAL_GENERATOR_AVAILABLE and visual_generator:
            await job_store.update_field(job_id, "phase", "generating_visuals", prefix="v3")
            print(f"[GENERATE-V3] Generating visuals for {len(slides)} slides...", flush=True)

            # Map style string to enum
            style_mapping = {
                "dark": DiagramStyle.DARK,
                "light": DiagramStyle.LIGHT,
                "colorful": DiagramStyle.COLORFUL,
            }
            style_enum = style_mapping.get(visual_style, DiagramStyle.DARK)

            visuals_generated = 0
            for idx, slide_data in enumerate(slides):
                try:
                    # Build content for detection
                    content = f"{slide_data.get('title', '')} {slide_data.get('voiceover_text', '')}"
                    if slide_data.get('code'):
                        content += f" Code: {slide_data['code']}"

                    # Generate visual
                    result = await visual_generator.generate_from_slide(
                        slide_content=slide_data,
                        lesson_context=script.title,
                        style=style_enum,
                    )

                    if result.success and result.file_path:
                        # Add visual asset to slide data
                        slide_data["visual_asset"] = {
                            "path": result.file_path,
                            "type": result.visual_type.value if result.visual_type else "unknown",
                            "renderer": result.renderer_used,
                            "duration": result.duration_seconds,
                        }
                        visuals_generated += 1
                        print(f"[GENERATE-V3] Visual generated for slide {idx}: {result.visual_type}", flush=True)

                except Exception as ve:
                    print(f"[GENERATE-V3] Visual generation warning for slide {idx}: {str(ve)}", flush=True)
                    # Continue without visual if generation fails

            await job_store.update_field(job_id, "visuals_generated", visuals_generated, prefix="v3")
            print(f"[GENERATE-V3] Generated {visuals_generated} visuals", flush=True)

        # Initialize scene statuses
        scene_statuses = [
            {"scene_index": i, "status": "pending", "sync_score": 0}
            for i in range(len(slides))
        ]
        await job_store.update_field(job_id, "scene_statuses", scene_statuses, prefix="v3")

        print(f"[GENERATE-V3] Processing {len(slides)} scenes for job {job_id}", flush=True)

        # Step 2: Run multi-agent video generation
        result = await generate_presentation_video(
            job_id=job_id,
            slides=slides,
            title=script.title,
            style=request.style.value,
            content_language=request.content_language
        )

        # Update final status
        if result.get("success"):
            status = "completed"
            phase = "completed"
        else:
            status = "failed"
            phase = "failed"

        # Update scene statuses from summary
        final_scene_statuses = scene_statuses.copy()
        if result.get("summary", {}).get("scene_packages"):
            for sp in result["summary"]["scene_packages"]:
                idx = sp.get("scene_index", 0)
                if idx < len(final_scene_statuses):
                    final_scene_statuses[idx] = {
                        "scene_index": idx,
                        "status": sp.get("sync_status", "unknown"),
                        "sync_score": sp.get("sync_score", 0)
                    }

        # Get scene videos for progressive download (from result or already in Redis)
        scene_videos = result.get("scene_videos", [])
        if not scene_videos:
            # Try to get from summary
            scene_videos = result.get("summary", {}).get("scene_videos", [])

        # CRITICAL: Update job status with retry - if this fails, the job would appear stuck
        final_update = {
            "status": status,
            "progress": 100,
            "phase": phase,
            "result": result,
            "output_url": result.get("output_url"),
            "duration": result.get("duration", 0),
            "summary": result.get("summary", {}),
            "completed_at": datetime.utcnow().isoformat(),
            "scene_statuses": final_scene_statuses,
            "scene_videos": scene_videos  # Individual lessons for progressive download
        }

        # Retry the status update up to 3 times to prevent stuck jobs
        update_success = False
        for attempt in range(3):
            update_success = await job_store.update_fields(job_id, final_update, prefix="v3")
            if update_success:
                break
            print(f"[GENERATE-V3] WARNING: Failed to update job status (attempt {attempt + 1}/3)", flush=True)
            await asyncio.sleep(0.5 * (attempt + 1))  # Backoff

        if not update_success:
            print(f"[GENERATE-V3] CRITICAL: Failed to update job {job_id} status after 3 attempts!", flush=True)
            # Still log completion for debugging
            print(f"[GENERATE-V3] Job {job_id} video generated but status update failed: {result.get('output_url')}", flush=True)
        else:
            print(f"[GENERATE-V3] Job {job_id} completed: {result.get('success')}", flush=True)

    except Exception as e:
        print(f"[GENERATE-V3] Error for job {job_id}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        await job_store.update_fields(job_id, {
            "status": "failed",
            "error": str(e),
            "phase": "error"
        }, prefix="v3")


@app.get("/api/v1/presentations/jobs/v3/{job_id}")
async def get_multiagent_job_status(job_id: str):
    """Get the status of a Multi-Agent presentation generation job (V3)."""
    try:
        job = await job_store.get(job_id, prefix="v3")
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except RedisConnectionError as e:
        print(f"[JOB_STATUS] Redis unavailable for V3 job {job_id}: {e}", flush=True)
        raise HTTPException(
            status_code=503,
            detail="Job store temporarily unavailable. Please retry.",
            headers={"Retry-After": "5"}
        )


@app.get("/api/v1/presentations/jobs/v3/{job_id}/lessons")
async def get_available_lessons(job_id: str):
    """
    Get individual lesson videos available for download (Progressive Download).

    This endpoint allows the frontend to poll for individual lesson videos
    as they become ready, without waiting for the entire presentation to complete.

    Returns:
        - lessons: List of lesson videos ready for download
        - total_lessons: Expected total number of lessons
        - completed: Number of lessons ready
        - status: Job status (processing, completed, failed)

    Each lesson includes:
        - scene_index: Lesson number (0-based)
        - title: Lesson title
        - video_url: Direct download URL
        - duration: Video duration in seconds
        - status: ready, failed, pending
        - ready_at: Timestamp when lesson became available
    """
    try:
        job = await job_store.get(job_id, prefix="v3")
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get scene videos (available lessons)
        scene_videos = job.get("scene_videos", [])

        # Get total expected lessons from scene_statuses or slides count
        scene_statuses = job.get("scene_statuses", [])
        total_lessons = len(scene_statuses) if scene_statuses else 0

        # If no scene_statuses yet, try to get from request slides
        if total_lessons == 0:
            request_data = job.get("request", {})
            total_lessons = len(request_data.get("slides", []))

        # Sort by scene_index
        scene_videos_sorted = sorted(
            scene_videos,
            key=lambda x: x.get("scene_index", 0)
        )

        return {
            "job_id": job_id,
            "status": job.get("status", "unknown"),
            "phase": job.get("phase", "unknown"),
            "progress": job.get("progress", 0),
            "total_lessons": total_lessons,
            "completed": len([v for v in scene_videos if v.get("status") == "ready"]),
            "lessons": scene_videos_sorted,
            "final_video_url": job.get("output_url") if job.get("status") == "completed" else None
        }

    except RedisConnectionError as e:
        print(f"[LESSONS] Redis unavailable for job {job_id}: {e}", flush=True)
        raise HTTPException(
            status_code=503,
            detail="Job store temporarily unavailable. Please retry.",
            headers={"Retry-After": "5"}
        )


# ==============================================================================
# JOB MANAGEMENT: RETRY, CANCEL, ERROR QUEUE
# ==============================================================================

from services.job_manager import job_manager


@app.get("/api/v1/presentations/jobs/v3/{job_id}/errors")
async def get_job_errors(job_id: str):
    """
    Get all errors for a job with editable content (Error Queue).

    Returns failed lessons with their original content that can be
    edited before retry.

    Response:
    {
        "job_id": "...",
        "status": "partial",
        "total_lessons": 10,
        "failed_count": 2,
        "errors": [
            {
                "scene_index": 3,
                "title": "Configuration Kafka",
                "error_type": "tts_failed",
                "error_message": "Voice generation timeout",
                "original_content": {
                    "voiceover_text": "...",
                    "code": "...",
                    "bullet_points": [...]
                },
                "editable": true,
                "retry_count": 0
            }
        ],
        "can_retry": true
    }
    """
    try:
        result = await job_manager.get_errors(job_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except RedisConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail="Job store temporarily unavailable. Please retry.",
            headers={"Retry-After": "5"}
        )


@app.patch("/api/v1/presentations/jobs/v3/{job_id}/lessons/{scene_index}")
async def update_lesson_content(
    job_id: str,
    scene_index: int,
    content: Dict[str, Any]
):
    """
    Update lesson content before retry.

    Allows users to fix errors in the voiceover text, code, or other
    content before retrying a failed lesson.

    Request body:
    {
        "voiceover_text": "Updated narration text...",
        "title": "New title",
        "code": "fixed_code()",
        "bullet_points": ["Point 1", "Point 2"]
    }

    Response:
    {
        "success": true,
        "message": "Lesson 3 content updated. Ready for retry.",
        "scene_index": 3
    }
    """
    try:
        result = await job_manager.update_lesson_content(job_id, scene_index, content)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except RedisConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail="Job store temporarily unavailable. Please retry.",
            headers={"Retry-After": "5"}
        )


@app.post("/api/v1/presentations/jobs/v3/{job_id}/lessons/{scene_index}/retry")
async def retry_lesson(
    job_id: str,
    scene_index: int,
    rebuild_final: bool = True
):
    """
    Retry a single failed lesson.

    Regenerates only the specified lesson and optionally rebuilds
    the final concatenated video.

    Parameters:
    - scene_index: Lesson index (0-based) to retry
    - rebuild_final: If true, rebuild the final video after retry (default: true)

    Response:
    {
        "success": true,
        "message": "Lesson 3 regenerated successfully",
        "scene_index": 3,
        "video_url": "https://olsitec.com/media/files/videos/abc_scene_003.mp4",
        "final_video_url": "https://olsitec.com/media/files/videos/abc_final.mp4"
    }
    """
    try:
        result = await job_manager.retry_lesson(job_id, scene_index, rebuild_final)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except RedisConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail="Job store temporarily unavailable. Please retry.",
            headers={"Retry-After": "5"}
        )


@app.post("/api/v1/presentations/jobs/v3/{job_id}/retry")
async def retry_failed_lessons(job_id: str):
    """
    Retry all failed lessons in a job.

    Regenerates all lessons that failed and rebuilds the final video.

    Response:
    {
        "success": true,
        "message": "Retried 2 lessons, 0 still failed",
        "retried": [3, 7],
        "failed": [],
        "final_video_url": "https://olsitec.com/media/files/videos/abc_final.mp4"
    }
    """
    try:
        result = await job_manager.retry_failed_lessons(job_id)
        return result
    except RedisConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail="Job store temporarily unavailable. Please retry.",
            headers={"Retry-After": "5"}
        )


@app.post("/api/v1/presentations/jobs/v3/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    keep_completed: bool = True
):
    """
    Cancel a job in progress (Graceful Cancellation).

    Parameters:
    - keep_completed: If true (default), keep completed lessons and mark
                     job as "partial". If false, mark entire job as cancelled.

    Response:
    {
        "success": true,
        "message": "Job cancelled. 5 lessons completed and available.",
        "status": "partial",
        "completed_lessons": [0, 1, 2, 3, 4],
        "cancelled_lessons": [5, 6, 7],
        "output_url": "https://olsitec.com/media/files/videos/abc_final.mp4"
    }

    The output_url contains a partial video with only the completed lessons
    if keep_completed=true.
    """
    try:
        result = await job_manager.cancel_job(job_id, keep_completed)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except RedisConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail="Job store temporarily unavailable. Please retry.",
            headers={"Retry-After": "5"}
        )


@app.post("/api/v1/presentations/jobs/v3/{job_id}/rebuild")
async def rebuild_final_video(job_id: str):
    """
    Rebuild the final video from all available scene videos.

    Useful after manual fixes or when the final concatenation failed
    but individual scenes are available.

    Response:
    {
        "success": true,
        "message": "Final video rebuilt",
        "output_url": "https://olsitec.com/media/files/videos/abc_final.mp4"
    }
    """
    try:
        result = await job_manager.rebuild_final_video(job_id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return result
    except RedisConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail="Job store temporarily unavailable. Please retry.",
            headers={"Retry-After": "5"}
        )


@app.get("/api/v1/presentations/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a presentation generation job.

    Returns current progress, stage, and results when complete.
    Checks all job stores: compositor, V2 (LangGraph), and V3 (MultiAgent).
    """
    try:
        # Check V3 (MultiAgent) jobs first as it's the most recent version
        v3_job = await job_store.get(job_id, prefix="v3")
        if v3_job is not None:
            return v3_job

        # Check V2 (LangGraph) jobs
        v2_job = await job_store.get(job_id, prefix="v2")
        if v2_job is not None:
            return v2_job
    except RedisConnectionError as e:
        # Redis is temporarily unavailable - return 503 to tell frontend to retry
        print(f"[JOB_STATUS] Redis unavailable for job {job_id}: {e}", flush=True)
        raise HTTPException(
            status_code=503,
            detail="Job store temporarily unavailable. Please retry.",
            headers={"Retry-After": "5"}  # Suggest retry after 5 seconds
        )

    # Check legacy compositor jobs (still in-memory for backwards compatibility)
    job = compositor.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job.job_id,
        "status": job.status,
        "current_stage": job.current_stage.value,
        "progress": job.progress,
        "message": job.message,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }

    # Add script info if available
    if job.script:
        response["script"] = {
            "title": job.script.title,
            "description": job.script.description,
            "slide_count": job.script.slide_count,
            "total_duration": job.script.total_duration
        }

    # Add output URLs if available
    if job.slide_images:
        response["slide_images"] = job.slide_images

    if job.voiceover_url:
        response["voiceover_url"] = job.voiceover_url

    if job.output_url:
        response["output_url"] = job.output_url

    # Phase 2: Add code execution results
    if job.code_execution_results:
        response["code_execution_results"] = job.code_execution_results

    # Phase 2: Add animation videos
    if job.animation_videos:
        response["animation_videos"] = job.animation_videos

    # Phase 2: Add avatar video URL
    if job.avatar_video_url:
        response["avatar_video_url"] = job.avatar_video_url

    # Add error info if failed
    if job.error:
        response["error"] = job.error
        response["error_details"] = job.error_details

    if job.completed_at:
        response["completed_at"] = job.completed_at.isoformat()

    return response


@app.get("/api/v1/presentations/jobs")
async def list_jobs(limit: int = 20):
    """
    List recent presentation generation jobs.
    """
    jobs = compositor.list_jobs(limit)

    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "title": job.script.title if job.script else None,
                "created_at": job.created_at.isoformat(),
            }
            for job in jobs
        ],
        "total": len(jobs)
    }


# ==============================================================================
# SLIDE PREVIEW
# ==============================================================================

@app.post("/api/v1/presentations/slides/preview")
async def preview_slide(request: SlidePreviewRequest):
    """
    Generate a preview image for a single slide.

    Useful for testing slide appearance before generating full presentation.
    """
    try:
        image_bytes = await slide_generator.generate_slide_image(
            request.slide,
            request.style
        )

        # Upload and get URL
        filename = f"preview_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png"
        image_url = await slide_generator.upload_to_cloudinary(image_bytes, filename)

        return SlidePreviewResponse(
            image_url=image_url,
            width=1920,
            height=1080
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# CONFIGURATION ENDPOINTS
# ==============================================================================

@app.get("/api/v1/presentations/languages")
async def get_supported_languages():
    """
    Get list of supported programming languages for code slides.
    """
    config_path = Path(__file__).parent / "config" / "languages.json"

    if not config_path.exists():
        return {"languages": []}

    with open(config_path, "r") as f:
        config = json.load(f)

    languages = []
    for lang_id, lang_data in config.get("languages", {}).items():
        languages.append(LanguageInfo(
            id=lang_id,
            name=lang_data.get("name", lang_id),
            file_extension=lang_data.get("file_extension", ""),
            supported=lang_data.get("supported", True),
            icon=lang_data.get("icon", "code")
        ))

    return {
        "languages": [lang.model_dump() for lang in languages],
        "total": len(languages)
    }


@app.get("/api/v1/presentations/styles")
async def get_available_styles():
    """
    Get list of available presentation styles/themes.
    """
    config_path = Path(__file__).parent / "config" / "languages.json"

    if not config_path.exists():
        return {"styles": []}

    with open(config_path, "r") as f:
        config = json.load(f)

    styles = []
    for style_id, style_data in config.get("styles", {}).items():
        styles.append(StyleInfo(
            id=style_id,
            name=style_data.get("name", style_id),
            preview_colors={
                "background": style_data.get("background_color", "#1e1e2e"),
                "text": style_data.get("text_color", "#ffffff"),
                "accent": style_data.get("accent_color", "#89b4fa"),
                "code_bg": style_data.get("code_background", "#181825")
            }
        ))

    return {
        "styles": [style.model_dump() for style in styles],
        "total": len(styles)
    }


# ==============================================================================
# QUICK TEST ENDPOINT
# ==============================================================================

class QuickTestRequest(BaseModel):
    """Quick test for slide generation"""
    code: str = Field(..., description="Code to render")
    language: str = Field(default="python")
    title: str = Field(default="Code Example")
    style: PresentationStyle = Field(default=PresentationStyle.DARK)


@app.post("/api/v1/presentations/test/code-slide")
async def test_code_slide(request: QuickTestRequest):
    """
    Quick test endpoint to generate a single code slide.

    Useful for testing syntax highlighting and styling.
    """
    from models.presentation_models import CodeBlock

    slide = Slide(
        type=SlideType.CODE,
        title=request.title,
        code_blocks=[
            CodeBlock(
                language=request.language,
                code=request.code,
                filename=f"example.{request.language}"
            )
        ],
        duration=10.0,
        voiceover_text=""
    )

    try:
        image_bytes = await slide_generator.generate_slide_image(slide, request.style)
        filename = f"test_code_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png"
        image_url = await slide_generator.upload_to_cloudinary(image_bytes, filename)

        return {
            "success": True,
            "image_url": image_url,
            "language": request.language,
            "style": request.style.value
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# STATIC FILE SERVING (for inter-service communication)
# ==============================================================================

@app.get("/files/presentations/{filename}")
async def serve_presentation_file(filename: str):
    """
    Serve generated presentation files (slides, etc.)
    This endpoint allows media-generator to access slide images.
    """
    import tempfile

    # Security: only allow specific file patterns
    if not filename.endswith(('.png', '.jpg', '.jpeg', '.mp4', '.mp3')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Sanitize filename
    safe_filename = Path(filename).name  # Remove any path components

    file_path = Path(tempfile.gettempdir()) / "presentations" / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine content type
    content_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.mp4': 'video/mp4',
        '.mp3': 'audio/mpeg'
    }
    suffix = file_path.suffix.lower()
    content_type = content_types.get(suffix, 'application/octet-stream')

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=safe_filename
    )


@app.get("/files/presentations/animations/{filename}")
async def serve_animation_file(filename: str):
    """
    Serve typing animation video files.
    This endpoint serves the generated typing animation videos.
    """
    import tempfile

    # Security: only allow video files
    if not filename.endswith(('.mp4', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Sanitize filename
    safe_filename = Path(filename).name

    file_path = Path(tempfile.gettempdir()) / "presentations" / "animations" / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Animation file not found")

    content_types = {
        '.mp4': 'video/mp4',
        '.webm': 'video/webm'
    }
    suffix = file_path.suffix.lower()
    content_type = content_types.get(suffix, 'video/mp4')

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=safe_filename
    )


@app.get("/files/presentations/output/{filename}")
async def serve_output_file(filename: str):
    """
    Serve composed video files from the output directory.
    This endpoint serves the final timeline videos for course-generator to download.
    """
    import tempfile

    # Security: only allow video files
    if not filename.endswith(('.mp4', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Sanitize filename
    safe_filename = Path(filename).name

    file_path = Path(tempfile.gettempdir()) / "presentations" / "output" / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    content_types = {
        '.mp4': 'video/mp4',
        '.webm': 'video/webm'
    }
    suffix = file_path.suffix.lower()
    content_type = content_types.get(suffix, 'video/mp4')

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=safe_filename
    )


@app.get("/files/videos/{filename}")
async def serve_video_file(filename: str):
    """
    Serve final composed video files from the CompositorAgent output directory.
    This is used by course-generator to access the generated lecture videos.
    """
    # Security: only allow video files
    if not filename.endswith(('.mp4', '.webm', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Sanitize filename
    safe_filename = Path(filename).name

    # Check in VIDEO_OUTPUT_DIR (where CompositorAgent saves files)
    video_dir = os.getenv("VIDEO_OUTPUT_DIR", "/tmp/viralify/videos")
    file_path = Path(video_dir) / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {safe_filename}")

    content_types = {
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
        '.mkv': 'video/x-matroska'
    }
    suffix = file_path.suffix.lower()
    content_type = content_types.get(suffix, 'video/mp4')

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=safe_filename
    )


# ==============================================================================
# VISUAL GENERATOR ENDPOINTS (Phase 6)
# ==============================================================================

@app.get("/api/v1/visuals/status")
async def get_visual_generator_status():
    """Check if Visual Generator module is available."""
    return {
        "available": VISUAL_GENERATOR_AVAILABLE,
        "initialized": visual_generator is not None,
    }


@app.get("/api/v1/visuals/types")
async def get_visual_types():
    """Get available visual/diagram types."""
    return {
        "diagram_types": [
            {
                "id": "flowchart",
                "name": "Flowchart",
                "description": "Process flows, decision trees",
                "renderer": "mermaid",
            },
            {
                "id": "sequence",
                "name": "Sequence Diagram",
                "description": "API calls, message passing",
                "renderer": "mermaid",
            },
            {
                "id": "class",
                "name": "Class Diagram",
                "description": "OOP class hierarchies",
                "renderer": "mermaid",
            },
            {
                "id": "state",
                "name": "State Diagram",
                "description": "State machines",
                "renderer": "mermaid",
            },
            {
                "id": "er",
                "name": "ER Diagram",
                "description": "Database schemas",
                "renderer": "mermaid",
            },
            {
                "id": "architecture",
                "name": "Architecture Diagram",
                "description": "System architecture",
                "renderer": "mermaid",
            },
            {
                "id": "line_chart",
                "name": "Line Chart",
                "description": "Trends over time",
                "renderer": "matplotlib",
            },
            {
                "id": "bar_chart",
                "name": "Bar Chart",
                "description": "Category comparisons",
                "renderer": "matplotlib",
            },
            {
                "id": "pie_chart",
                "name": "Pie Chart",
                "description": "Distribution breakdown",
                "renderer": "matplotlib",
            },
            {
                "id": "algorithm",
                "name": "Algorithm Animation",
                "description": "Sorting, searching visualizations",
                "renderer": "manim",
            },
            {
                "id": "data_structure",
                "name": "Data Structure Animation",
                "description": "Trees, graphs, linked lists",
                "renderer": "manim",
            },
        ],
        "styles": [
            {"id": "dark", "name": "Dark", "description": "Dark theme (recommended)"},
            {"id": "light", "name": "Light", "description": "Light theme"},
            {"id": "colorful", "name": "Colorful", "description": "Vibrant colors"},
        ],
    }


class VisualGenerateRequest(BaseModel):
    """Request to generate a visual/diagram."""
    content: str = Field(..., description="Description of what to visualize")
    slide_type: Optional[str] = Field(None, description="Type of slide (concept, code, etc.)")
    lesson_context: Optional[str] = Field(None, description="Context of the lesson")
    style: str = Field(default="dark", description="Visual style: dark, light, colorful")
    width: int = Field(default=1920, description="Image width")
    height: int = Field(default=1080, description="Image height")
    force_diagram_type: Optional[str] = Field(None, description="Force specific diagram type")


@app.post("/api/v1/visuals/generate")
async def generate_visual(request: VisualGenerateRequest):
    """
    Generate a visual/diagram from a description.

    The AI will analyze the content and automatically:
    1. Detect if a visualization is needed
    2. Determine the best diagram type
    3. Generate the appropriate visual (Mermaid, Matplotlib, or Manim)

    Returns the file path of the generated visual.
    """
    if not VISUAL_GENERATOR_AVAILABLE or not visual_generator:
        raise HTTPException(status_code=503, detail="Visual Generator not available")

    try:
        # Map style string to enum
        style_mapping = {
            "dark": DiagramStyle.DARK,
            "light": DiagramStyle.LIGHT,
            "colorful": DiagramStyle.COLORFUL,
        }
        style_enum = style_mapping.get(request.style, DiagramStyle.DARK)

        # Create visual generation request
        visual_request = VisualGenerationRequest(
            content=request.content,
            slide_type=request.slide_type,
            lesson_context=request.lesson_context,
            style=style_enum,
            width=request.width,
            height=request.height,
            force_diagram_type=request.force_diagram_type,
        )

        # Generate visual
        result = await visual_generator.generate(visual_request)

        if result.success:
            return {
                "success": True,
                "visual_type": result.visual_type.value if result.visual_type else None,
                "renderer_used": result.renderer_used,
                "file_path": result.file_path,
                "duration_seconds": result.duration_seconds,
                "is_animation": result.duration_seconds is not None and result.duration_seconds > 0,
            }
        else:
            return {
                "success": False,
                "needs_visual": result.needs_visual,
                "reason": "No visualization needed for this content" if not result.needs_visual else "Generation failed",
            }

    except Exception as e:
        print(f"[VISUAL] Error generating visual: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/visuals/detect")
async def detect_visual_need(content: str, slide_type: Optional[str] = None):
    """
    Detect if content needs a visualization and what type.

    This is a lighter endpoint that only runs detection, not generation.
    Useful for previewing what visual would be generated.
    """
    if not VISUAL_GENERATOR_AVAILABLE or not visual_generator:
        raise HTTPException(status_code=503, detail="Visual Generator not available")

    try:
        detection = await visual_generator.detector.detect(content, slide_type)

        return {
            "needs_visual": detection.needs_visual,
            "diagram_type": detection.diagram_type.value if detection.diagram_type else None,
            "confidence": detection.confidence,
            "reason": detection.reason,
        }

    except Exception as e:
        print(f"[VISUAL] Error detecting visual: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


class VisualFromSlideRequest(BaseModel):
    """Request to generate a visual from slide content."""
    slide_title: str = Field(..., description="Slide title")
    slide_content: str = Field(..., description="Slide content/voiceover")
    slide_type: str = Field(default="concept", description="Type of slide")
    code: Optional[str] = Field(None, description="Code on the slide if any")
    lesson_title: Optional[str] = Field(None, description="Parent lesson title for context")
    style: str = Field(default="dark")


@app.post("/api/v1/visuals/generate-for-slide")
async def generate_visual_for_slide(request: VisualFromSlideRequest):
    """
    Generate a visual specifically for a presentation slide.

    Combines slide content with lesson context to generate
    the most appropriate visualization.
    """
    if not VISUAL_GENERATOR_AVAILABLE or not visual_generator:
        raise HTTPException(status_code=503, detail="Visual Generator not available")

    try:
        # Build context
        combined_content = f"Title: {request.slide_title}\n"
        combined_content += f"Content: {request.slide_content}\n"
        if request.code:
            combined_content += f"Code: {request.code}\n"

        lesson_context = request.lesson_title or "General programming tutorial"

        # Map style
        style_mapping = {
            "dark": DiagramStyle.DARK,
            "light": DiagramStyle.LIGHT,
            "colorful": DiagramStyle.COLORFUL,
        }
        style_enum = style_mapping.get(request.style, DiagramStyle.DARK)

        # Generate
        result = await visual_generator.generate_from_slide(
            slide_content={
                "title": request.slide_title,
                "content": request.slide_content,
                "type": request.slide_type,
                "code": request.code,
            },
            lesson_context=lesson_context,
            style=style_enum,
        )

        if result.success:
            return {
                "success": True,
                "visual_type": result.visual_type.value if result.visual_type else None,
                "renderer_used": result.renderer_used,
                "file_path": result.file_path,
                "file_url": f"/files/visuals/{Path(result.file_path).name}" if result.file_path else None,
                "duration_seconds": result.duration_seconds,
            }
        else:
            return {
                "success": False,
                "needs_visual": result.needs_visual,
                "reason": "No visualization needed" if not result.needs_visual else "Generation failed",
            }

    except Exception as e:
        print(f"[VISUAL] Error generating visual for slide: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/visuals/{filename}")
async def serve_visual_file(filename: str):
    """
    Serve generated visual files (images and animations).
    """
    # Security: only allow specific file patterns
    if not filename.endswith(('.png', '.jpg', '.jpeg', '.mp4', '.webm', '.gif')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Sanitize filename
    safe_filename = Path(filename).name

    # Check in visual output directory
    output_dir = os.getenv("VISUAL_OUTPUT_DIR", "/tmp/viralify/visuals")
    file_path = Path(output_dir) / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Visual file not found")

    # Determine content type
    content_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
    }
    suffix = file_path.suffix.lower()
    content_type = content_types.get(suffix, 'application/octet-stream')

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=safe_filename
    )


# ==============================================================================
# VIRALIFY DIAGRAMS ENDPOINTS (Phase 8B)
# ==============================================================================

# Import Viralify Diagrams service
try:
    from services.viralify_diagram_service import (
        ViralifyDiagramService,
        ViralifyLayoutType,
        ViralifyExportFormat,
        get_viralify_diagram_service,
    )
    VIRALIFY_DIAGRAMS_AVAILABLE = True
    print("[STARTUP] Viralify Diagrams module loaded successfully", flush=True)
except ImportError as e:
    print(f"[STARTUP] Viralify Diagrams module not available: {e}", flush=True)
    VIRALIFY_DIAGRAMS_AVAILABLE = False


class ViralifyDiagramRequest(BaseModel):
    """Request for viralify diagram generation"""
    title: str = Field(..., description="Diagram title")
    description: str = Field(default="", description="Diagram description")
    nodes: List[Dict] = Field(..., description="List of nodes")
    edges: List[Dict] = Field(default_factory=list, description="List of edges")
    clusters: Optional[List[Dict]] = Field(default=None, description="Optional clusters")
    layout: str = Field(default="horizontal", description="Layout type: horizontal, vertical, grid, radial")
    theme: str = Field(default="dark", description="Theme name")
    export_format: str = Field(default="png_single", description="Export format: svg_static, svg_animated, png_frames, png_single")
    generate_narration: bool = Field(default=False, description="Generate narration script")
    narration_style: str = Field(default="educational", description="Narration style: educational, professional, casual, technical")
    width: int = Field(default=1920, description="Output width")
    height: int = Field(default=1080, description="Output height")
    max_nodes: int = Field(default=10, description="Max nodes (auto-simplification)")
    animation_config: Optional[Dict] = Field(default=None, description="Animation configuration")


class ViralifyAIDiagramRequest(BaseModel):
    """Request for AI-generated viralify diagram"""
    description: str = Field(..., description="Natural language description of the diagram")
    title: str = Field(..., description="Diagram title")
    diagram_type: str = Field(default="architecture", description="Diagram type: architecture, flowchart, process, hierarchy")
    layout: str = Field(default="horizontal", description="Layout type")
    theme: str = Field(default="dark", description="Theme name")
    export_format: str = Field(default="png_single", description="Export format")
    generate_narration: bool = Field(default=False, description="Generate narration script")
    target_audience: str = Field(default="senior", description="Target audience: beginner, senior, executive")
    width: int = Field(default=1920, description="Output width")
    height: int = Field(default=1080, description="Output height")


class ViralifyThemeRequest(BaseModel):
    """Request to register a custom theme"""
    theme_json: str = Field(..., description="Theme definition as JSON string")


@app.get("/api/v1/viralify-diagrams/health")
async def viralify_diagrams_health():
    """Check if Viralify Diagrams service is available"""
    return {
        "available": VIRALIFY_DIAGRAMS_AVAILABLE,
        "message": "Viralify Diagrams service is ready" if VIRALIFY_DIAGRAMS_AVAILABLE else "Service not available"
    }


@app.get("/api/v1/viralify-diagrams/themes")
async def list_viralify_themes():
    """List available themes for viralify diagrams"""
    if not VIRALIFY_DIAGRAMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Viralify Diagrams not available")

    service = get_viralify_diagram_service()
    themes = service.list_available_themes()

    return {
        "themes": themes,
        "total": len(themes)
    }


@app.get("/api/v1/viralify-diagrams/themes/{theme_name}")
async def get_viralify_theme(theme_name: str):
    """Get a theme definition as JSON"""
    if not VIRALIFY_DIAGRAMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Viralify Diagrams not available")

    service = get_viralify_diagram_service()
    theme_json = service.get_theme_json(theme_name)

    if not theme_json:
        raise HTTPException(status_code=404, detail=f"Theme '{theme_name}' not found")

    return {"theme": json.loads(theme_json)}


@app.post("/api/v1/viralify-diagrams/themes")
async def register_viralify_theme(request: ViralifyThemeRequest):
    """Register a custom theme"""
    if not VIRALIFY_DIAGRAMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Viralify Diagrams not available")

    service = get_viralify_diagram_service()
    success = service.register_custom_theme(request.theme_json)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to register theme. Check JSON format.")

    return {"success": True, "message": "Theme registered successfully"}


@app.post("/api/v1/viralify-diagrams/generate")
async def generate_viralify_diagram(request: ViralifyDiagramRequest):
    """
    Generate a diagram using viralify-diagrams library.

    Supports:
    - Multiple layouts (horizontal, vertical, grid, radial)
    - Custom themes
    - Multiple export formats (SVG static, SVG animated, PNG frames, PNG single)
    - Narration script generation
    """
    if not VIRALIFY_DIAGRAMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Viralify Diagrams not available")

    service = get_viralify_diagram_service()

    # Parse enums
    try:
        layout = ViralifyLayoutType(request.layout)
    except ValueError:
        layout = ViralifyLayoutType.HORIZONTAL

    try:
        export_format = ViralifyExportFormat(request.export_format)
    except ValueError:
        export_format = ViralifyExportFormat.PNG_SINGLE

    result = await service.generate_diagram(
        description=request.description,
        title=request.title,
        nodes=request.nodes,
        edges=request.edges,
        clusters=request.clusters,
        layout=layout,
        theme=request.theme,
        export_format=export_format,
        generate_narration=request.generate_narration,
        narration_style=request.narration_style,
        width=request.width,
        height=request.height,
        max_nodes=request.max_nodes,
        animation_config=request.animation_config
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Diagram generation failed")

    response = {
        "success": True,
        "file_path": result.file_path,
    }

    if result.svg_content:
        response["svg_content"] = result.svg_content
    if result.animation_timeline:
        response["animation_timeline"] = result.animation_timeline
    if result.narration_script:
        response["narration_script"] = result.narration_script
    if result.frame_manifest:
        response["frame_manifest"] = result.frame_manifest
    if result.error:
        response["warning"] = result.error

    return response


@app.post("/api/v1/viralify-diagrams/generate-ai")
async def generate_viralify_diagram_from_ai(request: ViralifyAIDiagramRequest):
    """
    Generate a diagram from natural language description using AI.

    GPT-4 extracts nodes, edges, and clusters from the description,
    then generates the diagram using viralify-diagrams.
    """
    if not VIRALIFY_DIAGRAMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Viralify Diagrams not available")

    service = get_viralify_diagram_service()

    # Parse enums
    try:
        layout = ViralifyLayoutType(request.layout)
    except ValueError:
        layout = ViralifyLayoutType.HORIZONTAL

    try:
        export_format = ViralifyExportFormat(request.export_format)
    except ValueError:
        export_format = ViralifyExportFormat.PNG_SINGLE

    result = await service.generate_from_ai_description(
        description=request.description,
        title=request.title,
        diagram_type=request.diagram_type,
        layout=layout,
        theme=request.theme,
        export_format=export_format,
        generate_narration=request.generate_narration,
        target_audience=request.target_audience,
        width=request.width,
        height=request.height
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Diagram generation failed")

    response = {
        "success": True,
        "file_path": result.file_path,
    }

    if result.svg_content:
        response["svg_content"] = result.svg_content
    if result.animation_timeline:
        response["animation_timeline"] = result.animation_timeline
    if result.narration_script:
        response["narration_script"] = result.narration_script
    if result.frame_manifest:
        response["frame_manifest"] = result.frame_manifest

    return response


@app.get("/api/v1/viralify-diagrams/files/{filename}")
async def get_viralify_diagram_file(filename: str):
    """Serve generated viralify diagram files"""
    if not VIRALIFY_DIAGRAMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Viralify Diagrams not available")

    # Validate filename
    allowed_extensions = {'.svg', '.png', '.jpg', '.jpeg'}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Sanitize filename
    safe_filename = Path(filename).name

    # Check in viralify output directory
    output_dir = "/tmp/presentations/viralify_diagrams"
    file_path = Path(output_dir) / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Diagram file not found")

    # Determine content type
    content_types = {
        '.svg': 'image/svg+xml',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    }
    content_type = content_types.get(ext, 'application/octet-stream')

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=safe_filename
    )


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8006"))
    workers = int(os.getenv("UVICORN_WORKERS", "2"))  # Multiple workers to handle concurrent requests

    if os.getenv("DEBUG", "false").lower() == "true":
        # Single worker with reload for development
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=True
        )
    else:
        # Multiple workers for production
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            workers=workers
        )
