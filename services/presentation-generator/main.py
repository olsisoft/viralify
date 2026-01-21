"""
Presentation Generator Service

FastAPI service for generating code presentations and tutorials.
Transforms prompts into professional video presentations with slides,
syntax-highlighted code, and voiceover narration.
"""
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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


# Global services
compositor: PresentationCompositorService = None
slide_generator: SlideGeneratorService = None
visual_generator: Optional[VisualGeneratorService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global compositor, slide_generator, visual_generator

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

    print("[STARTUP] Services initialized", flush=True)

    yield

    print("[SHUTDOWN] Cleaning up...", flush=True)


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


# ==============================================================================
# PRESENTATION GENERATION
# ==============================================================================

@app.post("/api/v1/presentations/generate", response_model=Dict)
async def generate_presentation(request: GeneratePresentationRequest):
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

    job_id = str(uuid.uuid4())

    # Store initial job status
    _langgraph_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "phase": "initializing",
        "created_at": datetime.utcnow().isoformat(),
        "request": request.model_dump()
    }

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


# In-memory storage for LangGraph jobs
_langgraph_jobs: Dict[str, Dict] = {}


async def _run_langgraph_generation(job_id: str, request: GeneratePresentationRequest):
    """Background task to run LangGraph generation"""
    from services.langgraph_orchestrator import LangGraphOrchestrator

    try:
        _langgraph_jobs[job_id]["status"] = "processing"

        orchestrator = LangGraphOrchestrator()

        async def on_progress(percent: int, message: str):
            _langgraph_jobs[job_id]["progress"] = percent
            _langgraph_jobs[job_id]["phase"] = message
            print(f"[GENERATE-V2] Job {job_id}: {percent}% - {message}", flush=True)

        result = await orchestrator.generate_video(request, job_id, on_progress)

        # Determine status: if video was generated, consider it successful even with warnings
        has_output = result.get("output_video_url") is not None
        has_critical_errors = any("failed" in str(e).lower() or "error" in str(e).lower()
                                   for e in result.get("errors", [])
                                   if "voiceover" not in str(e).lower())  # Voiceover failure is non-critical

        if has_output and not has_critical_errors:
            _langgraph_jobs[job_id]["status"] = "completed"
        elif has_output:
            _langgraph_jobs[job_id]["status"] = "completed_with_warnings"
        else:
            _langgraph_jobs[job_id]["status"] = "failed"

        _langgraph_jobs[job_id]["progress"] = 100
        _langgraph_jobs[job_id]["result"] = result
        _langgraph_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        print(f"[GENERATE-V2] Error for job {job_id}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        _langgraph_jobs[job_id]["status"] = "failed"
        _langgraph_jobs[job_id]["error"] = str(e)


@app.get("/api/v1/presentations/jobs/v2/{job_id}")
async def get_langgraph_job_status(job_id: str):
    """Get the status of a LangGraph presentation generation job (V2)."""
    if job_id not in _langgraph_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _langgraph_jobs[job_id]


# ==============================================================================
# V3: MULTI-AGENT SYSTEM (Scene-by-Scene with Perfect Sync)
# ==============================================================================

# In-memory storage for V3 jobs
_multiagent_jobs: Dict[str, Dict] = {}


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

    job_id = str(uuid.uuid4())

    # Store initial job status
    _multiagent_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "phase": "initializing",
        "created_at": datetime.utcnow().isoformat(),
        "request": request.model_dump(),
        "scene_statuses": [],
        "enable_visuals": enable_visuals,
        "visual_style": visual_style,
    }

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

    try:
        _multiagent_jobs[job_id]["status"] = "processing"
        _multiagent_jobs[job_id]["phase"] = "generating_script"
        _multiagent_jobs[job_id]["progress"] = 5

        # Step 1: Generate script/slides using GPT-4
        script_generator = ScriptGenerator()
        script = await script_generator.generate_script(
            topic=request.topic,
            language=request.language,
            style=request.style.value,
            duration=request.duration,
            execute_code=request.execute_code
        )

        _multiagent_jobs[job_id]["phase"] = "processing_scenes"
        _multiagent_jobs[job_id]["progress"] = 20
        _multiagent_jobs[job_id]["script"] = {
            "title": script.title,
            "slide_count": len(script.slides),
            "estimated_duration": script.total_duration
        }

        # Convert slides to format expected by multi-agent system
        slides = []
        for slide in script.slides:
            slide_data = {
                "title": slide.title,
                "type": slide.type.value,
                "voiceover_text": slide.voiceover_text,
                "duration": slide.duration
            }

            # Add code if present
            if slide.code_blocks:
                slide_data["code"] = slide.code_blocks[0].code
                slide_data["language"] = slide.code_blocks[0].language
                # Add expected output from code block
                if slide.code_blocks[0].expected_output:
                    slide_data["expected_output"] = slide.code_blocks[0].expected_output

            # Add bullet points
            if slide.bullet_points:
                slide_data["bullet_points"] = slide.bullet_points

            slides.append(slide_data)

        # Step 1.5: Generate visuals for slides if enabled (Phase 6)
        if enable_visuals and VISUAL_GENERATOR_AVAILABLE and visual_generator:
            _multiagent_jobs[job_id]["phase"] = "generating_visuals"
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

            _multiagent_jobs[job_id]["visuals_generated"] = visuals_generated
            print(f"[GENERATE-V3] Generated {visuals_generated} visuals", flush=True)

        # Initialize scene statuses
        _multiagent_jobs[job_id]["scene_statuses"] = [
            {"scene_index": i, "status": "pending", "sync_score": 0}
            for i in range(len(slides))
        ]

        print(f"[GENERATE-V3] Processing {len(slides)} scenes for job {job_id}", flush=True)

        # Step 2: Run multi-agent video generation
        result = await generate_presentation_video(
            job_id=job_id,
            slides=slides,
            title=script.title,
            style=request.style.value
        )

        # Update final status
        if result.get("success"):
            _multiagent_jobs[job_id]["status"] = "completed"
            _multiagent_jobs[job_id]["progress"] = 100
            _multiagent_jobs[job_id]["phase"] = "completed"
        else:
            _multiagent_jobs[job_id]["status"] = "failed"
            _multiagent_jobs[job_id]["phase"] = "failed"

        _multiagent_jobs[job_id]["result"] = result
        _multiagent_jobs[job_id]["output_url"] = result.get("output_url")
        _multiagent_jobs[job_id]["duration"] = result.get("duration", 0)
        _multiagent_jobs[job_id]["summary"] = result.get("summary", {})
        _multiagent_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

        # Update scene statuses from summary
        if result.get("summary", {}).get("scene_packages"):
            for sp in result["summary"]["scene_packages"]:
                idx = sp.get("scene_index", 0)
                if idx < len(_multiagent_jobs[job_id]["scene_statuses"]):
                    _multiagent_jobs[job_id]["scene_statuses"][idx] = {
                        "scene_index": idx,
                        "status": sp.get("sync_status", "unknown"),
                        "sync_score": sp.get("sync_score", 0)
                    }

        print(f"[GENERATE-V3] Job {job_id} completed: {result.get('success')}", flush=True)

    except Exception as e:
        print(f"[GENERATE-V3] Error for job {job_id}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        _multiagent_jobs[job_id]["status"] = "failed"
        _multiagent_jobs[job_id]["error"] = str(e)
        _multiagent_jobs[job_id]["phase"] = "error"


@app.get("/api/v1/presentations/jobs/v3/{job_id}")
async def get_multiagent_job_status(job_id: str):
    """Get the status of a Multi-Agent presentation generation job (V3)."""
    if job_id not in _multiagent_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _multiagent_jobs[job_id]


@app.get("/api/v1/presentations/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a presentation generation job.

    Returns current progress, stage, and results when complete.
    """
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
