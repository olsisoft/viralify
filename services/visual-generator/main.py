"""
Visual Generator Microservice

FastAPI service for generating professional diagrams and visualizations.
Isolated service with heavy dependencies (Graphviz, Matplotlib, etc.)

Port: 8003
"""

import os
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models.visual_models import (
    DiagramType,
    DiagramStyle,
    RenderFormat,
    DiagramRequest,
    DiagramResult,
    VisualGenerationRequest,
    VisualGenerationResult,
    DetectionResult,
)
from renderers.diagrams_renderer import DiagramsRenderer, DiagramProvider
from renderers.mermaid_renderer import MermaidRenderer
from renderers.matplotlib_renderer import MatplotlibRenderer
from services.diagram_detector import DiagramDetector


# Configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/viralify/diagrams")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    print("[VISUAL-GENERATOR] Starting up...", flush=True)
    print(f"[VISUAL-GENERATOR] Output directory: {OUTPUT_DIR}", flush=True)
    print(f"[VISUAL-GENERATOR] OpenAI API key configured: {bool(OPENAI_API_KEY)}", flush=True)
    yield
    print("[VISUAL-GENERATOR] Shutting down...", flush=True)


app = FastAPI(
    title="Visual Generator Service",
    description="Microservice for generating professional diagrams and visualizations",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files for serving generated diagrams
app.mount("/diagrams", StaticFiles(directory=OUTPUT_DIR), name="diagrams")

# Initialize renderers
diagrams_renderer = DiagramsRenderer(openai_api_key=OPENAI_API_KEY, output_dir=OUTPUT_DIR)
mermaid_renderer = MermaidRenderer()
diagram_detector = DiagramDetector()


# ============================================
# Request/Response Models
# ============================================

class TargetAudience(str, Enum):
    """Audience level for diagram complexity adjustment"""
    BEGINNER = "beginner"     # Simple, few nodes, high-level concepts
    SENIOR = "senior"         # Detailed, specific protocols, clusters
    EXECUTIVE = "executive"   # Value flow, system boundaries, minimal tech details


class RenderingEngine(str, Enum):
    """Rendering engine to use"""
    DIAGRAMS_PYTHON = "diagrams_python"  # Python Diagrams library (AWS/Azure/K8s icons)
    MERMAID = "mermaid"                   # Mermaid.js (flowcharts, sequences)
    MATPLOTLIB = "matplotlib"             # Matplotlib (data charts)


class DiagramGenerateRequest(BaseModel):
    """Request to generate a diagram."""
    description: str = Field(..., description="Natural language description of the diagram")
    diagram_type: DiagramType = Field(default=DiagramType.ARCHITECTURE)
    style: DiagramStyle = Field(default=DiagramStyle.DARK)
    provider: Optional[str] = Field(None, description="Cloud provider: aws, azure, gcp, k8s, onprem")
    title: Optional[str] = Field(None, description="Diagram title")
    context: Optional[str] = Field(None, description="Additional context")
    language: str = Field(default="en", description="Language for labels")
    format: RenderFormat = Field(default=RenderFormat.PNG)
    # New fields for enhanced generation
    audience: Optional[str] = Field(default="senior", description="Target audience: beginner, senior, executive")
    engine: Optional[str] = Field(default=None, description="Rendering engine: diagrams_python, mermaid, matplotlib")
    cheat_sheet: Optional[str] = Field(default=None, description="Valid imports cheat sheet for LLM")


class DiagramGenerateResponse(BaseModel):
    """Response from diagram generation."""
    success: bool
    diagram_type: DiagramType
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    generation_time_ms: int
    renderer_used: str
    error: Optional[str] = None


class MermaidRenderRequest(BaseModel):
    """Request to render Mermaid code."""
    code: str = Field(..., description="Mermaid diagram code")
    diagram_type: DiagramType = Field(default=DiagramType.FLOWCHART)
    title: Optional[str] = None
    theme: str = Field(default="dark")
    format: RenderFormat = Field(default=RenderFormat.PNG)


class DetectDiagramRequest(BaseModel):
    """Request to detect if content needs a diagram."""
    content: str = Field(..., description="Content to analyze")
    context: Optional[str] = None


# ============================================
# Health & Info Endpoints
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "visual-generator",
        "version": "1.0.0",
        "openai_configured": bool(OPENAI_API_KEY),
    }


@app.get("/info")
async def service_info():
    """Service information."""
    return {
        "service": "visual-generator",
        "description": "Professional diagram and visualization generation",
        "renderers": ["diagrams", "mermaid", "matplotlib"],
        "supported_types": [t.value for t in DiagramType],
        "supported_styles": [s.value for s in DiagramStyle],
        "supported_formats": [f.value for f in RenderFormat],
        "providers": ["aws", "azure", "gcp", "k8s", "onprem", "generic"],
    }


# ============================================
# Diagram Generation Endpoints
# ============================================

@app.post("/api/v1/diagrams/generate", response_model=DiagramGenerateResponse)
async def generate_diagram(request: DiagramGenerateRequest):
    """
    Generate a diagram from natural language description.

    Uses the Python Diagrams library for architecture diagrams,
    with official icons from AWS, Azure, GCP, Kubernetes, etc.
    """
    start_time = time.time()

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Cannot generate diagrams."
        )

    try:
        # Determine provider
        provider = None
        if request.provider:
            try:
                provider = DiagramProvider(request.provider)
            except ValueError:
                pass  # Use auto-detection

        # Log the request for debugging
        print(f"[VISUAL-GENERATOR] Request: type={request.diagram_type.value}, audience={request.audience}, engine={request.engine}", flush=True)

        # Generate diagram using Diagrams renderer with enhanced parameters
        result = await diagrams_renderer.generate_and_render(
            description=request.description,
            diagram_type=request.diagram_type,
            style=request.style,
            provider=provider,
            format=request.format,
            context=request.context,
            language=request.language,
            audience=request.audience,
            cheat_sheet=request.cheat_sheet,
        )

        generation_time = int((time.time() - start_time) * 1000)

        if result.success:
            # Build public URL
            filename = Path(result.file_path).name if result.file_path else None
            file_url = f"/diagrams/{filename}" if filename else None

            return DiagramGenerateResponse(
                success=True,
                diagram_type=request.diagram_type,
                file_path=result.file_path,
                file_url=file_url,
                generation_time_ms=generation_time,
                renderer_used="diagrams",
            )
        else:
            return DiagramGenerateResponse(
                success=False,
                diagram_type=request.diagram_type,
                generation_time_ms=generation_time,
                renderer_used="diagrams",
                error=result.error,
            )

    except Exception as e:
        generation_time = int((time.time() - start_time) * 1000)
        print(f"[VISUAL-GENERATOR] Error: {e}", flush=True)
        return DiagramGenerateResponse(
            success=False,
            diagram_type=request.diagram_type,
            generation_time_ms=generation_time,
            renderer_used="diagrams",
            error=str(e),
        )


@app.post("/api/v1/diagrams/mermaid", response_model=DiagramGenerateResponse)
async def render_mermaid(request: MermaidRenderRequest):
    """
    Render a Mermaid diagram from code.

    Uses Kroki API for rendering.
    """
    start_time = time.time()

    try:
        result = await mermaid_renderer.render(
            code=request.code,
            diagram_type=request.diagram_type,
            title=request.title,
            theme=request.theme,
        )

        generation_time = int((time.time() - start_time) * 1000)

        if result.get("success"):
            return DiagramGenerateResponse(
                success=True,
                diagram_type=request.diagram_type,
                file_path=result.get("file_path"),
                file_url=result.get("file_url"),
                generation_time_ms=generation_time,
                renderer_used="mermaid",
            )
        else:
            return DiagramGenerateResponse(
                success=False,
                diagram_type=request.diagram_type,
                generation_time_ms=generation_time,
                renderer_used="mermaid",
                error=result.get("error"),
            )

    except Exception as e:
        generation_time = int((time.time() - start_time) * 1000)
        return DiagramGenerateResponse(
            success=False,
            diagram_type=request.diagram_type,
            generation_time_ms=generation_time,
            renderer_used="mermaid",
            error=str(e),
        )


@app.post("/api/v1/diagrams/detect")
async def detect_diagram_need(request: DetectDiagramRequest):
    """
    Detect if content needs a diagram and suggest the best type.
    """
    try:
        result = await diagram_detector.detect(
            content=request.content,
            context=request.context,
        )
        return result
    except Exception as e:
        return DetectionResult(
            needs_diagram=False,
            confidence=0.0,
            reasoning=f"Detection failed: {str(e)}",
        )


@app.get("/api/v1/diagrams/{filename}")
async def get_diagram(filename: str):
    """
    Retrieve a generated diagram by filename.
    """
    file_path = Path(OUTPUT_DIR) / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Diagram not found")

    # Determine content type
    suffix = file_path.suffix.lower()
    content_types = {
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }
    content_type = content_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=filename,
    )


# ============================================
# Cleanup Endpoints
# ============================================

@app.delete("/api/v1/diagrams/{filename}")
async def delete_diagram(filename: str):
    """
    Delete a generated diagram.
    """
    file_path = Path(OUTPUT_DIR) / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Diagram not found")

    try:
        file_path.unlink()
        return {"success": True, "deleted": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


@app.post("/api/v1/diagrams/cleanup")
async def cleanup_old_diagrams(max_age_hours: int = 24):
    """
    Clean up diagrams older than specified hours.
    """
    import time as time_module

    deleted = 0
    cutoff = time_module.time() - (max_age_hours * 3600)

    for file_path in Path(OUTPUT_DIR).iterdir():
        if file_path.is_file() and file_path.stat().st_mtime < cutoff:
            try:
                file_path.unlink()
                deleted += 1
            except OSError:
                pass

    return {"success": True, "deleted_count": deleted}


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8003"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "production") == "development",
    )
