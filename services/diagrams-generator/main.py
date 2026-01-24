"""
Diagrams Generator Microservice

A dedicated service for generating professional-quality architecture diagrams
using the Python diagrams library. This service provides:

- Comprehensive import validation and auto-correction
- Safe code execution in isolated environment
- Support for all major cloud providers (AWS, Azure, GCP, etc.)
- High-quality PNG output with proper styling
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from models import (
    DiagramRequest,
    DiagramResponse,
    DiagramType,
    CloudProvider,
    ValidationResult,
)
from services import DiagramService, ImportValidator

# Initialize FastAPI app
app = FastAPI(
    title="Diagrams Generator",
    description="Professional architecture diagram generation service",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
diagram_service = DiagramService()


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "diagrams-generator"}


@app.get("/info")
async def service_info():
    """Get service information."""
    return {
        "service": "diagrams-generator",
        "version": "1.0.0",
        "supported_providers": [p.value for p in CloudProvider],
        "supported_diagram_types": [t.value for t in DiagramType],
    }


# =============================================================================
# Diagram Generation Endpoints
# =============================================================================

@app.post("/api/v1/diagrams/generate", response_model=DiagramResponse)
async def generate_diagram(request: DiagramRequest):
    """
    Generate a diagram from Python code.

    The code should use the Python 'diagrams' library.
    Invalid imports will be automatically corrected.

    Args:
        request: DiagramRequest with python_code

    Returns:
        DiagramResponse with base64-encoded image or error
    """
    if not request.python_code:
        raise HTTPException(
            status_code=400,
            detail="python_code is required"
        )

    result = diagram_service.generate(request)
    return result


@app.post("/api/v1/diagrams/validate", response_model=ValidationResult)
async def validate_code(request: DiagramRequest):
    """
    Validate diagram code without executing.

    Checks for:
    - Invalid imports (with auto-correction suggestions)
    - Syntax errors
    - Dangerous operations

    Args:
        request: DiagramRequest with python_code

    Returns:
        ValidationResult with corrected code if needed
    """
    if not request.python_code:
        raise HTTPException(
            status_code=400,
            detail="python_code is required"
        )

    result = diagram_service.validate_code(request.python_code)
    return result


@app.get("/api/v1/diagrams/file/{filename}")
async def get_diagram_file(filename: str):
    """
    Get a generated diagram file by filename.

    Args:
        filename: The diagram filename

    Returns:
        The PNG file
    """
    file_path = diagram_service.output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Diagram not found")

    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=filename
    )


# =============================================================================
# Icon Discovery Endpoints
# =============================================================================

@app.get("/api/v1/diagrams/icons/{provider}")
async def get_available_icons(provider: str):
    """
    Get available icons for a cloud provider.

    Args:
        provider: Cloud provider (aws, azure, gcp, kubernetes, onprem, generic)

    Returns:
        Dictionary of module paths to icon names
    """
    try:
        cloud_provider = CloudProvider(provider.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider. Must be one of: {[p.value for p in CloudProvider]}"
        )

    icons = diagram_service.get_available_icons(cloud_provider)
    return {"provider": provider, "icons": icons}


@app.get("/api/v1/diagrams/icons")
async def get_all_icons():
    """
    Get all available icons organized by provider.

    Returns:
        Dictionary of all valid imports
    """
    return {
        "total_modules": len(ImportValidator.VALID_IMPORTS),
        "imports": {
            module: sorted(list(icons))
            for module, icons in ImportValidator.VALID_IMPORTS.items()
        }
    }


@app.post("/api/v1/diagrams/suggest-icons")
async def suggest_icons(description: str):
    """
    Suggest icons based on a text description.

    Args:
        description: Description of the architecture

    Returns:
        List of suggested imports
    """
    suggestions = diagram_service.suggest_icons(description)
    provider = diagram_service.detect_provider(description)

    return {
        "detected_provider": provider.value,
        "suggestions": [
            {"module": module, "icon": icon}
            for module, icon in suggestions
        ]
    }


# =============================================================================
# Import Validation Endpoint
# =============================================================================

@app.post("/api/v1/diagrams/fix-imports")
async def fix_imports(code: str):
    """
    Fix imports in diagram code.

    Args:
        code: Python code with potentially invalid imports

    Returns:
        Fixed code with corrections applied
    """
    fixed_code, errors, warnings = ImportValidator.fix_imports(code)

    return {
        "original_code": code,
        "fixed_code": fixed_code,
        "errors": errors,
        "warnings": warnings,
        "has_changes": len(warnings) > 0 or len(errors) > 0
    }


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    print("=" * 60, flush=True)
    print("Diagrams Generator Service Starting...", flush=True)
    print(f"Output directory: {diagram_service.output_dir}", flush=True)
    print(f"Available providers: {[p.value for p in CloudProvider]}", flush=True)
    print(f"Total import modules: {len(ImportValidator.VALID_IMPORTS)}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
