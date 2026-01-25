"""Diagram models for the diagrams-generator service."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DiagramType(str, Enum):
    """Types of diagrams that can be generated."""
    ARCHITECTURE = "architecture"
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    NETWORK = "network"
    CLOUD = "cloud"
    DATA_PIPELINE = "data_pipeline"
    MICROSERVICES = "microservices"
    KUBERNETES = "kubernetes"
    HIERARCHY = "hierarchy"          # Organizational/tree structures
    PROCESS = "process"              # Process flow diagrams
    COMPARISON = "comparison"        # Side-by-side comparisons
    GENERIC = "generic"


class CloudProvider(str, Enum):
    """Cloud providers for icon selection."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    ONPREM = "onprem"
    GENERIC = "generic"
    AUTO = "auto"  # Auto-detect from description


class DiagramRequest(BaseModel):
    """Request model for diagram generation."""
    description: str = Field(..., description="Description of the diagram to generate")
    diagram_type: DiagramType = Field(default=DiagramType.ARCHITECTURE)
    cloud_provider: CloudProvider = Field(default=CloudProvider.AUTO)
    title: Optional[str] = Field(default=None, description="Title for the diagram")
    direction: str = Field(default="LR", description="Diagram direction: LR, RL, TB, BT")
    show_legend: bool = Field(default=False)
    style: Optional[Dict[str, Any]] = Field(default=None, description="Custom styling options")

    # For pre-generated code (optional)
    python_code: Optional[str] = Field(default=None, description="Pre-generated Python code to execute")


class ValidationResult(BaseModel):
    """Result of code validation."""
    is_valid: bool
    corrected_code: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DiagramResponse(BaseModel):
    """Response model for diagram generation."""
    success: bool
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    error: Optional[str] = None
    validation: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
