"""
Visual types and analysis models for intelligent visual generation.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class VisualType(str, Enum):
    """Types of visual content that can be generated."""
    DIAGRAM = "diagram"      # Mermaid.js technical diagrams
    CHART = "chart"          # Data visualization charts
    AVATAR = "avatar"        # D-ID lip-sync avatar video
    CONCEPT = "concept"      # DALL-E abstract/creative visuals
    STOCK = "stock"          # Pexels stock footage
    AI_IMAGE = "ai_image"    # DALL-E general images


class DiagramType(str, Enum):
    """Types of technical diagrams supported by Mermaid.js."""
    FLOWCHART = "flowchart"       # Process flows, workflows, decisions
    SEQUENCE = "sequence"         # API calls, interactions, message flows
    ARCHITECTURE = "architecture" # System components, microservices
    CLASS = "class"               # OOP classes, inheritance, relationships
    ER = "er"                     # Database schemas, entity relationships
    MINDMAP = "mindmap"           # Hierarchical concepts, brainstorming
    GANTT = "gantt"               # Project timelines, schedules
    STATE = "state"               # State machines, transitions
    PIE = "pie"                   # Simple pie charts
    JOURNEY = "journey"           # User journeys, experience maps


class ChartType(str, Enum):
    """Types of data visualization charts."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HEATMAP = "heatmap"


class VisualAnalysis(BaseModel):
    """Result of GPT-4 visual context analysis for a scene."""
    visual_type: VisualType = Field(
        description="Recommended visual type for this scene"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for the recommendation"
    )
    diagram_type: Optional[DiagramType] = Field(
        default=None,
        description="Specific diagram type if visual_type is DIAGRAM"
    )
    chart_type: Optional[ChartType] = Field(
        default=None,
        description="Specific chart type if visual_type is CHART"
    )
    requires_avatar: bool = Field(
        default=False,
        description="Whether the scene would benefit from an avatar presenter"
    )
    extracted_elements: List[str] = Field(
        default_factory=list,
        description="Key elements extracted from the description for diagram generation"
    )
    extracted_relationships: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Relationships between elements (from, to, label)"
    )
    mermaid_possible: bool = Field(
        default=False,
        description="Whether Mermaid.js can accurately represent this visual"
    )
    domain: Optional[str] = Field(
        default=None,
        description="Detected domain: software, architecture, business, data, etc."
    )
    reasoning: str = Field(
        default="",
        description="GPT-4's reasoning for the recommendation"
    )
    suggested_prompt: Optional[str] = Field(
        default=None,
        description="Enhanced prompt for DALL-E if using AI generation"
    )


class DiagramResult(BaseModel):
    """Result of diagram generation."""
    image_url: str = Field(description="URL or path to the generated diagram image")
    generator: str = Field(description="Generator used: 'mermaid' or 'dalle'")
    mermaid_code: Optional[str] = Field(
        default=None,
        description="Mermaid code if generated with Mermaid.js"
    )
    width: int = Field(default=1080, description="Image width in pixels")
    height: int = Field(default=1920, description="Image height in pixels")
    fallback_used: bool = Field(
        default=False,
        description="Whether DALL-E fallback was used due to Mermaid failure"
    )


class VisualGenerationRequest(BaseModel):
    """Request for visual generation."""
    description: str = Field(description="Scene description or content")
    script_context: Optional[str] = Field(
        default=None,
        description="Surrounding script context for better understanding"
    )
    preferred_type: Optional[VisualType] = Field(
        default=None,
        description="User-preferred visual type (overrides auto-detection)"
    )
    output_format: str = Field(
        default="9:16",
        description="Output aspect ratio"
    )
    style: Optional[str] = Field(
        default=None,
        description="Visual style preference"
    )
    avatar_id: Optional[str] = Field(
        default=None,
        description="Avatar ID if using avatar visual"
    )
    voiceover_url: Optional[str] = Field(
        default=None,
        description="Voiceover URL for avatar lip-sync"
    )
