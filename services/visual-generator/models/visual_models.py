"""
Visual Generator Models
Pydantic models for diagram detection, generation, and rendering.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


class DiagramType(str, Enum):
    """Types of diagrams that can be generated."""
    # Mermaid types
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    CLASS_DIAGRAM = "class"
    STATE_DIAGRAM = "state"
    ER_DIAGRAM = "er"
    GANTT = "gantt"
    PIE_CHART = "pie"
    MINDMAP = "mindmap"
    TIMELINE = "timeline"
    ARCHITECTURE = "architecture"
    HIERARCHY = "hierarchy"        # Tree/organizational structures
    PROCESS = "process"            # Process flow diagrams
    COMPARISON = "comparison"      # Side-by-side comparisons

    # Matplotlib types
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX_PLOT = "box_plot"

    # Manim types
    ANIMATION = "animation"
    MATH_FORMULA = "math"
    GRAPH_THEORY = "graph"
    TRANSFORMATION = "transform"
    CODE_VISUALIZATION = "code_viz"
    DATA_STRUCTURE = "data_structure"
    ALGORITHM = "algorithm"


class DiagramStyle(str, Enum):
    """Visual style for diagrams."""
    DARK = "dark"
    LIGHT = "light"
    NEUTRAL = "neutral"
    COLORFUL = "colorful"
    MINIMAL = "minimal"
    CORPORATE = "corporate"


class AnimationComplexity(str, Enum):
    """Complexity level for Manim animations."""
    SIMPLE = "simple"       # 5-10 seconds, basic transforms
    MODERATE = "moderate"   # 10-20 seconds, multiple elements
    COMPLEX = "complex"     # 20-45 seconds, full scene
    CINEMATIC = "cinematic" # 45-90 seconds, production quality


class RenderFormat(str, Enum):
    """Output format for rendered visuals."""
    PNG = "png"
    SVG = "svg"
    MP4 = "mp4"
    GIF = "gif"
    WEBM = "webm"


class DiagramRequest(BaseModel):
    """Base request for diagram generation."""
    description: str = Field(..., description="Natural language description of the diagram")
    diagram_type: Optional[DiagramType] = Field(None, description="Specific diagram type, auto-detected if not provided")
    style: DiagramStyle = Field(default=DiagramStyle.DARK, description="Visual style")
    width: int = Field(default=1920, ge=640, le=3840)
    height: int = Field(default=1080, ge=480, le=2160)
    format: RenderFormat = Field(default=RenderFormat.PNG)

    # Optional context for better generation
    context: Optional[str] = Field(None, description="Additional context (e.g., lesson content)")
    language: str = Field(default="en", description="Language for labels")


class NodeCoordinate(BaseModel):
    """Coordinates of a node in a diagram (extracted from Graphviz)."""
    name: str = Field(..., description="Node identifier/name")
    label: str = Field(default="", description="Display label of the node")
    x: float = Field(..., description="X coordinate (in points)")
    y: float = Field(..., description="Y coordinate (in points)")
    width: float = Field(default=0, description="Node width (in inches)")
    height: float = Field(default=0, description="Node height (in inches)")
    # Bounding box (llx, lly, urx, ury) in points
    bbox: Optional[List[float]] = Field(default=None, description="Bounding box [llx, lly, urx, ury]")
    # For camera animations
    center_x: Optional[float] = Field(default=None, description="Center X in pixels (scaled)")
    center_y: Optional[float] = Field(default=None, description="Center Y in pixels (scaled)")


class EdgeCoordinate(BaseModel):
    """Coordinates of an edge in a diagram (extracted from Graphviz)."""
    source: str = Field(..., description="Source node name")
    target: str = Field(..., description="Target node name")
    label: Optional[str] = Field(default=None, description="Edge label")
    # Spline points for the edge path
    points: List[List[float]] = Field(default_factory=list, description="Edge path points [[x,y], ...]")


class DiagramCoordinates(BaseModel):
    """Complete coordinate data for a diagram (for camera animations)."""
    nodes: List[NodeCoordinate] = Field(default_factory=list)
    edges: List[EdgeCoordinate] = Field(default_factory=list)
    # Graph bounding box
    graph_bbox: Optional[List[float]] = Field(default=None, description="Graph bounding box [llx, lly, urx, ury]")
    graph_width: float = Field(default=0, description="Graph width in points")
    graph_height: float = Field(default=0, description="Graph height in points")
    # DPI used for rendering (for pixel conversion)
    dpi: int = Field(default=96, description="DPI used for coordinate scaling")


class DiagramResult(BaseModel):
    """Result of diagram generation."""
    success: bool
    diagram_type: DiagramType
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    width: int = 0  # 0 indicates not generated (e.g., on error)
    height: int = 0
    format: RenderFormat = RenderFormat.PNG
    generation_time_ms: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # NEW: Node coordinates for camera animations
    coordinates: Optional[DiagramCoordinates] = Field(
        default=None,
        description="Node/edge coordinates extracted from Graphviz for camera animations"
    )


# Mermaid-specific models
class MermaidDiagram(BaseModel):
    """Mermaid diagram specification."""
    diagram_type: DiagramType
    code: str = Field(..., description="Mermaid syntax code")
    title: Optional[str] = None
    theme: str = Field(default="dark", description="Mermaid theme: dark, default, forest, neutral")
    background_color: str = Field(default="transparent")

    class Config:
        json_schema_extra = {
            "example": {
                "diagram_type": "flowchart",
                "code": "graph TD\n    A[Producer] --> B[Kafka Broker]\n    B --> C[Consumer]",
                "title": "Kafka Architecture",
                "theme": "dark"
            }
        }


# Matplotlib-specific models
class DataSeries(BaseModel):
    """Data series for charts."""
    name: str
    values: List[Union[int, float]]
    color: Optional[str] = None
    style: Optional[str] = None  # line style, bar pattern, etc.


class MatplotlibChart(BaseModel):
    """Matplotlib chart specification."""
    chart_type: DiagramType
    title: str
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_values: Optional[List[Union[str, int, float]]] = None
    data_series: List[DataSeries]
    legend: bool = Field(default=True)
    grid: bool = Field(default=True)
    style: str = Field(default="dark_background", description="Matplotlib style")

    # Advanced options
    annotations: Optional[List[Dict[str, Any]]] = None
    secondary_y_axis: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "chart_type": "line_chart",
                "title": "API Response Times",
                "x_label": "Time",
                "y_label": "Latency (ms)",
                "x_values": ["00:00", "06:00", "12:00", "18:00"],
                "data_series": [
                    {"name": "GET /api/users", "values": [45, 52, 48, 55]},
                    {"name": "POST /api/orders", "values": [120, 135, 128, 142]}
                ]
            }
        }


# Manim-specific models
class ManimScene(BaseModel):
    """A scene within a Manim animation."""
    name: str
    description: str
    duration_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    elements: List[Dict[str, Any]] = Field(default_factory=list)


class ManimAnimation(BaseModel):
    """Manim animation specification."""
    title: str
    description: str
    complexity: AnimationComplexity = Field(default=AnimationComplexity.MODERATE)
    scenes: List[ManimScene] = Field(default_factory=list)

    # Visual settings
    background_color: str = Field(default="#1e1e1e")
    resolution: str = Field(default="1080p", pattern="^(720p|1080p|1440p|4k)$")
    fps: int = Field(default=30, ge=24, le=60)

    # Content type hints
    includes_math: bool = False
    includes_code: bool = False
    includes_graph: bool = False

    # Generated Manim Python code (filled by generator)
    manim_code: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Binary Search Visualization",
                "description": "Animated step-by-step binary search on sorted array",
                "complexity": "moderate",
                "scenes": [
                    {
                        "name": "intro",
                        "description": "Show sorted array",
                        "duration_seconds": 3.0
                    },
                    {
                        "name": "search",
                        "description": "Highlight mid, compare, eliminate half",
                        "duration_seconds": 10.0
                    }
                ],
                "includes_code": True
            }
        }


# Detection models
class DetectionResult(BaseModel):
    """Result of diagram detection in content."""
    needs_diagram: bool
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_type: Optional[DiagramType] = None
    suggested_description: Optional[str] = None
    reasoning: Optional[str] = None

    # For complex content, may suggest multiple diagrams
    additional_suggestions: List[Dict[str, Any]] = Field(default_factory=list)


# Orchestrator models
class VisualGenerationRequest(BaseModel):
    """Request to the VisualGenerator orchestrator."""
    content: str = Field(..., description="Slide content or description")
    slide_type: Optional[str] = Field(None, description="Type of slide (title, concept, code, etc.)")
    lesson_context: Optional[str] = Field(None, description="Full lesson context for better generation")

    # Preferences
    preferred_type: Optional[DiagramType] = None
    style: DiagramStyle = Field(default=DiagramStyle.DARK)
    format: RenderFormat = Field(default=RenderFormat.PNG)
    width: int = Field(default=1920)
    height: int = Field(default=1080)

    # For animations
    max_duration_seconds: float = Field(default=30.0, description="Max duration for Manim animations")

    # Language
    language: str = Field(default="en")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Kafka is a distributed streaming platform with producers, brokers, and consumers",
                "slide_type": "concept",
                "lesson_context": "This lesson covers message queue fundamentals...",
                "preferred_type": "architecture",
                "style": "dark"
            }
        }


class VisualGenerationResult(BaseModel):
    """Result from the VisualGenerator orchestrator."""
    request_id: str
    success: bool

    # Detection info
    detection: DetectionResult

    # Generated visual
    visual_type: Optional[DiagramType] = None
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    format: RenderFormat = RenderFormat.PNG

    # For animations
    duration_seconds: Optional[float] = None

    # Metadata
    generation_time_ms: int
    renderer_used: Optional[str] = None  # "mermaid", "matplotlib", "manim"

    # Errors
    error: Optional[str] = None

    # Raw data (for debugging or re-rendering)
    raw_specification: Optional[Dict[str, Any]] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def needs_visual(self) -> bool:
        """Whether the content needs a visual based on detection."""
        return self.detection.needs_diagram if self.detection else False
