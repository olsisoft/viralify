"""
Viralify Diagrams Service

Integrates the viralify-diagrams library for video-optimized diagram generation.
Provides professional diagrams with:
- Theme customization (including user-uploaded themes)
- Animation support (SVG with CSS, PNG frames)
- Narration script generation for voiceover sync
"""

import os
import json
import uuid
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Viralify Diagrams library
from viralify_diagrams import (
    Diagram,
    Node,
    Edge,
    Cluster,
    Theme,
    ThemeManager,
    HorizontalLayout,
    VerticalLayout,
    GridLayout,
    RadialLayout,
    SVGExporter,
    AnimatedSVGExporter,
    PNGFrameExporter,
    DiagramNarrator,
    NarrationScript,
)
from viralify_diagrams.core.diagram import NodeShape, EdgeStyle, EdgeDirection
from viralify_diagrams.exporters.animated_svg_exporter import AnimationConfig, AnimationType
from viralify_diagrams.exporters.png_frame_exporter import FrameConfig
from viralify_diagrams.narration.diagram_narrator import NarrationStyle


class ViralifyLayoutType(str, Enum):
    """Layout types for viralify diagrams"""
    HORIZONTAL = "horizontal"  # Left to right flow
    VERTICAL = "vertical"      # Top to bottom flow
    GRID = "grid"              # Uniform grid
    RADIAL = "radial"          # Hub and spoke


class ViralifyExportFormat(str, Enum):
    """Export formats for viralify diagrams"""
    SVG_STATIC = "svg_static"           # Static SVG with named groups
    SVG_ANIMATED = "svg_animated"       # SVG with CSS animations
    PNG_FRAMES = "png_frames"           # PNG frames for video
    PNG_SINGLE = "png_single"           # Single PNG image


@dataclass
class ViralifyDiagramResult:
    """Result of diagram generation"""
    success: bool
    file_path: Optional[str] = None
    svg_content: Optional[str] = None
    animation_timeline: Optional[Dict] = None
    narration_script: Optional[Dict] = None
    frame_manifest: Optional[Dict] = None
    error: Optional[str] = None


class ViralifyDiagramService:
    """
    Service for generating video-optimized diagrams using viralify-diagrams.

    Features:
    - Multiple layout algorithms (horizontal, vertical, grid, radial)
    - Theme customization (built-in + custom JSON themes)
    - Three export modes (static SVG, animated SVG, PNG frames)
    - Narration script generation for voiceover
    """

    def __init__(self, output_dir: str = "/tmp/presentations/viralify_diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.theme_manager = ThemeManager()

        # Default animation config
        self.default_animation_config = AnimationConfig(
            duration=0.5,
            delay_between=0.3,
            easing="ease-out",
            stagger=True,
            initial_delay=0.5
        )

        # Default frame config for video
        self.default_frame_config = FrameConfig(
            fps=30,
            element_duration=0.5,
            width=1920,
            height=1080
        )

    def register_custom_theme(self, theme_json: str) -> bool:
        """
        Register a custom theme from JSON.

        Args:
            theme_json: JSON string with theme definition

        Returns:
            True if successful, False otherwise
        """
        try:
            theme = Theme.from_json(theme_json)
            self.theme_manager.register(theme)
            print(f"[VIRALIFY] Registered custom theme: {theme.name}", flush=True)
            return True
        except Exception as e:
            print(f"[VIRALIFY] Failed to register theme: {e}", flush=True)
            return False

    async def generate_diagram(
        self,
        description: str,
        title: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        clusters: Optional[List[Dict[str, Any]]] = None,
        layout: ViralifyLayoutType = ViralifyLayoutType.HORIZONTAL,
        theme: str = "dark",
        export_format: ViralifyExportFormat = ViralifyExportFormat.PNG_SINGLE,
        generate_narration: bool = False,
        narration_style: str = "educational",
        width: int = 1920,
        height: int = 1080,
        max_nodes: int = 10,
        animation_config: Optional[Dict] = None,
    ) -> ViralifyDiagramResult:
        """
        Generate a diagram with viralify-diagrams.

        Args:
            description: Description of the diagram
            title: Title for the diagram
            nodes: List of node definitions [{"id": "...", "label": "...", "shape": "...", "description": "..."}]
            edges: List of edge definitions [{"source": "...", "target": "...", "label": "...", "style": "..."}]
            clusters: Optional list of cluster definitions [{"id": "...", "label": "...", "node_ids": [...]}]
            layout: Layout algorithm to use
            theme: Theme name (built-in or custom)
            export_format: Export format
            generate_narration: Whether to generate narration script
            narration_style: Style for narration (educational, professional, casual, technical)
            width: Output width
            height: Output height
            max_nodes: Maximum number of nodes (auto-simplification)
            animation_config: Optional animation configuration override

        Returns:
            ViralifyDiagramResult with file path and metadata
        """
        try:
            print(f"[VIRALIFY] Generating diagram: {title} ({layout.value} layout, {theme} theme)", flush=True)

            # Create diagram
            diagram = Diagram(
                title=title,
                description=description,
                theme=theme,
                width=width,
                height=height,
                max_nodes=max_nodes
            )

            # Add nodes
            for node_data in nodes:
                shape = self._parse_node_shape(node_data.get("shape", "rounded"))
                diagram.add_node(
                    node_id=node_data["id"],
                    label=node_data.get("label", node_data["id"]),
                    description=node_data.get("description"),
                    shape=shape
                )

            # Add edges
            for edge_data in edges:
                style = self._parse_edge_style(edge_data.get("style", "solid"))
                diagram.add_edge(
                    source_id=edge_data["source"],
                    target_id=edge_data["target"],
                    label=edge_data.get("label"),
                    style=style
                )

            # Add clusters
            if clusters:
                for cluster_data in clusters:
                    diagram.add_cluster(
                        cluster_id=cluster_data["id"],
                        label=cluster_data.get("label", cluster_data["id"]),
                        node_ids=cluster_data.get("node_ids", []),
                        description=cluster_data.get("description")
                    )

            # Apply layout
            layout_engine = self._get_layout_engine(layout)
            diagram = layout_engine.layout(diagram)

            # Export based on format
            result = await self._export_diagram(
                diagram=diagram,
                export_format=export_format,
                animation_config=animation_config
            )

            # Generate narration if requested
            if generate_narration:
                result.narration_script = self._generate_narration(
                    diagram=diagram,
                    style=narration_style,
                    animation_timeline=result.animation_timeline
                )

            return result

        except Exception as e:
            print(f"[VIRALIFY] Error generating diagram: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return ViralifyDiagramResult(success=False, error=str(e))

    async def generate_from_ai_description(
        self,
        description: str,
        title: str,
        diagram_type: str = "architecture",
        layout: ViralifyLayoutType = ViralifyLayoutType.HORIZONTAL,
        theme: str = "dark",
        export_format: ViralifyExportFormat = ViralifyExportFormat.PNG_SINGLE,
        generate_narration: bool = False,
        target_audience: str = "senior",
        width: int = 1920,
        height: int = 1080,
    ) -> ViralifyDiagramResult:
        """
        Generate a diagram from an AI-generated description.
        Uses GPT-4 to extract nodes, edges, and clusters from the description.

        Args:
            description: Natural language description of the diagram
            title: Title for the diagram
            diagram_type: Type of diagram (architecture, flowchart, process, etc.)
            layout: Layout algorithm to use
            theme: Theme name
            export_format: Export format
            generate_narration: Whether to generate narration
            target_audience: Audience level for complexity
            width: Output width
            height: Output height

        Returns:
            ViralifyDiagramResult
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Determine max nodes based on audience
        max_nodes_map = {
            "beginner": 6,
            "senior": 10,
            "executive": 8
        }
        max_nodes = max_nodes_map.get(target_audience, 10)

        prompt = f"""Extract diagram structure from this description.

DESCRIPTION:
{description}

DIAGRAM TYPE: {diagram_type}
TARGET AUDIENCE: {target_audience}
MAX NODES: {max_nodes}

Return a JSON object with:
{{
    "nodes": [
        {{"id": "unique_id", "label": "Display Label", "shape": "rounded|rectangle|circle|diamond|hexagon|cylinder", "description": "optional description"}}
    ],
    "edges": [
        {{"source": "node_id", "target": "node_id", "label": "optional label", "style": "solid|dashed|dotted"}}
    ],
    "clusters": [
        {{"id": "cluster_id", "label": "Cluster Label", "node_ids": ["node1", "node2"], "description": "optional"}}
    ]
}}

Rules:
- Use short, clear labels (max 20 chars)
- Limit to {max_nodes} nodes maximum
- Group related nodes in clusters
- Use appropriate shapes for node types (cylinder for databases, hexagon for services)
- Use dashed edges for async/optional connections

Output ONLY valid JSON:"""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You extract diagram structures from descriptions. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1500
            )

            structure = json.loads(response.choices[0].message.content)

            return await self.generate_diagram(
                description=description,
                title=title,
                nodes=structure.get("nodes", []),
                edges=structure.get("edges", []),
                clusters=structure.get("clusters"),
                layout=layout,
                theme=theme,
                export_format=export_format,
                generate_narration=generate_narration,
                width=width,
                height=height,
                max_nodes=max_nodes
            )

        except Exception as e:
            print(f"[VIRALIFY] AI extraction failed: {e}", flush=True)
            return ViralifyDiagramResult(success=False, error=f"AI extraction failed: {e}")

    async def _export_diagram(
        self,
        diagram: Diagram,
        export_format: ViralifyExportFormat,
        animation_config: Optional[Dict] = None
    ) -> ViralifyDiagramResult:
        """Export diagram to the specified format"""

        file_id = uuid.uuid4().hex[:8]

        if export_format == ViralifyExportFormat.SVG_STATIC:
            return await self._export_svg_static(diagram, file_id)
        elif export_format == ViralifyExportFormat.SVG_ANIMATED:
            return await self._export_svg_animated(diagram, file_id, animation_config)
        elif export_format == ViralifyExportFormat.PNG_FRAMES:
            return await self._export_png_frames(diagram, file_id)
        else:  # PNG_SINGLE
            return await self._export_png_single(diagram, file_id)

    async def _export_svg_static(self, diagram: Diagram, file_id: str) -> ViralifyDiagramResult:
        """Export as static SVG"""
        exporter = SVGExporter()
        output_path = self.output_dir / f"diagram_{file_id}.svg"

        svg_content = exporter.export(diagram, str(output_path))

        # Get element metadata for external animation
        elements = exporter.get_elements()
        timeline = {
            "elements": [
                {
                    "id": e.id,
                    "type": e.element_type,
                    "order": e.order,
                    "metadata": e.metadata
                }
                for e in elements
            ]
        }

        return ViralifyDiagramResult(
            success=True,
            file_path=str(output_path),
            svg_content=svg_content,
            animation_timeline=timeline
        )

    async def _export_svg_animated(
        self,
        diagram: Diagram,
        file_id: str,
        animation_config: Optional[Dict] = None
    ) -> ViralifyDiagramResult:
        """Export as animated SVG with CSS animations"""

        # Parse animation config
        config = self.default_animation_config
        if animation_config:
            config = AnimationConfig(
                duration=animation_config.get("duration", 0.5),
                delay_between=animation_config.get("delay_between", 0.3),
                easing=animation_config.get("easing", "ease-out"),
                stagger=animation_config.get("stagger", True),
                initial_delay=animation_config.get("initial_delay", 0.5),
                loop=animation_config.get("loop", False)
            )

        exporter = AnimatedSVGExporter(config=config)
        output_path = self.output_dir / f"diagram_{file_id}_animated.svg"

        # Determine animation types
        node_anim = AnimationType.SCALE_IN
        edge_anim = AnimationType.DRAW
        cluster_anim = AnimationType.FADE_IN

        svg_content = exporter.export(
            diagram,
            output_path=str(output_path),
            node_animation=node_anim,
            edge_animation=edge_anim,
            cluster_animation=cluster_anim
        )

        # Get timing script
        timing = exporter.export_timing_script()

        return ViralifyDiagramResult(
            success=True,
            file_path=str(output_path),
            svg_content=svg_content,
            animation_timeline=timing
        )

    async def _export_png_frames(self, diagram: Diagram, file_id: str) -> ViralifyDiagramResult:
        """Export as PNG frames for video composition"""

        exporter = PNGFrameExporter(config=self.default_frame_config)
        frames_dir = self.output_dir / f"frames_{file_id}"

        try:
            frames = exporter.export(diagram, str(frames_dir))
            manifest = exporter.export_frame_manifest()

            return ViralifyDiagramResult(
                success=True,
                file_path=str(frames_dir),
                frame_manifest=manifest
            )
        except ImportError as e:
            # Fallback to single PNG if cairosvg not available
            print(f"[VIRALIFY] PNG frames export failed (missing deps): {e}, falling back to single PNG", flush=True)
            return await self._export_png_single(diagram, file_id)

    async def _export_png_single(self, diagram: Diagram, file_id: str) -> ViralifyDiagramResult:
        """Export as single PNG image"""

        # First export as SVG
        svg_exporter = SVGExporter()
        svg_content = svg_exporter.export(diagram)

        output_path = self.output_dir / f"diagram_{file_id}.png"

        try:
            # Try to convert SVG to PNG using cairosvg
            import cairosvg
            cairosvg.svg2png(
                bytestring=svg_content.encode('utf-8'),
                write_to=str(output_path),
                output_width=diagram.width,
                output_height=diagram.height
            )
        except ImportError:
            # Fallback: save SVG and note that PNG conversion requires cairosvg
            svg_path = self.output_dir / f"diagram_{file_id}.svg"
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)

            return ViralifyDiagramResult(
                success=True,
                file_path=str(svg_path),
                svg_content=svg_content,
                error="PNG export requires cairosvg. Exported as SVG instead."
            )

        return ViralifyDiagramResult(
            success=True,
            file_path=str(output_path)
        )

    def _generate_narration(
        self,
        diagram: Diagram,
        style: str,
        animation_timeline: Optional[Dict] = None
    ) -> Dict:
        """Generate narration script for the diagram"""

        # Map style string to enum
        style_map = {
            "educational": NarrationStyle.EDUCATIONAL,
            "professional": NarrationStyle.PROFESSIONAL,
            "casual": NarrationStyle.CASUAL,
            "technical": NarrationStyle.TECHNICAL
        }
        narration_style = style_map.get(style, NarrationStyle.EDUCATIONAL)

        narrator = DiagramNarrator(style=narration_style)
        script = narrator.generate_script(
            diagram,
            element_duration=2.0,
            include_intro=True,
            include_conclusion=True
        )

        # Sync with animation if available
        if animation_timeline and "elements" in animation_timeline:
            script = narrator.synchronize_with_animation(
                script,
                animation_timeline["elements"]
            )

        return {
            "total_duration": script.total_duration,
            "style": script.style.value,
            "segments": [
                {
                    "element_id": s.element_id,
                    "element_type": s.element_type,
                    "start_time": s.start_time,
                    "duration": s.duration,
                    "text": s.text,
                    "emphasis_words": s.emphasis_words,
                    "pause_after": s.pause_after
                }
                for s in script.segments
            ],
            "srt": script.to_srt(),
            "ssml": script.to_ssml()
        }

    def _get_layout_engine(self, layout: ViralifyLayoutType):
        """Get the appropriate layout engine"""
        if layout == ViralifyLayoutType.HORIZONTAL:
            return HorizontalLayout()
        elif layout == ViralifyLayoutType.VERTICAL:
            return VerticalLayout()
        elif layout == ViralifyLayoutType.GRID:
            return GridLayout()
        elif layout == ViralifyLayoutType.RADIAL:
            return RadialLayout()
        else:
            return HorizontalLayout()

    def _parse_node_shape(self, shape: str) -> NodeShape:
        """Parse node shape string to enum"""
        shape_map = {
            "rectangle": NodeShape.RECTANGLE,
            "rounded": NodeShape.ROUNDED,
            "circle": NodeShape.CIRCLE,
            "diamond": NodeShape.DIAMOND,
            "hexagon": NodeShape.HEXAGON,
            "cylinder": NodeShape.CYLINDER,
            "cloud": NodeShape.CLOUD
        }
        return shape_map.get(shape.lower(), NodeShape.ROUNDED)

    def _parse_edge_style(self, style: str) -> EdgeStyle:
        """Parse edge style string to enum"""
        style_map = {
            "solid": EdgeStyle.SOLID,
            "dashed": EdgeStyle.DASHED,
            "dotted": EdgeStyle.DOTTED
        }
        return style_map.get(style.lower(), EdgeStyle.SOLID)

    def list_available_themes(self) -> List[str]:
        """List all available themes"""
        return self.theme_manager.list_themes()

    def get_theme_json(self, theme_name: str) -> Optional[str]:
        """Get a theme as JSON for inspection/modification"""
        theme = self.theme_manager.get(theme_name)
        if theme:
            return theme.to_json()
        return None


# Singleton instance
_viralify_service: Optional[ViralifyDiagramService] = None


def get_viralify_diagram_service() -> ViralifyDiagramService:
    """Get the singleton ViralifyDiagramService instance"""
    global _viralify_service
    if _viralify_service is None:
        _viralify_service = ViralifyDiagramService()
    return _viralify_service
