"""
Viralify Diagrams Service

Integrates the viralify-diagrams library for video-optimized diagram generation.
Provides professional diagrams with:
- Intelligent taxonomy and template routing
- Theme customization (including user-uploaded themes)
- Animation support (SVG with CSS, PNG frames)
- Narration script generation for voiceover sync
- Hybrid Graphviz layout for 50+ component diagrams
- Enterprise edge management (100+ connections)
"""

import os
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Viralify Diagrams library - Core
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
    GraphvizLayout,
    GraphvizAlgorithm,
    auto_layout,
    SVGExporter,
    AnimatedSVGExporter,
    PNGFrameExporter,
    DiagramNarrator,
    NarrationScript,
    # Enterprise edge management
    apply_smart_routing,
    apply_edge_bundling,
    apply_channel_routing,
    EdgeRoutingMode,
)
from viralify_diagrams.core.diagram import NodeShape, EdgeStyle, EdgeDirection
from viralify_diagrams.exporters.animated_svg_exporter import AnimationConfig, AnimationType
from viralify_diagrams.exporters.png_frame_exporter import FrameConfig
from viralify_diagrams.narration.diagram_narrator import NarrationStyle

# Viralify Diagrams library - Taxonomy & Templates
from viralify_diagrams import (
    # Taxonomy
    DiagramDomain,
    DiagramCategory,
    DiagramType,
    TargetAudience,
    DiagramComplexity,
    AudienceType,
    RequestClassifier,
    ClassificationResult,
    DiagramRouter,
    RoutingResult,
    SlideOptimizer,
    OptimizationResult,
    # Templates
    get_template,
    get_template_registry,
    DiagramTemplate,
    # Specific templates
    C4ContextTemplate,
    C4ContainerTemplate,
    C4ComponentTemplate,
    C4DeploymentTemplate,
    UMLClassTemplate,
    UMLSequenceTemplate,
    UMLActivityTemplate,
    DFDTemplate,
    ERDTemplate,
    DataLineageTemplate,
    STRIDEThreatTemplate,
    CICDPipelineTemplate,
    KubernetesTemplate,
    BPMNProcessTemplate,
)


class ViralifyLayoutType(str, Enum):
    """Layout types for viralify diagrams"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    GRID = "grid"
    RADIAL = "radial"
    GRAPHVIZ_DOT = "dot"
    GRAPHVIZ_NEATO = "neato"
    GRAPHVIZ_FDP = "fdp"
    GRAPHVIZ_SFDP = "sfdp"
    GRAPHVIZ_CIRCO = "circo"
    GRAPHVIZ_TWOPI = "twopi"
    AUTO = "auto"


class ViralifyExportFormat(str, Enum):
    """Export formats for viralify diagrams"""
    SVG_STATIC = "svg_static"
    SVG_ANIMATED = "svg_animated"
    PNG_FRAMES = "png_frames"
    PNG_SINGLE = "png_single"


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
    # New: taxonomy metadata
    classification: Optional[Dict] = None
    template_used: Optional[str] = None
    slide_count: int = 1


@dataclass
class SlideResult:
    """Result for a single slide in multi-slide generation"""
    slide_number: int
    title: str
    file_path: Optional[str] = None
    svg_content: Optional[str] = None
    narration: Optional[str] = None
    element_count: int = 0


@dataclass
class MultiSlideResult:
    """Result of multi-slide diagram generation"""
    success: bool
    slides: List[SlideResult] = field(default_factory=list)
    total_slides: int = 0
    classification: Optional[Dict] = None
    optimization: Optional[Dict] = None
    error: Optional[str] = None


class ViralifyDiagramService:
    """
    Service for generating video-optimized diagrams using viralify-diagrams.

    Features:
    - Intelligent taxonomy-based diagram routing
    - Multiple templates (C4, UML, DFD, ERD, BPMN, STRIDE, K8s, CI/CD)
    - Automatic slide optimization by audience
    - Enterprise edge management for 100+ connections
    - Theme customization (built-in + custom JSON themes)
    - Three export modes (static SVG, animated SVG, PNG frames)
    - Narration script generation for voiceover
    """

    def __init__(self, output_dir: str = "/tmp/presentations/viralify_diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.theme_manager = ThemeManager()

        # Taxonomy components
        self.classifier = RequestClassifier()
        self.router = DiagramRouter()
        self.optimizer = SlideOptimizer()
        self.registry = get_template_registry()

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
        """Register a custom theme from JSON."""
        try:
            theme = Theme.from_json(theme_json)
            self.theme_manager.register(theme)
            print(f"[VIRALIFY] Registered custom theme: {theme.name}", flush=True)
            return True
        except Exception as e:
            print(f"[VIRALIFY] Failed to register theme: {e}", flush=True)
            return False

    def classify_request(self, description: str) -> ClassificationResult:
        """
        Classify a diagram request to determine the best template.

        Args:
            description: Natural language description of the diagram

        Returns:
            ClassificationResult with diagram type, domain, complexity, etc.
        """
        return self.classifier.classify(description)

    def get_slide_recommendation(
        self,
        description: str,
        target_audience: str = "developer",
        max_slides: Optional[int] = None
    ) -> Tuple[ClassificationResult, RoutingResult, OptimizationResult]:
        """
        Get recommendations for diagram generation including slide count.

        Args:
            description: Description of the diagram
            target_audience: Target audience (executive, architect, developer, student)
            max_slides: Maximum number of slides allowed

        Returns:
            Tuple of (classification, routing, optimization) results
        """
        # Classify
        classification = self.classifier.classify(description)
        print(f"[VIRALIFY] Classified as: {classification.diagram_type} "
              f"(domain: {classification.domain}, confidence: {classification.overall_confidence:.2f})", flush=True)

        # Route to template (router does its own classification internally)
        routing = self.router.route(
            description,
            audience=target_audience
        )
        template_id = getattr(routing.primary_template, 'template_id', None) if routing.primary_template else None
        print(f"[VIRALIFY] Routed to template: {template_id or 'default'}", flush=True)

        # Optimize slides
        # Estimate elements based on complexity
        complexity_elements = {
            "low": 5,
            "medium": 10,
            "high": 20,
            "very_high": 30
        }
        estimated_elements = complexity_elements.get(
            classification.complexity.value if classification.complexity else "medium",
            10
        )

        # Map string audience to AudienceType enum
        audience_mapping = {
            "executive": AudienceType.EXECUTIVE,
            "manager": AudienceType.MANAGER,
            "architect": AudienceType.ARCHITECT,
            "developer": AudienceType.DEVELOPER,
            "data_engineer": AudienceType.DATA_ENGINEER,
            "devops": AudienceType.DEVOPS,
            "security": AudienceType.SECURITY,
            "general": AudienceType.GENERAL,
        }
        audience_enum = audience_mapping.get(target_audience.lower(), AudienceType.GENERAL)

        # Get diagram type enum (or default to GENERIC_ARCH)
        diagram_type_enum = classification.diagram_type if classification.diagram_type else DiagramType.GENERIC_ARCH

        optimization = self.optimizer.optimize(
            element_count=estimated_elements,
            diagram_type=diagram_type_enum,
            audience=audience_enum
        )
        print(f"[VIRALIFY] Optimized: {optimization.slide_count} slides, "
              f"avg {optimization.avg_elements_per_slide:.1f} elements/slide", flush=True)

        return classification, routing, optimization

    async def generate_from_description_intelligent(
        self,
        description: str,
        title: str,
        target_audience: str = "developer",
        theme: str = "dark",
        export_format: ViralifyExportFormat = ViralifyExportFormat.PNG_SINGLE,
        generate_narration: bool = False,
        max_slides: Optional[int] = None,
        width: int = 1920,
        height: int = 1080,
    ) -> MultiSlideResult:
        """
        Generate diagrams intelligently using taxonomy and templates.

        This method:
        1. Classifies the request to determine diagram type
        2. Routes to the appropriate template
        3. Optimizes slide count based on audience
        4. Generates content using GPT-4
        5. Creates diagrams using the template

        Args:
            description: Natural language description
            title: Title for the diagram(s)
            target_audience: Target audience level
            theme: Visual theme
            export_format: Export format
            generate_narration: Whether to generate narration
            max_slides: Maximum slides allowed
            width: Output width
            height: Output height

        Returns:
            MultiSlideResult with generated slides
        """
        try:
            # Step 1-3: Classify, route, optimize
            classification, routing, optimization = self.get_slide_recommendation(
                description, target_audience, max_slides
            )

            # Step 4: Get template
            template_id = getattr(routing.primary_template, 'template_id', None) if routing.primary_template else None
            template = get_template(template_id) if template_id else None

            if template:
                print(f"[VIRALIFY] Using template: {template.name}", flush=True)

            # Step 5: Generate content with GPT-4
            structure = await self._generate_structure_with_ai(
                description=description,
                diagram_type=classification.diagram_type.value if classification.diagram_type else "architecture",
                template=template,
                target_audience=target_audience,
                max_elements=optimization.total_elements
            )

            if not structure:
                return MultiSlideResult(
                    success=False,
                    error="Failed to generate diagram structure"
                )

            # Step 6: Split into slides and generate
            slides = await self._generate_slides(
                structure=structure,
                template=template,
                optimization=optimization,
                title=title,
                theme=theme,
                export_format=export_format,
                generate_narration=generate_narration,
                width=width,
                height=height
            )

            return MultiSlideResult(
                success=True,
                slides=slides,
                total_slides=len(slides),
                classification={
                    "domain": classification.domain.value if classification.domain else None,
                    "category": classification.category.value if classification.category else None,
                    "diagram_type": classification.diagram_type.value if classification.diagram_type else None,
                    "complexity": classification.complexity.value if classification.complexity else None,
                    "confidence": classification.confidence
                },
                optimization={
                    "total_slides": optimization.slide_count,
                    "avg_elements_per_slide": optimization.avg_elements_per_slide,
                    "total_elements": optimization.total_elements,
                    "optimization_score": optimization.optimization_score
                }
            )

        except Exception as e:
            print(f"[VIRALIFY] Error in intelligent generation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return MultiSlideResult(success=False, error=str(e))

    async def _generate_structure_with_ai(
        self,
        description: str,
        diagram_type: str,
        template: Optional[DiagramTemplate],
        target_audience: str,
        max_elements: int
    ) -> Optional[Dict]:
        """Generate diagram structure using GPT-4."""

        # Get allowed element types from template
        element_types = []
        relation_types = []
        if template:
            element_types = template.get_element_types()
            relation_types = template.get_relation_types()

        # Build prompt with template awareness
        element_guidance = ""
        if element_types:
            element_guidance = f"""
ALLOWED ELEMENT TYPES for this diagram:
{', '.join(element_types)}

ALLOWED RELATION TYPES:
{', '.join(relation_types)}

Use ONLY these element and relation types."""

        # Try shared LLM provider
        try:
            from shared.llm_provider import get_llm_client, get_model_name
            client = get_llm_client()
            model = get_model_name("quality")
        except ImportError:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("OPENAI_MODEL", "gpt-4o")

        # Audience-specific limits
        max_nodes_map = {
            "executive": 8,
            "architect": 12,
            "developer": 10,
            "student": 6
        }
        max_nodes = min(max_nodes_map.get(target_audience, 10), max_elements)

        prompt = f"""Extract diagram structure from this description.

DESCRIPTION:
{description}

DIAGRAM TYPE: {diagram_type}
TARGET AUDIENCE: {target_audience}
MAX ELEMENTS: {max_nodes}
{element_guidance}

Return a JSON object with:
{{
    "nodes": [
        {{"id": "unique_id", "label": "Display Label", "element_type": "type_from_allowed_list", "description": "optional description", "properties": {{}}}}
    ],
    "edges": [
        {{"source": "node_id", "target": "node_id", "relation_type": "type_from_allowed_list", "label": "optional label"}}
    ],
    "clusters": [
        {{"id": "cluster_id", "label": "Cluster Label", "node_ids": ["node1", "node2"]}}
    ],
    "metadata": {{
        "suggested_layout": "horizontal|vertical|graphviz",
        "complexity": "low|medium|high"
    }}
}}

Rules:
- Use short, clear labels (max 25 chars)
- Limit to {max_nodes} nodes maximum
- Group related nodes in clusters when logical
- Use the EXACT element_type and relation_type from the allowed lists

Output ONLY valid JSON:"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You extract diagram structures from descriptions. Use the exact element and relation types provided. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"[VIRALIFY] AI structure generation failed: {e}", flush=True)
            return None

    async def _generate_slides(
        self,
        structure: Dict,
        template: Optional[DiagramTemplate],
        optimization: OptimizationResult,
        title: str,
        theme: str,
        export_format: ViralifyExportFormat,
        generate_narration: bool,
        width: int,
        height: int
    ) -> List[SlideResult]:
        """Generate slides from structure."""

        slides = []
        nodes = structure.get("nodes", [])
        edges = structure.get("edges", [])
        clusters = structure.get("clusters", [])

        # Calculate max elements per slide from optimization result
        # Use avg or default to 8 if no slides
        if optimization.slides:
            max_per_slide = max(
                len(s.element_ids) for s in optimization.slides if s.element_ids
            ) if any(s.element_ids for s in optimization.slides) else 8
        else:
            max_per_slide = 8
        total_slides = optimization.slide_count

        # Split nodes into groups
        node_groups = [nodes[i:i + max_per_slide] for i in range(0, len(nodes), max_per_slide)]

        # If only one group, use all nodes
        if len(node_groups) == 0:
            node_groups = [nodes]

        for slide_num, node_group in enumerate(node_groups, 1):
            node_ids = {n["id"] for n in node_group}

            # Filter edges to only those connecting nodes in this group
            slide_edges = [
                e for e in edges
                if e["source"] in node_ids and e["target"] in node_ids
            ]

            # Filter clusters
            slide_clusters = []
            for cluster in clusters:
                cluster_node_ids = [nid for nid in cluster.get("node_ids", []) if nid in node_ids]
                if cluster_node_ids:
                    slide_clusters.append({
                        **cluster,
                        "node_ids": cluster_node_ids
                    })

            # Generate slide title
            slide_title = f"{title}" if len(node_groups) == 1 else f"{title} ({slide_num}/{len(node_groups)})"

            # Generate diagram
            result = await self._generate_single_slide(
                nodes=node_group,
                edges=slide_edges,
                clusters=slide_clusters,
                template=template,
                title=slide_title,
                theme=theme,
                export_format=export_format,
                generate_narration=generate_narration,
                width=width,
                height=height,
                metadata=structure.get("metadata", {})
            )

            slides.append(SlideResult(
                slide_number=slide_num,
                title=slide_title,
                file_path=result.file_path,
                svg_content=result.svg_content,
                narration=result.narration_script.get("ssml") if result.narration_script else None,
                element_count=len(node_group)
            ))

        return slides

    async def _generate_single_slide(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        clusters: List[Dict],
        template: Optional[DiagramTemplate],
        title: str,
        theme: str,
        export_format: ViralifyExportFormat,
        generate_narration: bool,
        width: int,
        height: int,
        metadata: Dict
    ) -> ViralifyDiagramResult:
        """Generate a single slide diagram."""

        try:
            # Create diagram
            diagram = Diagram(
                title=title,
                theme=theme,
                width=width,
                height=height
            )

            # Add nodes using template if available
            for node_data in nodes:
                if template:
                    try:
                        elem = template.create_element(
                            element_type=node_data.get("element_type", "process"),
                            label=node_data.get("label", node_data["id"]),
                            properties={
                                "id": node_data["id"],
                                "description": node_data.get("description", ""),
                                **node_data.get("properties", {})
                            }
                        )
                        node = Node(
                            id=elem["id"],
                            label=elem["label"],
                            shape=self._map_shape(elem.get("shape", "rounded")),
                            fill_color=elem.get("fill_color"),
                            stroke_color=elem.get("stroke_color"),
                            description=node_data.get("description", "")
                        )
                    except Exception as e:
                        print(f"[VIRALIFY] Template element creation failed, using default: {e}", flush=True)
                        node = self._create_default_node(node_data)
                else:
                    node = self._create_default_node(node_data)

                diagram.add_node(node)

            # Add edges using template if available
            for edge_data in edges:
                if template:
                    try:
                        rel = template.create_relation(
                            relation_type=edge_data.get("relation_type", "uses"),
                            source_id=edge_data["source"],
                            target_id=edge_data["target"],
                            label=edge_data.get("label"),
                            properties=edge_data.get("properties", {})
                        )
                        edge = Edge(
                            source=rel["source"],
                            target=rel["target"],
                            label=rel.get("label", ""),
                            style=self._map_line_style(rel.get("line_style", "solid"))
                        )
                    except Exception as e:
                        print(f"[VIRALIFY] Template relation creation failed, using default: {e}", flush=True)
                        edge = self._create_default_edge(edge_data)
                else:
                    edge = self._create_default_edge(edge_data)

                diagram.add_edge(edge)

            # Add clusters
            for cluster_data in clusters:
                cluster = Cluster(
                    id=cluster_data["id"],
                    label=cluster_data.get("label", cluster_data["id"]),
                    nodes=cluster_data.get("node_ids", []),
                    description=cluster_data.get("description", "")
                )
                diagram.add_cluster(cluster)

            # Apply layout
            suggested_layout = metadata.get("suggested_layout", "horizontal")
            layout_engine = self._get_layout_engine(
                ViralifyLayoutType(suggested_layout) if suggested_layout in [e.value for e in ViralifyLayoutType] else ViralifyLayoutType.AUTO,
                num_nodes=len(nodes)
            )
            diagram = layout_engine.layout(diagram)

            # Apply enterprise edge routing for complex diagrams
            if len(edges) > 20:
                print(f"[VIRALIFY] Applying channel routing for {len(edges)} edges", flush=True)
                diagram = apply_channel_routing(diagram)
            elif len(edges) > 10:
                print(f"[VIRALIFY] Applying smart routing for {len(edges)} edges", flush=True)
                diagram = apply_smart_routing(diagram, mode=EdgeRoutingMode.ORTHOGONAL)

            # Export
            result = await self._export_diagram(
                diagram=diagram,
                export_format=export_format,
                animation_config=None
            )

            # Generate narration if requested
            if generate_narration:
                result.narration_script = self._generate_narration(
                    diagram=diagram,
                    style="educational",
                    animation_timeline=result.animation_timeline
                )

            return result

        except Exception as e:
            print(f"[VIRALIFY] Single slide generation error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return ViralifyDiagramResult(success=False, error=str(e))

    def _create_default_node(self, node_data: Dict) -> Node:
        """Create a default node when template is not available."""
        shape = self._parse_node_shape(node_data.get("shape", node_data.get("element_type", "rounded")))
        return Node(
            id=node_data["id"],
            label=node_data.get("label", node_data["id"]),
            description=node_data.get("description", ""),
            shape=shape
        )

    def _create_default_edge(self, edge_data: Dict) -> Edge:
        """Create a default edge when template is not available."""
        style = self._parse_edge_style(edge_data.get("style", "solid"))
        return Edge(
            source=edge_data["source"],
            target=edge_data["target"],
            label=edge_data.get("label"),
            style=style
        )

    def _map_shape(self, shape_str: str) -> NodeShape:
        """Map shape string to NodeShape enum."""
        shape_map = {
            "rectangle": NodeShape.RECTANGLE,
            "rounded": NodeShape.ROUNDED,
            "circle": NodeShape.CIRCLE,
            "diamond": NodeShape.DIAMOND,
            "hexagon": NodeShape.HEXAGON,
            "cylinder": NodeShape.CYLINDER,
            "cloud": NodeShape.CLOUD,
            "actor": NodeShape.CIRCLE,  # Fallback
            "lifeline": NodeShape.RECTANGLE,
            "component": NodeShape.RECTANGLE,
            "node": NodeShape.RECTANGLE,
            "queue": NodeShape.CYLINDER,
            "package": NodeShape.RECTANGLE,
        }
        return shape_map.get(shape_str.lower(), NodeShape.ROUNDED)

    def _map_line_style(self, style_str: str) -> EdgeStyle:
        """Map line style string to EdgeStyle enum."""
        style_map = {
            "solid": EdgeStyle.SOLID,
            "dashed": EdgeStyle.DASHED,
            "dotted": EdgeStyle.DOTTED
        }
        return style_map.get(style_str.lower(), EdgeStyle.SOLID)

    # =========================================================================
    # Legacy methods for backward compatibility
    # =========================================================================

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
        Generate a diagram with viralify-diagrams (legacy method).

        For new code, prefer generate_from_description_intelligent().
        """
        try:
            print(f"[VIRALIFY] Generating diagram: {title} ({layout.value} layout, {theme} theme)", flush=True)

            diagram = Diagram(
                title=title,
                description=description,
                theme=theme,
                width=width,
                height=height,
                max_nodes=max_nodes
            )

            for node_data in nodes:
                shape = self._parse_node_shape(node_data.get("shape", "rounded"))
                node = Node(
                    id=node_data["id"],
                    label=node_data.get("label", node_data["id"]),
                    description=node_data.get("description", ""),
                    shape=shape
                )
                diagram.add_node(node)

            for edge_data in edges:
                style = self._parse_edge_style(edge_data.get("style", "solid"))
                edge = Edge(
                    source=edge_data["source"],
                    target=edge_data["target"],
                    label=edge_data.get("label"),
                    style=style
                )
                diagram.add_edge(edge)

            if clusters:
                for cluster_data in clusters:
                    cluster = Cluster(
                        id=cluster_data["id"],
                        label=cluster_data.get("label", cluster_data["id"]),
                        nodes=cluster_data.get("node_ids", []),
                        description=cluster_data.get("description", "")
                    )
                    diagram.add_cluster(cluster)

            layout_engine = self._get_layout_engine(layout, num_nodes=len(nodes))
            diagram = layout_engine.layout(diagram)
            print(f"[VIRALIFY] Layout applied: {layout.value} with {len(nodes)} nodes", flush=True)

            result = await self._export_diagram(
                diagram=diagram,
                export_format=export_format,
                animation_config=animation_config
            )

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
        Generate a diagram from an AI-generated description (legacy method).

        For new code, prefer generate_from_description_intelligent().
        """
        # Map legacy audience names
        audience_map = {
            "beginner": "student",
            "senior": "developer",
            "executive": "executive"
        }
        mapped_audience = audience_map.get(target_audience, target_audience)

        # Use new intelligent method
        result = await self.generate_from_description_intelligent(
            description=description,
            title=title,
            target_audience=mapped_audience,
            theme=theme,
            export_format=export_format,
            generate_narration=generate_narration,
            max_slides=1,  # Legacy mode: single slide
            width=width,
            height=height
        )

        # Convert to legacy result format
        if result.success and result.slides:
            slide = result.slides[0]
            return ViralifyDiagramResult(
                success=True,
                file_path=slide.file_path,
                svg_content=slide.svg_content,
                narration_script={"ssml": slide.narration} if slide.narration else None,
                classification=result.classification,
                template_used=result.classification.get("diagram_type") if result.classification else None
            )
        else:
            return ViralifyDiagramResult(
                success=False,
                error=result.error
            )

    async def _export_diagram(
        self,
        diagram: Diagram,
        export_format: ViralifyExportFormat,
        animation_config: Optional[Dict] = None
    ) -> ViralifyDiagramResult:
        """Export diagram to the specified format."""

        file_id = uuid.uuid4().hex[:8]

        if export_format == ViralifyExportFormat.SVG_STATIC:
            return await self._export_svg_static(diagram, file_id)
        elif export_format == ViralifyExportFormat.SVG_ANIMATED:
            return await self._export_svg_animated(diagram, file_id, animation_config)
        elif export_format == ViralifyExportFormat.PNG_FRAMES:
            return await self._export_png_frames(diagram, file_id)
        else:
            return await self._export_png_single(diagram, file_id)

    async def _export_svg_static(self, diagram: Diagram, file_id: str) -> ViralifyDiagramResult:
        """Export as static SVG."""
        exporter = SVGExporter()
        output_path = self.output_dir / f"diagram_{file_id}.svg"

        svg_content = exporter.export(diagram, str(output_path))

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
        """Export as animated SVG with CSS animations."""

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

        svg_content = exporter.export(
            diagram,
            output_path=str(output_path),
            node_animation=AnimationType.SCALE_IN,
            edge_animation=AnimationType.DRAW,
            cluster_animation=AnimationType.FADE_IN
        )

        timing = exporter.export_timing_script()

        return ViralifyDiagramResult(
            success=True,
            file_path=str(output_path),
            svg_content=svg_content,
            animation_timeline=timing
        )

    async def _export_png_frames(self, diagram: Diagram, file_id: str) -> ViralifyDiagramResult:
        """Export as PNG frames for video composition."""

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
            print(f"[VIRALIFY] PNG frames export failed (missing deps): {e}", flush=True)
            return await self._export_png_single(diagram, file_id)

    async def _export_png_single(self, diagram: Diagram, file_id: str) -> ViralifyDiagramResult:
        """Export as single PNG image."""

        svg_exporter = SVGExporter()
        svg_content = svg_exporter.export(diagram)

        output_path = self.output_dir / f"diagram_{file_id}.png"

        try:
            import cairosvg
            cairosvg.svg2png(
                bytestring=svg_content.encode('utf-8'),
                write_to=str(output_path),
                output_width=diagram.width,
                output_height=diagram.height
            )
        except ImportError:
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
        """Generate narration script for the diagram."""

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

    def _get_layout_engine(self, layout: ViralifyLayoutType, num_nodes: int = 0):
        """Get the appropriate layout engine."""

        if layout == ViralifyLayoutType.HORIZONTAL:
            return HorizontalLayout()
        elif layout == ViralifyLayoutType.VERTICAL:
            return VerticalLayout()
        elif layout == ViralifyLayoutType.GRID:
            return GridLayout()
        elif layout == ViralifyLayoutType.RADIAL:
            return RadialLayout()
        elif layout == ViralifyLayoutType.GRAPHVIZ_DOT:
            return GraphvizLayout(algorithm="dot", rankdir="TB")
        elif layout == ViralifyLayoutType.GRAPHVIZ_NEATO:
            return GraphvizLayout(algorithm="neato")
        elif layout == ViralifyLayoutType.GRAPHVIZ_FDP:
            return GraphvizLayout(algorithm="fdp")
        elif layout == ViralifyLayoutType.GRAPHVIZ_SFDP:
            return GraphvizLayout(algorithm="sfdp")
        elif layout == ViralifyLayoutType.GRAPHVIZ_CIRCO:
            return GraphvizLayout(algorithm="circo")
        elif layout == ViralifyLayoutType.GRAPHVIZ_TWOPI:
            return GraphvizLayout(algorithm="twopi")
        elif layout == ViralifyLayoutType.AUTO:
            if num_nodes >= 10:
                print(f"[VIRALIFY] Auto-selecting Graphviz DOT for {num_nodes} nodes", flush=True)
                return GraphvizLayout(algorithm="dot", rankdir="TB")
            else:
                print(f"[VIRALIFY] Auto-selecting Horizontal layout for {num_nodes} nodes", flush=True)
                return HorizontalLayout()
        else:
            return GraphvizLayout(algorithm="dot", rankdir="TB")

    def _parse_node_shape(self, shape: str) -> NodeShape:
        """Parse node shape string to enum."""
        shape_map = {
            "rectangle": NodeShape.RECTANGLE,
            "rounded": NodeShape.ROUNDED,
            "circle": NodeShape.CIRCLE,
            "diamond": NodeShape.DIAMOND,
            "hexagon": NodeShape.HEXAGON,
            "cylinder": NodeShape.CYLINDER,
            "cloud": NodeShape.CLOUD,
            # Template element types mapping
            "person": NodeShape.CIRCLE,
            "software_system": NodeShape.ROUNDED,
            "external_system": NodeShape.ROUNDED,
            "container": NodeShape.ROUNDED,
            "database": NodeShape.CYLINDER,
            "message_queue": NodeShape.CYLINDER,
            "component": NodeShape.RECTANGLE,
            "process": NodeShape.ROUNDED,
            "data_store": NodeShape.CYLINDER,
            "external_entity": NodeShape.RECTANGLE,
        }
        return shape_map.get(shape.lower(), NodeShape.ROUNDED)

    def _parse_edge_style(self, style: str) -> EdgeStyle:
        """Parse edge style string to enum."""
        style_map = {
            "solid": EdgeStyle.SOLID,
            "dashed": EdgeStyle.DASHED,
            "dotted": EdgeStyle.DOTTED
        }
        return style_map.get(style.lower(), EdgeStyle.SOLID)

    def list_available_themes(self) -> List[str]:
        """List all available themes."""
        return self.theme_manager.list_themes()

    def list_available_templates(self) -> List[str]:
        """List all available diagram templates."""
        return self.registry.list_all()

    def list_templates_by_domain(self, domain: str) -> List[str]:
        """List templates for a specific domain."""
        try:
            domain_enum = DiagramDomain(domain)
            return self.registry.list_by_domain(domain_enum)
        except ValueError:
            return []

    def get_template_info(self, template_id: str) -> Optional[Dict]:
        """Get information about a specific template."""
        return self.registry.get_template_info(template_id)


# Singleton instance
_viralify_service: Optional[ViralifyDiagramService] = None


def get_viralify_diagram_service() -> ViralifyDiagramService:
    """Get the singleton ViralifyDiagramService instance."""
    global _viralify_service
    if _viralify_service is None:
        _viralify_service = ViralifyDiagramService()
    return _viralify_service
