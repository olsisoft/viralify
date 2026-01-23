"""
Diagram Generator Service

Generates visual diagrams for presentation slides using:
1. Python Diagrams library for professional architecture diagrams (PRIMARY)
   - Supports AWS, Azure, GCP, Kubernetes, On-Premise icons
   - Uses GPT-4o to generate Python code from descriptions
2. Mermaid.js via Kroki API for flowcharts and sequences (SECONDARY)
3. Pillow for fallback rendering (TERTIARY)
"""

import asyncio
import base64
import json
import math
import os
import re
import tempfile
import uuid
import zlib
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import httpx


class DiagramType(str, Enum):
    FLOWCHART = "flowchart"
    ARCHITECTURE = "architecture"
    SEQUENCE = "sequence"
    MINDMAP = "mindmap"
    COMPARISON = "comparison"
    HIERARCHY = "hierarchy"
    PROCESS = "process"
    TIMELINE = "timeline"


class DiagramProvider(str, Enum):
    """Cloud/Tech providers for diagram icons"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "k8s"
    ON_PREMISE = "onprem"
    GENERIC = "generic"
    PROGRAMMING = "programming"
    SAAS = "saas"


class DiagramStyle(str, Enum):
    """Visual style for diagrams"""
    DARK = "dark"
    LIGHT = "light"
    NEUTRAL = "neutral"
    COLORFUL = "colorful"


@dataclass
class DiagramNode:
    """A node in the diagram"""
    id: str
    label: str
    x: int = 0
    y: int = 0
    width: int = 200
    height: int = 80
    color: str = "#4A90D9"
    text_color: str = "#FFFFFF"
    shape: str = "rectangle"  # rectangle, oval, diamond, hexagon


@dataclass
class DiagramEdge:
    """An edge connecting two nodes"""
    from_node: str
    to_node: str
    label: Optional[str] = None
    style: str = "solid"  # solid, dashed, dotted
    color: str = "#666666"
    arrow: bool = True


class DiagramsRenderer:
    """
    Renders professional diagrams using the Python Diagrams library.
    PRIMARY rendering method for architecture diagrams.
    """

    def __init__(self, output_dir: str = "/tmp/presentations/diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = "gpt-4o"

    async def generate_and_render(
        self,
        description: str,
        diagram_type: DiagramType,
        title: str,
        style: DiagramStyle = DiagramStyle.DARK,
        provider: Optional[DiagramProvider] = None
    ) -> Optional[str]:
        """
        Generate Python Diagrams code and render to image.

        Returns:
            Path to generated PNG or None if failed
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Generate Python code
        code = await self._generate_diagrams_code(client, description, diagram_type, title, style, provider)
        if not code:
            return None

        # Render the code
        output_path = await self._execute_diagrams_code(code, title)
        return output_path

    async def _generate_diagrams_code(
        self,
        client,
        description: str,
        diagram_type: DiagramType,
        title: str,
        style: DiagramStyle,
        provider: Optional[DiagramProvider]
    ) -> Optional[str]:
        """Generate Python Diagrams code using GPT-4o."""

        # Style configurations
        style_configs = {
            DiagramStyle.DARK: 'graph_attr={"bgcolor": "#1e1e1e", "fontcolor": "white"}',
            DiagramStyle.LIGHT: 'graph_attr={"bgcolor": "#ffffff", "fontcolor": "black"}',
            DiagramStyle.NEUTRAL: 'graph_attr={"bgcolor": "#f5f5f5", "fontcolor": "#333333"}',
            DiagramStyle.COLORFUL: 'graph_attr={"bgcolor": "#1a1a2e", "fontcolor": "white"}',
        }

        style_config = style_configs.get(style, style_configs[DiagramStyle.DARK])

        system_prompt = f"""You are an expert at creating professional architecture diagrams using the Python 'diagrams' library.

Generate ONLY valid Python code that uses the 'diagrams' library. The code must:
1. Be syntactically correct Python that can execute without errors
2. Use proper imports from the diagrams library
3. Create clear, professional diagrams with meaningful labels
4. Use appropriate icons from the correct provider modules
5. Include proper clustering/grouping where logical
6. Have clear connection flows with Edge labels where helpful

CRITICAL RULES:
- Output ONLY Python code, no explanations or markdown
- Maximum 12-15 nodes for readability
- Use subgraphs/Clusters to group related components
- Always set show=False and filename parameter
- Use descriptive variable names

AVAILABLE PROVIDERS AND IMPORTS:
# AWS
from diagrams.aws.compute import EC2, Lambda, ECS, EKS, Fargate
from diagrams.aws.database import RDS, Aurora, DynamoDB, ElastiCache, Redshift
from diagrams.aws.network import APIGateway, CloudFront, ELB, ALB, NLB, Route53, VPC
from diagrams.aws.storage import S3, EBS, EFS
from diagrams.aws.integration import SQS, SNS, EventBridge, StepFunctions
from diagrams.aws.analytics import Kinesis, Glue, Athena, EMR
from diagrams.aws.ml import Sagemaker
from diagrams.aws.security import IAM, Cognito, WAF, KMS

# Azure
from diagrams.azure.compute import VM, FunctionApps, ContainerInstances, AKS
from diagrams.azure.database import SQLDatabases, CosmosDB, BlobStorage
from diagrams.azure.network import LoadBalancers, ApplicationGateway, VirtualNetworks

# GCP
from diagrams.gcp.compute import ComputeEngine, Functions, Run, GKE
from diagrams.gcp.database import SQL, Spanner, Bigtable, Firestore
from diagrams.gcp.network import LoadBalancing, CDN, DNS
from diagrams.gcp.storage import GCS
from diagrams.gcp.analytics import BigQuery, Dataflow, PubSub

# Kubernetes
from diagrams.k8s.compute import Pod, Deployment, ReplicaSet, StatefulSet, DaemonSet
from diagrams.k8s.network import Service, Ingress, NetworkPolicy
from diagrams.k8s.storage import PV, PVC, StorageClass

# On-Premise
from diagrams.onprem.compute import Server, Nomad
from diagrams.onprem.database import PostgreSQL, MySQL, MongoDB, Redis, Cassandra, Elasticsearch
from diagrams.onprem.network import Nginx, HAProxy, Traefik, Kong
from diagrams.onprem.queue import Kafka, RabbitMQ, Celery
from diagrams.onprem.container import Docker
from diagrams.onprem.ci import Jenkins, GitlabCI, GithubActions
from diagrams.onprem.monitoring import Prometheus, Grafana, Datadog
from diagrams.onprem.logging import Fluentd, Logstash

# Generic
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL as GenericSQL
from diagrams.generic.network import Firewall, Router, Switch
from diagrams.generic.device import Mobile, Tablet

# Programming Languages
from diagrams.programming.language import Python, Java, Go, Rust, JavaScript, TypeScript

# Always use
from diagrams import Diagram, Cluster, Edge

EXAMPLE - Microservices Architecture:
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import ECS
from diagrams.aws.database import RDS, ElastiCache
from diagrams.aws.network import ALB, Route53
from diagrams.aws.integration import SQS

with Diagram("Microservices", show=False, filename="diagram", direction="TB", {style_config}):
    dns = Route53("DNS")
    lb = ALB("Load Balancer")

    with Cluster("Application Layer"):
        svc_api = ECS("API Gateway")
        svc_user = ECS("User Service")
        svc_order = ECS("Order Service")

    with Cluster("Data Layer"):
        db_main = RDS("Main DB")
        cache = ElastiCache("Redis Cache")

    queue = SQS("Event Queue")

    dns >> lb >> svc_api
    svc_api >> [svc_user, svc_order]
    svc_user >> db_main
    svc_order >> [db_main, cache]
    svc_order >> queue

Now generate diagram code based on the user's description. Output ONLY the Python code."""

        user_content = f"""Create a professional diagram showing: {description}

Diagram type: {diagram_type.value}
Title: {title}
Style: {style.value}
Primary provider: {provider.value if provider else 'auto-detect from description'}"""

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            code = response.choices[0].message.content.strip()

            # Clean up code blocks if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()

            return code

        except Exception as e:
            print(f"[DIAGRAMS] Failed to generate code: {e}", flush=True)
            return None

    async def _execute_diagrams_code(self, code: str, title: str) -> Optional[str]:
        """Execute the generated Python code to render the diagram."""
        file_id = f"diagram_{uuid.uuid4().hex[:8]}"

        try:
            # Inject correct filename
            code = self._inject_filename(code, file_id)

            # Create temporary Python file
            temp_py_path = self.output_dir / f"{file_id}.py"
            with open(temp_py_path, 'w') as f:
                f.write(code)

            # Execute the Python script
            result = await asyncio.create_subprocess_exec(
                'python', str(temp_py_path),
                cwd=str(self.output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            # Clean up temp file
            if temp_py_path.exists():
                os.unlink(temp_py_path)

            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                print(f"[DIAGRAMS] Execution failed: {error_msg[:500]}", flush=True)
                return None

            # Find the generated PNG file
            output_path = self.output_dir / f"{file_id}.png"

            if not output_path.exists():
                # Diagrams library might add .png automatically
                for candidate in self.output_dir.glob(f"{file_id}*"):
                    if candidate.suffix in ['.png', '.svg']:
                        output_path = candidate
                        break

            if output_path.exists():
                print(f"[DIAGRAMS] Generated: {output_path}", flush=True)
                return str(output_path)

            print(f"[DIAGRAMS] Output file not found", flush=True)
            return None

        except Exception as e:
            print(f"[DIAGRAMS] Error: {e}", flush=True)
            return None

    def _inject_filename(self, code: str, filename: str) -> str:
        """Ensure the filename parameter is set correctly in the diagram code."""
        # Pattern to match Diagram(...) constructor
        pattern = r'Diagram\s*\([^)]*\)'

        def replace_filename(match):
            diagram_call = match.group(0)

            # Check if filename parameter exists
            if 'filename=' in diagram_call:
                # Replace existing filename
                diagram_call = re.sub(
                    r'filename\s*=\s*["\'][^"\']*["\']',
                    f'filename="{filename}"',
                    diagram_call
                )
            else:
                # Add filename parameter before the closing parenthesis
                diagram_call = diagram_call.rstrip(')')
                diagram_call += f', filename="{filename}")'

            # Ensure show=False
            if 'show=' not in diagram_call:
                diagram_call = diagram_call.rstrip(')')
                diagram_call += ', show=False)'
            elif 'show=True' in diagram_call:
                diagram_call = diagram_call.replace('show=True', 'show=False')

            return diagram_call

        return re.sub(pattern, replace_filename, code)

    def _detect_provider(self, description: str) -> Optional[DiagramProvider]:
        """Auto-detect cloud provider from description."""
        desc_lower = description.lower()

        if any(kw in desc_lower for kw in ['aws', 'amazon', 'ec2', 's3', 'lambda', 'dynamodb', 'sqs', 'sns']):
            return DiagramProvider.AWS
        if any(kw in desc_lower for kw in ['azure', 'microsoft', 'cosmos', 'blob']):
            return DiagramProvider.AZURE
        if any(kw in desc_lower for kw in ['gcp', 'google cloud', 'bigquery', 'gcs', 'cloud run']):
            return DiagramProvider.GCP
        if any(kw in desc_lower for kw in ['kubernetes', 'k8s', 'pod', 'deployment', 'ingress']):
            return DiagramProvider.KUBERNETES
        if any(kw in desc_lower for kw in ['docker', 'nginx', 'kafka', 'redis', 'postgres', 'mysql', 'mongo']):
            return DiagramProvider.ON_PREMISE

        return None


class DiagramGeneratorService:
    """
    Service for generating visual diagrams.
    Uses Python Diagrams as PRIMARY, Mermaid as SECONDARY, PIL as TERTIARY.
    """

    def __init__(self, output_dir: str = "/tmp/presentations/diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Diagrams renderer
        self.diagrams_renderer = DiagramsRenderer(output_dir)

        # Colors for different themes
        self.themes = {
            "tech": {
                "background": "#1E1E1E",
                "node_colors": ["#4A90D9", "#50C878", "#FF6B6B", "#FFA500", "#9B59B6"],
                "text_color": "#FFFFFF",
                "edge_color": "#888888",
            },
            "light": {
                "background": "#FFFFFF",
                "node_colors": ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6"],
                "text_color": "#333333",
                "edge_color": "#666666",
            },
            "gradient": {
                "background": "#0F0F23",
                "node_colors": ["#667EEA", "#764BA2", "#F093FB", "#F5576C", "#4FACFE"],
                "text_color": "#FFFFFF",
                "edge_color": "#AAAAAA",
            }
        }

        # Try to load a good font
        self.fonts = self._load_fonts()

    def _load_fonts(self) -> Dict[str, ImageFont.FreeTypeFont]:
        """Load fonts for diagram text"""
        fonts = {}
        font_paths = [
            "/app/fonts/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]

        for size_name, size in [("title", 32), ("label", 20), ("small", 14)]:
            fonts[size_name] = ImageFont.load_default()
            for path in font_paths:
                if os.path.exists(path):
                    try:
                        fonts[size_name] = ImageFont.truetype(path, size)
                        break
                    except Exception:
                        continue

        return fonts

    async def generate_diagram(
        self,
        diagram_type: DiagramType,
        description: str,
        title: str,
        job_id: str,
        slide_index: int,
        theme: str = "tech",
        width: int = 1920,
        height: int = 1080
    ) -> Optional[str]:
        """
        Generate a diagram image.

        PRIMARY: Uses Python Diagrams library for architecture diagrams (AWS/Azure/GCP/K8s icons).
        SECONDARY: Uses Mermaid.js via Kroki API for flowcharts and sequences.
        TERTIARY: Uses Pillow for basic rendering if all else fails.

        Args:
            diagram_type: Type of diagram to generate
            description: Text description of what the diagram should show
            title: Title for the diagram
            job_id: Job ID for file naming
            slide_index: Slide index for file naming
            theme: Color theme
            width: Image width
            height: Image height

        Returns:
            Path to generated diagram image
        """
        print(f"[DIAGRAM] Generating {diagram_type.value} diagram: {title}", flush=True)
        output_path = self.output_dir / f"{job_id}_diagram_{slide_index}.png"

        # Map theme to DiagramStyle
        style_map = {
            "tech": DiagramStyle.DARK,
            "light": DiagramStyle.LIGHT,
            "gradient": DiagramStyle.COLORFUL,
        }
        style = style_map.get(theme, DiagramStyle.DARK)

        # PRIMARY: Try Python Diagrams library (best for architecture diagrams)
        # Good for: architecture, hierarchy, process with infrastructure
        if diagram_type in [DiagramType.ARCHITECTURE, DiagramType.HIERARCHY, DiagramType.PROCESS]:
            try:
                print(f"[DIAGRAM] Trying Diagrams library (PRIMARY)...", flush=True)
                diagrams_path = await self.diagrams_renderer.generate_and_render(
                    description=description,
                    diagram_type=diagram_type,
                    title=title,
                    style=style,
                    provider=self.diagrams_renderer._detect_provider(description)
                )
                if diagrams_path:
                    # Add title overlay and resize to target dimensions
                    final_image = await self._post_process_diagram(
                        diagrams_path, title, theme, width, height
                    )
                    if final_image:
                        final_image.save(str(output_path), "PNG")
                        print(f"[DIAGRAM] Generated via Diagrams library: {output_path}", flush=True)
                        return str(output_path)
            except Exception as e:
                print(f"[DIAGRAM] Diagrams library failed: {e}, trying Mermaid...", flush=True)

        # SECONDARY: Try Mermaid rendering via Kroki
        # Good for: flowcharts, sequences, mindmaps, timelines
        try:
            print(f"[DIAGRAM] Trying Mermaid (SECONDARY)...", flush=True)
            mermaid_image = await self._generate_mermaid_diagram(
                diagram_type=diagram_type,
                description=description,
                title=title,
                theme=theme,
                width=width,
                height=height
            )
            if mermaid_image:
                mermaid_image.save(str(output_path), "PNG")
                print(f"[DIAGRAM] Generated via Mermaid: {output_path}", flush=True)
                return str(output_path)
        except Exception as e:
            print(f"[DIAGRAM] Mermaid rendering failed: {e}, falling back to PIL", flush=True)

        # TERTIARY: Use PIL-based rendering
        print(f"[DIAGRAM] Using PIL fallback (TERTIARY) for {diagram_type.value}", flush=True)

        # Parse the description to extract structure
        structure = await self._parse_diagram_description(description, diagram_type)

        if diagram_type == DiagramType.FLOWCHART:
            image = await self._generate_flowchart(structure, title, theme, width, height)
        elif diagram_type == DiagramType.ARCHITECTURE:
            image = await self._generate_architecture(structure, title, theme, width, height)
        elif diagram_type == DiagramType.COMPARISON:
            image = await self._generate_comparison(structure, title, theme, width, height)
        elif diagram_type == DiagramType.PROCESS:
            image = await self._generate_process(structure, title, theme, width, height)
        elif diagram_type == DiagramType.HIERARCHY:
            image = await self._generate_hierarchy(structure, title, theme, width, height)
        else:
            # Default to a labeled diagram
            image = await self._generate_labeled_diagram(structure, title, theme, width, height)

        # Save the image
        image.save(str(output_path), "PNG")

        print(f"[DIAGRAM] Generated via PIL fallback: {output_path}", flush=True)
        return str(output_path)

    async def _post_process_diagram(
        self,
        diagram_path: str,
        title: str,
        theme: str,
        width: int,
        height: int
    ) -> Optional[Image.Image]:
        """
        Post-process a diagram generated by Diagrams library.
        Adds title and resizes to target dimensions.
        """
        try:
            # Load the generated diagram
            diagram_image = Image.open(diagram_path)

            # Convert to RGB if needed
            if diagram_image.mode == 'RGBA':
                # Create background
                colors = self.themes.get(theme, self.themes["tech"])
                background = Image.new('RGB', diagram_image.size, colors["background"])
                background.paste(diagram_image, mask=diagram_image.split()[3])
                diagram_image = background
            elif diagram_image.mode != 'RGB':
                diagram_image = diagram_image.convert('RGB')

            # Create final canvas
            colors = self.themes.get(theme, self.themes["tech"])
            final_image = Image.new("RGB", (width, height), colors["background"])
            draw = ImageDraw.Draw(final_image)

            # Add title at top
            self._draw_centered_text(draw, title, width // 2, 40, self.fonts["title"], colors["text_color"])

            # Calculate diagram area
            diagram_area_top = 100
            diagram_area_height = height - 120
            diagram_area_width = width - 100

            # Scale diagram to fit
            scale = min(
                diagram_area_width / diagram_image.width,
                diagram_area_height / diagram_image.height
            )
            scale = min(scale, 2.0)  # Limit scale

            new_width = int(diagram_image.width * scale)
            new_height = int(diagram_image.height * scale)
            diagram_image = diagram_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Center the diagram
            x_offset = (width - new_width) // 2
            y_offset = diagram_area_top + (diagram_area_height - new_height) // 2

            final_image.paste(diagram_image, (x_offset, y_offset))

            return final_image

        except Exception as e:
            print(f"[DIAGRAM] Post-processing failed: {e}", flush=True)
            return None

    async def _generate_mermaid_diagram(
        self,
        diagram_type: DiagramType,
        description: str,
        title: str,
        theme: str,
        width: int,
        height: int
    ) -> Optional[Image.Image]:
        """
        Generate a diagram using Mermaid.js via Kroki API.

        Steps:
        1. Generate Mermaid code using GPT-4
        2. Render Mermaid to PNG using Kroki API
        3. Add title and styling
        """
        # Step 1: Generate Mermaid code
        mermaid_code = await self._generate_mermaid_code(diagram_type, description, title, theme)
        if not mermaid_code:
            return None

        print(f"[DIAGRAM] Generated Mermaid code:\n{mermaid_code[:200]}...", flush=True)

        # Step 2: Render via Kroki
        diagram_image = await self._render_mermaid_via_kroki(mermaid_code, theme)
        if not diagram_image:
            return None

        # Step 3: Create final image with title and proper sizing
        colors = self.themes.get(theme, self.themes["tech"])
        final_image = Image.new("RGB", (width, height), colors["background"])

        # Add title at top
        draw = ImageDraw.Draw(final_image)
        self._draw_centered_text(draw, title, width // 2, 40, self.fonts["title"], colors["text_color"])

        # Scale and center the diagram
        diagram_area_top = 100
        diagram_area_height = height - 120
        diagram_area_width = width - 100

        # Scale diagram to fit
        scale = min(
            diagram_area_width / diagram_image.width,
            diagram_area_height / diagram_image.height
        )
        # Limit scale to avoid over-enlargement
        scale = min(scale, 2.0)

        new_width = int(diagram_image.width * scale)
        new_height = int(diagram_image.height * scale)
        diagram_image = diagram_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center the diagram
        x_offset = (width - new_width) // 2
        y_offset = diagram_area_top + (diagram_area_height - new_height) // 2

        # Paste diagram (handle transparency if present)
        if diagram_image.mode == 'RGBA':
            final_image.paste(diagram_image, (x_offset, y_offset), diagram_image)
        else:
            final_image.paste(diagram_image, (x_offset, y_offset))

        return final_image

    def _sanitize_mermaid_label(self, text: str) -> str:
        """Sanitize text for use in Mermaid labels"""
        # Remove or escape problematic characters
        sanitized = text.replace('"', "'")
        sanitized = sanitized.replace('[', '(')
        sanitized = sanitized.replace(']', ')')
        sanitized = sanitized.replace('{', '(')
        sanitized = sanitized.replace('}', ')')
        sanitized = sanitized.replace('<', '')
        sanitized = sanitized.replace('>', '')
        sanitized = sanitized.replace('|', '-')
        sanitized = sanitized.replace('#', '')
        sanitized = sanitized.replace('&', 'and')
        sanitized = sanitized.replace('\n', ' ')
        sanitized = sanitized.replace('\r', '')
        # Limit length
        if len(sanitized) > 30:
            sanitized = sanitized[:27] + "..."
        return sanitized.strip()

    def _validate_mermaid_code(self, code: str) -> bool:
        """Basic validation of Mermaid syntax"""
        if not code or len(code) < 10:
            return False

        lines = code.strip().split('\n')
        # Must have at least diagram type declaration
        has_diagram_type = False
        valid_starts = ['flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram',
                       'erDiagram', 'gantt', 'pie', 'mindmap', 'timeline', 'graph', '%%{init']

        for line in lines:
            line = line.strip()
            if any(line.startswith(s) for s in valid_starts):
                has_diagram_type = True
                break

        return has_diagram_type

    def _get_fallback_mermaid_code(self, diagram_type: DiagramType, title: str, theme: str) -> str:
        """Generate a simple fallback Mermaid diagram"""
        safe_title = self._sanitize_mermaid_label(title)

        theme_directive = "%%{init: {'theme': 'dark'}}%%"

        if diagram_type == DiagramType.SEQUENCE:
            return f"""{theme_directive}
sequenceDiagram
    participant A as Component A
    participant B as Component B
    participant C as Component C
    A->>B: Request
    B->>C: Process
    C-->>B: Response
    B-->>A: Result"""

        elif diagram_type == DiagramType.ARCHITECTURE:
            return f"""{theme_directive}
flowchart TB
    subgraph Layer1[Presentation]
        A[Client]
    end
    subgraph Layer2[Application]
        B[Service]
    end
    subgraph Layer3[Data]
        C[(Database)]
    end
    A --> B
    B --> C"""

        else:
            # Default flowchart
            return f"""{theme_directive}
flowchart TD
    A[Start] --> B[Process 1]
    B --> C[Process 2]
    C --> D[Process 3]
    D --> E[End]"""

    async def _generate_mermaid_code(
        self,
        diagram_type: DiagramType,
        description: str,
        title: str,
        theme: str
    ) -> Optional[str]:
        """
        Generate Mermaid diagram code using GPT-4.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Sanitize description to avoid syntax issues
        safe_description = self._sanitize_mermaid_label(description[:200])

        # Map diagram types to Mermaid syntax
        mermaid_type_hints = {
            DiagramType.FLOWCHART: "flowchart TD",
            DiagramType.ARCHITECTURE: "flowchart TB with subgraphs",
            DiagramType.SEQUENCE: "sequenceDiagram",
            DiagramType.MINDMAP: "flowchart TD",  # mindmap has issues, use flowchart
            DiagramType.COMPARISON: "flowchart LR",
            DiagramType.HIERARCHY: "flowchart TD",
            DiagramType.PROCESS: "flowchart LR",
            DiagramType.TIMELINE: "flowchart LR",  # timeline has issues, use flowchart
        }

        hint = mermaid_type_hints.get(diagram_type, "flowchart TD")

        # Simplified theme directive (avoid complex escaping issues)
        theme_directive = "%%{init: {'theme': 'dark'}}%%"

        prompt = f"""Generate VALID Mermaid diagram code.

TYPE: {hint}
TOPIC: {safe_description}

STRICT RULES:
1. Start with: {theme_directive}
2. Use ONLY simple ASCII characters in labels
3. Use short node IDs: A, B, C, D, E, F
4. Keep labels under 20 characters
5. NO special characters in labels: no quotes, brackets, pipes
6. Use --> for arrows
7. For subgraphs use: subgraph Name

SIMPLE EXAMPLE:
{theme_directive}
flowchart TD
    A[Start] --> B[Step 1]
    B --> C[Step 2]
    C --> D[End]

Generate diagram now (ONLY the code, nothing else):"""

        try:
            response = await client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "Output ONLY valid Mermaid code. No explanations. No markdown blocks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )

            mermaid_code = response.choices[0].message.content.strip()

            # Clean up the response - remove any markdown code blocks
            if "```" in mermaid_code:
                lines = mermaid_code.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                mermaid_code = "\n".join(lines).strip()

            # Remove any leading/trailing whitespace from lines
            lines = [l.rstrip() for l in mermaid_code.split('\n')]
            mermaid_code = '\n'.join(lines)

            # Ensure theme directive is at the start
            if not mermaid_code.startswith("%%{init"):
                mermaid_code = theme_directive + "\n" + mermaid_code

            # Validate the code
            if not self._validate_mermaid_code(mermaid_code):
                print(f"[DIAGRAM] Generated code failed validation, using fallback", flush=True)
                return self._get_fallback_mermaid_code(diagram_type, title, theme)

            return mermaid_code

        except Exception as e:
            print(f"[DIAGRAM] Failed to generate Mermaid code: {e}", flush=True)
            return self._get_fallback_mermaid_code(diagram_type, title, theme)

    async def _render_mermaid_via_kroki(
        self,
        mermaid_code: str,
        theme: str = "tech"
    ) -> Optional[Image.Image]:
        """
        Render Mermaid code to PNG using Kroki API.

        Kroki is a free, open-source diagram rendering service.
        API: POST https://kroki.io/mermaid/png with the diagram in the body
        """
        # Kroki API endpoint
        kroki_url = os.getenv("KROKI_URL", "https://kroki.io")

        # Clean the code before sending
        clean_code = mermaid_code.strip()

        # Encode diagram for Kroki
        # Kroki accepts URL-safe base64 encoded, zlib compressed diagrams
        compressed = zlib.compress(clean_code.encode('utf-8'), 9)
        encoded = base64.urlsafe_b64encode(compressed).decode('ascii')

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Method 1: GET with encoded diagram
                url = f"{kroki_url}/mermaid/png/{encoded}"
                response = await client.get(url)

                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    # Convert to RGBA for transparency support
                    if image.mode != 'RGBA':
                        image = image.convert('RGBA')
                    print(f"[DIAGRAM] Kroki render successful", flush=True)
                    return image
                else:
                    error_text = response.text[:200] if response.text else "No error message"
                    print(f"[DIAGRAM] Kroki GET failed: {response.status_code} - {error_text}", flush=True)

                    # Method 2: POST with raw diagram
                    response = await client.post(
                        f"{kroki_url}/mermaid/png",
                        content=clean_code.encode('utf-8'),
                        headers={"Content-Type": "text/plain"}
                    )

                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        if image.mode != 'RGBA':
                            image = image.convert('RGBA')
                        print(f"[DIAGRAM] Kroki POST render successful", flush=True)
                        return image
                    else:
                        error_text = response.text[:200] if response.text else "No error message"
                        print(f"[DIAGRAM] Kroki POST failed: {response.status_code} - {error_text}", flush=True)

                        # Try with a minimal fallback diagram
                        fallback_code = """%%{init: {'theme': 'dark'}}%%
flowchart TD
    A[Start] --> B[Process]
    B --> C[End]"""
                        fallback_compressed = zlib.compress(fallback_code.encode('utf-8'), 9)
                        fallback_encoded = base64.urlsafe_b64encode(fallback_compressed).decode('ascii')

                        fallback_response = await client.get(f"{kroki_url}/mermaid/png/{fallback_encoded}")
                        if fallback_response.status_code == 200:
                            print(f"[DIAGRAM] Using minimal fallback diagram", flush=True)
                            image = Image.open(BytesIO(fallback_response.content))
                            if image.mode != 'RGBA':
                                image = image.convert('RGBA')
                            return image

                        return None

        except Exception as e:
            print(f"[DIAGRAM] Kroki rendering error: {e}", flush=True)
            return None

    async def _parse_diagram_description(
        self,
        description: str,
        diagram_type: DiagramType
    ) -> Dict:
        """
        Parse a text description into diagram structure.
        Uses GPT-4 to extract nodes and edges.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = f"""Parse this diagram description into structured data.

DIAGRAM TYPE: {diagram_type.value}

DESCRIPTION:
{description}

Return a JSON object with:
- nodes: array of {{"id": "n1", "label": "Node Label"}}
- edges: array of {{"from": "n1", "to": "n2", "label": "optional edge label"}}
- groups: optional array of {{"name": "Group Name", "nodes": ["n1", "n2"]}}

For comparison diagrams, use:
- items: array of {{"name": "Item", "left": "value1", "right": "value2"}}

For process diagrams, use:
- steps: array of {{"number": 1, "title": "Step", "description": "Details"}}

Extract the key concepts and relationships from the description.
Output ONLY valid JSON."""

        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You extract diagram structures from descriptions. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Return a simple structure if parsing fails
            return {
                "nodes": [{"id": "n1", "label": description[:50]}],
                "edges": []
            }

    async def _generate_flowchart(
        self,
        structure: Dict,
        title: str,
        theme: str,
        width: int,
        height: int
    ) -> Image.Image:
        """Generate a flowchart diagram"""
        colors = self.themes.get(theme, self.themes["tech"])
        image = Image.new("RGB", (width, height), colors["background"])
        draw = ImageDraw.Draw(image)

        nodes = structure.get("nodes", [])
        edges = structure.get("edges", [])

        if not nodes:
            nodes = [{"id": "n1", "label": "Start"}, {"id": "n2", "label": "End"}]
            edges = [{"from": "n1", "to": "n2"}]

        # Calculate node positions in a top-down flow
        margin = 100
        node_width = 250
        node_height = 80
        v_spacing = 120

        # Position nodes
        node_positions = {}
        rows = self._arrange_nodes_in_rows(nodes, edges)

        y_offset = margin + 80  # Leave room for title
        for row_idx, row in enumerate(rows):
            row_width = len(row) * (node_width + 50) - 50
            x_start = (width - row_width) // 2

            for col_idx, node in enumerate(row):
                x = x_start + col_idx * (node_width + 50)
                y = y_offset + row_idx * v_spacing
                node_positions[node["id"]] = (x, y)

        # Draw title
        self._draw_centered_text(draw, title, width // 2, 50, self.fonts["title"], colors["text_color"])

        # Draw edges first (so they appear behind nodes)
        for edge in edges:
            from_pos = node_positions.get(edge.get("from"))
            to_pos = node_positions.get(edge.get("to"))
            if from_pos and to_pos:
                self._draw_arrow(
                    draw,
                    (from_pos[0] + node_width // 2, from_pos[1] + node_height),
                    (to_pos[0] + node_width // 2, to_pos[1]),
                    colors["edge_color"],
                    edge.get("label")
                )

        # Draw nodes
        for i, node in enumerate(nodes):
            pos = node_positions.get(node["id"])
            if pos:
                color = colors["node_colors"][i % len(colors["node_colors"])]
                self._draw_rounded_rect(
                    draw,
                    pos[0], pos[1],
                    pos[0] + node_width, pos[1] + node_height,
                    radius=15,
                    fill=color
                )
                self._draw_centered_text(
                    draw,
                    node["label"],
                    pos[0] + node_width // 2,
                    pos[1] + node_height // 2,
                    self.fonts["label"],
                    colors["text_color"]
                )

        return image

    async def _generate_architecture(
        self,
        structure: Dict,
        title: str,
        theme: str,
        width: int,
        height: int
    ) -> Image.Image:
        """Generate an architecture diagram with layers"""
        colors = self.themes.get(theme, self.themes["tech"])
        image = Image.new("RGB", (width, height), colors["background"])
        draw = ImageDraw.Draw(image)

        nodes = structure.get("nodes", [])
        groups = structure.get("groups", [])

        # Draw title
        self._draw_centered_text(draw, title, width // 2, 40, self.fonts["title"], colors["text_color"])

        # If we have groups, draw them as layers
        if groups:
            layer_height = (height - 150) // len(groups)
            y_offset = 100

            for i, group in enumerate(groups):
                # Draw layer background
                layer_color = colors["node_colors"][i % len(colors["node_colors"])]
                self._draw_rounded_rect(
                    draw,
                    50, y_offset,
                    width - 50, y_offset + layer_height - 20,
                    radius=10,
                    fill=self._adjust_alpha(layer_color, 0.3),
                    outline=layer_color
                )

                # Draw layer label
                self._draw_text(draw, group["name"], 70, y_offset + 10, self.fonts["label"], layer_color)

                # Draw nodes in this layer
                group_nodes = [n for n in nodes if n["id"] in group.get("nodes", [])]
                if group_nodes:
                    node_width = min(200, (width - 200) // len(group_nodes))
                    x_start = 100
                    for j, node in enumerate(group_nodes):
                        x = x_start + j * (node_width + 30)
                        y = y_offset + 50
                        self._draw_rounded_rect(
                            draw,
                            x, y,
                            x + node_width - 20, y + 60,
                            radius=8,
                            fill=layer_color
                        )
                        self._draw_centered_text(
                            draw,
                            node["label"][:20],
                            x + (node_width - 20) // 2,
                            y + 30,
                            self.fonts["small"],
                            colors["text_color"]
                        )

                y_offset += layer_height
        else:
            # No groups, just draw nodes in a grid
            await self._draw_nodes_grid(draw, nodes, colors, width, height, 100)

        return image

    async def _generate_comparison(
        self,
        structure: Dict,
        title: str,
        theme: str,
        width: int,
        height: int
    ) -> Image.Image:
        """Generate a comparison table/diagram"""
        colors = self.themes.get(theme, self.themes["tech"])
        image = Image.new("RGB", (width, height), colors["background"])
        draw = ImageDraw.Draw(image)

        items = structure.get("items", [])

        # Draw title
        self._draw_centered_text(draw, title, width // 2, 40, self.fonts["title"], colors["text_color"])

        if not items:
            # Fallback if no items
            items = [
                {"name": "Feature", "left": "Option A", "right": "Option B"},
            ]

        # Draw comparison table
        table_width = width - 200
        table_x = 100
        table_y = 120
        row_height = min(80, (height - 200) // (len(items) + 1))
        col_width = table_width // 3

        # Header
        headers = ["Feature", "Left", "Right"]
        header_colors = [colors["node_colors"][0], colors["node_colors"][1], colors["node_colors"][2]]
        for i, header in enumerate(headers):
            x = table_x + i * col_width
            self._draw_rounded_rect(
                draw,
                x, table_y,
                x + col_width - 5, table_y + row_height - 5,
                radius=5,
                fill=header_colors[i]
            )
            self._draw_centered_text(
                draw,
                header,
                x + col_width // 2,
                table_y + row_height // 2,
                self.fonts["label"],
                colors["text_color"]
            )

        # Rows
        for i, item in enumerate(items):
            y = table_y + (i + 1) * row_height
            row_color = "#2A2A2A" if i % 2 == 0 else "#333333"

            values = [item.get("name", ""), item.get("left", ""), item.get("right", "")]
            for j, value in enumerate(values):
                x = table_x + j * col_width
                self._draw_rounded_rect(
                    draw,
                    x, y,
                    x + col_width - 5, y + row_height - 5,
                    radius=3,
                    fill=row_color
                )
                self._draw_centered_text(
                    draw,
                    str(value)[:30],
                    x + col_width // 2,
                    y + row_height // 2,
                    self.fonts["small"],
                    colors["text_color"]
                )

        return image

    async def _generate_process(
        self,
        structure: Dict,
        title: str,
        theme: str,
        width: int,
        height: int
    ) -> Image.Image:
        """Generate a process/steps diagram"""
        colors = self.themes.get(theme, self.themes["tech"])
        image = Image.new("RGB", (width, height), colors["background"])
        draw = ImageDraw.Draw(image)

        steps = structure.get("steps", [])

        # Draw title
        self._draw_centered_text(draw, title, width // 2, 40, self.fonts["title"], colors["text_color"])

        if not steps:
            # Extract from nodes if steps not provided
            nodes = structure.get("nodes", [])
            steps = [{"number": i + 1, "title": n["label"], "description": ""} for i, n in enumerate(nodes)]

        if not steps:
            steps = [{"number": 1, "title": "Step 1", "description": "Description"}]

        # Draw horizontal process flow
        margin = 100
        step_width = (width - 2 * margin) // len(steps)
        y_center = height // 2

        for i, step in enumerate(steps):
            x = margin + i * step_width + step_width // 2
            color = colors["node_colors"][i % len(colors["node_colors"])]

            # Draw circle with number
            radius = 40
            draw.ellipse(
                [x - radius, y_center - 100 - radius, x + radius, y_center - 100 + radius],
                fill=color
            )
            self._draw_centered_text(
                draw,
                str(step.get("number", i + 1)),
                x, y_center - 100,
                self.fonts["title"],
                colors["text_color"]
            )

            # Draw connector to next step
            if i < len(steps) - 1:
                next_x = margin + (i + 1) * step_width + step_width // 2
                draw.line(
                    [x + radius + 10, y_center - 100, next_x - radius - 10, y_center - 100],
                    fill=colors["edge_color"],
                    width=3
                )
                # Arrow head
                self._draw_arrow_head(draw, next_x - radius - 10, y_center - 100, "right", colors["edge_color"])

            # Draw title and description
            self._draw_centered_text(
                draw,
                step.get("title", f"Step {i + 1}")[:25],
                x, y_center,
                self.fonts["label"],
                colors["text_color"]
            )

            desc = step.get("description", "")
            if desc:
                self._draw_centered_text(
                    draw,
                    desc[:40],
                    x, y_center + 40,
                    self.fonts["small"],
                    "#888888"
                )

        return image

    async def _generate_hierarchy(
        self,
        structure: Dict,
        title: str,
        theme: str,
        width: int,
        height: int
    ) -> Image.Image:
        """Generate a hierarchy/tree diagram"""
        colors = self.themes.get(theme, self.themes["tech"])
        image = Image.new("RGB", (width, height), colors["background"])
        draw = ImageDraw.Draw(image)

        # Draw title
        self._draw_centered_text(draw, title, width // 2, 40, self.fonts["title"], colors["text_color"])

        nodes = structure.get("nodes", [])
        edges = structure.get("edges", [])

        if not nodes:
            return image

        # Build tree structure
        tree = self._build_tree(nodes, edges)

        # Draw tree
        await self._draw_tree(draw, tree, width, height, colors, 100)

        return image

    async def _generate_labeled_diagram(
        self,
        structure: Dict,
        title: str,
        theme: str,
        width: int,
        height: int
    ) -> Image.Image:
        """Generate a simple labeled diagram as fallback"""
        colors = self.themes.get(theme, self.themes["tech"])
        image = Image.new("RGB", (width, height), colors["background"])
        draw = ImageDraw.Draw(image)

        # Draw title
        self._draw_centered_text(draw, title, width // 2, 40, self.fonts["title"], colors["text_color"])

        nodes = structure.get("nodes", [])
        await self._draw_nodes_grid(draw, nodes, colors, width, height, 100)

        return image

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _arrange_nodes_in_rows(self, nodes: List[Dict], edges: List[Dict]) -> List[List[Dict]]:
        """Arrange nodes into rows based on dependencies"""
        if not edges:
            # No edges, arrange in single row or multiple rows
            max_per_row = 4
            rows = []
            for i in range(0, len(nodes), max_per_row):
                rows.append(nodes[i:i + max_per_row])
            return rows if rows else [[]]

        # Build dependency graph
        children = {n["id"]: [] for n in nodes}
        parents = {n["id"]: [] for n in nodes}

        for edge in edges:
            from_id, to_id = edge.get("from"), edge.get("to")
            if from_id in children:
                children[from_id].append(to_id)
            if to_id in parents:
                parents[to_id].append(from_id)

        # Find roots (nodes with no parents)
        roots = [n for n in nodes if not parents.get(n["id"])]
        if not roots:
            roots = nodes[:1]

        # BFS to arrange in levels
        rows = []
        visited = set()
        current_row = roots

        while current_row:
            rows.append(current_row)
            for node in current_row:
                visited.add(node["id"])

            next_row = []
            for node in current_row:
                for child_id in children.get(node["id"], []):
                    if child_id not in visited:
                        child_node = next((n for n in nodes if n["id"] == child_id), None)
                        if child_node and child_node not in next_row:
                            next_row.append(child_node)

            current_row = next_row

        # Add any unvisited nodes
        remaining = [n for n in nodes if n["id"] not in visited]
        if remaining:
            rows.append(remaining)

        return rows

    def _build_tree(self, nodes: List[Dict], edges: List[Dict]) -> Dict:
        """Build a tree structure from nodes and edges"""
        # Find root
        children_ids = {e.get("to") for e in edges}
        parent_ids = {e.get("from") for e in edges}
        roots = [n for n in nodes if n["id"] not in children_ids]

        if not roots:
            roots = nodes[:1] if nodes else []

        # Build tree recursively
        def build_subtree(node_id: str, depth: int = 0) -> Dict:
            node = next((n for n in nodes if n["id"] == node_id), {"id": node_id, "label": node_id})
            children = [e.get("to") for e in edges if e.get("from") == node_id]
            return {
                "node": node,
                "children": [build_subtree(c, depth + 1) for c in children if depth < 10]
            }

        return build_subtree(roots[0]["id"]) if roots else {"node": {"id": "root", "label": "Root"}, "children": []}

    async def _draw_tree(
        self,
        draw: ImageDraw.ImageDraw,
        tree: Dict,
        width: int,
        height: int,
        colors: Dict,
        y_start: int
    ):
        """Draw a tree structure"""
        def draw_node(node_data: Dict, x: int, y: int, level: int, x_range: Tuple[int, int]):
            node = node_data["node"]
            children = node_data.get("children", [])

            # Draw this node
            color = colors["node_colors"][level % len(colors["node_colors"])]
            node_w, node_h = 160, 50
            self._draw_rounded_rect(
                draw,
                x - node_w // 2, y,
                x + node_w // 2, y + node_h,
                radius=8,
                fill=color
            )
            self._draw_centered_text(
                draw,
                node["label"][:20],
                x, y + node_h // 2,
                self.fonts["small"],
                colors["text_color"]
            )

            # Draw children
            if children:
                child_y = y + 100
                x_start, x_end = x_range
                child_spacing = (x_end - x_start) // (len(children) + 1)

                for i, child in enumerate(children):
                    child_x = x_start + (i + 1) * child_spacing
                    child_range = (
                        x_start + i * child_spacing,
                        x_start + (i + 2) * child_spacing
                    )

                    # Draw edge
                    draw.line(
                        [x, y + node_h, child_x, child_y],
                        fill=colors["edge_color"],
                        width=2
                    )

                    draw_node(child, child_x, child_y, level + 1, child_range)

        draw_node(tree, width // 2, y_start, 0, (100, width - 100))

    async def _draw_nodes_grid(
        self,
        draw: ImageDraw.ImageDraw,
        nodes: List[Dict],
        colors: Dict,
        width: int,
        height: int,
        y_start: int
    ):
        """Draw nodes in a grid layout"""
        if not nodes:
            return

        cols = min(4, len(nodes))
        rows = math.ceil(len(nodes) / cols)

        node_w = (width - 200) // cols
        node_h = min(100, (height - y_start - 100) // rows)

        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols

            x = 100 + col * node_w + 20
            y = y_start + row * (node_h + 30)

            color = colors["node_colors"][i % len(colors["node_colors"])]
            self._draw_rounded_rect(
                draw,
                x, y,
                x + node_w - 40, y + node_h,
                radius=10,
                fill=color
            )
            self._draw_centered_text(
                draw,
                node["label"][:30],
                x + (node_w - 40) // 2,
                y + node_h // 2,
                self.fonts["label"],
                colors["text_color"]
            )

    def _draw_rounded_rect(
        self,
        draw: ImageDraw.ImageDraw,
        x1: int, y1: int, x2: int, y2: int,
        radius: int = 10,
        fill: str = None,
        outline: str = None
    ):
        """Draw a rounded rectangle"""
        # Draw the rounded rectangle using arcs and rectangles
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline)

    def _draw_centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int, y: int,
        font: ImageFont.FreeTypeFont,
        color: str
    ):
        """Draw centered text"""
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y - text_height // 2), text, fill=color, font=font)

    def _draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int, y: int,
        font: ImageFont.FreeTypeFont,
        color: str
    ):
        """Draw text at position"""
        draw.text((x, y), text, fill=color, font=font)

    def _draw_arrow(
        self,
        draw: ImageDraw.ImageDraw,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: str,
        label: Optional[str] = None
    ):
        """Draw an arrow from start to end"""
        draw.line([start, end], fill=color, width=2)

        # Arrow head
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        arrow_len = 15
        arrow_angle = math.pi / 6

        x1 = end[0] - arrow_len * math.cos(angle - arrow_angle)
        y1 = end[1] - arrow_len * math.sin(angle - arrow_angle)
        x2 = end[0] - arrow_len * math.cos(angle + arrow_angle)
        y2 = end[1] - arrow_len * math.sin(angle + arrow_angle)

        draw.polygon([end, (x1, y1), (x2, y2)], fill=color)

        # Label
        if label:
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            draw.text((mid_x, mid_y - 10), label, fill=color, font=self.fonts["small"])

    def _draw_arrow_head(
        self,
        draw: ImageDraw.ImageDraw,
        x: int, y: int,
        direction: str,
        color: str
    ):
        """Draw an arrow head"""
        size = 10
        if direction == "right":
            draw.polygon([(x, y), (x - size, y - size), (x - size, y + size)], fill=color)
        elif direction == "left":
            draw.polygon([(x, y), (x + size, y - size), (x + size, y + size)], fill=color)
        elif direction == "down":
            draw.polygon([(x, y), (x - size, y - size), (x + size, y - size)], fill=color)
        elif direction == "up":
            draw.polygon([(x, y), (x - size, y + size), (x + size, y + size)], fill=color)

    def _adjust_alpha(self, hex_color: str, alpha: float) -> str:
        """Create a darker/lighter version of a color"""
        # Simple approximation - mix with background
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)

        r = int(r * alpha)
        g = int(g * alpha)
        b = int(b * alpha)

        return f"#{r:02x}{g:02x}{b:02x}"
