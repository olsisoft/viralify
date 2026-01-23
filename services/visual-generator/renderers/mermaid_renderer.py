"""
Mermaid Renderer Service

Generates diagrams using Mermaid syntax with multiple rendering backends:
1. PRIMARY: Kroki self-hosted (configurable via KROKI_URL)
2. FALLBACK: mermaid.ink public API

Kroki advantages:
- Self-hosted = no external dependency
- Supports 20+ diagram types (Mermaid, PlantUML, D2, GraphViz, etc.)
- Better privacy (diagrams stay in your infrastructure)

Docker setup for Kroki:
  docker run -d -p 8000:8000 yuzutech/kroki
"""

import os
import json
import base64
import zlib
import httpx
import uuid
import time
from typing import Optional, Dict, Any
from pathlib import Path
from openai import AsyncOpenAI

from models.visual_models import (
    DiagramType,
    DiagramStyle,
    RenderFormat,
    MermaidDiagram,
    DiagramResult,
)


class MermaidRenderer:
    """
    Renders Mermaid diagrams with Kroki (primary) and mermaid.ink (fallback).
    Can also generate Mermaid code from natural language descriptions.
    """

    # Kroki self-hosted (PRIMARY) - configurable via environment
    KROKI_URL = os.getenv("KROKI_URL", "http://kroki:8000")

    # mermaid.ink public API (FALLBACK)
    MERMAID_INK_URL = "https://mermaid.ink"

    # Whether to use Kroki as primary renderer
    USE_KROKI = os.getenv("USE_KROKI", "true").lower() == "true"

    # Theme mappings
    THEME_MAP = {
        DiagramStyle.DARK: "dark",
        DiagramStyle.LIGHT: "default",
        DiagramStyle.NEUTRAL: "neutral",
        DiagramStyle.COLORFUL: "forest",
        DiagramStyle.MINIMAL: "neutral",
        DiagramStyle.CORPORATE: "default",
    }

    # Background colors by style
    BG_COLORS = {
        DiagramStyle.DARK: "#1e1e1e",
        DiagramStyle.LIGHT: "#ffffff",
        DiagramStyle.NEUTRAL: "#f5f5f5",
        DiagramStyle.COLORFUL: "#1a1a2e",
        DiagramStyle.MINIMAL: "#fafafa",
        DiagramStyle.CORPORATE: "#ffffff",
    }

    # Diagram type to Mermaid syntax prefix
    TYPE_PREFIX = {
        DiagramType.FLOWCHART: "graph TD",
        DiagramType.SEQUENCE: "sequenceDiagram",
        DiagramType.CLASS_DIAGRAM: "classDiagram",
        DiagramType.STATE_DIAGRAM: "stateDiagram-v2",
        DiagramType.ER_DIAGRAM: "erDiagram",
        DiagramType.GANTT: "gantt",
        DiagramType.PIE_CHART: "pie",
        DiagramType.MINDMAP: "mindmap",
        DiagramType.TIMELINE: "timeline",
        DiagramType.ARCHITECTURE: "graph TB",
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        output_dir: str = "/tmp/viralify/diagrams"
    ):
        """Initialize the Mermaid renderer."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def generate_from_description(
        self,
        description: str,
        diagram_type: DiagramType,
        style: DiagramStyle = DiagramStyle.DARK,
        context: Optional[str] = None,
        language: str = "en"
    ) -> MermaidDiagram:
        """
        Generate Mermaid code from a natural language description using GPT-4.

        Args:
            description: What the diagram should show
            diagram_type: Type of Mermaid diagram
            style: Visual style
            context: Additional context
            language: Language for labels

        Returns:
            MermaidDiagram with generated code
        """
        if not self.client:
            raise ValueError("OpenAI API key required for diagram generation")

        system_prompt = f"""You are a Mermaid diagram expert. Generate valid Mermaid syntax for diagrams.

Rules:
1. Output ONLY the Mermaid code, no explanations
2. Use the correct diagram type prefix
3. Make diagrams clear and readable
4. Use descriptive but concise labels
5. Language for labels: {language}
6. Keep diagrams focused - max 10-12 nodes for readability

Diagram type: {diagram_type.value}
Prefix to use: {self.TYPE_PREFIX.get(diagram_type, "graph TD")}

Style guide for {style.value}:
- Use clear, professional labels
- Logical flow direction (top-down or left-right)
- Group related items when possible"""

        user_content = f"Create a {diagram_type.value} diagram showing: {description}"
        if context:
            user_content += f"\n\nContext: {context[:500]}"

        response = await self.client.chat.completions.create(
            model="gpt-4o",  # Upgraded from gpt-4o-mini for better diagram quality
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        mermaid_code = response.choices[0].message.content.strip()

        # Clean up code blocks if present
        if mermaid_code.startswith("```"):
            lines = mermaid_code.split("\n")
            mermaid_code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        return MermaidDiagram(
            diagram_type=diagram_type,
            code=mermaid_code,
            title=description[:50],
            theme=self.THEME_MAP.get(style, "dark"),
            background_color=self.BG_COLORS.get(style, "#1e1e1e")
        )

    def _encode_mermaid(self, diagram: MermaidDiagram) -> str:
        """
        Encode Mermaid diagram for mermaid.ink URL.
        Uses pako compression and base64 encoding.
        """
        # Create the configuration object
        config = {
            "code": diagram.code,
            "mermaid": {
                "theme": diagram.theme
            }
        }

        # Encode to JSON and compress
        json_str = json.dumps(config)
        compressed = zlib.compress(json_str.encode("utf-8"), level=9)

        # Base64 encode (URL-safe)
        encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")

        return encoded

    def _encode_for_kroki(self, code: str) -> str:
        """
        Encode diagram code for Kroki API.
        Uses deflate compression and base64 encoding.
        """
        compressed = zlib.compress(code.encode("utf-8"), level=9)
        encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")
        return encoded

    async def _render_with_kroki(
        self,
        diagram: MermaidDiagram,
        format: RenderFormat = RenderFormat.PNG
    ) -> Optional[bytes]:
        """
        Render diagram using Kroki self-hosted instance.

        Args:
            diagram: The Mermaid diagram to render
            format: Output format (png or svg)

        Returns:
            Image bytes if successful, None otherwise
        """
        try:
            # Kroki supports POST with JSON body
            output_format = "svg" if format == RenderFormat.SVG else "png"

            # Add theme directive to mermaid code
            themed_code = diagram.code
            if not themed_code.strip().startswith("%%"):
                themed_code = f"%%{{init: {{'theme': '{diagram.theme}'}}}}%%\n{diagram.code}"

            response = await self.http_client.post(
                f"{self.KROKI_URL}/mermaid/{output_format}",
                content=themed_code,
                headers={"Content-Type": "text/plain"},
                timeout=30.0
            )
            response.raise_for_status()

            print(f"[MERMAID] Rendered via Kroki ({len(response.content)} bytes)", flush=True)
            return response.content

        except httpx.ConnectError as e:
            print(f"[MERMAID] Kroki not available: {e}", flush=True)
            return None
        except httpx.HTTPStatusError as e:
            print(f"[MERMAID] Kroki error {e.response.status_code}: {e.response.text[:200]}", flush=True)
            return None
        except Exception as e:
            print(f"[MERMAID] Kroki failed: {e}", flush=True)
            return None

    async def _render_with_mermaid_ink(
        self,
        diagram: MermaidDiagram,
        format: RenderFormat = RenderFormat.PNG,
        width: int = 1920,
        height: int = 1080
    ) -> Optional[bytes]:
        """
        Render diagram using mermaid.ink public API (fallback).

        Args:
            diagram: The Mermaid diagram to render
            format: Output format
            width: Image width
            height: Image height

        Returns:
            Image bytes if successful, None otherwise
        """
        try:
            image_url = self.get_image_url(diagram, format, width, height)
            response = await self.http_client.get(image_url, timeout=30.0)
            response.raise_for_status()

            print(f"[MERMAID] Rendered via mermaid.ink ({len(response.content)} bytes)", flush=True)
            return response.content

        except Exception as e:
            print(f"[MERMAID] mermaid.ink failed: {e}", flush=True)
            return None

    def get_image_url(
        self,
        diagram: MermaidDiagram,
        format: RenderFormat = RenderFormat.PNG,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> str:
        """
        Get the mermaid.ink URL for rendering the diagram.

        Args:
            diagram: The Mermaid diagram to render
            format: Output format (png or svg)
            width: Optional width constraint
            height: Optional height constraint

        Returns:
            URL to the rendered diagram
        """
        encoded = self._encode_mermaid(diagram)

        # Build URL
        if format == RenderFormat.SVG:
            url = f"{self.MERMAID_INK_URL}/svg/pako:{encoded}"
        else:
            url = f"{self.MERMAID_INK_URL}/img/pako:{encoded}"

        # Add size parameters if specified
        params = []
        if width:
            params.append(f"width={width}")
        if height:
            params.append(f"height={height}")
        if diagram.background_color and diagram.background_color != "transparent":
            bg = diagram.background_color.replace("#", "")
            params.append(f"bgColor={bg}")

        if params:
            url += "?" + "&".join(params)

        return url

    async def render(
        self,
        diagram: MermaidDiagram,
        format: RenderFormat = RenderFormat.PNG,
        width: int = 1920,
        height: int = 1080,
        save_to_file: bool = True
    ) -> DiagramResult:
        """
        Render a Mermaid diagram to an image file.

        Rendering priority:
        1. Kroki self-hosted (if USE_KROKI=true and available)
        2. mermaid.ink public API (fallback)

        Args:
            diagram: The Mermaid diagram to render
            format: Output format
            width: Image width
            height: Image height
            save_to_file: Whether to save locally

        Returns:
            DiagramResult with file path and URL
        """
        start_time = time.time()
        renderer_used = "unknown"
        image_content = None

        try:
            # PRIMARY: Try Kroki self-hosted first
            if self.USE_KROKI:
                image_content = await self._render_with_kroki(diagram, format)
                if image_content:
                    renderer_used = "kroki"

            # FALLBACK: Use mermaid.ink if Kroki failed or disabled
            if image_content is None:
                image_content = await self._render_with_mermaid_ink(diagram, format, width, height)
                if image_content:
                    renderer_used = "mermaid.ink"

            # Both failed
            if image_content is None:
                generation_time = int((time.time() - start_time) * 1000)
                return DiagramResult(
                    success=False,
                    diagram_type=diagram.diagram_type,
                    width=width,
                    height=height,
                    format=format,
                    generation_time_ms=generation_time,
                    error="All rendering backends failed (Kroki + mermaid.ink)"
                )

            # Save to file
            file_path = None
            if save_to_file:
                file_id = str(uuid.uuid4())[:8]
                extension = "svg" if format == RenderFormat.SVG else "png"
                file_path = self.output_dir / f"mermaid_{file_id}.{extension}"

                with open(file_path, "wb") as f:
                    f.write(image_content)

            generation_time = int((time.time() - start_time) * 1000)

            # Build file URL for serving
            file_url = f"/diagrams/{file_path.name}" if file_path else None

            return DiagramResult(
                success=True,
                diagram_type=diagram.diagram_type,
                file_path=str(file_path) if file_path else None,
                file_url=file_url,
                width=width,
                height=height,
                format=format,
                generation_time_ms=generation_time,
                metadata={
                    "mermaid_theme": diagram.theme,
                    "code_length": len(diagram.code),
                    "renderer": renderer_used
                }
            )

        except Exception as e:
            generation_time = int((time.time() - start_time) * 1000)
            return DiagramResult(
                success=False,
                diagram_type=diagram.diagram_type,
                width=width,
                height=height,
                format=format,
                generation_time_ms=generation_time,
                error=str(e)
            )

    async def generate_and_render(
        self,
        description: str,
        diagram_type: DiagramType,
        style: DiagramStyle = DiagramStyle.DARK,
        format: RenderFormat = RenderFormat.PNG,
        width: int = 1920,
        height: int = 1080,
        context: Optional[str] = None,
        language: str = "en"
    ) -> DiagramResult:
        """
        Generate Mermaid code from description and render to image.

        Convenience method that combines generation and rendering.
        """
        # Generate the Mermaid code
        diagram = await self.generate_from_description(
            description=description,
            diagram_type=diagram_type,
            style=style,
            context=context,
            language=language
        )

        # Render to image
        return await self.render(
            diagram=diagram,
            format=format,
            width=width,
            height=height
        )

    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Predefined diagram templates
class MermaidTemplates:
    """Common diagram templates for quick generation."""

    @staticmethod
    def microservices_architecture(services: list[str], gateway: str = "API Gateway") -> str:
        """Generate a microservices architecture diagram."""
        code = "graph TB\n"
        code += f"    Client[Client] --> GW[{gateway}]\n"
        for svc in services:
            code += f"    GW --> {svc.replace(' ', '')}[{svc}]\n"
        code += "    subgraph Services\n"
        for svc in services:
            code += f"        {svc.replace(' ', '')}\n"
        code += "    end"
        return code

    @staticmethod
    def kafka_architecture() -> str:
        """Standard Kafka architecture diagram."""
        return """graph LR
    subgraph Producers
        P1[Producer 1]
        P2[Producer 2]
    end

    subgraph Kafka Cluster
        B1[Broker 1]
        B2[Broker 2]
        B3[Broker 3]
    end

    subgraph Consumers
        C1[Consumer Group 1]
        C2[Consumer Group 2]
    end

    P1 --> B1
    P2 --> B2
    B1 <--> B2
    B2 <--> B3
    B1 --> C1
    B2 --> C1
    B3 --> C2"""

    @staticmethod
    def rest_api_sequence(client: str, server: str, endpoint: str) -> str:
        """REST API request/response sequence."""
        return f"""sequenceDiagram
    participant C as {client}
    participant S as {server}

    C->>S: {endpoint} Request
    activate S
    S-->>S: Process Request
    S-->>C: Response (200 OK)
    deactivate S"""

    @staticmethod
    def state_machine(states: list[tuple[str, str, str]]) -> str:
        """
        Generate a state diagram.
        states: list of (from_state, event, to_state) tuples
        """
        code = "stateDiagram-v2\n"
        for from_state, event, to_state in states:
            code += f"    {from_state} --> {to_state}: {event}\n"
        return code

    @staticmethod
    def class_diagram(classes: list[dict]) -> str:
        """
        Generate a class diagram.
        classes: list of {"name": str, "attributes": list, "methods": list, "extends": str|None}
        """
        code = "classDiagram\n"
        for cls in classes:
            code += f"    class {cls['name']} {{\n"
            for attr in cls.get("attributes", []):
                code += f"        {attr}\n"
            for method in cls.get("methods", []):
                code += f"        {method}()\n"
            code += "    }\n"
            if cls.get("extends"):
                code += f"    {cls['extends']} <|-- {cls['name']}\n"
        return code
