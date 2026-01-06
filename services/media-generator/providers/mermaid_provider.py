"""
Mermaid Provider - Generate technical diagrams using Mermaid.js

Uses GPT-4 to generate Mermaid code from descriptions and
mermaid-cli (mmdc) to render diagrams to images.
"""

import os
import json
import uuid
import asyncio
import logging
import tempfile
import subprocess
from typing import Optional, Tuple
from openai import AsyncOpenAI

from models.visual_types import DiagramType

logger = logging.getLogger(__name__)


class MermaidProvider:
    """Provider for generating Mermaid.js diagrams."""

    # Mermaid diagram syntax templates
    DIAGRAM_PREFIXES = {
        DiagramType.FLOWCHART: "flowchart TD",
        DiagramType.SEQUENCE: "sequenceDiagram",
        DiagramType.ARCHITECTURE: "flowchart TD",  # Architecture uses flowchart with subgraphs
        DiagramType.CLASS: "classDiagram",
        DiagramType.ER: "erDiagram",
        DiagramType.MINDMAP: "mindmap",
        DiagramType.STATE: "stateDiagram-v2",
        DiagramType.GANTT: "gantt",
        DiagramType.PIE: "pie",
        DiagramType.JOURNEY: "journey",
    }

    def __init__(
        self,
        openai_api_key: str,
        output_dir: str = "/tmp/mermaid",
        mmdc_path: str = "mmdc"
    ):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.output_dir = output_dir
        self.mmdc_path = mmdc_path
        os.makedirs(output_dir, exist_ok=True)

    async def generate_mermaid_code(
        self,
        description: str,
        diagram_type: DiagramType,
        elements: Optional[list] = None,
        relationships: Optional[list] = None
    ) -> str:
        """
        Generate Mermaid.js code from description using GPT-4.

        Args:
            description: Scene description
            diagram_type: Type of diagram to generate
            elements: Pre-extracted elements to include
            relationships: Pre-extracted relationships

        Returns:
            Valid Mermaid.js code string
        """
        prefix = self.DIAGRAM_PREFIXES.get(diagram_type, "flowchart TD")

        # Build context from extracted elements
        context_parts = []
        if elements:
            context_parts.append(f"Key elements to include: {', '.join(elements)}")
        if relationships:
            rels = [f"{r.get('from')} -> {r.get('to')}: {r.get('label', '')}" for r in relationships]
            context_parts.append(f"Relationships: {'; '.join(rels)}")

        context = "\n".join(context_parts) if context_parts else ""

        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": self._get_mermaid_system_prompt(diagram_type)
                },
                {
                    "role": "user",
                    "content": self._get_mermaid_user_prompt(description, prefix, context)
                }
            ],
            temperature=0.2,
            max_tokens=2000
        )

        # Extract and clean Mermaid code
        code = response.choices[0].message.content
        code = self._clean_mermaid_code(code)

        # Validate the code
        if not self._validate_mermaid_code(code, diagram_type):
            logger.warning(f"Generated invalid Mermaid code, attempting fix...")
            code = self._fix_mermaid_code(code, diagram_type)

        return code

    def _get_mermaid_system_prompt(self, diagram_type: DiagramType) -> str:
        """System prompt for Mermaid code generation."""
        examples = self._get_diagram_examples(diagram_type)

        return f"""You are an expert at creating Mermaid.js diagrams. Generate clean, valid Mermaid code for technical diagrams.

RULES:
1. Output ONLY the Mermaid code, no explanations or markdown code blocks
2. Use clear, concise node labels (max 30 chars)
3. Use proper Mermaid syntax for the diagram type
4. Ensure the diagram is readable and not too complex
5. Use subgraphs for grouping related components
6. Add styling for better visual appearance

{examples}

IMPORTANT:
- Do NOT include ```mermaid or ``` markers
- Do NOT include any text before or after the diagram code
- Ensure all node IDs are valid (no spaces, use underscores)
- For flowcharts, use meaningful shapes: [] for process, {{}} for decision, () for rounded"""

    def _get_diagram_examples(self, diagram_type: DiagramType) -> str:
        """Get example code for each diagram type."""
        examples = {
            DiagramType.FLOWCHART: """
FLOWCHART EXAMPLE:
flowchart TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Process]
    B -->|No| D[Alternative]
    C --> E[End]
    D --> E
""",
            DiagramType.SEQUENCE: """
SEQUENCE DIAGRAM EXAMPLE:
sequenceDiagram
    participant C as Client
    participant S as Server
    participant DB as Database
    C->>S: Request
    S->>DB: Query
    DB-->>S: Results
    S-->>C: Response
""",
            DiagramType.ARCHITECTURE: """
ARCHITECTURE EXAMPLE:
flowchart TD
    subgraph Frontend
        A[Web App]
        B[Mobile App]
    end
    subgraph Backend
        C[API Gateway]
        D[Auth Service]
        E[Core Service]
    end
    subgraph Data
        F[(Database)]
        G[(Cache)]
    end
    A --> C
    B --> C
    C --> D
    C --> E
    E --> F
    E --> G
""",
            DiagramType.CLASS: """
CLASS DIAGRAM EXAMPLE:
classDiagram
    class User {
        +String name
        +String email
        +login()
        +logout()
    }
    class Order {
        +int id
        +Date date
        +process()
    }
    User "1" --> "*" Order : places
""",
            DiagramType.ER: """
ER DIAGRAM EXAMPLE:
erDiagram
    USER ||--o{ ORDER : places
    ORDER ||--|{ LINE_ITEM : contains
    PRODUCT ||--o{ LINE_ITEM : "ordered in"
    USER {
        int id PK
        string name
        string email
    }
""",
            DiagramType.MINDMAP: """
MINDMAP EXAMPLE:
mindmap
    root((Main Topic))
        Branch1
            Sub1
            Sub2
        Branch2
            Sub3
        Branch3
""",
            DiagramType.STATE: """
STATE DIAGRAM EXAMPLE:
stateDiagram-v2
    [*] --> Draft
    Draft --> Review
    Review --> Approved
    Review --> Rejected
    Approved --> Published
    Rejected --> Draft
    Published --> [*]
"""
        }
        return examples.get(diagram_type, examples[DiagramType.FLOWCHART])

    def _get_mermaid_user_prompt(self, description: str, prefix: str, context: str) -> str:
        """User prompt for Mermaid generation."""
        return f"""Create a {prefix.split()[0]} diagram for this concept:

DESCRIPTION: {description}

{f'CONTEXT: {context}' if context else ''}

Generate clean Mermaid code starting with: {prefix}
Output only the Mermaid code, nothing else."""

    def _clean_mermaid_code(self, code: str) -> str:
        """Clean generated Mermaid code."""
        # Remove markdown code blocks
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first and last lines if they're code block markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        # Remove any leading/trailing whitespace
        code = code.strip()

        return code

    def _validate_mermaid_code(self, code: str, diagram_type: DiagramType) -> bool:
        """Basic validation of Mermaid code."""
        if not code:
            return False

        # Check for diagram type prefix
        expected_prefixes = {
            DiagramType.FLOWCHART: ["flowchart", "graph"],
            DiagramType.SEQUENCE: ["sequenceDiagram"],
            DiagramType.ARCHITECTURE: ["flowchart", "graph"],
            DiagramType.CLASS: ["classDiagram"],
            DiagramType.ER: ["erDiagram"],
            DiagramType.MINDMAP: ["mindmap"],
            DiagramType.STATE: ["stateDiagram"],
            DiagramType.GANTT: ["gantt"],
            DiagramType.PIE: ["pie"],
            DiagramType.JOURNEY: ["journey"],
        }

        prefixes = expected_prefixes.get(diagram_type, ["flowchart"])
        first_line = code.split("\n")[0].strip().lower()

        return any(first_line.startswith(p.lower()) for p in prefixes)

    def _fix_mermaid_code(self, code: str, diagram_type: DiagramType) -> str:
        """Attempt to fix invalid Mermaid code."""
        prefix = self.DIAGRAM_PREFIXES.get(diagram_type, "flowchart TD")

        # If no valid prefix, add it
        if not self._validate_mermaid_code(code, diagram_type):
            code = f"{prefix}\n{code}"

        return code

    async def render_to_image(
        self,
        mermaid_code: str,
        width: int = 1080,
        height: int = 1920,
        background: str = "transparent",
        theme: str = "dark"
    ) -> str:
        """
        Render Mermaid code to an image using mermaid-cli.

        Args:
            mermaid_code: Valid Mermaid.js code
            width: Output image width
            height: Output image height
            background: Background color or 'transparent'
            theme: Mermaid theme (default, dark, forest, neutral)

        Returns:
            Path to the generated image file
        """
        # Create unique filenames
        file_id = str(uuid.uuid4())[:8]
        input_file = os.path.join(self.output_dir, f"diagram_{file_id}.mmd")
        output_file = os.path.join(self.output_dir, f"diagram_{file_id}.png")
        config_file = os.path.join(self.output_dir, f"config_{file_id}.json")

        try:
            # Write Mermaid code to file
            with open(input_file, "w", encoding="utf-8") as f:
                f.write(mermaid_code)

            # Create config file for styling
            config = {
                "theme": theme,
                "themeVariables": {
                    "primaryColor": "#4F46E5",
                    "primaryTextColor": "#FFFFFF",
                    "primaryBorderColor": "#6366F1",
                    "lineColor": "#94A3B8",
                    "secondaryColor": "#1E293B",
                    "tertiaryColor": "#0F172A",
                    "background": background if background != "transparent" else "transparent",
                    "mainBkg": "#1E293B",
                    "nodeBorder": "#4F46E5",
                    "clusterBkg": "#1E293B",
                    "clusterBorder": "#4F46E5",
                    "titleColor": "#FFFFFF",
                    "edgeLabelBackground": "#1E293B"
                }
            }
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f)

            # Build mmdc command
            cmd = [
                self.mmdc_path,
                "-i", input_file,
                "-o", output_file,
                "-w", str(width),
                "-H", str(height),
                "-c", config_file,
                "-b", background,
                "--pdfFit"
            ]

            # Run mermaid-cli
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Mermaid CLI failed: {error_msg}")
                raise RuntimeError(f"Mermaid rendering failed: {error_msg}")

            if not os.path.exists(output_file):
                raise RuntimeError("Mermaid output file not created")

            logger.info(f"Mermaid diagram rendered: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Mermaid render error: {e}")
            raise

        finally:
            # Cleanup temp files (keep output)
            for f in [input_file, config_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass

    async def generate_and_render(
        self,
        description: str,
        diagram_type: DiagramType,
        width: int = 1080,
        height: int = 1920,
        elements: Optional[list] = None,
        relationships: Optional[list] = None
    ) -> Tuple[str, str]:
        """
        Generate Mermaid code and render to image in one step.

        Returns:
            Tuple of (image_path, mermaid_code)
        """
        # Generate code
        code = await self.generate_mermaid_code(
            description=description,
            diagram_type=diagram_type,
            elements=elements,
            relationships=relationships
        )

        # Render to image
        image_path = await self.render_to_image(
            mermaid_code=code,
            width=width,
            height=height
        )

        return image_path, code

    def preview_code(self, code: str) -> str:
        """
        Generate a preview URL for Mermaid code using mermaid.live.

        Args:
            code: Mermaid code to preview

        Returns:
            URL to mermaid.live editor with the code
        """
        import base64
        import zlib

        # Encode for mermaid.live
        json_str = json.dumps({
            "code": code,
            "mermaid": {"theme": "dark"},
            "updateEditor": True
        })
        compressed = zlib.compress(json_str.encode("utf-8"), 9)
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")

        return f"https://mermaid.live/edit#pako:{encoded}"
