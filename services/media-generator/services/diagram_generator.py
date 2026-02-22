"""
Technical Diagram Generator Service
Generates professional technical diagrams from descriptions using:
- Mermaid.js (flowcharts, sequence, class, ER, state diagrams)
- Python Diagrams (cloud architecture with real service icons)
- Graphviz (custom graphs, dependency trees)
"""

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

import httpx

# Use shared LLM provider for model name resolution
try:
    from shared.llm_provider import get_model_name as _get_model_name
    _HAS_SHARED_LLM = True
except ImportError:
    _HAS_SHARED_LLM = False
    _get_model_name = lambda tier: {"fast": "gpt-4o-mini", "quality": "gpt-4o"}.get(tier, "gpt-4o-mini")


class DiagramType(str, Enum):
    # Mermaid types
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    CLASS = "class"
    ER = "er"
    STATE = "state"
    GANTT = "gantt"
    PIE = "pie"
    MINDMAP = "mindmap"

    # Architecture types (Python Diagrams)
    CLOUD_AWS = "cloud_aws"
    CLOUD_GCP = "cloud_gcp"
    CLOUD_AZURE = "cloud_azure"
    CLOUD_K8S = "cloud_k8s"
    CLOUD_GENERIC = "cloud_generic"
    MICROSERVICES = "microservices"

    # Graphviz types
    DEPENDENCY = "dependency"
    NETWORK = "network"
    TREE = "tree"
    GRAPH = "graph"


class DiagramGenerator:
    """Generates technical diagrams from natural language descriptions"""

    def __init__(self, openai_api_key: str):
        self.openai_key = openai_api_key
        self.temp_dir = Path(tempfile.mkdtemp(prefix="diagrams_"))
        self.output_dir = Path("/tmp/diagrams")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mermaid config for better rendering
        self.mermaid_config = {
            "theme": "dark",
            "themeVariables": {
                "primaryColor": "#6366f1",
                "primaryTextColor": "#ffffff",
                "primaryBorderColor": "#818cf8",
                "lineColor": "#a5b4fc",
                "secondaryColor": "#4f46e5",
                "tertiaryColor": "#312e81",
                "background": "#1e1b4b",
                "mainBkg": "#1e1b4b",
                "nodeBorder": "#818cf8",
                "clusterBkg": "#312e81",
                "clusterBorder": "#6366f1",
                "titleColor": "#ffffff",
                "edgeLabelBackground": "#312e81"
            },
            "flowchart": {
                "curve": "basis",
                "padding": 20,
                "nodeSpacing": 50,
                "rankSpacing": 50
            },
            "sequence": {
                "actorMargin": 80,
                "boxMargin": 10,
                "boxTextMargin": 5,
                "noteMargin": 10,
                "messageMargin": 35
            }
        }

    async def generate_diagram(
        self,
        description: str,
        diagram_type: Optional[DiagramType] = None,
        style: str = "professional",
        aspect_ratio: str = "16:9"
    ) -> Optional[str]:
        """
        Generate a technical diagram from a description.
        Returns path to the generated PNG image.
        """
        try:
            # Auto-detect diagram type if not specified
            if diagram_type is None:
                diagram_type = await self._detect_diagram_type(description)

            print(f"Generating {diagram_type} diagram for: {description[:100]}...")

            # Route to appropriate generator
            if diagram_type in [DiagramType.FLOWCHART, DiagramType.SEQUENCE,
                               DiagramType.CLASS, DiagramType.ER, DiagramType.STATE,
                               DiagramType.GANTT, DiagramType.PIE, DiagramType.MINDMAP]:
                return await self._generate_mermaid_diagram(description, diagram_type, style, aspect_ratio)

            elif diagram_type in [DiagramType.CLOUD_AWS, DiagramType.CLOUD_GCP,
                                 DiagramType.CLOUD_AZURE, DiagramType.CLOUD_K8S,
                                 DiagramType.CLOUD_GENERIC, DiagramType.MICROSERVICES]:
                return await self._generate_cloud_diagram(description, diagram_type, style, aspect_ratio)

            else:
                return await self._generate_graphviz_diagram(description, diagram_type, style, aspect_ratio)

        except Exception as e:
            print(f"Diagram generation error: {e}")
            try:
                return await self._generate_mermaid_diagram(description, DiagramType.FLOWCHART, style, aspect_ratio)
            except Exception:
                return None

    async def _detect_diagram_type(self, description: str) -> DiagramType:
        """Use GPT to detect the best diagram type for the description"""

        prompt = f"""Analyze this description and determine the best diagram type to visualize it.

Description: {description}

Choose ONE from these options:
- flowchart: For processes, workflows, decision trees, algorithms
- sequence: For interactions between components/services over time, API calls
- class: For OOP class structures, inheritance, relationships
- er: For database schemas, entity relationships
- state: For state machines, lifecycle diagrams
- cloud_aws: For AWS architecture diagrams
- cloud_gcp: For Google Cloud architecture
- cloud_azure: For Azure architecture
- cloud_k8s: For Kubernetes deployments
- cloud_generic: For generic cloud/infrastructure diagrams
- microservices: For microservices architecture
- dependency: For dependency graphs, module relationships
- network: For network topology diagrams
- tree: For hierarchical structures, org charts

Respond with ONLY the diagram type name, nothing else."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": _get_model_name("fast"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0
                },
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"].strip().lower()

                type_map = {
                    "flowchart": DiagramType.FLOWCHART,
                    "sequence": DiagramType.SEQUENCE,
                    "class": DiagramType.CLASS,
                    "er": DiagramType.ER,
                    "state": DiagramType.STATE,
                    "gantt": DiagramType.GANTT,
                    "pie": DiagramType.PIE,
                    "mindmap": DiagramType.MINDMAP,
                    "cloud_aws": DiagramType.CLOUD_AWS,
                    "cloud_gcp": DiagramType.CLOUD_GCP,
                    "cloud_azure": DiagramType.CLOUD_AZURE,
                    "cloud_k8s": DiagramType.CLOUD_K8S,
                    "cloud_generic": DiagramType.CLOUD_GENERIC,
                    "microservices": DiagramType.MICROSERVICES,
                    "dependency": DiagramType.DEPENDENCY,
                    "network": DiagramType.NETWORK,
                    "tree": DiagramType.TREE,
                    "graph": DiagramType.GRAPH
                }

                return type_map.get(result, DiagramType.FLOWCHART)

        return DiagramType.FLOWCHART

    async def _generate_mermaid_diagram(
        self,
        description: str,
        diagram_type: DiagramType,
        style: str,
        aspect_ratio: str
    ) -> Optional[str]:
        """Generate a diagram using Mermaid.js"""

        mermaid_code = await self._generate_mermaid_code(description, diagram_type)

        if not mermaid_code:
            print("Failed to generate Mermaid code")
            return None

        print(f"Generated Mermaid code:\n{mermaid_code[:500]}...")

        dimensions = self._get_dimensions(aspect_ratio)

        diagram_id = uuid.uuid4().hex[:8]
        mmd_file = self.temp_dir / f"diagram_{diagram_id}.mmd"
        output_file = self.output_dir / f"diagram_{diagram_id}.png"
        config_file = self.temp_dir / f"config_{diagram_id}.json"

        with open(mmd_file, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)

        with open(config_file, 'w') as f:
            json.dump(self.mermaid_config, f)

        cmd = [
            "mmdc",
            "-i", str(mmd_file),
            "-o", str(output_file),
            "-c", str(config_file),
            "-w", str(dimensions[0]),
            "-H", str(dimensions[1]),
            "-b", "transparent",
            "-p", "/usr/bin/chromium"
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"Mermaid rendering error: {stderr.decode()}")
            cmd_simple = [
                "mmdc",
                "-i", str(mmd_file),
                "-o", str(output_file),
                "-w", str(dimensions[0]),
                "-H", str(dimensions[1]),
                "-b", "#1a1a2e",
                "-p", "/usr/bin/chromium"
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd_simple,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

        if output_file.exists():
            print(f"Mermaid diagram generated: {output_file}")
            return str(output_file)

        return None

    async def _generate_mermaid_code(self, description: str, diagram_type: DiagramType) -> Optional[str]:
        """Use GPT to generate Mermaid diagram code"""

        type_examples = {
            DiagramType.FLOWCHART: """flowchart TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Process 1]
    B -->|No| D[Process 2]
    C --> E[End]
    D --> E""",
            DiagramType.SEQUENCE: """sequenceDiagram
    participant Client
    participant API
    participant Database
    Client->>API: Request
    API->>Database: Query
    Database-->>API: Results
    API-->>Client: Response""",
            DiagramType.CLASS: """classDiagram
    class Animal {
        +String name
        +makeSound()
    }
    class Dog {
        +bark()
    }
    Animal <|-- Dog""",
            DiagramType.ER: """erDiagram
    USER ||--o{ ORDER : places
    ORDER ||--|{ ITEM : contains
    USER {
        int id PK
        string name
        string email
    }""",
            DiagramType.STATE: """stateDiagram-v2
    [*] --> Idle
    Idle --> Processing : start
    Processing --> Completed : success
    Processing --> Failed : error
    Completed --> [*]
    Failed --> Idle : retry"""
        }

        example = type_examples.get(diagram_type, type_examples[DiagramType.FLOWCHART])

        prompt = f"""Generate a Mermaid diagram for this description. Use the {diagram_type.value} type.

Description: {description}

Example format:
{example}

Rules:
1. Output ONLY valid Mermaid code, no explanations
2. Use clear, concise labels (max 30 chars per label)
3. Use proper Mermaid syntax for {diagram_type.value}
4. Include 5-15 nodes/elements for good visual balance
5. Use meaningful connections and relationships
6. Do NOT wrap in code blocks or markdown

Generate the Mermaid code:"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": _get_model_name("quality"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.7
                },
                timeout=60.0
            )

            if response.status_code == 200:
                code = response.json()["choices"][0]["message"]["content"].strip()
                code = code.replace("```mermaid", "").replace("```", "").strip()
                return code

        return None

    async def _generate_cloud_diagram(
        self,
        description: str,
        diagram_type: DiagramType,
        style: str,
        aspect_ratio: str
    ) -> Optional[str]:
        """Generate cloud architecture diagram using Python Diagrams library"""

        diagram_code = await self._generate_diagrams_code(description, diagram_type)

        if not diagram_code:
            print("Failed to generate Diagrams code, falling back to Mermaid")
            return await self._generate_mermaid_diagram(description, DiagramType.FLOWCHART, style, aspect_ratio)

        print(f"Generated Diagrams code:\n{diagram_code[:500]}...")

        diagram_id = uuid.uuid4().hex[:8]
        script_file = self.temp_dir / f"diagram_script_{diagram_id}.py"

        full_code = f'''
import os
os.chdir("{self.output_dir}")

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, ECS, Lambda, EKS
from diagrams.aws.database import RDS, Dynamodb, ElastiCache, Redshift
# Aliases for common GPT-generated naming variations
DynamoDB = Dynamodb
from diagrams.aws.network import ELB, APIGateway, CloudFront, Route53, VPC
from diagrams.aws.storage import S3
from diagrams.aws.integration import SQS, SNS
from diagrams.aws.security import IAM, Cognito, WAF
from diagrams.aws.analytics import Kinesis, Athena
from diagrams.aws.management import Cloudwatch

from diagrams.gcp.compute import GCE, GKE, Functions, Run
from diagrams.gcp.database import SQL, Spanner, Firestore, Bigtable
from diagrams.gcp.network import LoadBalancing, CDN, DNS
from diagrams.gcp.storage import GCS

from diagrams.azure.compute import VM, AKS, FunctionApps
from diagrams.azure.database import SQLDatabases, CosmosDb
from diagrams.azure.network import LoadBalancers, ApplicationGateway

from diagrams.k8s.compute import Pod, Deployment, ReplicaSet, StatefulSet
from diagrams.k8s.network import Service, Ingress
from diagrams.k8s.storage import PV, PVC, StorageClass

from diagrams.onprem.client import Users, Client
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL, MySQL, MongoDB
from diagrams.onprem.inmemory import Redis, Memcached
RedisCache = Redis  # Alias for compatibility
from diagrams.onprem.network import Nginx, HAProxy
from diagrams.onprem.queue import Kafka, RabbitMQ
from diagrams.onprem.container import Docker
from diagrams.onprem.ci import Jenkins, GithubActions

from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL as GenericSQL
from diagrams.generic.network import Firewall
from diagrams.generic.storage import Storage

graph_attr = {{
    "bgcolor": "#1a1a2e",
    "fontcolor": "white",
    "fontsize": "16",
    "pad": "0.5",
    "splines": "spline",
    "nodesep": "0.8",
    "ranksep": "1.0"
}}

node_attr = {{
    "fontcolor": "white",
    "fontsize": "12"
}}

edge_attr = {{
    "color": "#6366f1",
    "penwidth": "2.0"
}}

{diagram_code}
'''

        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(full_code)

        process = await asyncio.create_subprocess_exec(
            "python", str(script_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.output_dir)
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"Diagrams execution error: {stderr.decode()}")
            return await self._generate_mermaid_diagram(description, DiagramType.FLOWCHART, style, aspect_ratio)

        # Find the generated PNG file
        import time
        time.sleep(0.5)  # Wait for file to be written

        for f in self.output_dir.iterdir():
            if f.name.endswith('.png') and f.stat().st_mtime > (script_file.stat().st_mtime - 10):
                print(f"Cloud diagram generated: {f}")
                return str(f)

        return None

    async def _generate_diagrams_code(self, description: str, diagram_type: DiagramType) -> Optional[str]:
        """Use GPT to generate Python Diagrams library code"""

        provider_hint = {
            DiagramType.CLOUD_AWS: "AWS services (EC2, RDS, S3, Lambda, etc.)",
            DiagramType.CLOUD_GCP: "Google Cloud services (GCE, GKE, Cloud SQL, etc.)",
            DiagramType.CLOUD_AZURE: "Azure services (VM, AKS, SQL Database, etc.)",
            DiagramType.CLOUD_K8S: "Kubernetes resources (Pod, Deployment, Service, Ingress, etc.)",
            DiagramType.CLOUD_GENERIC: "generic infrastructure components",
            DiagramType.MICROSERVICES: "microservices with databases, queues, and API gateways"
        }.get(diagram_type, "cloud infrastructure")

        diagram_id = uuid.uuid4().hex[:8]

        prompt = f"""Generate Python code using the 'diagrams' library to create an architecture diagram.

Description: {description}
Focus on: {provider_hint}

Example format:
with Diagram("Architecture Name", show=False, filename="diagram_{diagram_id}", direction="LR", graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr):
    users = Users("Users")

    with Cluster("Frontend"):
        cdn = CloudFront("CDN")
        lb = ELB("Load Balancer")

    with Cluster("Backend"):
        api = ECS("API Service")
        worker = Lambda("Worker")

    with Cluster("Data"):
        db = RDS("Database")
        cache = ElastiCache("Cache")

    users >> cdn >> lb >> api
    api >> db
    api >> cache
    worker >> db

Rules:
1. Output ONLY the code starting with 'with Diagram(...'
2. Use show=False and filename="diagram_{diagram_id}"
3. Use Cluster() for logical groupings
4. Use >> for connections (left to right)
5. Use << for reverse connections
6. Use Edge() for labeled connections: node1 >> Edge(label="HTTP") >> node2
7. Include 5-12 components for good visual balance
8. Use direction="LR" for horizontal or "TB" for vertical layouts
9. Always include graph_attr, node_attr, edge_attr in Diagram()

Generate the Python diagrams code:"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": _get_model_name("quality"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.7
                },
                timeout=60.0
            )

            if response.status_code == 200:
                code = response.json()["choices"][0]["message"]["content"].strip()
                code = code.replace("```python", "").replace("```", "").strip()
                if "with Diagram" in code:
                    idx = code.find("with Diagram")
                    code = code[idx:]
                return code

        return None

    async def _generate_graphviz_diagram(
        self,
        description: str,
        diagram_type: DiagramType,
        style: str,
        aspect_ratio: str
    ) -> Optional[str]:
        """Generate diagram using Graphviz DOT language"""

        dot_code = await self._generate_dot_code(description, diagram_type)

        if not dot_code:
            print("Failed to generate DOT code, falling back to Mermaid")
            return await self._generate_mermaid_diagram(description, DiagramType.FLOWCHART, style, aspect_ratio)

        print(f"Generated DOT code:\n{dot_code[:500]}...")

        dimensions = self._get_dimensions(aspect_ratio)
        dpi = 150

        diagram_id = uuid.uuid4().hex[:8]
        dot_file = self.temp_dir / f"diagram_{diagram_id}.dot"
        output_file = self.output_dir / f"graphviz_{diagram_id}.png"

        with open(dot_file, 'w', encoding='utf-8') as f:
            f.write(dot_code)

        cmd = [
            "dot",
            "-Tpng",
            f"-Gdpi={dpi}",
            f"-Gsize={dimensions[0]/dpi},{dimensions[1]/dpi}",
            "-o", str(output_file),
            str(dot_file)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"Graphviz rendering error: {stderr.decode()}")
            return await self._generate_mermaid_diagram(description, DiagramType.FLOWCHART, style, aspect_ratio)

        if output_file.exists():
            print(f"Graphviz diagram generated: {output_file}")
            return str(output_file)

        return None

    async def _generate_dot_code(self, description: str, diagram_type: DiagramType) -> Optional[str]:
        """Use GPT to generate Graphviz DOT code"""

        graph_type = "digraph" if diagram_type in [DiagramType.DEPENDENCY, DiagramType.TREE] else "graph"

        prompt = f"""Generate Graphviz DOT code for this description.

Description: {description}
Type: {diagram_type.value}

Example format:
{graph_type} G {{
    bgcolor="#1a1a2e"
    fontcolor="white"
    node [shape=box, style="rounded,filled", fillcolor="#4f46e5", fontcolor="white", fontsize=14]
    edge [color="#6366f1", penwidth=2]

    A [label="Component A"]
    B [label="Component B"]
    C [label="Component C"]

    A -> B [label="connects"]
    B -> C
}}

Rules:
1. Output ONLY valid DOT code
2. Use dark theme colors (bgcolor="#1a1a2e", fillcolor="#4f46e5")
3. Use white text (fontcolor="white")
4. Include 5-15 nodes for good balance
5. Use descriptive labels
6. Use subgraph for groupings if needed

Generate the DOT code:"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": _get_model_name("fast"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.7
                },
                timeout=60.0
            )

            if response.status_code == 200:
                code = response.json()["choices"][0]["message"]["content"].strip()
                code = code.replace("```dot", "").replace("```graphviz", "").replace("```", "").strip()
                return code

        return None

    def _get_dimensions(self, aspect_ratio: str) -> tuple:
        """Get pixel dimensions for aspect ratio"""
        dimensions = {
            "9:16": (1080, 1920),
            "16:9": (1920, 1080),
            "1:1": (1080, 1080)
        }
        return dimensions.get(aspect_ratio, (1920, 1080))

    async def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except OSError:
            pass
