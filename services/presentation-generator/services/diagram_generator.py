"""
Diagram Generator Service

Generates visual diagrams for presentation slides using:
1. Visual Generator Microservice (PRIMARY) - HTTP calls to visual-generator:8003
   - Supports AWS, Azure, GCP, Kubernetes, On-Premise icons via Python Diagrams
   - Isolated service with Graphviz and other heavy dependencies
2. Mermaid.js via Kroki API for flowcharts and sequences (SECONDARY)
3. Pillow for fallback rendering (TERTIARY)

Architecture:
- presentation-generator (this service) -> HTTP -> visual-generator:8003
- visual-generator handles code generation and rendering

Key Features:
- Import cheat sheet to prevent LLM hallucinations
- Audience-based complexity adjustment (beginner, senior, executive)
- Type-specific routing (architecture -> Python, flowchart -> Mermaid)
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

# Career-based diagram focus
from models.tech_domains import TechCareer, get_diagram_instructions_for_career

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import (
        get_llm_client,
        get_model_name,
        get_provider,
    )
    USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    USE_SHARED_LLM = False
    print("[DIAGRAM] Warning: shared.llm_provider not found, using direct OpenAI", flush=True)

# Training data logger (optional - for fine-tuning data collection)
try:
    from shared.training_logger import log_training_example, TaskType
    TRAINING_LOGGER_AVAILABLE = True
except ImportError:
    log_training_example = None
    TRAINING_LOGGER_AVAILABLE = False


# Visual Generator Service URL
VISUAL_GENERATOR_URL = os.getenv("VISUAL_GENERATOR_URL", "http://visual-generator:8003")

# Diagrams Generator Service URL (dedicated Python Diagrams service)
DIAGRAMS_GENERATOR_URL = os.getenv("DIAGRAMS_GENERATOR_URL", "http://diagrams-generator:8009")


class DiagramType(str, Enum):
    FLOWCHART = "flowchart"
    ARCHITECTURE = "architecture"
    SEQUENCE = "sequence"
    MINDMAP = "mindmap"
    COMPARISON = "comparison"
    HIERARCHY = "hierarchy"
    PROCESS = "process"
    TIMELINE = "timeline"
    DATA_CHART = "data_chart"


class TargetAudience(str, Enum):
    """Audience level for diagram complexity adjustment"""
    BEGINNER = "beginner"     # Simple, few nodes, high-level concepts
    SENIOR = "senior"         # Detailed, specific protocols, clusters
    EXECUTIVE = "executive"   # Value flow, system boundaries, minimal tech details


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


# =============================================================================
# CHEAT SHEET TO PREVENT LLM IMPORT HALLUCINATIONS
# =============================================================================
# This is the KEY to stability - LLM picks from this list instead of guessing

DIAGRAMS_CHEAT_SHEET = """
VALID IMPORTS FOR PYTHON DIAGRAMS (Use ONLY these - DO NOT invent imports):

# Core
from diagrams import Diagram, Cluster, Edge

# AWS
from diagrams.aws.compute import EC2, Lambda, ECS, EKS, Fargate, Batch
from diagrams.aws.database import RDS, Aurora, Dynamodb, ElastiCache, Redshift, Neptune
from diagrams.aws.network import ELB, ALB, NLB, APIGateway, CloudFront, VPC, Route53, DirectConnect
from diagrams.aws.storage import S3, EBS, EFS, FSx
from diagrams.aws.integration import SQS, SNS, Eventbridge, StepFunctions, AppSync
from diagrams.aws.analytics import Kinesis, Glue, Athena, EMR, Quicksight
from diagrams.aws.ml import Sagemaker, Rekognition, Comprehend
from diagrams.aws.security import IAM, Cognito, WAF, KMS, SecretsManager

# Azure
from diagrams.azure.compute import VM, FunctionApps, ContainerInstances, AKS, AppServices
from diagrams.azure.database import SQLDatabases, CosmosDB, DatabaseForPostgresql, CacheForRedis
from diagrams.azure.network import LoadBalancers, ApplicationGateway, VirtualNetworks, Firewall
from diagrams.azure.storage import BlobStorage, StorageAccounts, DataLakeStorage
from diagrams.azure.integration import ServiceBus, EventGrid, LogicApps

# GCP
from diagrams.gcp.compute import ComputeEngine, Functions, Run, GKE, AppEngine
from diagrams.gcp.database import SQL, Spanner, Bigtable, Firestore, Memorystore
from diagrams.gcp.network import LoadBalancing, CDN, DNS, VPC as GVPC
from diagrams.gcp.storage import GCS, Filestore
from diagrams.gcp.analytics import BigQuery, Dataflow, PubSub, Dataproc

# Kubernetes
from diagrams.k8s.compute import Pod, Deployment, ReplicaSet, StatefulSet, DaemonSet, Job, CronJob
from diagrams.k8s.network import Service, Ingress, NetworkPolicy
from diagrams.k8s.storage import PV, PVC, StorageClass
from diagrams.k8s.rbac import ServiceAccount, Role, ClusterRole
from diagrams.k8s.controlplane import APIServer, Scheduler, ControllerManager
from diagrams.k8s.infra import Node, Master

# On-Premise / Self-Hosted
from diagrams.onprem.compute import Server, Nomad
from diagrams.onprem.database import PostgreSQL, Mysql, MongoDB, Redis, Cassandra, Elasticsearch, InfluxDB, Neo4J
from diagrams.onprem.queue import Kafka, RabbitMQ, Celery, ActiveMQ
from diagrams.onprem.network import Nginx, HAProxy, Traefik, Kong, Envoy, Istio
from diagrams.onprem.container import Docker, Containerd
from diagrams.onprem.ci import Jenkins, GitlabCI, GithubActions, CircleCI, TravisCI
from diagrams.onprem.monitoring import Prometheus, Grafana, Datadog, Nagios, Splunk
from diagrams.onprem.logging import Fluentbit, Loki, Graylog, SyslogNg
from diagrams.onprem.aggregator import Fluentd, Vector
from diagrams.elastic.elasticsearch import Logstash, Elasticsearch as ElasticSearch, Kibana, Beats
from diagrams.onprem.vcs import Git, Github, Gitlab
from diagrams.onprem.client import User, Users, Client

# Programming
from diagrams.programming.language import Python, Go, Java, Rust, JavaScript, TypeScript, Cpp, Csharp, Kotlin, Swift
from diagrams.programming.framework import React, Vue, Angular, Django, Spring, FastAPI, Flask, Rails

# SaaS
from diagrams.saas.chat import Slack, Teams, Discord
from diagrams.saas.cdn import Cloudflare, Fastly
from diagrams.saas.identity import Auth0, Okta
from diagrams.saas.alerting import Pagerduty, Opsgenie

# Generic
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL as GenericSQL
from diagrams.generic.network import Firewall, Router, Switch, VPN
from diagrams.generic.device import Mobile, Tablet
from diagrams.generic.storage import Storage
from diagrams.generic.os import LinuxGeneral, Windows, Ubuntu, Centos, RedHat, Suse, Android, IOS

# Custom (for icons not in library)
from diagrams.custom import Custom
"""

# =============================================================================
# AUDIENCE-SPECIFIC COMPLEXITY INSTRUCTIONS
# =============================================================================

AUDIENCE_INSTRUCTIONS = {
    TargetAudience.BEGINNER: """
COMPLEXITY LEVEL: BEGINNER
- Keep it SIMPLE: Maximum 5-7 nodes total
- Group related items in generic Clusters with clear labels
- Do NOT show networking details (no VPCs, subnets, load balancers)
- Use high-level concepts: "Database" not "PostgreSQL Primary + Read Replica"
- Focus on the WHAT, not the HOW
- Large, readable labels (short names)
- Minimal arrows - show main data flow only
""",
    TargetAudience.SENIOR: """
COMPLEXITY LEVEL: SENIOR/EXPERT
- Be DETAILED: 10-15 nodes is acceptable
- Use Clusters for VPCs, Subnets, Kubernetes namespaces
- Show caching layers, load balancers, message queues
- Include proper data flow directions with Edge labels
- Show redundancy patterns (primary/replica, multi-AZ)
- Use specific service names (not generic)
- Include monitoring and logging components if relevant
""",
    TargetAudience.EXECUTIVE: """
COMPLEXITY LEVEL: EXECUTIVE/BUSINESS
- Focus on VALUE FLOW and system boundaries
- Maximum 6-8 nodes - keep it scannable
- Show: Users -> System -> Value/Output
- Hide implementation details (no queues, caches, internal DBs)
- Use business terms, not tech jargon
- Emphasize external integrations and data sources
- Show costs/billing boundaries if relevant
""",
}


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
    Renders professional diagrams via the dedicated diagrams-generator microservice.
    PRIMARY rendering method - calls diagrams-generator:8009 via HTTP.

    This service:
    - Has comprehensive import validation (fixes LLM hallucinations)
    - Executes Python diagrams code in isolated environment
    - Has Graphviz and all cloud provider icons pre-installed

    Key improvements:
    - Generates Python code with DIAGRAMS_CHEAT_SHEET context
    - Sends code to dedicated service for validation + execution
    - Service auto-fixes invalid imports before running
    """

    def __init__(self, output_dir: str = "/tmp/presentations/diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # PRIMARY: Dedicated diagrams-generator service
        self.diagrams_service_url = DIAGRAMS_GENERATOR_URL
        # FALLBACK: Visual generator service
        self.visual_service_url = VISUAL_GENERATOR_URL
        self.timeout = 120.0  # Diagram generation can take time

    async def generate_and_render(
        self,
        description: str,
        diagram_type: DiagramType,
        title: str,
        style: DiagramStyle = DiagramStyle.DARK,
        provider: Optional[DiagramProvider] = None,
        audience: TargetAudience = TargetAudience.SENIOR,
        career: Optional[TechCareer] = None
    ) -> Optional[str]:
        """
        Generate diagram via dedicated diagrams-generator microservice.

        Flow:
        1. Generate Python diagrams code using GPT-4
        2. Send code to diagrams-generator for validation and execution
        3. Fall back to visual-generator if dedicated service fails

        Args:
            description: What the diagram should show
            diagram_type: Type of diagram (architecture, flowchart, etc.)
            title: Title for the diagram
            style: Visual style (dark, light, etc.)
            provider: Cloud provider for icons (aws, azure, etc.)
            audience: Target audience for complexity adjustment
            career: Target career for diagram focus (developer, architect, etc.)

        Returns:
            Path to generated PNG or None if failed
        """
        career_label = career.value if career else "generic"
        print(f"[DIAGRAMS] Generating {diagram_type.value} for {audience.value} audience, {career_label} career focus", flush=True)

        # Build enhanced description with cheat sheet, audience, and career-based focus instructions
        enhanced_description = self._build_enhanced_description(
            description=description,
            diagram_type=diagram_type,
            audience=audience,
            provider=provider,
            career=career
        )

        # Determine the best rendering engine for this diagram type
        engine = self._get_rendering_engine(diagram_type)

        # Only use dedicated service for Python Diagrams engine
        if engine == "diagrams_python":
            result = await self._generate_via_dedicated_service(
                enhanced_description=enhanced_description,
                diagram_type=diagram_type,
                title=title,
                style=style,
                provider=provider,
                audience=audience
            )
            if result:
                return result
            print(f"[DIAGRAMS] Dedicated service failed, falling back to visual-generator", flush=True)

        # Fallback to visual-generator service
        return await self._generate_via_visual_service(
            enhanced_description=enhanced_description,
            diagram_type=diagram_type,
            title=title,
            style=style,
            provider=provider,
            audience=audience,
            engine=engine
        )

    async def _generate_via_dedicated_service(
        self,
        enhanced_description: str,
        diagram_type: DiagramType,
        title: str,
        style: DiagramStyle,
        provider: Optional[DiagramProvider],
        audience: TargetAudience
    ) -> Optional[str]:
        """
        Generate diagram via dedicated diagrams-generator service.

        1. Generate Python code using GPT-4
        2. Send to diagrams-generator for validation and execution
        """
        try:
            # Step 1: Generate Python code
            python_code = await self._generate_python_diagrams_code(
                description=enhanced_description,
                diagram_type=diagram_type,
                title=title,
                provider=provider
            )

            if not python_code:
                print(f"[DIAGRAMS] Failed to generate Python code", flush=True)
                return None

            print(f"[DIAGRAMS] Generated Python code ({len(python_code)} chars), sending to dedicated service", flush=True)

            # Step 2: Send to diagrams-generator service for validation + execution
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.diagrams_service_url}/api/v1/diagrams/generate",
                    json={
                        "description": enhanced_description,
                        "diagram_type": diagram_type.value,
                        "title": title,
                        "python_code": python_code,
                    }
                )

                if response.status_code != 200:
                    print(f"[DIAGRAMS] Dedicated service returned {response.status_code}: {response.text[:200]}", flush=True)
                    return None

                result = response.json()

                if not result.get("success"):
                    error = result.get("error", "Unknown error")
                    validation = result.get("validation", {})
                    if validation.get("errors"):
                        print(f"[DIAGRAMS] Validation errors: {validation['errors']}", flush=True)
                    if validation.get("warnings"):
                        print(f"[DIAGRAMS] Import fixes applied: {validation['warnings']}", flush=True)
                    print(f"[DIAGRAMS] Dedicated service error: {error}", flush=True)
                    return None

                # Get the base64 image and save locally
                image_base64 = result.get("image_base64")
                if not image_base64:
                    print(f"[DIAGRAMS] No image_base64 in response", flush=True)
                    return None

                # Decode and save
                image_data = base64.b64decode(image_base64)
                local_path = self.output_dir / f"diagram_{uuid.uuid4().hex[:8]}.png"
                local_path.write_bytes(image_data)

                metadata = result.get("metadata", {})
                imports_fixed = metadata.get("imports_fixed", 0)
                if imports_fixed > 0:
                    print(f"[DIAGRAMS] Service auto-fixed {imports_fixed} invalid imports", flush=True)

                print(f"[DIAGRAMS] Generated via dedicated service: {local_path}", flush=True)
                return str(local_path)

        except httpx.TimeoutException:
            print(f"[DIAGRAMS] Dedicated service timeout after {self.timeout}s", flush=True)
            return None
        except httpx.ConnectError as e:
            print(f"[DIAGRAMS] Cannot connect to diagrams-generator: {e}", flush=True)
            return None
        except Exception as e:
            print(f"[DIAGRAMS] Error with dedicated service: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    async def _generate_python_diagrams_code(
        self,
        description: str,
        diagram_type: DiagramType,
        title: str,
        provider: Optional[DiagramProvider]
    ) -> Optional[str]:
        """
        Generate Python diagrams code using GPT-4.

        Uses DIAGRAMS_CHEAT_SHEET to guide valid imports.
        """
        # Use configured LLM provider (Groq, OpenAI, DeepSeek, etc.)
        if USE_SHARED_LLM:
            client = get_llm_client()
            model = get_model_name("quality")
            provider_name = get_provider().value if get_provider() else "unknown"
            print(f"[DIAGRAMS] Using {provider_name} with model {model}", flush=True)
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("OPENAI_MODEL", "gpt-4o")

        # Determine provider for icon selection
        detected_provider = provider or self._detect_provider(description)
        provider_hint = ""
        if detected_provider:
            provider_hints = {
                DiagramProvider.AWS: "Use diagrams.aws.* imports",
                DiagramProvider.AZURE: "Use diagrams.azure.* imports",
                DiagramProvider.GCP: "Use diagrams.gcp.* imports",
                DiagramProvider.KUBERNETES: "Use diagrams.k8s.* imports",
                DiagramProvider.ON_PREMISE: "Use diagrams.onprem.* imports",
            }
            provider_hint = provider_hints.get(detected_provider, "")

        prompt = f"""Generate Python code using the 'diagrams' library.

CRITICAL RULES - READ CAREFULLY:
1. The diagram MUST be SPECIFIC to the topic described below
2. DO NOT generate generic "cloud architecture" diagrams with AWS/Azure services unless the topic explicitly mentions them
3. Use the EXACT components, concepts, and relationships described in the DIAGRAM REQUEST
4. If the topic is about a programming concept (e.g., decorators, async/await, OOP), create a CONCEPTUAL diagram showing the concept's structure/flow - NOT a cloud infrastructure diagram
5. For non-cloud topics, use diagrams.onprem, diagrams.programming, or diagrams.generic imports

{DIAGRAMS_CHEAT_SHEET}

DIAGRAM REQUEST (MUST FOLLOW EXACTLY):
{description}

REQUIREMENTS:
- Title: "{title}"
- Use ONLY imports from the cheat sheet above
- {provider_hint if "aws" in description.lower() or "azure" in description.lower() or "gcp" in description.lower() or "cloud" in description.lower() else "For programming/conceptual topics, prefer: diagrams.programming.*, diagrams.onprem.*, or diagrams.generic.*"}
- MATCH the diagram to the topic - if it's about Python decorators, show decorator flow, NOT AWS Lambda
- Use Cluster() for logical grouping
- Add Edge() connections with meaningful labels that describe the relationships
- Node labels MUST use terminology from the DIAGRAM REQUEST, not generic terms
- The code should be self-contained and executable

TOPIC-SPECIFIC GUIDANCE:
- Programming concepts (decorators, classes, functions): Use diagrams.programming.language.* and show conceptual flow
- Data pipelines: Use appropriate data/analytics icons (Kafka, Spark, databases)
- Web architecture: Use Server, Nginx, databases, queues
- Cloud infrastructure: ONLY use AWS/Azure/GCP if explicitly mentioned in the request

OUTPUT FORMAT:
Return ONLY the Python code, no explanations or markdown blocks.
The code should start with imports and end with the Diagram context manager.

Example structure:
```
from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL

with Diagram("{title}", show=False, direction="TB"):
    with Cluster("..."):
        ...
```

Generate the code now:"""

        try:
            system_msg = "You are an expert at generating Python diagrams code. Output ONLY valid Python code, no explanations."
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            code = response.choices[0].message.content.strip()

            # Log successful diagram code generation for training data
            if TRAINING_LOGGER_AVAILABLE and log_training_example:
                log_training_example(
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    response=code,
                    task_type=TaskType.DIAGRAM_GENERATION,
                    model=model,
                    input_tokens=getattr(response.usage, 'prompt_tokens', None),
                    output_tokens=getattr(response.usage, 'completion_tokens', None),
                    metadata={
                        "diagram_type": "python_diagrams",
                        "title": title,
                        "provider": provider.value if provider else "auto",
                    }
                )

            # Clean up markdown code blocks if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()

            return code

        except Exception as e:
            print(f"[DIAGRAMS] Failed to generate Python code: {e}", flush=True)
            return None

    async def _generate_via_visual_service(
        self,
        enhanced_description: str,
        diagram_type: DiagramType,
        title: str,
        style: DiagramStyle,
        provider: Optional[DiagramProvider],
        audience: TargetAudience,
        engine: str
    ) -> Optional[str]:
        """
        Fallback: Generate diagram via visual-generator service.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.visual_service_url}/api/v1/diagrams/generate",
                    json={
                        "description": enhanced_description,
                        "diagram_type": diagram_type.value,
                        "style": style.value,
                        "provider": provider.value if provider else self._detect_provider(enhanced_description).value if self._detect_provider(enhanced_description) else None,
                        "title": title,
                        "language": "en",
                        "format": "png",
                        "audience": audience.value,
                        "engine": engine,
                        "cheat_sheet": DIAGRAMS_CHEAT_SHEET if engine == "diagrams_python" else None,
                    }
                )

                if response.status_code != 200:
                    print(f"[DIAGRAMS] Visual service returned {response.status_code}: {response.text[:200]}", flush=True)
                    return None

                result = response.json()

                if not result.get("success"):
                    print(f"[DIAGRAMS] Visual service failed: {result.get('error')}", flush=True)
                    return None

                # Download the generated image
                file_url = result.get("file_url")
                if not file_url:
                    print(f"[DIAGRAMS] No file_url in response", flush=True)
                    return None

                # Build full URL
                if file_url.startswith("/"):
                    file_url = f"{self.visual_service_url}{file_url}"

                # Download the image
                img_response = await client.get(file_url)
                if img_response.status_code != 200:
                    print(f"[DIAGRAMS] Failed to download image: {img_response.status_code}", flush=True)
                    return None

                # Save locally
                local_path = self.output_dir / f"diagram_{uuid.uuid4().hex[:8]}.png"
                local_path.write_bytes(img_response.content)

                print(f"[DIAGRAMS] Generated via visual-generator: {local_path} (engine={engine})", flush=True)
                return str(local_path)

        except httpx.TimeoutException:
            print(f"[DIAGRAMS] Visual service timeout after {self.timeout}s", flush=True)
            return None
        except httpx.ConnectError as e:
            print(f"[DIAGRAMS] Cannot connect to visual-generator: {e}", flush=True)
            return None
        except Exception as e:
            print(f"[DIAGRAMS] Error calling visual-generator: {e}", flush=True)
            return None

    def _build_enhanced_description(
        self,
        description: str,
        diagram_type: DiagramType,
        audience: TargetAudience,
        provider: Optional[DiagramProvider],
        career: Optional[TechCareer] = None
    ) -> str:
        """
        Build an enhanced description with audience and career-based focus instructions.
        This helps the LLM generate appropriate complexity and perspective.

        Args:
            description: Original diagram description
            diagram_type: Type of diagram
            audience: Target audience level (beginner, senior, executive)
            provider: Cloud provider for icons
            career: Target career role for diagram focus (developer, architect, etc.)

        Returns:
            Enhanced description with all context
        """
        # Get audience-specific instructions (complexity level)
        audience_instruction = AUDIENCE_INSTRUCTIONS.get(audience, AUDIENCE_INSTRUCTIONS[TargetAudience.SENIOR])

        # Get career-specific focus instructions (what perspective to show)
        career_instruction = ""
        if career:
            career_instruction = get_diagram_instructions_for_career(career)

        # Build enhanced description
        enhanced = f"""
DIAGRAM REQUEST:
{description}

{audience_instruction}
"""
        # Add career focus if available
        if career_instruction:
            enhanced += f"""
{career_instruction}
"""

        enhanced += f"""
ADDITIONAL CONTEXT:
- Diagram Type: {diagram_type.value}
- Provider Focus: {provider.value if provider else 'auto-detect from description'}
- Target Career: {career.value if career else 'general audience'}
- Style: Professional, clean, readable
"""
        return enhanced.strip()

    def _get_rendering_engine(self, diagram_type: DiagramType) -> str:
        """
        Determine the best rendering engine for the diagram type.

        PRIORITY: Python Diagrams is PRIMARY for professional quality.
        Only use Mermaid for diagram types that Python Diagrams cannot render.

        - Python Diagrams: architecture, hierarchy, flowchart, process, comparison
        - Mermaid: sequence (requires arrows between actors), mindmap (radial layout), timeline
        - Matplotlib: data_chart (statistics)
        """
        # Mermaid is ONLY for diagram types Python Diagrams truly cannot render
        mermaid_only_types = [DiagramType.SEQUENCE, DiagramType.MINDMAP, DiagramType.TIMELINE]

        if diagram_type in mermaid_only_types:
            return "mermaid"
        elif diagram_type == DiagramType.DATA_CHART:
            return "matplotlib"
        else:
            # Python Diagrams for everything else (architecture, hierarchy, flowchart, process, comparison)
            return "diagrams_python"

    def _detect_provider(self, description: str) -> Optional[DiagramProvider]:
        """Auto-detect cloud provider from description."""
        desc_lower = description.lower()

        # AWS keywords
        if any(kw in desc_lower for kw in ['aws', 'amazon', 'ec2', 's3', 'lambda', 'dynamodb', 'sqs', 'sns', 'rds', 'eks', 'ecs', 'fargate', 'cloudfront']):
            return DiagramProvider.AWS
        # Azure keywords
        if any(kw in desc_lower for kw in ['azure', 'microsoft', 'cosmos', 'blob', 'aks', 'app service', 'functions']):
            return DiagramProvider.AZURE
        # GCP keywords
        if any(kw in desc_lower for kw in ['gcp', 'google cloud', 'bigquery', 'gcs', 'cloud run', 'gke', 'pubsub', 'spanner']):
            return DiagramProvider.GCP
        # Kubernetes keywords
        if any(kw in desc_lower for kw in ['kubernetes', 'k8s', 'pod', 'deployment', 'ingress', 'helm', 'kubectl']):
            return DiagramProvider.KUBERNETES
        # On-premise keywords
        if any(kw in desc_lower for kw in ['docker', 'nginx', 'kafka', 'redis', 'postgres', 'mysql', 'mongo', 'rabbitmq', 'elasticsearch']):
            return DiagramProvider.ON_PREMISE

        return None

    def _detect_audience_from_context(self, context: Optional[str]) -> TargetAudience:
        """
        Detect target audience from course/presentation context.
        """
        if not context:
            return TargetAudience.SENIOR

        context_lower = context.lower()

        # Beginner indicators
        if any(kw in context_lower for kw in ['beginner', 'introduction', 'basics', 'getting started', 'fundamentals', '101', 'for dummies', 'first steps']):
            return TargetAudience.BEGINNER

        # Executive indicators
        if any(kw in context_lower for kw in ['executive', 'cto', 'ceo', 'business', 'strategy', 'overview', 'high-level', 'management']):
            return TargetAudience.EXECUTIVE

        # Default to senior for technical content
        return TargetAudience.SENIOR

    async def check_service_health(self) -> bool:
        """Check if the visual-generator service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.service_url}/health")
                return response.status_code == 200
        except Exception:
            return False


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
        height: int = 1080,
        target_audience: str = "senior",
        target_career: Optional[str] = None
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
            target_audience: Target audience level (beginner, senior, executive)
            target_career: Target career role for diagram focus (e.g., "data_engineer", "cloud_architect")

        Returns:
            Path to generated diagram image
        """
        career_label = target_career if target_career else "general"
        print(f"[DIAGRAM] Generating {diagram_type.value} diagram for {target_audience} audience, {career_label} career: {title}", flush=True)

        # Ensure job_id is never None to prevent filenames like "None_diagram_0.png"
        safe_job_id = job_id if job_id else "unknown"
        output_path = self.output_dir / f"{safe_job_id}_diagram_{slide_index}.png"

        # Map theme to DiagramStyle
        style_map = {
            "tech": DiagramStyle.DARK,
            "light": DiagramStyle.LIGHT,
            "gradient": DiagramStyle.COLORFUL,
        }
        style = style_map.get(theme, DiagramStyle.DARK)

        # Map target_audience string to TargetAudience enum
        audience_map = {
            "beginner": TargetAudience.BEGINNER,
            "absolute beginner": TargetAudience.BEGINNER,
            "novice": TargetAudience.BEGINNER,
            "senior": TargetAudience.SENIOR,
            "expert": TargetAudience.SENIOR,
            "advanced": TargetAudience.SENIOR,
            "intermediate": TargetAudience.SENIOR,
            "executive": TargetAudience.EXECUTIVE,
            "business": TargetAudience.EXECUTIVE,
            "manager": TargetAudience.EXECUTIVE,
        }
        # Extract audience level from target_audience string
        audience_lower = target_audience.lower() if target_audience else "senior"
        audience = TargetAudience.SENIOR  # default
        for key, value in audience_map.items():
            if key in audience_lower:
                audience = value
                break

        # Parse target_career string to TechCareer enum
        career: Optional[TechCareer] = None
        if target_career:
            try:
                # Try direct enum lookup
                career = TechCareer(target_career.lower())
            except ValueError:
                # Try matching by name (e.g., "DATA_ENGINEER" -> TechCareer.DATA_ENGINEER)
                try:
                    career = TechCareer[target_career.upper()]
                except KeyError:
                    print(f"[DIAGRAM] Unknown career '{target_career}', using default focus", flush=True)
                    career = None

        # PRIMARY: Try Python Diagrams library for professional quality
        # Use for ALL diagram types except Mermaid-specific ones (sequence, mindmap, timeline)
        # Python Diagrams produces enterprise-grade diagrams with real AWS/Azure/GCP icons
        mermaid_only_types = [DiagramType.SEQUENCE, DiagramType.MINDMAP, DiagramType.TIMELINE]

        if diagram_type not in mermaid_only_types:
            try:
                print(f"[DIAGRAM] PRIMARY: Python Diagrams library ({diagram_type.value}, {audience.value} complexity, career={career.value if career else 'none'})", flush=True)
                diagrams_path = await self.diagrams_renderer.generate_and_render(
                    description=description,
                    diagram_type=diagram_type,
                    title=title,
                    style=style,
                    provider=self.diagrams_renderer._detect_provider(description),
                    audience=audience,
                    career=career
                )
                if diagrams_path:
                    # Add title overlay and resize to target dimensions
                    final_image = await self._post_process_diagram(
                        diagrams_path, title, theme, width, height
                    )
                    if final_image:
                        final_image.save(str(output_path), "PNG")
                        print(f"[DIAGRAM] SUCCESS via Python Diagrams: {output_path}", flush=True)
                        return str(output_path)
                    else:
                        print(f"[DIAGRAM] Python Diagrams returned path but post-processing failed", flush=True)
                else:
                    print(f"[DIAGRAM] Python Diagrams returned no path", flush=True)
            except Exception as e:
                print(f"[DIAGRAM] Python Diagrams FAILED: {e}", flush=True)
                import traceback
                traceback.print_exc()

        # SECONDARY: Try Mermaid rendering via Kroki (for sequence/mindmap/timeline or as fallback)
        # Note: Mermaid quality is lower than Python Diagrams
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
        Generate Mermaid diagram code using configured LLM provider.
        """
        # Use configured LLM provider (Groq, OpenAI, DeepSeek, etc.)
        if USE_SHARED_LLM:
            client = get_llm_client()
            model = get_model_name("fast")  # Use fast model for simple Mermaid generation
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Cheaper fallback

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

        system_msg = "Output ONLY valid Mermaid code. No explanations. No markdown blocks."
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
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

            # Log successful Mermaid code generation for training data
            if TRAINING_LOGGER_AVAILABLE and log_training_example:
                log_training_example(
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    response=mermaid_code,
                    task_type=TaskType.DIAGRAM_GENERATION,
                    model=model,
                    input_tokens=getattr(response.usage, 'prompt_tokens', None),
                    output_tokens=getattr(response.usage, 'completion_tokens', None),
                    metadata={
                        "diagram_type": "mermaid",
                        "mermaid_type": diagram_type.value,
                        "title": title,
                    }
                )

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
        Uses configured LLM to extract nodes and edges.
        """
        # Use configured LLM provider (Groq, OpenAI, DeepSeek, etc.)
        if USE_SHARED_LLM:
            client = get_llm_client()
            model = get_model_name("fast")  # Use fast model for JSON extraction
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Cheaper fallback

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

        system_msg = "You extract diagram structures from descriptions. Output only valid JSON."
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000
        )

        response_content = response.choices[0].message.content
        try:
            parsed_result = json.loads(response_content)

            # Log successful diagram structure parsing for training data
            if TRAINING_LOGGER_AVAILABLE and log_training_example:
                log_training_example(
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    response=response_content,
                    task_type=TaskType.DIAGRAM_GENERATION,
                    model=model,
                    input_tokens=getattr(response.usage, 'prompt_tokens', None),
                    output_tokens=getattr(response.usage, 'completion_tokens', None),
                    metadata={
                        "diagram_type": diagram_type.value,
                        "parsing_type": "structure_extraction",
                    }
                )

            return parsed_result
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
