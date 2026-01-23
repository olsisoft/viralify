"""
Professional Diagram Renderer using Diagrams (Python)

Generates enterprise-grade architecture diagrams with official icons from:
- AWS (EC2, S3, Lambda, RDS, etc.)
- Azure (VM, Blob, Functions, etc.)
- GCP (Compute, Storage, Cloud Run, etc.)
- Kubernetes (Pod, Service, Deployment, etc.)
- On-Premise (servers, networks, databases)
- Generic (users, mobile, web)

Uses GPT-4o to generate Python Diagrams code from natural language descriptions.
"""

import os
import json
import uuid
import time
import tempfile
import subprocess
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from openai import AsyncOpenAI
from enum import Enum

from ..models.visual_models import (
    DiagramType,
    DiagramStyle,
    RenderFormat,
    DiagramResult,
)


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
    ELASTIC = "elastic"
    FIREBASE = "firebase"
    OUTSCALE = "outscale"
    ALIBABACLOUD = "alibabacloud"
    ORACLE = "oracle"
    OPENSTACK = "openstack"
    IBM = "ibm"
    DIGITALOCEAN = "digitalocean"


class DiagramsRenderer:
    """
    Renders professional diagrams using the Diagrams Python library.
    Generates Python code via GPT-4o, then executes it to produce images.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        output_dir: str = "/tmp/viralify/diagrams"
    ):
        """Initialize the Diagrams renderer."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = "gpt-4o"  # Use GPT-4o for better code generation

    async def generate_diagram_code(
        self,
        description: str,
        diagram_type: DiagramType,
        style: DiagramStyle = DiagramStyle.DARK,
        provider: Optional[DiagramProvider] = None,
        context: Optional[str] = None,
        language: str = "en"
    ) -> str:
        """
        Generate Python Diagrams code from natural language description.

        Args:
            description: What the diagram should show
            diagram_type: Type of diagram
            style: Visual style
            provider: Primary cloud/tech provider for icons
            context: Additional context
            language: Language for labels

        Returns:
            Valid Python code using the diagrams library
        """
        if not self.client:
            raise ValueError("OpenAI API key required for diagram generation")

        system_prompt = self._build_system_prompt(diagram_type, style, provider, language)

        user_content = f"""Create a professional diagram showing: {description}

Diagram type: {diagram_type.value}
Style: {style.value}
Primary provider: {provider.value if provider else 'auto-detect from description'}
Label language: {language}"""

        if context:
            user_content += f"\n\nAdditional context:\n{context[:1000]}"

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,  # Low temperature for consistent code
            max_tokens=2000
        )

        code = response.choices[0].message.content.strip()

        # Clean up code blocks if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code

    def _build_system_prompt(
        self,
        diagram_type: DiagramType,
        style: DiagramStyle,
        provider: Optional[DiagramProvider],
        language: str
    ) -> str:
        """Build the system prompt for diagram code generation."""

        # Style configurations
        style_config = {
            DiagramStyle.DARK: 'graph_attr={"bgcolor": "#1e1e1e", "fontcolor": "white"}',
            DiagramStyle.LIGHT: 'graph_attr={"bgcolor": "#ffffff", "fontcolor": "black"}',
            DiagramStyle.NEUTRAL: 'graph_attr={"bgcolor": "#f5f5f5", "fontcolor": "#333333"}',
            DiagramStyle.COLORFUL: 'graph_attr={"bgcolor": "#1a1a2e", "fontcolor": "white"}',
        }

        return f"""You are an expert at creating professional architecture diagrams using the Python 'diagrams' library.

Generate ONLY valid Python code that uses the 'diagrams' library. The code must:
1. Be syntactically correct Python that can execute without errors
2. Use proper imports from the diagrams library
3. Create clear, professional diagrams with meaningful labels
4. Use appropriate icons from the correct provider modules
5. Include proper clustering/grouping where logical
6. Have clear connection flows with Edge labels where helpful

CRITICAL RULES:
- Output ONLY Python code, no explanations
- Labels should be in {language}
- Maximum 12-15 nodes for readability
- Use subgraphs/Clusters to group related components
- Always set show=False and filename parameter
- Use descriptive variable names

AVAILABLE PROVIDERS AND IMPORTS:
```python
# AWS
from diagrams.aws.compute import EC2, Lambda, ECS, EKS, Fargate, Batch
from diagrams.aws.database import RDS, Aurora, DynamoDB, ElastiCache, Redshift
from diagrams.aws.network import APIGateway, CloudFront, ELB, ALB, NLB, Route53, VPC
from diagrams.aws.storage import S3, EBS, EFS
from diagrams.aws.integration import SQS, SNS, EventBridge, StepFunctions
from diagrams.aws.analytics import Kinesis, Glue, Athena, EMR
from diagrams.aws.ml import Sagemaker, Comprehend, Rekognition
from diagrams.aws.security import IAM, Cognito, WAF, Shield, KMS

# Azure
from diagrams.azure.compute import VM, FunctionApps, ContainerInstances, AKS
from diagrams.azure.database import SQLDatabases, CosmosDB, BlobStorage
from diagrams.azure.network import LoadBalancers, ApplicationGateway, VirtualNetworks
from diagrams.azure.integration import ServiceBus, EventGrid
from diagrams.azure.ml import MachineLearningServiceWorkspaces

# GCP
from diagrams.gcp.compute import ComputeEngine, Functions, Run, GKE
from diagrams.gcp.database import SQL, Spanner, Bigtable, Firestore
from diagrams.gcp.network import LoadBalancing, CDN, DNS
from diagrams.gcp.storage import GCS
from diagrams.gcp.analytics import BigQuery, Dataflow, PubSub
from diagrams.gcp.ml import AIHub, AutoML

# Kubernetes
from diagrams.k8s.compute import Pod, Deployment, ReplicaSet, StatefulSet, DaemonSet
from diagrams.k8s.network import Service, Ingress, NetworkPolicy
from diagrams.k8s.storage import PV, PVC, StorageClass
from diagrams.k8s.rbac import ServiceAccount, Role, ClusterRole
from diagrams.k8s.group import Namespace

# On-Premise
from diagrams.onprem.compute import Server, Nomad
from diagrams.onprem.database import PostgreSQL, MySQL, MongoDB, Redis, Cassandra, Elasticsearch
from diagrams.onprem.network import Nginx, HAProxy, Traefik, Kong
from diagrams.onprem.queue import Kafka, RabbitMQ, Celery
from diagrams.onprem.container import Docker
from diagrams.onprem.ci import Jenkins, GitlabCI, GithubActions, CircleCI
from diagrams.onprem.monitoring import Prometheus, Grafana, Datadog
from diagrams.onprem.logging import Fluentd, Logstash
from diagrams.onprem.mlops import Mlflow, Kubeflow

# Generic
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL as GenericSQL
from diagrams.generic.network import Firewall, Router, Switch
from diagrams.generic.os import Linux, Windows
from diagrams.generic.device import Mobile, Tablet

# Programming Languages
from diagrams.programming.language import Python, Java, Go, Rust, JavaScript, TypeScript

# SaaS
from diagrams.saas.chat import Slack, Teams
from diagrams.saas.cdn import Cloudflare
from diagrams.saas.identity import Auth0, Okta

# Custom/Generic shapes
from diagrams.custom import Custom
from diagrams import Diagram, Cluster, Edge
```

EXAMPLE - Microservices Architecture:
```python
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import ECS
from diagrams.aws.database import RDS, ElastiCache
from diagrams.aws.network import ALB, Route53
from diagrams.aws.integration import SQS

with Diagram("Microservices Architecture", show=False, filename="diagram", direction="TB", {style_config.get(style, '')}):
    dns = Route53("DNS")
    lb = ALB("Load Balancer")

    with Cluster("Application Layer"):
        svc_api = ECS("API Gateway")
        svc_user = ECS("User Service")
        svc_order = ECS("Order Service")
        svc_payment = ECS("Payment Service")

    with Cluster("Data Layer"):
        db_main = RDS("Main DB")
        cache = ElastiCache("Redis Cache")

    queue = SQS("Event Queue")

    dns >> lb >> svc_api
    svc_api >> [svc_user, svc_order]
    svc_order >> svc_payment
    svc_user >> db_main
    svc_order >> [db_main, cache]
    svc_payment >> queue
```

EXAMPLE - Kubernetes Deployment:
```python
from diagrams import Diagram, Cluster
from diagrams.k8s.compute import Pod, Deployment
from diagrams.k8s.network import Service, Ingress
from diagrams.k8s.storage import PVC

with Diagram("Kubernetes Deployment", show=False, filename="diagram", direction="LR"):
    ingress = Ingress("ingress")

    with Cluster("Namespace: production"):
        svc = Service("service")

        with Cluster("Deployment"):
            pods = [Pod("pod-1"), Pod("pod-2"), Pod("pod-3")]

        pvc = PVC("persistent-volume")

    ingress >> svc >> pods
    pods >> pvc
```

EXAMPLE - Data Pipeline:
```python
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.analytics import Kinesis, Glue, Athena
from diagrams.aws.storage import S3
from diagrams.aws.database import Redshift
from diagrams.onprem.analytics import Spark

with Diagram("Data Pipeline", show=False, filename="diagram", direction="LR"):
    with Cluster("Ingestion"):
        stream = Kinesis("Kinesis Stream")

    with Cluster("Processing"):
        etl = Glue("Glue ETL")
        spark = Spark("Spark Jobs")

    with Cluster("Storage"):
        lake = S3("Data Lake")
        warehouse = Redshift("Data Warehouse")

    with Cluster("Analytics"):
        query = Athena("Athena")

    stream >> Edge(label="raw data") >> etl
    etl >> Edge(label="transform") >> lake
    lake >> spark >> warehouse
    warehouse >> query
```

Now generate the diagram code based on the user's description."""

    async def render(
        self,
        code: str,
        filename: Optional[str] = None,
        format: RenderFormat = RenderFormat.PNG
    ) -> DiagramResult:
        """
        Execute the generated Python code to render the diagram.

        Args:
            code: Python code using the diagrams library
            filename: Output filename (without extension)
            format: Output format (PNG recommended)

        Returns:
            DiagramResult with file path and metadata
        """
        start_time = time.time()
        file_id = filename or f"diagram_{uuid.uuid4().hex[:8]}"

        try:
            # Create a temporary Python file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                dir=str(self.output_dir)
            ) as f:
                # Ensure the filename is set correctly in the code
                modified_code = self._inject_filename(code, file_id)
                f.write(modified_code)
                temp_py_path = f.name

            # Execute the Python script
            result = await asyncio.create_subprocess_exec(
                'python', temp_py_path,
                cwd=str(self.output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            # Clean up temp file
            os.unlink(temp_py_path)

            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                print(f"[DIAGRAMS] Execution failed: {error_msg}", flush=True)
                return DiagramResult(
                    success=False,
                    diagram_type=DiagramType.ARCHITECTURE,
                    generation_time_ms=int((time.time() - start_time) * 1000),
                    error=f"Code execution failed: {error_msg[:500]}"
                )

            # Find the generated file (diagrams creates .png by default)
            output_path = self.output_dir / f"{file_id}.png"

            if not output_path.exists():
                # Try to find any generated file
                for ext in ['.png', '.svg', '.jpg', '.pdf']:
                    candidate = self.output_dir / f"{file_id}{ext}"
                    if candidate.exists():
                        output_path = candidate
                        break

            if not output_path.exists():
                return DiagramResult(
                    success=False,
                    diagram_type=DiagramType.ARCHITECTURE,
                    generation_time_ms=int((time.time() - start_time) * 1000),
                    error="Output file not generated"
                )

            generation_time = int((time.time() - start_time) * 1000)

            return DiagramResult(
                success=True,
                diagram_type=DiagramType.ARCHITECTURE,
                file_path=str(output_path),
                file_url=f"/diagrams/{output_path.name}",
                format=format,
                generation_time_ms=generation_time,
                metadata={
                    "renderer": "diagrams",
                    "code_length": len(code)
                }
            )

        except Exception as e:
            generation_time = int((time.time() - start_time) * 1000)
            print(f"[DIAGRAMS] Error: {e}", flush=True)
            return DiagramResult(
                success=False,
                diagram_type=DiagramType.ARCHITECTURE,
                generation_time_ms=generation_time,
                error=str(e)
            )

    def _inject_filename(self, code: str, filename: str) -> str:
        """Ensure the filename parameter is set correctly in the diagram code."""
        import re

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

    async def generate_and_render(
        self,
        description: str,
        diagram_type: DiagramType,
        style: DiagramStyle = DiagramStyle.DARK,
        provider: Optional[DiagramProvider] = None,
        format: RenderFormat = RenderFormat.PNG,
        context: Optional[str] = None,
        language: str = "en",
        max_retries: int = 2
    ) -> DiagramResult:
        """
        Generate diagram code from description and render to image.
        Includes retry logic for code generation failures.
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Generate the code
                code = await self.generate_diagram_code(
                    description=description,
                    diagram_type=diagram_type,
                    style=style,
                    provider=provider,
                    context=context,
                    language=language
                )

                print(f"[DIAGRAMS] Generated code (attempt {attempt + 1}):\n{code[:500]}...", flush=True)

                # Render the diagram
                result = await self.render(code, format=format)

                if result.success:
                    return result

                last_error = result.error
                print(f"[DIAGRAMS] Render failed (attempt {attempt + 1}): {last_error}", flush=True)

            except Exception as e:
                last_error = str(e)
                print(f"[DIAGRAMS] Error (attempt {attempt + 1}): {e}", flush=True)

        # All retries failed
        return DiagramResult(
            success=False,
            diagram_type=diagram_type,
            generation_time_ms=0,
            error=f"Failed after {max_retries + 1} attempts: {last_error}"
        )

    async def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate Python diagrams code without executing.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"


# Predefined templates for common architectures
class DiagramTemplates:
    """Ready-to-use templates for common architecture patterns."""

    @staticmethod
    def three_tier_web_app(app_name: str = "Web Application") -> str:
        return f'''from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB, Route53

with Diagram("{app_name}", show=False, filename="diagram", direction="TB"):
    dns = Route53("DNS")
    lb = ELB("Load Balancer")

    with Cluster("Web Tier"):
        web = [EC2("Web 1"), EC2("Web 2")]

    with Cluster("App Tier"):
        app = [EC2("App 1"), EC2("App 2")]

    with Cluster("Data Tier"):
        db_primary = RDS("Primary")
        db_replica = RDS("Replica")

    dns >> lb >> web >> app
    app >> db_primary
    db_primary - db_replica
'''

    @staticmethod
    def microservices_k8s(service_count: int = 4) -> str:
        services = [f'Pod("svc-{i+1}")' for i in range(service_count)]
        return f'''from diagrams import Diagram, Cluster
from diagrams.k8s.compute import Pod, Deployment
from diagrams.k8s.network import Service, Ingress
from diagrams.k8s.storage import PVC
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.queue import Kafka

with Diagram("Microservices on Kubernetes", show=False, filename="diagram", direction="TB"):
    ingress = Ingress("API Gateway")

    with Cluster("Services"):
        svcs = [{', '.join(services)}]

    with Cluster("Data"):
        db = PostgreSQL("Database")
        queue = Kafka("Event Bus")

    ingress >> svcs
    svcs >> db
    svcs >> queue
'''

    @staticmethod
    def data_pipeline_aws() -> str:
        return '''from diagrams import Diagram, Cluster, Edge
from diagrams.aws.analytics import Kinesis, Glue, Athena
from diagrams.aws.storage import S3
from diagrams.aws.database import Redshift
from diagrams.aws.ml import Sagemaker

with Diagram("AWS Data Pipeline", show=False, filename="diagram", direction="LR"):
    with Cluster("Ingestion"):
        kinesis = Kinesis("Kinesis")

    with Cluster("Processing"):
        glue = Glue("Glue ETL")

    with Cluster("Storage"):
        s3_raw = S3("Raw Data")
        s3_processed = S3("Processed")

    with Cluster("Analytics"):
        redshift = Redshift("Warehouse")
        athena = Athena("Ad-hoc Query")

    with Cluster("ML"):
        sagemaker = Sagemaker("ML Models")

    kinesis >> glue >> s3_raw
    s3_raw >> glue >> s3_processed
    s3_processed >> [redshift, athena]
    s3_processed >> sagemaker
'''

    @staticmethod
    def ci_cd_pipeline() -> str:
        return '''from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.vcs import Github
from diagrams.onprem.ci import GithubActions, Jenkins
from diagrams.onprem.container import Docker
from diagrams.k8s.compute import Deployment
from diagrams.aws.compute import ECS
from diagrams.generic.storage import Storage

with Diagram("CI/CD Pipeline", show=False, filename="diagram", direction="LR"):
    vcs = Github("Source Code")

    with Cluster("CI"):
        ci = GithubActions("GitHub Actions")
        test = Storage("Tests")

    with Cluster("Build"):
        docker = Docker("Container Build")
        registry = Storage("Registry")

    with Cluster("Deploy"):
        staging = Deployment("Staging")
        prod = ECS("Production")

    vcs >> ci >> test
    test >> docker >> registry
    registry >> staging >> Edge(label="promote") >> prod
'''
