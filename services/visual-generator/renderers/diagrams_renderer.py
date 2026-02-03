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
import ast
from typing import Optional, Dict, Any, List, Tuple, Set
from pathlib import Path
from openai import AsyncOpenAI
from enum import Enum

from models.visual_models import (
    DiagramType,
    DiagramStyle,
    RenderFormat,
    DiagramResult,
    DiagramCoordinates,
    NodeCoordinate,
    EdgeCoordinate,
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


class CodeSecurityValidator:
    """
    Validates generated Python code for security before execution.

    Uses AST parsing to analyze code without executing it.
    Blocks dangerous imports, functions, and patterns that could
    compromise server security.
    """

    # Allowed module prefixes - ONLY diagrams library
    ALLOWED_IMPORT_PREFIXES: Set[str] = {
        'diagrams',      # All diagrams.* modules
    }

    # Explicitly blocked imports - dangerous modules
    BLOCKED_IMPORTS: Set[str] = {
        # System access
        'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        'platform', 'ctypes', 'signal',
        # File/IO
        'io', 'tempfile', 'fileinput', 'glob',
        # Network
        'socket', 'requests', 'urllib', 'http', 'ftplib',
        'smtplib', 'poplib', 'imaplib', 'telnetlib', 'ssl',
        'asyncio',  # Could be used for network ops
        # Serialization (code execution risks)
        'pickle', 'shelve', 'marshal', 'dill',
        # Code execution
        'code', 'codeop', 'compileall', 'importlib',
        'builtins', '__builtins__', 'types',
        # Process/Threading
        'multiprocessing', 'threading', 'concurrent',
        # Other dangerous
        'pty', 'tty', 'termios', 'resource',
        'gc', 'inspect', 'dis', 'traceback',
    }

    # Dangerous built-in functions
    BLOCKED_FUNCTIONS: Set[str] = {
        # Code execution
        'exec', 'eval', 'compile', '__import__',
        # File operations
        'open', 'file', 'input',
        # Reflection (can bypass restrictions)
        'getattr', 'setattr', 'delattr', 'hasattr',
        'globals', 'locals', 'vars', 'dir',
        # System
        'exit', 'quit', 'breakpoint',
        'print',  # Block print to prevent info leaks (diagram code shouldn't need it)
        # Memory/Object manipulation
        'id', 'hash', 'memoryview', 'bytearray',
    }

    # Dangerous attribute access patterns
    BLOCKED_ATTRIBUTES: Set[str] = {
        '__class__', '__bases__', '__subclasses__',
        '__mro__', '__globals__', '__code__',
        '__builtins__', '__import__', '__loader__',
        '__spec__', '__dict__', '__module__',
        '__reduce__', '__reduce_ex__',
    }

    @classmethod
    def validate(cls, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python code for security.

        Args:
            code: Python source code to validate

        Returns:
            Tuple of (is_safe, error_message)
            - (True, None) if code is safe
            - (False, "error description") if code is dangerous
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

        for node in ast.walk(tree):
            # Check import statements: import x, import x.y
            if isinstance(node, ast.Import):
                for alias in node.names:
                    is_safe, error = cls._check_import(alias.name)
                    if not is_safe:
                        return False, error

            # Check from imports: from x import y
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    is_safe, error = cls._check_import(node.module)
                    if not is_safe:
                        return False, error

            # Check function calls
            elif isinstance(node, ast.Call):
                is_safe, error = cls._check_call(node)
                if not is_safe:
                    return False, error

            # Check attribute access (e.g., obj.__class__)
            elif isinstance(node, ast.Attribute):
                is_safe, error = cls._check_attribute(node)
                if not is_safe:
                    return False, error

            # Check string literals for suspicious patterns
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                is_safe, error = cls._check_string_literal(node.value)
                if not is_safe:
                    return False, error

        return True, None

    @classmethod
    def _check_import(cls, module_name: str) -> Tuple[bool, Optional[str]]:
        """Check if an import is allowed."""
        # Get the top-level module
        top_module = module_name.split('.')[0]

        # Check if explicitly blocked
        if top_module in cls.BLOCKED_IMPORTS:
            return False, f"SECURITY: Blocked import '{module_name}' - not allowed in diagram code"

        # Check if in allowed list (must start with allowed prefix)
        is_allowed = any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in cls.ALLOWED_IMPORT_PREFIXES
        )

        if not is_allowed:
            return False, f"SECURITY: Unauthorized import '{module_name}' - only 'diagrams.*' imports allowed"

        return True, None

    @classmethod
    def _check_call(cls, node: ast.Call) -> Tuple[bool, Optional[str]]:
        """Check if a function call is safe."""
        func_name = None

        # Direct function call: func()
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

        # Method call on a name: something.func()
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name and func_name in cls.BLOCKED_FUNCTIONS:
            return False, f"SECURITY: Blocked function '{func_name}()' - not allowed in diagram code"

        return True, None

    @classmethod
    def _check_attribute(cls, node: ast.Attribute) -> Tuple[bool, Optional[str]]:
        """Check for dangerous attribute access patterns."""
        attr_name = node.attr

        # Block dangerous dunder attributes
        if attr_name in cls.BLOCKED_ATTRIBUTES:
            return False, f"SECURITY: Blocked attribute access '{attr_name}' - potential code injection"

        return True, None

    @classmethod
    def _check_string_literal(cls, value: str) -> Tuple[bool, Optional[str]]:
        """Check string literals for suspicious patterns."""
        suspicious_patterns = [
            '/bin/', '/usr/bin/', '/etc/',  # System paths
            'rm -rf', 'sudo', 'chmod', 'chown',  # Shell commands
            '$(', '`',  # Command substitution
            '127.0.0.1', 'localhost',  # Network access attempts
            '.env', 'password', 'secret', 'token', 'key',  # Credential access
        ]

        value_lower = value.lower()
        for pattern in suspicious_patterns:
            if pattern.lower() in value_lower:
                # Allow these patterns only if they look like diagram labels
                # (short strings that might legitimately contain these words)
                if len(value) > 100:  # Long strings are suspicious
                    return False, f"SECURITY: Suspicious string pattern detected: contains '{pattern}'"

        return True, None

    @classmethod
    def get_security_report(cls, code: str) -> Dict[str, Any]:
        """
        Generate a detailed security report for the code.

        Returns a dict with validation results and detected patterns.
        """
        is_safe, error = cls.validate(code)

        # Count imports
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
        except SyntaxError:
            pass

        return {
            "is_safe": is_safe,
            "error": error,
            "imports_detected": imports,
            "imports_count": len(imports),
        }
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
        language: str = "en",
        audience: Optional[str] = None,
        cheat_sheet: Optional[str] = None
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
            audience: Target audience level (beginner, senior, executive)
            cheat_sheet: Valid imports cheat sheet to prevent LLM hallucinations

        Returns:
            Valid Python code using the diagrams library
        """
        if not self.client:
            raise ValueError("OpenAI API key required for diagram generation")

        system_prompt = self._build_system_prompt(
            diagram_type, style, provider, language, audience, cheat_sheet
        )

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

        # Post-process to fix common Edge chain issues
        code = self._fix_edge_chains(code)

        return code

    def _fix_edge_chains(self, code: str) -> str:
        """
        Fix common Edge chain issues that cause runtime errors.

        Problems:
        1. Long chains: a >> Edge() >> b >> Edge() >> c causes issues
        2. Multiple Edges in one line can fail

        Solution: Split complex chains into multiple lines.
        """
        import re

        lines = code.split('\n')
        fixed_lines = []

        for line in lines:
            # Check if line has multiple Edge() calls
            edge_count = line.count('Edge(')

            if edge_count > 1:
                # This line has multiple Edges - split it
                # Pattern: node1 >> Edge(...) >> node2 >> Edge(...) >> node3
                print(f"[DIAGRAMS] Fixing multi-Edge line: {line.strip()[:80]}...", flush=True)

                # Try to split the chain
                # Find all segments: node >> Edge(...) >> node
                pattern = r'(\w+)\s*>>\s*Edge\([^)]*\)\s*>>\s*(\w+)'
                matches = list(re.finditer(pattern, line))

                if len(matches) >= 2:
                    # Get indentation
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent

                    # Extract all node-edge-node segments
                    segments = []
                    remaining = line.strip()

                    # Split by >> but preserve Edge calls
                    parts = re.split(r'\s*>>\s*', remaining)

                    current_node = None
                    current_edge = None

                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue

                        if part.startswith('Edge('):
                            current_edge = part
                        else:
                            if current_node and current_edge:
                                # We have a complete segment
                                segments.append((current_node, current_edge, part))
                                current_edge = None
                            elif current_node:
                                # Direct connection without Edge
                                segments.append((current_node, None, part))
                            current_node = part

                    # Generate split lines
                    for src, edge, dst in segments:
                        if edge:
                            fixed_lines.append(f"{indent_str}{src} >> {edge} >> {dst}")
                        else:
                            fixed_lines.append(f"{indent_str}{src} >> {dst}")

                    continue

            # Line is OK, keep as is
            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _build_system_prompt(
        self,
        diagram_type: DiagramType,
        style: DiagramStyle,
        provider: Optional[DiagramProvider],
        language: str,
        audience: Optional[str] = None,
        cheat_sheet: Optional[str] = None
    ) -> str:
        """Build the system prompt for diagram code generation."""

        # Style configurations
        style_config = {
            DiagramStyle.DARK: 'graph_attr={"bgcolor": "#1e1e1e", "fontcolor": "white"}',
            DiagramStyle.LIGHT: 'graph_attr={"bgcolor": "#ffffff", "fontcolor": "black"}',
            DiagramStyle.NEUTRAL: 'graph_attr={"bgcolor": "#f5f5f5", "fontcolor": "#333333"}',
            DiagramStyle.COLORFUL: 'graph_attr={"bgcolor": "#1a1a2e", "fontcolor": "white"}',
        }

        # Audience-specific complexity instructions
        audience_instructions = self._get_audience_instructions(audience)

        # Use provided cheat sheet or default imports
        imports_section = cheat_sheet if cheat_sheet else self._get_default_imports()

        return f"""You are an expert at creating professional architecture diagrams using the Python 'diagrams' library.

Generate ONLY valid Python code that uses the 'diagrams' library. The code must:
1. Be syntactically correct Python that can execute without errors
2. Use ONLY imports from the provided list below - DO NOT invent or guess imports
3. Create clear, professional diagrams with meaningful labels
4. Use appropriate icons from the correct provider modules
5. Include proper clustering/grouping where logical
6. Have clear connection flows with Edge labels where helpful

{audience_instructions}

CRITICAL RULES:
- Output ONLY Python code, no explanations
- Labels should be in {language}
- Use subgraphs/Clusters to group related components
- Always set show=False and filename parameter
- Use descriptive variable names
- NEVER import modules not in the list below - use alternatives or generic icons instead

COMMON IMPORT MISTAKES TO AVOID:
- Fluentd is in diagrams.onprem.AGGREGATOR, NOT in logging!
- Redis is in diagrams.onprem.INMEMORY, NOT in database!
- Elasticsearch/Kibana/Logstash are in diagrams.ELASTIC.elasticsearch, NOT in onprem!
- GenericSQL: use "from diagrams.generic.database import SQL as GenericSQL" (import SQL, alias it)
- Always check the import list below before using any icon
- COPY IMPORTS EXACTLY as shown - do not guess or modify them!

VARIABLE NAMING - CASE SENSITIVE:
- Use EXACT class names: PostgreSQL (not postgresql), MySQL (not mysql), MongoDB (not mongodb)
- Python is case-sensitive! "postgresql" and "PostgreSQL" are different variables
- Define variables BEFORE using them in connections

EDGE SYNTAX - CRITICAL:
- Simple connection: node1 >> node2
- With label: node1 >> Edge(label="text") >> node2
- NEVER chain multiple Edges: node1 >> Edge() >> node2 >> Edge() >> node3  # WRONG!
- Instead, split into multiple lines:
    node1 >> Edge(label="step 1") >> node2
    node2 >> Edge(label="step 2") >> node3
- For lists: node >> [node2, node3] is OK (one node to many)
- FORBIDDEN: [list] >> [list]  # TypeError! Cannot connect list to list!
- FORBIDDEN: [node1, node2] >> [node3, node4]  # WRONG!
- If you need many-to-many, use individual connections or a hub node
- Keep connection chains SHORT (max 3 nodes per line)

{imports_section}
```python
# AWS
from diagrams.aws.compute import EC2, Lambda, ECS, EKS, Fargate, Batch
from diagrams.aws.database import RDS, Aurora, Dynamodb, ElastiCache, Redshift
from diagrams.aws.network import APIGateway, CloudFront, ELB, ALB, NLB, Route53, VPC
from diagrams.aws.storage import S3, EBS, EFS
from diagrams.aws.integration import SQS, SNS, Eventbridge, StepFunctions
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
from diagrams.onprem.database import PostgreSQL, Mysql, MongoDB, Cassandra, Couchdb, Mariadb
from diagrams.onprem.inmemory import Redis, Memcached  # Redis is here, NOT in database!
from diagrams.onprem.network import Nginx, HAProxy, Traefik, Kong
from diagrams.onprem.queue import Kafka, RabbitMQ, Celery
from diagrams.onprem.container import Docker
from diagrams.onprem.ci import Jenkins, GitlabCI, GithubActions, CircleCI
from diagrams.onprem.monitoring import Prometheus, Grafana, Datadog
from diagrams.onprem.logging import Fluentbit, Loki, Graylog, SyslogNg
from diagrams.onprem.aggregator import Fluentd, Vector
from diagrams.onprem.mlops import Mlflow, Kubeflow

# Elastic Stack (Elasticsearch, Kibana, Logstash)
from diagrams.elastic.elasticsearch import Elasticsearch, Kibana, Logstash, Beats

# Generic
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL as GenericSQL
from diagrams.generic.network import Firewall, Router, Switch
from diagrams.generic.os import LinuxGeneral, Windows
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

    def _get_audience_instructions(self, audience: Optional[str]) -> str:
        """Get complexity instructions based on target audience."""
        if audience == "beginner":
            return """
COMPLEXITY LEVEL: BEGINNER
- Keep it SIMPLE: Maximum 5-7 nodes total
- Group related items in generic Clusters with clear labels
- Do NOT show networking details (no VPCs, subnets, load balancers)
- Use high-level concepts: "Database" not "PostgreSQL Primary + Read Replica"
- Focus on the WHAT, not the HOW
- Large, readable labels (short names)
- Minimal arrows - show main data flow only
"""
        elif audience == "executive":
            return """
COMPLEXITY LEVEL: EXECUTIVE/BUSINESS
- Focus on VALUE FLOW and system boundaries
- Maximum 6-8 nodes - keep it scannable
- Show: Users -> System -> Value/Output
- Hide implementation details (no queues, caches, internal DBs)
- Use business terms, not tech jargon
- Emphasize external integrations and data sources
- Show costs/billing boundaries if relevant
"""
        else:  # senior (default)
            return """
COMPLEXITY LEVEL: SENIOR/EXPERT
- Be DETAILED: 10-15 nodes is acceptable
- Use Clusters for VPCs, Subnets, Kubernetes namespaces
- Show caching layers, load balancers, message queues
- Include proper data flow directions with Edge labels
- Show redundancy patterns (primary/replica, multi-AZ)
- Use specific service names (not generic)
- Include monitoring and logging components if relevant
"""

    def _get_default_imports(self) -> str:
        """Get default imports cheat sheet if none provided."""
        return """AVAILABLE PROVIDERS AND IMPORTS:
```python
# AWS
from diagrams.aws.compute import EC2, Lambda, ECS, EKS, Fargate, Batch
from diagrams.aws.database import RDS, Aurora, Dynamodb, ElastiCache, Redshift
from diagrams.aws.network import APIGateway, CloudFront, ELB, ALB, NLB, Route53, VPC
from diagrams.aws.storage import S3, EBS, EFS
from diagrams.aws.integration import SQS, SNS, Eventbridge, StepFunctions
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
from diagrams.onprem.database import PostgreSQL, Mysql, MongoDB, Cassandra, Couchdb, Mariadb
from diagrams.onprem.inmemory import Redis, Memcached  # Redis is here, NOT in database!
from diagrams.onprem.network import Nginx, HAProxy, Traefik, Kong
from diagrams.onprem.queue import Kafka, RabbitMQ, Celery
from diagrams.onprem.container import Docker
from diagrams.onprem.ci import Jenkins, GitlabCI, GithubActions, CircleCI
from diagrams.onprem.monitoring import Prometheus, Grafana, Datadog
from diagrams.onprem.logging import Fluentbit, Loki, Graylog, SyslogNg
from diagrams.onprem.aggregator import Fluentd, Vector
from diagrams.onprem.mlops import Mlflow, Kubeflow

# Elastic Stack (Elasticsearch, Kibana, Logstash)
from diagrams.elastic.elasticsearch import Elasticsearch, Kibana, Logstash, Beats
from diagrams.onprem.client import User, Users, Client

# Generic
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL as GenericSQL
from diagrams.generic.network import Firewall, Router, Switch
from diagrams.generic.os import LinuxGeneral, Windows
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
```"""

    async def _extract_coordinates_from_dot(self, dot_path: str) -> Optional[DiagramCoordinates]:
        """
        Extract node/edge coordinates from DOT file using Graphviz's JSON output.

        Uses `dot -Tjson` to get precise coordinates calculated by Graphviz's
        layout engine (30+ years of R&D in graph layout algorithms).

        Args:
            dot_path: Path to the .dot file

        Returns:
            DiagramCoordinates with node positions for camera animations
        """
        try:
            # Run dot -Tjson to get coordinates
            result = await asyncio.create_subprocess_exec(
                'dot', '-Tjson', dot_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                print(f"[DIAGRAMS] dot -Tjson failed: {stderr.decode()}", flush=True)
                return None

            # Parse JSON output
            graph_data = json.loads(stdout.decode())

            nodes = []
            edges = []

            # Extract graph bounding box
            graph_bbox = None
            graph_width = 0.0
            graph_height = 0.0

            if 'bb' in graph_data:
                # bb format: "llx,lly,urx,ury"
                bb_parts = graph_data['bb'].split(',')
                if len(bb_parts) == 4:
                    graph_bbox = [float(x) for x in bb_parts]
                    graph_width = graph_bbox[2] - graph_bbox[0]
                    graph_height = graph_bbox[3] - graph_bbox[1]

            # Extract node coordinates
            for obj in graph_data.get('objects', []):
                if 'name' not in obj:
                    continue

                # Skip cluster/subgraph nodes
                if obj.get('name', '').startswith('cluster_'):
                    continue

                # Parse position "x,y"
                pos = obj.get('pos', '0,0')
                pos_parts = pos.split(',')
                x = float(pos_parts[0]) if pos_parts else 0.0
                y = float(pos_parts[1]) if len(pos_parts) > 1 else 0.0

                # Parse dimensions
                width = float(obj.get('width', 0))
                height = float(obj.get('height', 0))

                # Parse bounding box if available
                node_bbox = None
                if 'bb' in obj:
                    bb_parts = obj['bb'].split(',')
                    if len(bb_parts) == 4:
                        node_bbox = [float(b) for b in bb_parts]

                node = NodeCoordinate(
                    name=obj.get('name', ''),
                    label=obj.get('label', obj.get('name', '')),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    bbox=node_bbox,
                    # Center coordinates (same as pos for nodes)
                    center_x=x,
                    center_y=y
                )
                nodes.append(node)

            # Extract edge coordinates
            for edge_obj in graph_data.get('edges', []):
                source = str(edge_obj.get('tail', edge_obj.get('head', '')))
                target = str(edge_obj.get('head', edge_obj.get('tail', '')))

                # Parse spline points
                points = []
                pos = edge_obj.get('pos', '')
                if pos:
                    # Format: "e,x,y x1,y1 x2,y2 ..." or "s,x,y ..."
                    # Remove endpoint markers
                    pos_clean = pos
                    for marker in ['e,', 's,']:
                        if pos_clean.startswith(marker):
                            # Skip the marker and its coordinates
                            parts = pos_clean.split(' ', 1)
                            if len(parts) > 1:
                                pos_clean = parts[1]
                            break

                    for point_str in pos_clean.split(' '):
                        point_parts = point_str.split(',')
                        if len(point_parts) >= 2:
                            try:
                                points.append([float(point_parts[0]), float(point_parts[1])])
                            except ValueError:
                                pass

                edge = EdgeCoordinate(
                    source=source,
                    target=target,
                    label=edge_obj.get('label'),
                    points=points
                )
                edges.append(edge)

            print(f"[DIAGRAMS] Extracted coordinates: {len(nodes)} nodes, {len(edges)} edges", flush=True)

            return DiagramCoordinates(
                nodes=nodes,
                edges=edges,
                graph_bbox=graph_bbox,
                graph_width=graph_width,
                graph_height=graph_height,
                dpi=96  # Default Graphviz DPI
            )

        except json.JSONDecodeError as e:
            print(f"[DIAGRAMS] Failed to parse Graphviz JSON: {e}", flush=True)
            return None
        except Exception as e:
            print(f"[DIAGRAMS] Error extracting coordinates: {e}", flush=True)
            return None

    async def render(
        self,
        code: str,
        filename: Optional[str] = None,
        format: RenderFormat = RenderFormat.PNG,
        extract_coordinates: bool = True
    ) -> DiagramResult:
        """
        Execute the generated Python code to render the diagram.

        SECURITY: Code is validated via AST analysis before execution.
        Only 'diagrams' library imports are allowed. Dangerous functions
        (exec, eval, open, os.*, etc.) are blocked.

        NEW: Uses dot -Tjson to extract node coordinates for camera animations.

        Args:
            code: Python code using the diagrams library
            filename: Output filename (without extension)
            format: Output format (PNG recommended)
            extract_coordinates: Whether to extract node coordinates via dot -Tjson

        Returns:
            DiagramResult with file path, metadata, and optionally coordinates
        """
        start_time = time.time()
        file_id = filename or f"diagram_{uuid.uuid4().hex[:8]}"

        # ============================================
        # SECURITY VALIDATION - MUST PASS BEFORE EXECUTION
        # ============================================
        is_safe, security_error = CodeSecurityValidator.validate(code)
        if not is_safe:
            print(f"[DIAGRAMS] SECURITY BLOCKED: {security_error}", flush=True)
            print(f"[DIAGRAMS] Rejected code:\n{code[:500]}...", flush=True)
            return DiagramResult(
                success=False,
                diagram_type=DiagramType.ARCHITECTURE,
                generation_time_ms=int((time.time() - start_time) * 1000),
                error=f"Security validation failed: {security_error}"
            )

        # Log security report for monitoring
        security_report = CodeSecurityValidator.get_security_report(code)
        print(f"[DIAGRAMS] Security OK - {security_report['imports_count']} imports validated", flush=True)

        coordinates = None

        try:
            # ============================================
            # STEP 1: Generate DOT file first (for coordinates)
            # ============================================
            if extract_coordinates:
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.py',
                    delete=False,
                    dir=str(self.output_dir)
                ) as f:
                    # Generate DOT format for coordinate extraction
                    dot_code = self._inject_filename(code, file_id, outformat="dot")
                    f.write(dot_code)
                    temp_dot_py_path = f.name

                # Execute to generate .dot file
                result = await asyncio.create_subprocess_exec(
                    'python', temp_dot_py_path,
                    cwd=str(self.output_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await result.communicate()
                os.unlink(temp_dot_py_path)

                # Extract coordinates from DOT file
                dot_path = self.output_dir / f"{file_id}.dot"
                if dot_path.exists():
                    coordinates = await self._extract_coordinates_from_dot(str(dot_path))
                    # Clean up DOT file (we'll regenerate PNG directly)
                    # Keep it for debugging: os.unlink(dot_path)

            # ============================================
            # STEP 2: Generate PNG file
            # ============================================
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                dir=str(self.output_dir)
            ) as f:
                # Generate PNG format
                modified_code = self._inject_filename(code, file_id, outformat="png")
                f.write(modified_code)
                temp_py_path = f.name

            # Execute the Python script (safe - validated above)
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
                    "code_length": len(code),
                    "coordinates_extracted": coordinates is not None
                },
                coordinates=coordinates  # NEW: Include coordinates for camera animations
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

    def _inject_filename(self, code: str, filename: str, outformat: str = "png") -> str:
        """
        Ensure the filename and outformat parameters are set correctly in the diagram code.

        Args:
            code: Python code using the diagrams library
            filename: Output filename (without extension)
            outformat: Output format - "png" for image, "dot" for DOT file (coordinate extraction)

        Returns:
            Modified code with correct filename and outformat
        """
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

            # Set outformat for DOT extraction or PNG generation
            if 'outformat=' in diagram_call:
                diagram_call = re.sub(
                    r'outformat\s*=\s*["\'][^"\']*["\']',
                    f'outformat="{outformat}"',
                    diagram_call
                )
            else:
                diagram_call = diagram_call.rstrip(')')
                diagram_call += f', outformat="{outformat}")'

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
        max_retries: int = 2,
        audience: Optional[str] = None,
        cheat_sheet: Optional[str] = None,
        extract_coordinates: bool = True
    ) -> DiagramResult:
        """
        Generate diagram code from description and render to image.
        Includes retry logic for code generation failures.

        NEW: Extracts node coordinates via Graphviz's dot -Tjson for camera animations.

        Args:
            description: What the diagram should show
            diagram_type: Type of diagram
            style: Visual style
            provider: Primary cloud/tech provider for icons
            format: Output format
            context: Additional context
            language: Language for labels
            max_retries: Number of retries on failure
            audience: Target audience (beginner, senior, executive)
            cheat_sheet: Valid imports to prevent LLM hallucinations
            extract_coordinates: Extract node coordinates for camera animations (default: True)

        Returns:
            DiagramResult with image and optionally coordinates for camera zoom/pan
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Generate the code with audience and cheat_sheet
                code = await self.generate_diagram_code(
                    description=description,
                    diagram_type=diagram_type,
                    style=style,
                    provider=provider,
                    context=context,
                    language=language,
                    audience=audience,
                    cheat_sheet=cheat_sheet
                )

                print(f"[DIAGRAMS] Generated code (attempt {attempt + 1}):\n{code[:500]}...", flush=True)

                # Render the diagram with coordinate extraction
                result = await self.render(
                    code,
                    format=format,
                    extract_coordinates=extract_coordinates
                )

                if result.success:
                    if result.coordinates:
                        print(f"[DIAGRAMS] Coordinates available: {len(result.coordinates.nodes)} nodes for camera animations", flush=True)
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

    async def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python diagrams code for syntax AND security.

        Performs:
        1. AST parsing (syntax check)
        2. Security validation (import whitelist, blocked functions)

        Returns:
            Tuple of (is_valid, error_message)
        """
        return CodeSecurityValidator.validate(code)


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
