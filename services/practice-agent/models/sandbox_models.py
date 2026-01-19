"""
Sandbox Models for Practice Agent

Defines models for sandbox environments where code is executed safely.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SandboxType(str, Enum):
    """Types of sandbox environments"""
    DOCKER = "docker"           # Docker container execution
    KUBERNETES = "kubernetes"   # K8s cluster (kind/k3d)
    TERRAFORM = "terraform"     # Terraform workspace
    PYTHON = "python"           # Python interpreter
    BASH = "bash"               # Bash shell
    DATABASE = "database"       # PostgreSQL/MySQL sandbox
    COMPOSE = "compose"         # Docker Compose environment
    ANSIBLE = "ansible"         # Ansible playbook execution


class SandboxStatus(str, Enum):
    """Status of a sandbox environment"""
    CREATING = "creating"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DESTROYED = "destroyed"


class ResourceLimits(BaseModel):
    """Resource limits for sandbox"""
    cpu_cores: float = Field(default=1.0, description="CPU cores limit")
    memory_mb: int = Field(default=512, description="Memory limit in MB")
    disk_mb: int = Field(default=1024, description="Disk space limit in MB")
    network_enabled: bool = Field(default=True, description="Network access allowed")
    timeout_seconds: int = Field(default=300, description="Max execution time")


class SandboxConfig(BaseModel):
    """Configuration for creating a sandbox"""
    sandbox_type: SandboxType

    # Docker-specific
    base_image: Optional[str] = Field(None, description="Docker image to use")
    dockerfile: Optional[str] = Field(None, description="Custom Dockerfile content")

    # Kubernetes-specific
    k8s_version: Optional[str] = Field(None, description="Kubernetes version")
    preset_resources: Optional[Dict[str, Any]] = Field(None, description="Pre-deployed K8s resources")

    # Terraform-specific
    terraform_version: Optional[str] = Field(None)
    provider_config: Optional[Dict[str, Any]] = Field(None)

    # Common settings
    environment_vars: Dict[str, str] = Field(default_factory=dict)
    mount_files: Dict[str, str] = Field(default_factory=dict, description="Files to mount: {path: content}")
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)

    # Pre-execution setup
    setup_commands: List[str] = Field(default_factory=list, description="Commands to run on setup")

    class Config:
        json_schema_extra = {
            "example": {
                "sandbox_type": "docker",
                "base_image": "python:3.11-slim",
                "environment_vars": {"DEBUG": "1"},
                "resource_limits": {"cpu_cores": 1, "memory_mb": 512}
            }
        }


class SandboxState(BaseModel):
    """Current state of a sandbox"""
    sandbox_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sandbox_type: SandboxType
    status: SandboxStatus = Field(default=SandboxStatus.CREATING)

    # Container/resource IDs
    container_id: Optional[str] = Field(None, description="Docker container ID")
    cluster_name: Optional[str] = Field(None, description="K8s cluster name")
    workspace_path: Optional[str] = Field(None, description="Workspace directory path")

    # State info
    current_directory: str = Field(default="/workspace")
    files_created: List[str] = Field(default_factory=list)
    processes_running: List[str] = Field(default_factory=list)

    # Resource usage
    cpu_usage_percent: float = Field(default=0.0)
    memory_usage_mb: float = Field(default=0.0)
    disk_usage_mb: float = Field(default=0.0)

    # Network info
    exposed_ports: Dict[int, int] = Field(default_factory=dict, description="container:host port mapping")
    internal_ip: Optional[str] = Field(None)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_execution_at: Optional[datetime] = Field(None)
    expires_at: Optional[datetime] = Field(None)

    # Error tracking
    error_message: Optional[str] = Field(None)
    error_count: int = Field(default=0)


class ExecutionRequest(BaseModel):
    """Request to execute code/commands in sandbox"""
    sandbox_id: str

    # What to execute
    command: Optional[str] = Field(None, description="Shell command to run")
    code: Optional[str] = Field(None, description="Code to execute")
    language: Optional[str] = Field(None, description="Language for code execution")

    # Files to create/update before execution
    files: Dict[str, str] = Field(default_factory=dict, description="{filename: content}")

    # Execution options
    working_directory: Optional[str] = Field(None)
    environment_vars: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=30)

    # Stream options
    stream_output: bool = Field(default=False, description="Stream output via WebSocket")
    capture_stderr: bool = Field(default=True)


class ExecutionResult(BaseModel):
    """Result from code execution"""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sandbox_id: str

    # Status
    success: bool = Field(default=False)
    exit_code: int = Field(default=-1)

    # Output
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    combined_output: str = Field(default="")

    # Files created/modified
    files_changed: List[str] = Field(default_factory=list)

    # Timing
    execution_time_ms: int = Field(default=0)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)

    # Resource usage during execution
    peak_memory_mb: float = Field(default=0.0)
    cpu_time_ms: int = Field(default=0)

    # Error details
    error_type: Optional[str] = Field(None, description="timeout, oom, permission, etc.")
    error_message: Optional[str] = Field(None)


class SandboxResult(BaseModel):
    """Complete result from sandbox operation including validation"""
    execution: ExecutionResult

    # Validation results
    validation_passed: bool = Field(default=False)
    checks_results: Dict[str, bool] = Field(default_factory=dict, description="Check name -> passed")
    score: int = Field(default=0)
    max_score: int = Field(default=100)

    # Detailed feedback
    feedback_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {check, passed, message, points}"
    )

    # State after execution
    final_state: Optional[SandboxState] = Field(None)


# Docker-specific models

class DockerBuildResult(BaseModel):
    """Result from building a Docker image"""
    success: bool
    image_id: Optional[str] = None
    image_tag: Optional[str] = None
    build_logs: str = ""
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    build_time_seconds: float = 0.0


class DockerRunConfig(BaseModel):
    """Configuration for running a Docker container"""
    image: str
    command: Optional[str] = None
    entrypoint: Optional[str] = None
    environment: Dict[str, str] = Field(default_factory=dict)
    ports: Dict[str, int] = Field(default_factory=dict, description="container_port: host_port")
    volumes: Dict[str, str] = Field(default_factory=dict, description="host_path: container_path")
    network: Optional[str] = None
    detach: bool = False
    remove: bool = True
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)


# Kubernetes-specific models

class K8sResourceState(BaseModel):
    """State of a Kubernetes resource"""
    kind: str
    name: str
    namespace: str = "default"
    status: str
    ready: bool = False
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list)


class K8sClusterState(BaseModel):
    """State of a Kubernetes cluster"""
    cluster_name: str
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    namespaces: List[str] = Field(default_factory=list)
    resources: List[K8sResourceState] = Field(default_factory=list)
    kubeconfig_path: str
    api_server_url: Optional[str] = None


class K8sValidationCheck(BaseModel):
    """Validation check for Kubernetes exercises"""
    name: str
    description: str

    # What to check
    resource_kind: str = Field(..., description="Pod, Deployment, Service, etc.")
    resource_name: Optional[str] = Field(None, description="Specific resource name")
    namespace: str = Field(default="default")

    # Conditions
    must_exist: bool = Field(default=True)
    must_be_ready: bool = Field(default=False)
    field_checks: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSONPath -> expected value"
    )
    label_selector: Optional[str] = Field(None)

    # Points
    points: int = Field(default=10)


# Terraform-specific models

class TerraformState(BaseModel):
    """State of a Terraform workspace"""
    workspace_name: str
    workspace_path: str

    # State info
    resources_managed: int = Field(default=0)
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    outputs: Dict[str, Any] = Field(default_factory=dict)

    # Last operations
    last_plan: Optional[str] = Field(None)
    last_apply: Optional[str] = Field(None)

    # Status
    initialized: bool = Field(default=False)
    has_changes: bool = Field(default=False)


class TerraformValidationCheck(BaseModel):
    """Validation check for Terraform exercises"""
    name: str
    description: str

    # What to check
    check_type: str = Field(..., description="resource, output, state, plan")
    resource_type: Optional[str] = Field(None, description="aws_instance, azurerm_virtual_machine, etc.")
    resource_name: Optional[str] = Field(None)

    # Conditions
    must_exist: bool = Field(default=True)
    attribute_checks: Dict[str, Any] = Field(
        default_factory=dict,
        description="attribute_path -> expected value"
    )
    output_checks: Dict[str, Any] = Field(default_factory=dict)

    # Points
    points: int = Field(default=10)
