"""
Kubernetes Sandbox Environment

Creates ephemeral Kubernetes clusters for K8s exercises.
Uses kind (Kubernetes in Docker) for lightweight clusters.
"""

import asyncio
import os
import subprocess
import tempfile
import shutil
import yaml
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from models.sandbox_models import (
    SandboxConfig,
    SandboxState,
    SandboxStatus,
    SandboxType,
    ExecutionRequest,
    ExecutionResult,
    K8sClusterState,
    K8sResourceState,
    K8sValidationCheck,
    ResourceLimits,
)
from .base_sandbox import BaseSandbox


class KubernetesSandbox(BaseSandbox):
    """
    Kubernetes sandbox using kind (Kubernetes in Docker).

    Creates ephemeral clusters for:
    - Deployment exercises
    - Service configuration
    - Pod management
    - ConfigMaps/Secrets
    - RBAC exercises
    """

    # kind cluster configuration template
    KIND_CONFIG_TEMPLATE = """
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 30000
    hostPort: {host_port}
    protocol: TCP
"""

    def __init__(self, config: Optional[SandboxConfig] = None):
        super().__init__(config)
        self.cluster_name = f"sandbox-{self.sandbox_id[:8]}"
        self.kubeconfig_path = None
        self.workspace_dir = None
        self.cluster_state = None

    def _get_sandbox_type(self) -> str:
        return SandboxType.KUBERNETES.value

    async def create(self) -> SandboxState:
        """Create a kind cluster"""
        try:
            self.state.status = SandboxStatus.CREATING

            # Create workspace directory
            self.workspace_dir = tempfile.mkdtemp(prefix="k8s_sandbox_")
            self.state.workspace_path = self.workspace_dir

            # Create kind config file
            host_port = 30000 + (hash(self.sandbox_id) % 1000)
            kind_config = self.KIND_CONFIG_TEMPLATE.format(host_port=host_port)
            kind_config_path = os.path.join(self.workspace_dir, "kind-config.yaml")
            with open(kind_config_path, "w") as f:
                f.write(kind_config)

            # Create the cluster
            print(f"[K8S_SANDBOX] Creating kind cluster: {self.cluster_name}")
            result = await self._run_command(
                f"kind create cluster --name {self.cluster_name} --config {kind_config_path} --wait 60s"
            )

            if result["exit_code"] != 0:
                raise RuntimeError(f"Failed to create cluster: {result['stderr']}")

            # Get kubeconfig
            self.kubeconfig_path = os.path.join(self.workspace_dir, "kubeconfig")
            await self._run_command(
                f"kind get kubeconfig --name {self.cluster_name} > {self.kubeconfig_path}"
            )

            # Apply preset resources if configured
            if self.config and self.config.preset_resources:
                await self._apply_preset_resources(self.config.preset_resources)

            # Initialize cluster state
            self.cluster_state = K8sClusterState(
                cluster_name=self.cluster_name,
                kubeconfig_path=self.kubeconfig_path,
                namespaces=["default", "kube-system"],
            )
            self.state.cluster_name = self.cluster_name
            self.state.status = SandboxStatus.READY

            print(f"[K8S_SANDBOX] Cluster ready: {self.cluster_name}")
            return self.state

        except Exception as e:
            self.state.status = SandboxStatus.ERROR
            self.state.error_message = str(e)
            print(f"[K8S_SANDBOX] Create error: {e}")
            # Try to cleanup
            await self.destroy()
            raise

    async def destroy(self) -> bool:
        """Delete the kind cluster"""
        try:
            print(f"[K8S_SANDBOX] Destroying cluster: {self.cluster_name}")

            # Delete cluster
            await self._run_command(f"kind delete cluster --name {self.cluster_name}")

            # Cleanup workspace
            if self.workspace_dir and os.path.exists(self.workspace_dir):
                shutil.rmtree(self.workspace_dir, ignore_errors=True)

            self.state.status = SandboxStatus.DESTROYED
            print(f"[K8S_SANDBOX] Destroyed: {self.cluster_name}")
            return True

        except Exception as e:
            print(f"[K8S_SANDBOX] Destroy error: {e}")
            return False

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute kubectl commands or apply manifests"""
        start_time = datetime.utcnow()
        result = ExecutionResult(
            sandbox_id=self.sandbox_id,
            started_at=start_time,
        )

        try:
            self.state.status = SandboxStatus.RUNNING

            # Write any files (manifests) first
            for file_path, content in request.files.items():
                await self.write_file(file_path, content)

            # Determine what to execute
            if request.command:
                # Run kubectl command
                cmd = self._prepare_kubectl_command(request.command)
            elif request.code:
                # Apply YAML manifest
                manifest_path = os.path.join(self.workspace_dir, "manifest.yaml")
                with open(manifest_path, "w") as f:
                    f.write(request.code)
                cmd = self._prepare_kubectl_command(f"apply -f {manifest_path}")
            else:
                result.error_message = "No command or manifest provided"
                result.success = False
                return result

            # Execute with timeout
            try:
                exec_result = await asyncio.wait_for(
                    self._run_command(cmd),
                    timeout=request.timeout_seconds,
                )

                result.exit_code = exec_result["exit_code"]
                result.stdout = exec_result["stdout"]
                result.stderr = exec_result["stderr"]
                result.combined_output = result.stdout + result.stderr
                result.success = result.exit_code == 0

            except asyncio.TimeoutError:
                result.error_type = "timeout"
                result.error_message = f"Execution timed out after {request.timeout_seconds}s"
                result.success = False

            result.completed_at = datetime.utcnow()
            result.execution_time_ms = int(
                (result.completed_at - start_time).total_seconds() * 1000
            )

            self.state.status = SandboxStatus.READY
            return result

        except Exception as e:
            result.success = False
            result.error_type = "execution_error"
            result.error_message = str(e)
            self.state.status = SandboxStatus.ERROR
            return result

    async def get_state(self) -> SandboxState:
        """Get current cluster state"""
        if not self.kubeconfig_path:
            return self.state

        try:
            # Get namespaces
            ns_result = await self._run_kubectl("get namespaces -o json")
            if ns_result["exit_code"] == 0:
                ns_data = json.loads(ns_result["stdout"])
                self.cluster_state.namespaces = [
                    item["metadata"]["name"] for item in ns_data.get("items", [])
                ]

            # Get all resources in default namespace
            resources_result = await self._run_kubectl(
                "get all -n default -o json"
            )
            if resources_result["exit_code"] == 0:
                resources_data = json.loads(resources_result["stdout"])
                self.cluster_state.resources = [
                    K8sResourceState(
                        kind=item["kind"],
                        name=item["metadata"]["name"],
                        namespace=item["metadata"].get("namespace", "default"),
                        status=self._extract_status(item),
                        ready=self._is_resource_ready(item),
                    )
                    for item in resources_data.get("items", [])
                ]

        except Exception as e:
            print(f"[K8S_SANDBOX] Error getting state: {e}")

        return self.state

    async def validate_k8s_check(self, check: K8sValidationCheck) -> Dict[str, Any]:
        """Validate a specific Kubernetes check"""
        result = {
            "name": check.name,
            "passed": False,
            "message": "",
            "points": 0,
        }

        try:
            # Get the resource
            cmd = f"get {check.resource_kind}"
            if check.resource_name:
                cmd += f" {check.resource_name}"
            if check.label_selector:
                cmd += f" -l {check.label_selector}"
            cmd += f" -n {check.namespace} -o json"

            query_result = await self._run_kubectl(cmd)

            if query_result["exit_code"] != 0:
                if check.must_exist:
                    result["message"] = f"Resource not found: {check.resource_kind}"
                    return result
                else:
                    result["passed"] = True
                    result["message"] = "Resource correctly does not exist"
                    result["points"] = check.points
                    return result

            resource_data = json.loads(query_result["stdout"])

            # Handle list vs single resource
            if resource_data.get("kind", "").endswith("List"):
                items = resource_data.get("items", [])
                if not items:
                    if check.must_exist:
                        result["message"] = "No matching resources found"
                        return result
                resource_data = items[0]  # Check first matching resource

            # Check if must exist
            if check.must_exist:
                result["message"] = "Resource exists"

            # Check if must be ready
            if check.must_be_ready:
                if not self._is_resource_ready(resource_data):
                    result["message"] = "Resource exists but is not ready"
                    return result

            # Check field values
            for json_path, expected_value in check.field_checks.items():
                actual_value = self._get_json_path_value(resource_data, json_path)
                if actual_value != expected_value:
                    result["message"] = f"Field {json_path}: expected {expected_value}, got {actual_value}"
                    return result

            result["passed"] = True
            result["points"] = check.points
            if not result["message"]:
                result["message"] = "All checks passed"

        except Exception as e:
            result["message"] = f"Validation error: {str(e)}"

        return result

    async def write_file(self, path: str, content: str) -> bool:
        """Write a file to the workspace"""
        if not self.workspace_dir:
            return False

        if path.startswith("/"):
            path = path[1:]

        full_path = os.path.join(self.workspace_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w") as f:
            f.write(content)

        self.state.files_created.append(path)
        return True

    async def _run_command(self, command: str) -> Dict[str, Any]:
        """Run a shell command"""
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "KUBECONFIG": self.kubeconfig_path} if self.kubeconfig_path else None,
        )
        stdout, stderr = await process.communicate()

        return {
            "exit_code": process.returncode,
            "stdout": stdout.decode("utf-8"),
            "stderr": stderr.decode("utf-8"),
        }

    async def _run_kubectl(self, args: str) -> Dict[str, Any]:
        """Run a kubectl command"""
        return await self._run_command(self._prepare_kubectl_command(args))

    def _prepare_kubectl_command(self, command: str) -> str:
        """Prepare kubectl command with kubeconfig"""
        if command.startswith("kubectl"):
            base_cmd = command
        else:
            base_cmd = f"kubectl {command}"

        if self.kubeconfig_path:
            return f"KUBECONFIG={self.kubeconfig_path} {base_cmd}"
        return base_cmd

    async def _apply_preset_resources(self, resources: Dict[str, Any]):
        """Apply preset Kubernetes resources"""
        if isinstance(resources, dict):
            # Single resource
            manifest = yaml.dump(resources)
        elif isinstance(resources, list):
            # Multiple resources
            manifest = yaml.dump_all(resources)
        else:
            return

        manifest_path = os.path.join(self.workspace_dir, "preset.yaml")
        with open(manifest_path, "w") as f:
            f.write(manifest)

        await self._run_kubectl(f"apply -f {manifest_path}")

    def _extract_status(self, resource: Dict[str, Any]) -> str:
        """Extract status from Kubernetes resource"""
        status = resource.get("status", {})

        # Try different status fields
        if "phase" in status:
            return status["phase"]
        if "conditions" in status:
            for cond in status["conditions"]:
                if cond.get("type") == "Ready" and cond.get("status") == "True":
                    return "Ready"
        return "Unknown"

    def _is_resource_ready(self, resource: Dict[str, Any]) -> bool:
        """Check if a Kubernetes resource is ready"""
        kind = resource.get("kind", "")
        status = resource.get("status", {})

        if kind == "Pod":
            return status.get("phase") == "Running"
        elif kind == "Deployment":
            ready = status.get("readyReplicas", 0)
            desired = resource.get("spec", {}).get("replicas", 1)
            return ready >= desired
        elif kind == "Service":
            return True  # Services are always "ready"
        elif kind in ["ConfigMap", "Secret"]:
            return True

        # Default: check conditions
        for cond in status.get("conditions", []):
            if cond.get("type") == "Ready":
                return cond.get("status") == "True"

        return False

    def _get_json_path_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation"""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                value = value[int(key)]
            else:
                return None
        return value
