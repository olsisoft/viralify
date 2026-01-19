"""
Docker Sandbox Environment

Executes code in isolated Docker containers for DevOps exercises.
"""

import asyncio
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import uuid

import docker
from docker.errors import ContainerError, ImageNotFound, APIError

from models.sandbox_models import (
    SandboxConfig,
    SandboxState,
    SandboxStatus,
    SandboxType,
    ExecutionRequest,
    ExecutionResult,
    DockerBuildResult,
    DockerRunConfig,
    ResourceLimits,
)
from .base_sandbox import BaseSandbox


class DockerSandbox(BaseSandbox):
    """
    Docker-based sandbox for executing DevOps exercises.

    Supports:
    - Running commands in containers
    - Building Dockerfiles
    - Multi-container setups
    - Resource limiting
    - Timeout enforcement
    """

    # Default images for different exercise types
    DEFAULT_IMAGES = {
        "python": "python:3.11-slim",
        "node": "node:20-slim",
        "bash": "debian:bookworm-slim",
        "docker": "docker:24-dind",
        "terraform": "hashicorp/terraform:latest",
        "ansible": "ansible/ansible-runner:latest",
        "go": "golang:1.21-alpine",
    }

    def __init__(self, config: Optional[SandboxConfig] = None):
        super().__init__(config)
        self.client = docker.from_env()
        self.container = None
        self.workspace_dir = None
        self._network = None

    def _get_sandbox_type(self) -> str:
        return SandboxType.DOCKER.value

    async def create(self) -> SandboxState:
        """Create the Docker sandbox environment"""
        try:
            self.state.status = SandboxStatus.CREATING

            # Create workspace directory
            self.workspace_dir = tempfile.mkdtemp(prefix="sandbox_")
            self.state.workspace_path = self.workspace_dir

            # Determine image
            image = self._get_image()

            # Pull image if needed
            try:
                self.client.images.get(image)
            except ImageNotFound:
                print(f"[DOCKER_SANDBOX] Pulling image: {image}")
                self.client.images.pull(image)

            # Create network for isolation
            network_name = f"sandbox_{self.sandbox_id[:8]}"
            try:
                self._network = self.client.networks.create(
                    network_name,
                    driver="bridge",
                    internal=not self.config.resource_limits.network_enabled if self.config else False,
                )
            except APIError:
                # Network might already exist
                self._network = self.client.networks.get(network_name)

            # Create container
            resource_config = self._get_resource_config()

            self.container = self.client.containers.create(
                image=image,
                command="sleep infinity",  # Keep container running
                detach=True,
                working_dir="/workspace",
                volumes={
                    self.workspace_dir: {"bind": "/workspace", "mode": "rw"}
                },
                network=self._network.name if self._network else None,
                environment=self.config.environment_vars if self.config else {},
                **resource_config,
            )

            # Start container
            self.container.start()

            # Run setup commands
            if self.config and self.config.setup_commands:
                for cmd in self.config.setup_commands:
                    await self._exec_in_container(cmd)

            # Mount preset files
            if self.config and self.config.mount_files:
                for file_path, content in self.config.mount_files.items():
                    await self.write_file(file_path, content)

            self.state.status = SandboxStatus.READY
            self.state.container_id = self.container.id
            self.state.internal_ip = self._get_container_ip()

            print(f"[DOCKER_SANDBOX] Created: {self.sandbox_id}")
            return self.state

        except Exception as e:
            self.state.status = SandboxStatus.ERROR
            self.state.error_message = str(e)
            print(f"[DOCKER_SANDBOX] Create error: {e}")
            raise

    async def destroy(self) -> bool:
        """Destroy the Docker sandbox"""
        try:
            if self.container:
                try:
                    self.container.stop(timeout=5)
                except:
                    pass
                try:
                    self.container.remove(force=True)
                except:
                    pass

            if self._network:
                try:
                    self._network.remove()
                except:
                    pass

            if self.workspace_dir and os.path.exists(self.workspace_dir):
                shutil.rmtree(self.workspace_dir, ignore_errors=True)

            self.state.status = SandboxStatus.DESTROYED
            print(f"[DOCKER_SANDBOX] Destroyed: {self.sandbox_id}")
            return True

        except Exception as e:
            print(f"[DOCKER_SANDBOX] Destroy error: {e}")
            return False

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute code or commands in the container"""
        start_time = datetime.utcnow()
        result = ExecutionResult(
            sandbox_id=self.sandbox_id,
            started_at=start_time,
        )

        try:
            self.state.status = SandboxStatus.RUNNING
            self.state.last_execution_at = start_time

            # Write files if provided
            for file_path, content in request.files.items():
                await self.write_file(file_path, content)

            # Determine what to execute
            if request.command:
                cmd = request.command
            elif request.code and request.language:
                cmd = self._build_code_command(request.code, request.language)
                # Write code to file first
                code_file = self._get_code_filename(request.language)
                await self.write_file(f"/workspace/{code_file}", request.code)
            else:
                result.error_message = "No command or code provided"
                result.success = False
                return result

            # Execute with timeout
            try:
                exec_result = await asyncio.wait_for(
                    self._exec_in_container(
                        cmd,
                        workdir=request.working_directory,
                        environment=request.environment_vars,
                    ),
                    timeout=request.timeout_seconds,
                )

                result.exit_code = exec_result["exit_code"]
                result.stdout = exec_result["stdout"]
                result.stderr = exec_result["stderr"]
                result.combined_output = exec_result["stdout"] + exec_result["stderr"]
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
            result.stderr = str(e)
            self.state.status = SandboxStatus.ERROR
            return result

    async def _exec_in_container(
        self,
        command: str,
        workdir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a command in the container"""
        if not self.container:
            raise RuntimeError("Container not created")

        # Refresh container state
        self.container.reload()

        exec_result = self.container.exec_run(
            cmd=["sh", "-c", command],
            workdir=workdir or "/workspace",
            environment=environment or {},
            demux=True,
        )

        stdout = exec_result.output[0].decode("utf-8") if exec_result.output[0] else ""
        stderr = exec_result.output[1].decode("utf-8") if exec_result.output[1] else ""

        return {
            "exit_code": exec_result.exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }

    async def write_file(self, path: str, content: str) -> bool:
        """Write a file to the container"""
        if not self.workspace_dir:
            return False

        # Ensure path is relative or handle absolute
        if path.startswith("/workspace/"):
            rel_path = path[11:]
        elif path.startswith("/"):
            rel_path = path[1:]
        else:
            rel_path = path

        full_path = os.path.join(self.workspace_dir, rel_path)

        # Create parent directories
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w") as f:
            f.write(content)

        self.state.files_created.append(path)
        return True

    async def read_file(self, path: str) -> str:
        """Read a file from the container"""
        if not self.workspace_dir:
            return ""

        if path.startswith("/workspace/"):
            rel_path = path[11:]
        elif path.startswith("/"):
            rel_path = path[1:]
        else:
            rel_path = path

        full_path = os.path.join(self.workspace_dir, rel_path)

        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return f.read()
        return ""

    async def list_files(self, path: str = "/workspace") -> List[str]:
        """List files in directory"""
        if not self.workspace_dir:
            return []

        if path.startswith("/workspace"):
            rel_path = path[10:].lstrip("/")
        else:
            rel_path = path.lstrip("/")

        target_dir = os.path.join(self.workspace_dir, rel_path)

        if os.path.exists(target_dir):
            return os.listdir(target_dir)
        return []

    async def get_state(self) -> SandboxState:
        """Get current sandbox state"""
        if self.container:
            try:
                self.container.reload()
                stats = self.container.stats(stream=False)

                # Update resource usage
                if "memory_stats" in stats:
                    usage = stats["memory_stats"].get("usage", 0)
                    self.state.memory_usage_mb = usage / (1024 * 1024)

                if "cpu_stats" in stats:
                    # Calculate CPU percentage
                    cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                               stats["precpu_stats"]["cpu_usage"]["total_usage"]
                    system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                                  stats["precpu_stats"]["system_cpu_usage"]
                    if system_delta > 0:
                        self.state.cpu_usage_percent = (cpu_delta / system_delta) * 100

            except Exception:
                pass

        return self.state

    async def build_dockerfile(self, dockerfile_content: str, tag: str = None) -> DockerBuildResult:
        """Build a Docker image from Dockerfile content"""
        result = DockerBuildResult(success=False)
        start_time = datetime.utcnow()

        try:
            # Write Dockerfile
            dockerfile_path = os.path.join(self.workspace_dir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)

            # Build image
            tag = tag or f"sandbox_{self.sandbox_id[:8]}:latest"

            image, build_logs = self.client.images.build(
                path=self.workspace_dir,
                tag=tag,
                rm=True,
                forcerm=True,
            )

            # Collect logs
            logs = []
            for log in build_logs:
                if "stream" in log:
                    logs.append(log["stream"])
                if "error" in log:
                    result.errors.append(log["error"])

            result.build_logs = "".join(logs)
            result.image_id = image.id
            result.image_tag = tag
            result.success = True

        except Exception as e:
            result.errors.append(str(e))

        result.build_time_seconds = (datetime.utcnow() - start_time).total_seconds()
        return result

    def _get_image(self) -> str:
        """Get the Docker image to use"""
        if self.config and self.config.base_image:
            return self.config.base_image

        # Default to bash image
        return self.DEFAULT_IMAGES.get("bash", "debian:bookworm-slim")

    def _get_resource_config(self) -> Dict[str, Any]:
        """Get Docker resource configuration"""
        limits = self.config.resource_limits if self.config else ResourceLimits()

        config = {
            "cpu_period": 100000,
            "cpu_quota": int(limits.cpu_cores * 100000),
            "mem_limit": f"{limits.memory_mb}m",
        }

        if not limits.network_enabled:
            config["network_disabled"] = True

        return config

    def _get_container_ip(self) -> Optional[str]:
        """Get the container's IP address"""
        if not self.container:
            return None

        try:
            self.container.reload()
            networks = self.container.attrs.get("NetworkSettings", {}).get("Networks", {})
            for network_info in networks.values():
                ip = network_info.get("IPAddress")
                if ip:
                    return ip
        except:
            pass
        return None

    def _build_code_command(self, code: str, language: str) -> str:
        """Build the command to execute code"""
        commands = {
            "python": "python /workspace/code.py",
            "bash": "bash /workspace/code.sh",
            "javascript": "node /workspace/code.js",
            "go": "go run /workspace/code.go",
        }
        return commands.get(language, f"cat /workspace/code.{language}")

    def _get_code_filename(self, language: str) -> str:
        """Get the filename for code"""
        extensions = {
            "python": "code.py",
            "bash": "code.sh",
            "javascript": "code.js",
            "go": "code.go",
            "dockerfile": "Dockerfile",
        }
        return extensions.get(language, f"code.{language}")
