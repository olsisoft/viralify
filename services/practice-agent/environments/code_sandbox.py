"""
Code Sandbox Environment

Lightweight sandbox for executing code snippets (Python, Bash, etc.)
without full Docker overhead.
"""

import asyncio
import os
import sys
import tempfile
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional
import subprocess

from models.sandbox_models import (
    SandboxConfig,
    SandboxState,
    SandboxStatus,
    SandboxType,
    ExecutionRequest,
    ExecutionResult,
    ResourceLimits,
)
from .base_sandbox import BaseSandbox


class CodeSandbox(BaseSandbox):
    """
    Lightweight code execution sandbox.

    Uses subprocess with resource limits for quick code execution.
    Good for simple coding exercises that don't need containers.
    """

    SUPPORTED_LANGUAGES = {
        "python": {
            "extension": ".py",
            "command": ["python3", "{file}"],
            "compile": None,
        },
        "bash": {
            "extension": ".sh",
            "command": ["bash", "{file}"],
            "compile": None,
        },
        "javascript": {
            "extension": ".js",
            "command": ["node", "{file}"],
            "compile": None,
        },
        "go": {
            "extension": ".go",
            "command": ["go", "run", "{file}"],
            "compile": None,
        },
        "rust": {
            "extension": ".rs",
            "command": ["./{output}"],
            "compile": ["rustc", "{file}", "-o", "{output}"],
        },
    }

    def __init__(self, config: Optional[SandboxConfig] = None):
        super().__init__(config)
        self.workspace_dir = None
        self.language = "python"  # Default

    def _get_sandbox_type(self) -> str:
        return SandboxType.PYTHON.value

    async def create(self) -> SandboxState:
        """Create the code sandbox (workspace directory)"""
        try:
            self.state.status = SandboxStatus.CREATING

            # Create workspace directory
            self.workspace_dir = tempfile.mkdtemp(prefix="code_sandbox_")
            self.state.workspace_path = self.workspace_dir
            self.state.current_directory = self.workspace_dir

            # Write any preset files
            if self.config and self.config.mount_files:
                for file_path, content in self.config.mount_files.items():
                    await self.write_file(file_path, content)

            self.state.status = SandboxStatus.READY
            print(f"[CODE_SANDBOX] Created: {self.sandbox_id}")
            return self.state

        except Exception as e:
            self.state.status = SandboxStatus.ERROR
            self.state.error_message = str(e)
            raise

    async def destroy(self) -> bool:
        """Clean up the sandbox"""
        try:
            if self.workspace_dir and os.path.exists(self.workspace_dir):
                shutil.rmtree(self.workspace_dir, ignore_errors=True)

            self.state.status = SandboxStatus.DESTROYED
            print(f"[CODE_SANDBOX] Destroyed: {self.sandbox_id}")
            return True

        except Exception as e:
            print(f"[CODE_SANDBOX] Destroy error: {e}")
            return False

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute code in the sandbox"""
        start_time = datetime.utcnow()
        result = ExecutionResult(
            sandbox_id=self.sandbox_id,
            started_at=start_time,
        )

        try:
            self.state.status = SandboxStatus.RUNNING

            # Write any provided files
            for file_path, content in request.files.items():
                await self.write_file(file_path, content)

            # Determine language and prepare execution
            language = request.language or self.language
            if language not in self.SUPPORTED_LANGUAGES:
                result.error_message = f"Unsupported language: {language}"
                result.success = False
                return result

            lang_config = self.SUPPORTED_LANGUAGES[language]

            # Write code to file if provided
            if request.code:
                code_file = f"code{lang_config['extension']}"
                code_path = os.path.join(self.workspace_dir, code_file)
                with open(code_path, "w") as f:
                    f.write(request.code)
            else:
                code_file = None

            # Build command
            if request.command:
                cmd = request.command
            elif code_file:
                # Compile if needed
                if lang_config.get("compile"):
                    output_file = "code_output"
                    compile_cmd = [
                        c.format(file=code_path, output=os.path.join(self.workspace_dir, output_file))
                        for c in lang_config["compile"]
                    ]
                    compile_result = await self._run_process(compile_cmd, request.timeout_seconds)
                    if compile_result["exit_code"] != 0:
                        result.exit_code = compile_result["exit_code"]
                        result.stderr = compile_result["stderr"]
                        result.error_type = "compilation_error"
                        result.success = False
                        return result

                # Run
                run_cmd = [
                    c.format(
                        file=code_path,
                        output=os.path.join(self.workspace_dir, "code_output")
                    )
                    for c in lang_config["command"]
                ]
                cmd = run_cmd
            else:
                result.error_message = "No code or command provided"
                result.success = False
                return result

            # Execute
            try:
                exec_result = await asyncio.wait_for(
                    self._run_process(
                        cmd if isinstance(cmd, list) else cmd.split(),
                        request.timeout_seconds,
                        env=request.environment_vars,
                        cwd=request.working_directory or self.workspace_dir,
                    ),
                    timeout=request.timeout_seconds + 5,
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
            result.stderr = str(e)
            self.state.status = SandboxStatus.ERROR
            return result

    async def _run_process(
        self,
        cmd: List[str],
        timeout: int,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a subprocess with resource limits"""
        # Prepare environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Apply resource limits via ulimit (Unix only)
        if sys.platform != "win32":
            # Prepend ulimit commands for memory and CPU
            limits = self.config.resource_limits if self.config else ResourceLimits()
            ulimit_prefix = [
                "sh", "-c",
                f"ulimit -v {limits.memory_mb * 1024} 2>/dev/null; " + " ".join(cmd)
            ]
            cmd = ulimit_prefix

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
                cwd=cwd or self.workspace_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return {
                "exit_code": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }

        except asyncio.TimeoutError:
            if process:
                process.kill()
                await process.wait()
            raise

    async def write_file(self, path: str, content: str) -> bool:
        """Write a file to the workspace"""
        if not self.workspace_dir:
            return False

        # Handle paths
        if path.startswith("/"):
            path = path[1:]

        full_path = os.path.join(self.workspace_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w") as f:
            f.write(content)

        self.state.files_created.append(path)
        return True

    async def read_file(self, path: str) -> str:
        """Read a file from the workspace"""
        if not self.workspace_dir:
            return ""

        if path.startswith("/"):
            path = path[1:]

        full_path = os.path.join(self.workspace_dir, path)

        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return f.read()
        return ""

    async def list_files(self, path: str = ".") -> List[str]:
        """List files in directory"""
        if not self.workspace_dir:
            return []

        target_dir = os.path.join(self.workspace_dir, path.lstrip("/"))

        if os.path.exists(target_dir):
            return os.listdir(target_dir)
        return []

    async def get_state(self) -> SandboxState:
        """Get current sandbox state"""
        # Update disk usage
        if self.workspace_dir and os.path.exists(self.workspace_dir):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.workspace_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            self.state.disk_usage_mb = total_size / (1024 * 1024)

        return self.state

    def set_language(self, language: str):
        """Set the default programming language"""
        if language in self.SUPPORTED_LANGUAGES:
            self.language = language
        else:
            raise ValueError(f"Unsupported language: {language}")
