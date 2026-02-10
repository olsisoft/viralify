"""
Code Executor Service

Executes code snippets securely and captures output.
Supports local execution (subprocess) and Replicate API.
"""
import asyncio
import os
import subprocess
import tempfile
from typing import Optional
from pydantic import BaseModel


class ExecutionResult(BaseModel):
    """Result of code execution"""
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    error: Optional[str] = None


class CodeExecutorService:
    """Service for executing code snippets"""

    # Supported languages and their execution commands
    LANGUAGE_CONFIGS = {
        "python": {
            "extension": ".py",
            "command": ["python3", "-u"],
            "timeout": 30
        },
        "javascript": {
            "extension": ".js",
            "command": ["node"],
            "timeout": 30
        },
        "bash": {
            "extension": ".sh",
            "command": ["bash"],
            "timeout": 30
        },
        "shell": {
            "extension": ".sh",
            "command": ["sh"],
            "timeout": 30
        }
    }

    def __init__(self):
        self.replicate_api_key = os.getenv("REPLICATE_API_TOKEN")
        self.use_replicate = os.getenv("USE_REPLICATE_EXECUTOR", "false").lower() == "true"

    async def execute(
        self,
        code: str,
        language: str,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute code and capture output.

        Args:
            code: The code to execute
            language: Programming language (python, javascript, bash)
            timeout: Execution timeout in seconds (default: 30)

        Returns:
            ExecutionResult with stdout, stderr, and status
        """
        language = language.lower()

        if language not in self.LANGUAGE_CONFIGS:
            return ExecutionResult(
                success=False,
                error=f"Unsupported language: {language}. Supported: {list(self.LANGUAGE_CONFIGS.keys())}"
            )

        config = self.LANGUAGE_CONFIGS[language]
        timeout = timeout or config["timeout"]

        if self.use_replicate and self.replicate_api_key:
            return await self._execute_replicate(code, language, timeout)
        else:
            return await self._execute_local(code, language, timeout)

    async def _execute_local(
        self,
        code: str,
        language: str,
        timeout: int
    ) -> ExecutionResult:
        """Execute code locally using subprocess"""
        import time

        config = self.LANGUAGE_CONFIGS[language]
        start_time = time.time()

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=config["extension"],
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Build command
                cmd = config["command"] + [temp_file]

                # Execute with timeout
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tempfile.gettempdir()
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout
                    )

                    execution_time = time.time() - start_time

                    return ExecutionResult(
                        success=process.returncode == 0,
                        stdout=stdout.decode('utf-8', errors='replace').strip(),
                        stderr=stderr.decode('utf-8', errors='replace').strip(),
                        exit_code=process.returncode,
                        execution_time=execution_time
                    )

                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return ExecutionResult(
                        success=False,
                        error=f"Execution timeout after {timeout} seconds",
                        exit_code=-1,
                        execution_time=timeout
                    )

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def _execute_replicate(
        self,
        code: str,
        language: str,
        timeout: int
    ) -> ExecutionResult:
        """Execute code using Replicate API"""
        import time
        import httpx

        start_time = time.time()

        # Replicate models for code execution
        # Note: These are example model names - actual models may vary
        MODELS = {
            "python": "replicate/python:latest",
            "javascript": "replicate/nodejs:latest",
            "bash": "replicate/bash:latest"
        }

        model = MODELS.get(language)
        if not model:
            return ExecutionResult(
                success=False,
                error=f"No Replicate model for language: {language}"
            )

        try:
            async with httpx.AsyncClient(timeout=timeout + 10) as client:
                # Start prediction
                response = await client.post(
                    "https://api.replicate.com/v1/predictions",
                    headers={
                        "Authorization": f"Token {self.replicate_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "version": model,
                        "input": {
                            "code": code,
                            "timeout": timeout
                        }
                    }
                )

                if response.status_code != 201:
                    return ExecutionResult(
                        success=False,
                        error=f"Replicate API error: {response.status_code} - {response.text}"
                    )

                prediction = response.json()
                prediction_id = prediction["id"]

                # Poll for result
                max_polls = timeout * 2  # Poll every 0.5 seconds
                for _ in range(max_polls):
                    await asyncio.sleep(0.5)

                    status_response = await client.get(
                        f"https://api.replicate.com/v1/predictions/{prediction_id}",
                        headers={"Authorization": f"Token {self.replicate_api_key}"}
                    )

                    status_data = status_response.json()
                    status = status_data.get("status")

                    if status == "succeeded":
                        output = status_data.get("output", {})
                        return ExecutionResult(
                            success=True,
                            stdout=output.get("stdout", ""),
                            stderr=output.get("stderr", ""),
                            exit_code=output.get("exit_code", 0),
                            execution_time=time.time() - start_time
                        )
                    elif status == "failed":
                        return ExecutionResult(
                            success=False,
                            error=status_data.get("error", "Unknown error"),
                            execution_time=time.time() - start_time
                        )

                return ExecutionResult(
                    success=False,
                    error="Replicate execution timeout",
                    execution_time=time.time() - start_time
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def validate_code(self, code: str, language: str) -> bool:
        """
        Validate code syntax without executing it.

        Args:
            code: The code to validate
            language: Programming language

        Returns:
            True if syntax is valid
        """
        if language.lower() == "python":
            try:
                compile(code, "<string>", "exec")
                return True
            except SyntaxError:
                return False

        # For other languages, assume valid (could add more validators)
        return True
