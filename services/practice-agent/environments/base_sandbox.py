"""
Base Sandbox Interface

Abstract base class for all sandbox environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import uuid

from models.sandbox_models import (
    SandboxConfig,
    SandboxState,
    SandboxStatus,
    ExecutionRequest,
    ExecutionResult,
    SandboxResult,
    ResourceLimits,
)


class BaseSandbox(ABC):
    """
    Abstract base class for sandbox environments.

    All sandbox implementations must provide:
    - Environment creation and destruction
    - Code/command execution
    - State management
    - Result validation
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.sandbox_id = str(uuid.uuid4())
        self.config = config
        self.state = SandboxState(
            sandbox_id=self.sandbox_id,
            sandbox_type=self._get_sandbox_type(),
            status=SandboxStatus.CREATING,
        )

    @abstractmethod
    def _get_sandbox_type(self) -> str:
        """Return the sandbox type identifier"""
        pass

    @abstractmethod
    async def create(self) -> SandboxState:
        """
        Create and initialize the sandbox environment.
        Returns the sandbox state.
        """
        pass

    @abstractmethod
    async def destroy(self) -> bool:
        """
        Destroy the sandbox and clean up resources.
        Returns True if successful.
        """
        pass

    @abstractmethod
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Execute code or commands in the sandbox.
        """
        pass

    @abstractmethod
    async def get_state(self) -> SandboxState:
        """
        Get the current state of the sandbox.
        """
        pass

    async def write_file(self, path: str, content: str) -> bool:
        """Write a file to the sandbox filesystem"""
        raise NotImplementedError("File operations not supported for this sandbox")

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox filesystem"""
        raise NotImplementedError("File operations not supported for this sandbox")

    async def list_files(self, path: str = "/workspace") -> List[str]:
        """List files in a directory"""
        raise NotImplementedError("File operations not supported for this sandbox")

    async def validate(
        self,
        expected_outputs: List[Dict[str, Any]],
        actual_result: ExecutionResult,
    ) -> SandboxResult:
        """
        Validate execution results against expected outputs.
        Default implementation - can be overridden.
        """
        validation_passed = True
        checks_results = {}
        score = 0
        max_score = 0
        feedback_items = []

        for expected in expected_outputs:
            check_name = expected.get("name", "check")
            check_type = expected.get("type", "stdout")
            points = expected.get("points", 10)
            max_score += points

            passed = False
            message = ""

            if check_type == "stdout":
                # Check stdout output
                if expected.get("exact_match"):
                    passed = actual_result.stdout.strip() == expected["exact_match"].strip()
                    message = "Output exact match" if passed else "Output doesn't match expected"
                elif expected.get("contains"):
                    passed = all(s in actual_result.stdout for s in expected["contains"])
                    message = "Output contains required strings" if passed else "Missing required output"
                elif expected.get("pattern"):
                    import re
                    passed = bool(re.search(expected["pattern"], actual_result.stdout))
                    message = "Output matches pattern" if passed else "Output doesn't match pattern"

            elif check_type == "exit_code":
                expected_code = expected.get("expected_value", 0)
                passed = actual_result.exit_code == expected_code
                message = f"Exit code is {expected_code}" if passed else f"Expected {expected_code}, got {actual_result.exit_code}"

            elif check_type == "no_error":
                passed = actual_result.success and not actual_result.stderr
                message = "No errors" if passed else "Execution had errors"

            elif check_type == "file_exists":
                # Check if file was created - would need to implement in specific sandbox
                passed = expected.get("file_path", "") in self.state.files_created
                message = "File created" if passed else "File not found"

            checks_results[check_name] = passed
            if passed:
                score += points
            if not passed:
                validation_passed = False

            feedback_items.append({
                "check": check_name,
                "passed": passed,
                "message": message,
                "points": points if passed else 0,
            })

        return SandboxResult(
            execution=actual_result,
            validation_passed=validation_passed,
            checks_results=checks_results,
            score=score,
            max_score=max_score,
            feedback_items=feedback_items,
            final_state=self.state,
        )

    def _apply_resource_limits(self, limits: ResourceLimits) -> Dict[str, Any]:
        """Convert resource limits to container/process constraints"""
        return {
            "cpu_quota": int(limits.cpu_cores * 100000),
            "cpu_period": 100000,
            "mem_limit": f"{limits.memory_mb}m",
            "storage_opt": {"size": f"{limits.disk_mb}m"} if limits.disk_mb else None,
            "network_disabled": not limits.network_enabled,
        }
