"""
Sandbox Manager

Manages sandbox lifecycle and execution across different sandbox types.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from models.sandbox_models import (
    SandboxType,
    SandboxConfig,
    SandboxState,
    SandboxStatus,
    ExecutionRequest,
    ExecutionResult,
    SandboxResult,
    ResourceLimits,
)
from environments.base_sandbox import BaseSandbox
from environments.docker_sandbox import DockerSandbox
from environments.kubernetes_sandbox import KubernetesSandbox
from environments.code_sandbox import CodeSandbox


class SandboxManager:
    """
    Manages sandbox environments for practice exercises.

    Features:
    - Sandbox creation and pooling
    - Automatic cleanup
    - Resource monitoring
    - Multi-sandbox type support
    """

    # Maximum sandboxes per user
    MAX_SANDBOXES_PER_USER = 3

    # Sandbox timeout (auto-destroy after inactivity)
    SANDBOX_TIMEOUT_MINUTES = 30

    def __init__(self):
        # Active sandboxes: {sandbox_id: sandbox}
        self._sandboxes: Dict[str, BaseSandbox] = {}
        # User sandboxes: {user_id: [sandbox_ids]}
        self._user_sandboxes: Dict[str, List[str]] = {}
        # Cleanup task
        self._cleanup_task = None

    async def start(self):
        """Start the sandbox manager background tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        print("[SANDBOX_MANAGER] Started")

    async def stop(self):
        """Stop and cleanup all sandboxes"""
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Destroy all sandboxes
        for sandbox in list(self._sandboxes.values()):
            await sandbox.destroy()

        self._sandboxes.clear()
        self._user_sandboxes.clear()
        print("[SANDBOX_MANAGER] Stopped")

    async def create_sandbox(
        self,
        sandbox_type: SandboxType,
        user_id: str,
        config: Optional[SandboxConfig] = None,
    ) -> BaseSandbox:
        """Create a new sandbox for a user"""
        # Check user limits
        user_sandboxes = self._user_sandboxes.get(user_id, [])
        if len(user_sandboxes) >= self.MAX_SANDBOXES_PER_USER:
            # Destroy oldest sandbox
            oldest_id = user_sandboxes[0]
            await self.destroy_sandbox(oldest_id)

        # Create sandbox config if not provided
        if not config:
            config = SandboxConfig(sandbox_type=sandbox_type)

        # Create appropriate sandbox type
        sandbox = self._create_sandbox_instance(sandbox_type, config)

        # Initialize sandbox
        await sandbox.create()

        # Register sandbox
        self._sandboxes[sandbox.sandbox_id] = sandbox
        if user_id not in self._user_sandboxes:
            self._user_sandboxes[user_id] = []
        self._user_sandboxes[user_id].append(sandbox.sandbox_id)

        print(f"[SANDBOX_MANAGER] Created {sandbox_type.value} sandbox: {sandbox.sandbox_id}")
        return sandbox

    async def get_sandbox(self, sandbox_id: str) -> Optional[BaseSandbox]:
        """Get a sandbox by ID"""
        return self._sandboxes.get(sandbox_id)

    async def destroy_sandbox(self, sandbox_id: str) -> bool:
        """Destroy a sandbox"""
        sandbox = self._sandboxes.get(sandbox_id)
        if not sandbox:
            return False

        try:
            await sandbox.destroy()
        except Exception as e:
            print(f"[SANDBOX_MANAGER] Destroy error for {sandbox_id}: {e}")

        # Unregister
        del self._sandboxes[sandbox_id]
        for user_id, sandboxes in self._user_sandboxes.items():
            if sandbox_id in sandboxes:
                sandboxes.remove(sandbox_id)
                break

        return True

    async def execute(
        self,
        sandbox_type: str,
        code: str,
        exercise_config: Dict[str, Any],
        timeout: int = 60,
        user_id: str = "default",
    ) -> SandboxResult:
        """
        Execute code in a sandbox.

        Creates a temporary sandbox, executes, validates, and cleans up.
        """
        # Map string to enum
        try:
            s_type = SandboxType(sandbox_type)
        except ValueError:
            s_type = SandboxType.DOCKER

        # Create sandbox config from exercise
        config = SandboxConfig(
            sandbox_type=s_type,
            base_image=exercise_config.get("base_image"),
            environment_vars=exercise_config.get("environment", {}),
            setup_commands=exercise_config.get("setup_commands", []),
            resource_limits=ResourceLimits(
                cpu_cores=exercise_config.get("cpu_limit", 1.0),
                memory_mb=exercise_config.get("memory_limit", 512),
                timeout_seconds=timeout,
            ),
        )

        sandbox = None
        try:
            # Create sandbox
            sandbox = await self.create_sandbox(s_type, user_id, config)

            # Prepare execution request
            request = ExecutionRequest(
                sandbox_id=sandbox.sandbox_id,
                code=code,
                language=exercise_config.get("language", "bash"),
                files=exercise_config.get("files", {}),
                timeout_seconds=timeout,
            )

            # Execute
            result = await sandbox.execute(request)

            # Validate against expected outputs
            expected_outputs = exercise_config.get("expected_outputs", [])
            sandbox_result = await sandbox.validate(expected_outputs, result)

            return sandbox_result

        except Exception as e:
            print(f"[SANDBOX_MANAGER] Execution error: {e}")
            # Return error result
            return SandboxResult(
                execution=ExecutionResult(
                    sandbox_id=sandbox.sandbox_id if sandbox else "unknown",
                    success=False,
                    error_type="execution_error",
                    error_message=str(e),
                ),
                validation_passed=False,
                score=0,
                max_score=100,
            )

        finally:
            # Cleanup sandbox
            if sandbox:
                await self.destroy_sandbox(sandbox.sandbox_id)

    async def get_user_sandboxes(self, user_id: str) -> List[SandboxState]:
        """Get all sandbox states for a user"""
        sandbox_ids = self._user_sandboxes.get(user_id, [])
        states = []
        for sid in sandbox_ids:
            sandbox = self._sandboxes.get(sid)
            if sandbox:
                states.append(await sandbox.get_state())
        return states

    def _create_sandbox_instance(
        self,
        sandbox_type: SandboxType,
        config: SandboxConfig,
    ) -> BaseSandbox:
        """Create the appropriate sandbox instance"""
        sandbox_classes = {
            SandboxType.DOCKER: DockerSandbox,
            SandboxType.KUBERNETES: KubernetesSandbox,
            SandboxType.PYTHON: CodeSandbox,
            SandboxType.BASH: CodeSandbox,
        }

        sandbox_class = sandbox_classes.get(sandbox_type, DockerSandbox)
        return sandbox_class(config)

    async def _cleanup_loop(self):
        """Background task to cleanup inactive sandboxes"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()
                timeout_threshold = now - timedelta(minutes=self.SANDBOX_TIMEOUT_MINUTES)

                sandboxes_to_destroy = []
                for sandbox_id, sandbox in self._sandboxes.items():
                    state = await sandbox.get_state()
                    if state.last_execution_at and state.last_execution_at < timeout_threshold:
                        sandboxes_to_destroy.append(sandbox_id)

                for sandbox_id in sandboxes_to_destroy:
                    print(f"[SANDBOX_MANAGER] Auto-cleanup inactive sandbox: {sandbox_id}")
                    await self.destroy_sandbox(sandbox_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[SANDBOX_MANAGER] Cleanup error: {e}")


# Global instance
_sandbox_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    """Get the global sandbox manager instance"""
    global _sandbox_manager
    if _sandbox_manager is None:
        _sandbox_manager = SandboxManager()
    return _sandbox_manager
