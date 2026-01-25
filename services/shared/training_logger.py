"""
Training Data Logger

Logs all validated LLM outputs to a JSONL file for future model fine-tuning.
Each line contains a complete training example with:
- Input messages (system + user prompts)
- Output (assistant response)
- Metadata (model, provider, task type, timestamp)

Usage:
    from shared.training_logger import TrainingLogger, log_training_example

    # Option 1: Use the singleton
    log_training_example(
        messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
        response="...",
        task_type="course_generation",
        metadata={"course_id": "123"}
    )

    # Option 2: Use the class directly
    logger = TrainingLogger()
    logger.log(messages, response, task_type, metadata)

Environment Variables:
    TRAINING_DATA_PATH: Path to the JSONL file (default: /app/data/training_dataset.jsonl)
    TRAINING_LOGGER_ENABLED: Enable/disable logging (default: true)
"""

import os
import json
import fcntl
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class TaskType(str, Enum):
    """Types of tasks for categorizing training data."""
    COURSE_OUTLINE = "course_outline"
    COURSE_GENERATION = "course_generation"
    LESSON_GENERATION = "lesson_generation"
    QUIZ_GENERATION = "quiz_generation"
    PRESENTATION_PLANNING = "presentation_planning"
    SLIDE_GENERATION = "slide_generation"
    SCRIPT_GENERATION = "script_generation"
    DIAGRAM_GENERATION = "diagram_generation"
    CODE_GENERATION = "code_generation"
    TRANSLATION = "translation"
    ELEMENT_SUGGESTION = "element_suggestion"
    CATEGORY_DETECTION = "category_detection"
    VOICEOVER_GENERATION = "voiceover_generation"
    OTHER = "other"


@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""
    # Core training data
    messages: List[Dict[str, str]]  # [{"role": "system/user/assistant", "content": "..."}]
    response: str  # The validated LLM output

    # Metadata for filtering and analysis
    task_type: str
    provider: str
    model: str
    timestamp: str

    # Optional context
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    # Quality indicators
    was_successful: bool = True
    required_retry: bool = False
    retry_count: int = 0


class TrainingLogger:
    """
    Thread-safe logger for collecting LLM training data.

    Uses file locking for safe concurrent writes from multiple processes.
    """

    _instance: Optional['TrainingLogger'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._enabled = os.getenv("TRAINING_LOGGER_ENABLED", "true").lower() == "true"
        self._file_path = Path(os.getenv(
            "TRAINING_DATA_PATH",
            "/app/data/training_dataset.jsonl"
        ))
        self._write_lock = threading.Lock()

        # Get current provider info
        self._provider = os.getenv("LLM_PROVIDER", "openai")
        self._model = None  # Will be set on first log

        # Ensure directory exists
        if self._enabled:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[TRAINING_LOGGER] Initialized - logging to {self._file_path}", flush=True)
            print(f"[TRAINING_LOGGER] Provider: {self._provider}", flush=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def file_path(self) -> Path:
        return self._file_path

    def log(
        self,
        messages: List[Dict[str, str]],
        response: str,
        task_type: str = "other",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        latency_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        was_successful: bool = True,
        required_retry: bool = False,
        retry_count: int = 0,
    ) -> bool:
        """
        Log a training example to the JSONL file.

        Args:
            messages: The input messages sent to the LLM
            response: The validated LLM response
            task_type: Type of task (from TaskType enum or string)
            model: Model name used (auto-detected if not provided)
            provider: Provider name (auto-detected if not provided)
            input_tokens: Number of input tokens (if available)
            output_tokens: Number of output tokens (if available)
            latency_ms: Request latency in milliseconds
            metadata: Additional metadata (course_id, language, etc.)
            was_successful: Whether the response was valid/successful
            required_retry: Whether retries were needed
            retry_count: Number of retries performed

        Returns:
            True if logged successfully, False otherwise
        """
        if not self._enabled:
            return False

        # Skip empty or invalid responses
        if not response or not messages:
            return False

        # Auto-detect model if not provided
        if model is None:
            model = self._get_current_model()

        # Create training example
        example = TrainingExample(
            messages=messages,
            response=response,
            task_type=task_type,
            provider=provider or self._provider,
            model=model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            metadata=metadata,
            was_successful=was_successful,
            required_retry=required_retry,
            retry_count=retry_count,
        )

        return self._write_example(example)

    def log_conversation(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_response: str,
        task_type: str = "other",
        **kwargs
    ) -> bool:
        """
        Convenience method to log a simple conversation.

        Args:
            system_prompt: The system message
            user_prompt: The user message
            assistant_response: The assistant's response
            task_type: Type of task
            **kwargs: Additional arguments passed to log()
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        return self.log(
            messages=messages,
            response=assistant_response,
            task_type=task_type,
            **kwargs
        )

    def _get_current_model(self) -> str:
        """Get the current model from the LLM provider."""
        if self._model:
            return self._model

        try:
            from shared.llm_provider import get_model_name
            self._model = get_model_name("quality")
        except ImportError:
            self._model = os.getenv("OPENAI_MODEL", "gpt-4o")

        return self._model

    def _write_example(self, example: TrainingExample) -> bool:
        """Write a training example to the JSONL file with file locking."""
        try:
            with self._write_lock:
                # Convert to dict and serialize
                data = asdict(example)
                json_line = json.dumps(data, ensure_ascii=False) + "\n"

                # Write with file locking for multi-process safety
                with open(self._file_path, "a", encoding="utf-8") as f:
                    # Try to acquire exclusive lock (non-blocking on Windows)
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    except (AttributeError, OSError):
                        # fcntl not available on Windows, use thread lock only
                        pass

                    try:
                        f.write(json_line)
                    finally:
                        try:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        except (AttributeError, OSError):
                            pass

                return True

        except Exception as e:
            print(f"[TRAINING_LOGGER] Error writing example: {e}", flush=True)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collected training data."""
        if not self._file_path.exists():
            return {"total_examples": 0, "file_size_mb": 0}

        total = 0
        task_counts = {}
        provider_counts = {}

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        total += 1
                        try:
                            data = json.loads(line)
                            task = data.get("task_type", "unknown")
                            provider = data.get("provider", "unknown")
                            task_counts[task] = task_counts.get(task, 0) + 1
                            provider_counts[provider] = provider_counts.get(provider, 0) + 1
                        except json.JSONDecodeError:
                            pass

            file_size = self._file_path.stat().st_size / (1024 * 1024)

            return {
                "total_examples": total,
                "file_size_mb": round(file_size, 2),
                "by_task_type": task_counts,
                "by_provider": provider_counts,
                "file_path": str(self._file_path),
            }
        except Exception as e:
            return {"error": str(e)}

    def export_for_openai(self, output_path: Optional[str] = None) -> str:
        """
        Export training data in OpenAI fine-tuning format.

        OpenAI format:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

        Returns:
            Path to the exported file
        """
        if output_path is None:
            output_path = str(self._file_path).replace(".jsonl", "_openai.jsonl")

        exported = 0
        with open(self._file_path, "r", encoding="utf-8") as f_in:
            with open(output_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("was_successful", True):
                                # Build OpenAI format
                                messages = data.get("messages", [])
                                messages.append({
                                    "role": "assistant",
                                    "content": data.get("response", "")
                                })
                                openai_example = {"messages": messages}
                                f_out.write(json.dumps(openai_example, ensure_ascii=False) + "\n")
                                exported += 1
                        except json.JSONDecodeError:
                            pass

        print(f"[TRAINING_LOGGER] Exported {exported} examples to {output_path}", flush=True)
        return output_path

    def reset(self):
        """Reset the singleton (for testing)."""
        self._initialized = False
        TrainingLogger._instance = None


# Convenience functions for easy access

def get_training_logger() -> TrainingLogger:
    """Get the training logger singleton."""
    return TrainingLogger()


def log_training_example(
    messages: List[Dict[str, str]],
    response: str,
    task_type: str = "other",
    **kwargs
) -> bool:
    """Log a training example (convenience function)."""
    return get_training_logger().log(messages, response, task_type, **kwargs)


def log_conversation(
    system_prompt: str,
    user_prompt: str,
    assistant_response: str,
    task_type: str = "other",
    **kwargs
) -> bool:
    """Log a simple conversation (convenience function)."""
    return get_training_logger().log_conversation(
        system_prompt, user_prompt, assistant_response, task_type, **kwargs
    )


def get_training_stats() -> Dict[str, Any]:
    """Get training data statistics."""
    return get_training_logger().get_stats()
