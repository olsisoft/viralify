"""
Shared utilities for Viralify services.

This module provides common functionality used across multiple services,
including LLM provider configuration and training data logging.
"""

from .llm_provider import (
    # Classes
    LLMProvider,
    ProviderConfig,
    LLMClientManager,

    # Functions
    get_llm_manager,
    get_llm_client,
    get_llm_client_sync,
    get_model_name,
    get_provider,
    get_provider_config,
    estimate_cost,
    print_provider_info,

    # Backwards compatibility
    create_openai_client,
    get_openai_model,

    # Constants
    PROVIDER_CONFIGS,
)

from .training_logger import (
    # Classes
    TrainingLogger,
    TrainingExample,
    TaskType,

    # Functions
    get_training_logger,
    log_training_example,
    log_conversation,
    get_training_stats,
)

__all__ = [
    # LLM Provider
    "LLMProvider",
    "ProviderConfig",
    "LLMClientManager",
    "get_llm_manager",
    "get_llm_client",
    "get_llm_client_sync",
    "get_model_name",
    "get_provider",
    "get_provider_config",
    "estimate_cost",
    "print_provider_info",
    "create_openai_client",
    "get_openai_model",
    "PROVIDER_CONFIGS",
    # Training Logger
    "TrainingLogger",
    "TrainingExample",
    "TaskType",
    "get_training_logger",
    "log_training_example",
    "log_conversation",
    "get_training_stats",
]
