"""
Multi-Provider LLM Configuration

Centralized configuration for switching between LLM providers:
- OpenAI (GPT-4o, GPT-4o-mini)
- DeepSeek (V3.2 - 90% cheaper than OpenAI)
- Groq (Llama 3.3 - Ultra-fast inference)
- Mistral (Medium 3.1 - Good for French)
- Together AI (Llama, Mixtral)
- xAI Grok (2M context window)

Usage:
    from shared.llm_provider import get_llm_client, get_model_name, LLMProvider

    # Get configured client
    client = get_llm_client()

    # Get model names
    fast_model = get_model_name("fast")
    quality_model = get_model_name("quality")

    # Use in API calls
    response = await client.chat.completions.create(
        model=fast_model,
        messages=[...]
    )

Environment Variables:
    LLM_PROVIDER: "openai" | "deepseek" | "groq" | "mistral" | "together" | "xai"
    LLM_PROVIDER_API_KEY: API key for the selected provider (falls back to provider-specific keys)

    # Provider-specific keys (fallbacks)
    OPENAI_API_KEY: OpenAI API key
    DEEPSEEK_API_KEY: DeepSeek API key
    GROQ_API_KEY: Groq API key
    MISTRAL_API_KEY: Mistral API key
    TOGETHER_API_KEY: Together AI API key
    XAI_API_KEY: xAI/Grok API key
"""

import os
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
from openai import AsyncOpenAI, OpenAI


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    MISTRAL = "mistral"
    TOGETHER = "together"
    XAI = "xai"
    OLLAMA = "ollama"           # Self-hosted via Ollama
    RUNPOD = "runpod"           # RunPod serverless GPU


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider"""
    name: str
    base_url: str
    api_key_env: str
    model_fast: str
    model_quality: str
    model_reasoning: Optional[str]
    model_embedding: Optional[str]
    max_context: int
    supports_tools: bool
    supports_vision: bool
    cost_per_1m_input: float
    cost_per_1m_output: float
    timeout: float = 120.0          # Request timeout in seconds
    max_retries: int = 2            # Number of retries on failure


# Provider configurations
PROVIDER_CONFIGS: Dict[LLMProvider, ProviderConfig] = {
    LLMProvider.OPENAI: ProviderConfig(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model_fast="gpt-4o-mini",
        model_quality="gpt-4o",
        model_reasoning="o1-preview",
        model_embedding="text-embedding-3-small",
        max_context=128000,
        supports_tools=True,
        supports_vision=True,
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
    ),
    LLMProvider.DEEPSEEK: ProviderConfig(
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        model_fast="deepseek-chat",
        model_quality="deepseek-chat",
        model_reasoning="deepseek-reasoner",
        model_embedding=None,  # Use OpenAI for embeddings
        max_context=128000,
        supports_tools=True,
        supports_vision=False,
        cost_per_1m_input=0.28,
        cost_per_1m_output=0.42,
    ),
    LLMProvider.GROQ: ProviderConfig(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model_fast="llama-3.3-70b-versatile",
        model_quality="llama-3.3-70b-versatile",
        model_reasoning="llama-3.3-70b-versatile",
        model_embedding=None,
        max_context=128000,
        supports_tools=True,
        supports_vision=False,
        cost_per_1m_input=0.59,
        cost_per_1m_output=0.79,
    ),
    LLMProvider.MISTRAL: ProviderConfig(
        name="Mistral",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        model_fast="mistral-small-latest",
        model_quality="mistral-large-latest",
        model_reasoning="mistral-large-latest",
        model_embedding="mistral-embed",
        max_context=64000,
        supports_tools=True,
        supports_vision=False,
        cost_per_1m_input=0.40,
        cost_per_1m_output=2.00,
    ),
    LLMProvider.TOGETHER: ProviderConfig(
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        model_fast="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_quality="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_reasoning="deepseek-ai/DeepSeek-R1",
        model_embedding="togethercomputer/m2-bert-80M-8k-retrieval",
        max_context=128000,
        supports_tools=True,
        supports_vision=False,
        cost_per_1m_input=0.80,
        cost_per_1m_output=0.80,
    ),
    LLMProvider.XAI: ProviderConfig(
        name="xAI Grok",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        model_fast="grok-2-1212",
        model_quality="grok-2-1212",
        model_reasoning="grok-2-1212",
        model_embedding=None,
        max_context=131072,
        supports_tools=True,
        supports_vision=True,
        cost_per_1m_input=2.00,
        cost_per_1m_output=10.00,
    ),
    LLMProvider.OLLAMA: ProviderConfig(
        name="Ollama (Self-Hosted)",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key_env="OLLAMA_API_KEY",  # Usually "ollama" or empty
        model_fast="llama3.1:8b",
        model_quality="llama3.1:70b",
        model_reasoning="llama3.1:70b",
        model_embedding="nomic-embed-text",
        max_context=128000,
        supports_tools=True,
        supports_vision=False,
        cost_per_1m_input=0.0,          # Free (self-hosted)
        cost_per_1m_output=0.0,
        timeout=300.0,                   # Longer timeout for self-hosted
        max_retries=3,
    ),
    LLMProvider.RUNPOD: ProviderConfig(
        name="RunPod Serverless",
        base_url=os.getenv("RUNPOD_LLM_URL", "https://api.runpod.ai/v2"),
        api_key_env="RUNPOD_API_KEY",
        model_fast="llama3.1:8b",
        model_quality="llama3.1:70b",
        model_reasoning="llama3.1:70b",
        model_embedding=None,
        max_context=128000,
        supports_tools=True,
        supports_vision=False,
        cost_per_1m_input=0.10,          # ~$0.10/M with RunPod GPU
        cost_per_1m_output=0.10,
        timeout=180.0,                   # Longer timeout for serverless cold starts
        max_retries=3,
    ),
}


class LLMClientManager:
    """
    Manages LLM client instances with provider switching capability.

    Singleton pattern ensures consistent client usage across the application.
    """

    _instance: Optional['LLMClientManager'] = None
    _async_client: Optional[AsyncOpenAI] = None
    _sync_client: Optional[OpenAI] = None
    _provider: Optional[LLMProvider] = None
    _config: Optional[ProviderConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._provider is None:
            self._initialize()

    def _initialize(self):
        """Initialize the LLM client based on environment configuration"""
        # Get provider from environment
        provider_name = os.getenv("LLM_PROVIDER", "openai").lower()

        try:
            self._provider = LLMProvider(provider_name)
        except ValueError:
            print(f"[LLM] Unknown provider '{provider_name}', falling back to OpenAI", flush=True)
            self._provider = LLMProvider.OPENAI

        self._config = PROVIDER_CONFIGS[self._provider]

        # Get API key (check generic key first, then provider-specific)
        api_key = os.getenv("LLM_PROVIDER_API_KEY") or os.getenv(self._config.api_key_env)

        if not api_key:
            print(f"[LLM] WARNING: No API key found for {self._config.name}", flush=True)
            print(f"[LLM] Set LLM_PROVIDER_API_KEY or {self._config.api_key_env}", flush=True)

        # Get timeout and retries from config (with defaults for older configs)
        timeout = getattr(self._config, 'timeout', 120.0)
        max_retries = getattr(self._config, 'max_retries', 2)

        # For Ollama, API key can be empty or "ollama"
        if self._provider == LLMProvider.OLLAMA and not api_key:
            api_key = "ollama"

        # Create clients with timeout and retry configuration
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self._config.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._sync_client = OpenAI(
            api_key=api_key,
            base_url=self._config.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        print(f"[LLM] Initialized with {self._config.name} provider", flush=True)
        print(f"[LLM] Base URL: {self._config.base_url}", flush=True)
        print(f"[LLM] Fast model: {self._config.model_fast}", flush=True)
        print(f"[LLM] Quality model: {self._config.model_quality}", flush=True)
        print(f"[LLM] Timeout: {timeout}s, Retries: {max_retries}", flush=True)
        print(f"[LLM] Cost: ${self._config.cost_per_1m_input}/${self._config.cost_per_1m_output} per 1M tokens (in/out)", flush=True)

    @property
    def async_client(self) -> AsyncOpenAI:
        """Get the async OpenAI-compatible client"""
        return self._async_client

    @property
    def sync_client(self) -> OpenAI:
        """Get the sync OpenAI-compatible client"""
        return self._sync_client

    @property
    def provider(self) -> LLMProvider:
        """Get the current provider"""
        return self._provider

    @property
    def config(self) -> ProviderConfig:
        """Get the current provider config"""
        return self._config

    def get_model(self, tier: str = "quality") -> str:
        """
        Get the model name for the specified tier.

        Args:
            tier: "fast", "quality", or "reasoning"

        Returns:
            Model name string
        """
        if tier == "fast":
            return self._config.model_fast
        elif tier == "reasoning":
            return self._config.model_reasoning or self._config.model_quality
        else:
            return self._config.model_quality

    def get_embedding_client(self) -> AsyncOpenAI:
        """
        Get a client for embeddings.

        Some providers don't support embeddings, so we fall back to OpenAI.
        """
        if self._config.model_embedding:
            return self._async_client

        # Fall back to OpenAI for embeddings
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            return AsyncOpenAI(api_key=openai_key)

        raise ValueError("No embedding model available. Set OPENAI_API_KEY for fallback.")

    def get_embedding_model(self) -> str:
        """Get the embedding model name"""
        if self._config.model_embedding:
            return self._config.model_embedding
        return "text-embedding-3-small"  # OpenAI fallback

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the cost for a request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self._config.cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * self._config.cost_per_1m_output
        return input_cost + output_cost

    def reset(self):
        """Reset the client (useful for testing or provider switching)"""
        self._async_client = None
        self._sync_client = None
        self._provider = None
        self._config = None
        LLMClientManager._instance = None


# Convenience functions for easy access

def get_llm_manager() -> LLMClientManager:
    """Get the LLM client manager singleton"""
    return LLMClientManager()


def get_llm_client() -> AsyncOpenAI:
    """Get the async LLM client"""
    return get_llm_manager().async_client


def get_llm_client_sync() -> OpenAI:
    """Get the sync LLM client"""
    return get_llm_manager().sync_client


def get_model_name(tier: str = "quality") -> str:
    """
    Get the model name for the specified tier.

    Args:
        tier: "fast", "quality", or "reasoning"

    Returns:
        Model name string
    """
    return get_llm_manager().get_model(tier)


def get_provider() -> LLMProvider:
    """Get the current LLM provider"""
    return get_llm_manager().provider


def get_provider_config() -> ProviderConfig:
    """Get the current provider configuration"""
    return get_llm_manager().config


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate the cost for a request"""
    return get_llm_manager().estimate_cost(input_tokens, output_tokens)


# For backwards compatibility with existing code
def create_openai_client() -> AsyncOpenAI:
    """
    Create an OpenAI-compatible client.

    This function maintains backwards compatibility while using the
    configured provider.
    """
    return get_llm_client()


def get_openai_model(use_mini: bool = False) -> str:
    """
    Get the appropriate model name.

    This function maintains backwards compatibility while using the
    configured provider.

    Args:
        use_mini: If True, return the fast model; otherwise quality model
    """
    return get_model_name("fast" if use_mini else "quality")


# Provider info for logging/debugging
def print_provider_info():
    """Print current provider information"""
    manager = get_llm_manager()
    config = manager.config

    print("=" * 60, flush=True)
    print(f"LLM Provider: {config.name}", flush=True)
    print(f"Base URL: {config.base_url}", flush=True)
    print(f"Models:", flush=True)
    print(f"  - Fast: {config.model_fast}", flush=True)
    print(f"  - Quality: {config.model_quality}", flush=True)
    print(f"  - Reasoning: {config.model_reasoning}", flush=True)
    print(f"  - Embedding: {config.model_embedding or 'N/A (using OpenAI)'}", flush=True)
    print(f"Max Context: {config.max_context:,} tokens", flush=True)
    print(f"Pricing: ${config.cost_per_1m_input}/${config.cost_per_1m_output} per 1M (in/out)", flush=True)
    print(f"Features:", flush=True)
    print(f"  - Tools: {'Yes' if config.supports_tools else 'No'}", flush=True)
    print(f"  - Vision: {'Yes' if config.supports_vision else 'No'}", flush=True)
    print("=" * 60, flush=True)
