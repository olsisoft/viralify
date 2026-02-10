"""
MAESTRO LLM Provider Interface
Interface agnostique pour différents fournisseurs LLM

Updated with Groq rate limiting support to handle ERR-017.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import json
import logging
import os
import time
import re
import sys


logger = logging.getLogger(__name__)


# Add shared module to path for rate limiter import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

try:
    from groq_rate_limiter import get_groq_rate_limiter, record_groq_usage
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False
    logger.warning("[LLM_PROVIDER] Groq rate limiter not available, using basic rate limiting")


def sanitize_json_string(content: str) -> str:
    """
    Sanitize a JSON string to handle common LLM output issues.

    Handles:
    - Raw control characters inside strings (newlines, tabs, etc.)
    - Double-escaped characters
    - Trailing commas
    - Missing quotes around keys
    """
    # First, try to identify if we're inside a JSON string and escape control chars
    # This regex finds string values and escapes control characters inside them

    def escape_control_chars_in_string(match):
        """Escape control characters inside a JSON string value."""
        s = match.group(0)
        # Keep the quotes, escape the content
        inner = s[1:-1]  # Remove surrounding quotes

        # Escape actual control characters (not already escaped sequences)
        # Order matters: don't double-escape already escaped chars
        result = ""
        i = 0
        while i < len(inner):
            char = inner[i]
            # Check if this is an escape sequence
            if char == '\\' and i + 1 < len(inner):
                next_char = inner[i + 1]
                # Valid escape sequences - keep as is
                if next_char in 'nrtbf"\\/u':
                    result += char + next_char
                    i += 2
                    continue
                # Double backslash for actual backslash
                elif next_char == '\\':
                    result += '\\\\'
                    i += 2
                    continue

            # Escape actual control characters
            if char == '\n':
                result += '\\n'
            elif char == '\r':
                result += '\\r'
            elif char == '\t':
                result += '\\t'
            elif char == '\b':
                result += '\\b'
            elif char == '\f':
                result += '\\f'
            elif ord(char) < 32:
                # Other control characters - escape as unicode
                result += f'\\u{ord(char):04x}'
            else:
                result += char
            i += 1

        return '"' + result + '"'

    # Regex to match JSON string values (handles escaped quotes inside)
    # This is a simplified pattern that works for most cases
    json_string_pattern = r'"(?:[^"\\]|\\.)*"'

    try:
        sanitized = re.sub(json_string_pattern, escape_control_chars_in_string, content, flags=re.DOTALL)
        return sanitized
    except Exception as e:
        logger.warning(f"JSON sanitization regex failed: {e}, returning original")
        return content


def try_parse_json(content: str) -> Dict[str, Any]:
    """
    Try multiple strategies to parse JSON content.
    """
    errors = []

    # Strategy 1: Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        errors.append(f"Direct parse: {e}")

    # Strategy 2: Sanitize control characters
    try:
        sanitized = sanitize_json_string(content)
        return json.loads(sanitized)
    except json.JSONDecodeError as e:
        errors.append(f"After sanitization: {e}")

    # Strategy 3: Fix double-escaped characters
    try:
        fixed = content.replace('\\\\n', '\\n').replace('\\\\t', '\\t').replace('\\\\r', '\\r')
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        errors.append(f"After double-escape fix: {e}")

    # Strategy 4: Remove trailing commas (common LLM mistake)
    try:
        # Remove trailing commas before ] or }
        no_trailing = re.sub(r',\s*([}\]])', r'\1', content)
        return json.loads(no_trailing)
    except json.JSONDecodeError as e:
        errors.append(f"After trailing comma fix: {e}")

    # Strategy 5: Combined fixes
    try:
        fixed = content.replace('\\\\n', '\\n').replace('\\\\t', '\\t').replace('\\\\r', '\\r')
        sanitized = sanitize_json_string(fixed)
        no_trailing = re.sub(r',\s*([}\]])', r'\1', sanitized)
        return json.loads(no_trailing)
    except json.JSONDecodeError as e:
        errors.append(f"After all fixes: {e}")

    # All strategies failed
    raise ValueError(f"Failed to parse JSON after all strategies. Errors: {'; '.join(errors)}")


class LLMProvider(Enum):
    """Fournisseurs LLM supportés"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """Configuration pour le provider LLM"""
    provider: LLMProvider
    api_key: str = ""
    model: str = ""
    base_url: Optional[str] = None
    
    # Paramètres de génération par défaut
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    
    # Rate limiting
    requests_per_minute: int = 30
    tokens_per_minute: int = 100000
    
    # Retry
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Optimisation tokens
    enable_caching: bool = True
    
    @classmethod
    def for_groq(cls, api_key: str, model: str = "llama-3.3-70b-versatile") -> "LLMConfig":
        """Configuration optimisée pour Groq"""
        return cls(
            provider=LLMProvider.GROQ,
            api_key=api_key,
            model=model,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.7,
            max_tokens=4096,
            requests_per_minute=30,
            tokens_per_minute=6000,  # Groq free tier limit
        )
    
    @classmethod
    def for_openai(cls, api_key: str, model: str = "gpt-4o") -> "LLMConfig":
        return cls(
            provider=LLMProvider.OPENAI,
            api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=4096,
        )
    
    @classmethod
    def for_anthropic(cls, api_key: str, model: str = "claude-sonnet-4-20250514") -> "LLMConfig":
        return cls(
            provider=LLMProvider.ANTHROPIC,
            api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=4096,
        )
    
    @classmethod
    def for_ollama(cls, model: str = "llama3.1", base_url: str = "http://localhost:11434") -> "LLMConfig":
        return cls(
            provider=LLMProvider.OLLAMA,
            model=model,
            base_url=base_url,
            temperature=0.7,
            max_tokens=4096,
            requests_per_minute=1000,  # Local = pas de limite
            tokens_per_minute=1000000,
        )


@dataclass
class LLMMessage:
    """Message pour l'API LLM"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Réponse du LLM"""
    content: str
    model: str
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw_response: Optional[Dict] = None


@dataclass
class TokenBudget:
    """Gestion du budget de tokens"""
    total_budget: int
    used_tokens: int = 0
    
    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self.used_tokens)
    
    @property
    def usage_percent(self) -> float:
        return (self.used_tokens / self.total_budget) * 100 if self.total_budget > 0 else 0
    
    def consume(self, tokens: int):
        self.used_tokens += tokens
    
    def has_budget(self, required: int = 1000) -> bool:
        return self.remaining >= required


class BaseLLMProvider(ABC):
    """Interface abstraite pour tous les providers LLM"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._request_timestamps: List[float] = []
        self._token_usage: int = 0
        self._cache: Dict[str, str] = {}
    
    @abstractmethod
    def _call_api(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Appel API spécifique au provider"""
        pass
    
    def generate(self, 
                 messages: List[LLMMessage],
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 json_mode: bool = False) -> LLMResponse:
        """
        Génère une réponse avec gestion du rate limiting et retry.
        """
        # Check cache
        cache_key = self._get_cache_key(messages)
        if self.config.enable_caching and cache_key in self._cache:
            logger.debug("Cache hit for request")
            return LLMResponse(content=self._cache[cache_key], model=self.config.model)
        
        # Rate limiting
        self._apply_rate_limit()
        
        # Retry logic
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._call_api(
                    messages,
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    json_mode=json_mode,
                )
                
                # Track usage
                self._token_usage += response.tokens_used
                
                # Cache response
                if self.config.enable_caching:
                    self._cache[cache_key] = response.content
                
                return response
                
            except Exception as e:
                last_error = e
                error_str = str(e)

                # Don't retry on 400 errors (client errors) - they won't succeed on retry
                # This includes json_validate_failed from Groq
                if "400" in error_str or "json_validate_failed" in error_str:
                    logger.warning(f"LLM call failed with client error (not retrying): {e}")
                    raise

                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))

        raise RuntimeError(f"LLM call failed after {self.config.max_retries} attempts: {last_error}")
    
    def generate_json(self, 
                      messages: List[LLMMessage],
                      schema_hint: str = "",
                      **kwargs) -> Dict[str, Any]:
        """
        Génère une réponse JSON structurée.
        """
        # Ajouter instruction JSON au dernier message user
        if schema_hint:
            messages = messages.copy()
            last_user_idx = None
            for i, msg in enumerate(messages):
                if msg.role == "user":
                    last_user_idx = i
            
            if last_user_idx is not None:
                messages[last_user_idx] = LLMMessage(
                    role="user",
                    content=messages[last_user_idx].content + f"\n\nRespond with valid JSON only. Expected schema:\n{schema_hint}"
                )
        
        # Retry logic for empty responses (common with Groq rate limiting)
        # Increased to 5 retries with longer backoff for Groq stability
        max_json_retries = 5
        last_error = None

        for json_attempt in range(max_json_retries):
            # Try with JSON mode first, fallback to non-JSON mode if it fails
            # (Groq sometimes fails with json_validate_failed when code has newlines)
            try:
                response = self.generate(messages, json_mode=True, **kwargs)
            except Exception as e:
                error_str = str(e)
                if "json_validate_failed" in error_str:
                    logger.warning(f"JSON mode failed with json_validate_failed")

                    # Try to extract failed_generation from error (Groq includes the generated JSON)
                    try:
                        import re
                        # Look for 'failed_generation': '...' in the error
                        match = re.search(r"'failed_generation':\s*'(.*?)'(?=\s*\})", error_str, re.DOTALL)
                        if match:
                            failed_json = match.group(1)
                            # Unescape the JSON string
                            failed_json = failed_json.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            logger.info(f"Extracted failed_generation, attempting to parse...")
                            try:
                                return try_parse_json(failed_json)
                            except Exception as parse_err:
                                logger.warning(f"Failed to parse extracted failed_generation: {parse_err}")
                    except Exception as extract_err:
                        logger.warning(f"Could not extract failed_generation: {extract_err}")

                    # Fallback: retry without JSON mode
                    logger.info(f"Retrying without JSON mode...")
                    response = self.generate(messages, json_mode=False, **kwargs)
                else:
                    raise

            # Check for empty response BEFORE parsing
            if not response.content or not response.content.strip():
                last_error = ValueError("LLM returned empty response")
                backoff_time = 3.0 * (json_attempt + 1)  # Backoff: 3s, 6s, 9s, 12s, 15s
                logger.warning(f"Empty LLM response (attempt {json_attempt + 1}/{max_json_retries}), retrying in {backoff_time}s...")
                if json_attempt < max_json_retries - 1:
                    time.sleep(backoff_time)
                    continue
                else:
                    raise ValueError(f"LLM returned empty response after {max_json_retries} attempts. This may indicate rate limiting or API issues.")

            # Parse JSON with robust error handling
            try:
                # Nettoyer la réponse (enlever les markers de code)
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                # Final empty check after cleaning
                if not content:
                    last_error = ValueError("LLM response empty after cleaning code markers")
                    backoff_time = 3.0 * (json_attempt + 1)
                    logger.warning(f"Response empty after cleaning (attempt {json_attempt + 1}/{max_json_retries}), retrying in {backoff_time}s...")
                    if json_attempt < max_json_retries - 1:
                        time.sleep(backoff_time)
                        continue
                    else:
                        raise ValueError(f"LLM response empty after cleaning, after {max_json_retries} attempts")

                # Use robust JSON parser with multiple strategies
                return try_parse_json(content)

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                backoff_time = 3.0 * (json_attempt + 1)
                logger.warning(f"Failed to parse JSON (attempt {json_attempt + 1}/{max_json_retries}): {e}")
                # Log raw response to help debug Groq empty responses
                raw_preview = response.content[:200] if response.content else 'EMPTY'
                logger.warning(f"Raw response preview: {repr(raw_preview)}")
                if json_attempt < max_json_retries - 1:
                    logger.info(f"Retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                    continue
                else:
                    raise ValueError(f"Invalid JSON response from LLM after {max_json_retries} attempts: {e}")
    
    def _apply_rate_limit(self):
        """Applique le rate limiting"""
        now = time.time()
        minute_ago = now - 60
        
        # Nettoyer les anciennes requêtes
        self._request_timestamps = [t for t in self._request_timestamps if t > minute_ago]
        
        # Vérifier la limite
        if len(self._request_timestamps) >= self.config.requests_per_minute:
            sleep_time = 60 - (now - self._request_timestamps[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self._request_timestamps.append(now)
    
    def _get_cache_key(self, messages: List[LLMMessage]) -> str:
        """Génère une clé de cache pour les messages"""
        content = "".join(f"{m.role}:{m.content}" for m in messages)
        return str(hash(content))
    
    @property
    def total_tokens_used(self) -> int:
        return self._token_usage
    
    def clear_cache(self):
        self._cache.clear()


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Provider compatible OpenAI API.
    Fonctionne avec OpenAI, Groq, et tout service compatible.

    For Groq: Uses the shared rate limiter to handle ERR-017 (rate limits
    causing 40-50 second pauses) with intelligent throttling and key rotation.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self._rate_limiter = None
        self._current_api_key = config.api_key

        # Initialize Groq rate limiter if available and using Groq
        if config.provider == LLMProvider.GROQ and RATE_LIMITER_AVAILABLE:
            try:
                self._rate_limiter = get_groq_rate_limiter()
                if self._rate_limiter.has_keys:
                    logger.info(f"[GROQ] Rate limiter enabled with {self._rate_limiter.key_count} key(s)")
                else:
                    logger.info("[GROQ] Rate limiter initialized but no keys configured, using config key")
            except Exception as e:
                logger.warning(f"[GROQ] Failed to initialize rate limiter: {e}")
                self._rate_limiter = None

    def _get_client(self, api_key: Optional[str] = None):
        """Get or create OpenAI client, optionally with a different API key"""
        key_to_use = api_key or self._current_api_key

        # If key changed, recreate client
        if self._client is not None and api_key and api_key != self._current_api_key:
            self._client = None
            self._current_api_key = api_key

        if self._client is None:
            try:
                from openai import OpenAI

                # Get timeout from environment variable (fixes ERR-014: ReadTimeout)
                timeout = float(os.getenv("LLM_TIMEOUT", "120"))

                kwargs = {
                    "api_key": key_to_use,
                    "timeout": timeout,
                    "max_retries": 2,
                }
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url

                self._client = OpenAI(**kwargs)
                self._current_api_key = key_to_use
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

        return self._client

    def _estimate_tokens(self, messages: List[LLMMessage], max_tokens: int) -> int:
        """Estimate total tokens for a request (input + expected output)"""
        # Rough estimation: ~4 chars per token for input
        input_chars = sum(len(m.content) for m in messages)
        estimated_input_tokens = input_chars // 4
        # Add expected output tokens
        return estimated_input_tokens + max_tokens

    def _call_api(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        api_key_used = self._current_api_key

        # Use rate limiter for Groq if available
        if self.config.provider == LLMProvider.GROQ and self._rate_limiter and self._rate_limiter.has_keys:
            estimated_tokens = self._estimate_tokens(messages, max_tokens)

            # Acquire a rate-limited API key (blocks if necessary)
            api_key_used = self._rate_limiter.acquire_sync(estimated_tokens)
            client = self._get_client(api_key_used)
        else:
            client = self._get_client()

        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        api_kwargs = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": max_tokens,
        }

        # JSON mode si supporté
        if kwargs.get("json_mode") and self.config.provider in [LLMProvider.OPENAI, LLMProvider.GROQ]:
            api_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**api_kwargs)

        # Record actual token usage for rate limiting
        tokens_used = response.usage.total_tokens if response.usage else 0
        if self.config.provider == LLMProvider.GROQ and self._rate_limiter and self._rate_limiter.has_keys:
            record_groq_usage(api_key_used, tokens_used)

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            tokens_used=tokens_used,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )

    def get_rate_limiter_stats(self) -> Optional[Dict]:
        """Get rate limiter statistics (Groq only)"""
        if self._rate_limiter:
            return self._rate_limiter.get_total_stats()
        return None


class AnthropicProvider(BaseLLMProvider):
    """Provider pour l'API Anthropic (Claude)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client
    
    def _call_api(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        client = self._get_client()
        
        # Séparer le system message
        system_content = ""
        user_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_content += msg.content + "\n"
            else:
                user_messages.append({"role": msg.role, "content": msg.content})
        
        api_kwargs = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": user_messages,
        }
        
        if system_content:
            api_kwargs["system"] = system_content.strip()
        
        response = client.messages.create(**api_kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
        )


class OllamaProvider(BaseLLMProvider):
    """Provider pour Ollama (modèles locaux)"""
    
    def _call_api(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        import requests
        
        url = f"{self.config.base_url}/api/chat"
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        # Get timeout from environment variable (fixes ERR-014: ReadTimeout)
        timeout = float(os.getenv("LLM_TIMEOUT", "120"))
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["message"]["content"],
            model=self.config.model,
            tokens_used=data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )


def create_llm_provider(config: LLMConfig) -> BaseLLMProvider:
    """
    Factory pour créer le provider approprié.
    """
    if config.provider == LLMProvider.ANTHROPIC:
        return AnthropicProvider(config)
    elif config.provider == LLMProvider.OLLAMA:
        return OllamaProvider(config)
    else:
        # OpenAI, Groq, et autres compatibles
        return OpenAICompatibleProvider(config)


# Raccourcis pour création rapide
def groq_provider(api_key: str, model: str = "llama-3.3-70b-versatile") -> BaseLLMProvider:
    """Crée un provider Groq"""
    return create_llm_provider(LLMConfig.for_groq(api_key, model))


def openai_provider(api_key: str, model: str = "gpt-4o") -> BaseLLMProvider:
    """Crée un provider OpenAI"""
    return create_llm_provider(LLMConfig.for_openai(api_key, model))


def anthropic_provider(api_key: str, model: str = "claude-sonnet-4-20250514") -> BaseLLMProvider:
    """Crée un provider Anthropic"""
    return create_llm_provider(LLMConfig.for_anthropic(api_key, model))


def ollama_provider(model: str = "llama3.1", base_url: str = "http://localhost:11434") -> BaseLLMProvider:
    """Crée un provider Ollama"""
    return create_llm_provider(LLMConfig.for_ollama(model, base_url))
