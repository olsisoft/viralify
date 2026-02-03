"""
Groq API Rate Limiter with API Key Rotation

Handles ERR-017: Rate limits causing 40-50 second pauses by implementing:
- Request per minute tracking (default: 30 req/min for free tier)
- Token per minute tracking (default: 6000 tokens/min)
- Intelligent throttling (waits just enough time to stay under limit)
- Multiple API key rotation (round-robin)

Environment Variables:
    GROQ_API_KEYS: Comma-separated list of API keys for rotation
    GROQ_REQUESTS_PER_MINUTE: Max requests per minute per key (default: 30)
    GROQ_TOKENS_PER_MINUTE: Max tokens per minute per key (default: 6000)
    GROQ_RATE_LIMIT_BUFFER: Safety buffer percentage (default: 0.9 = 90% of limit)

Usage:
    from shared.groq_rate_limiter import GroqRateLimiter, get_groq_rate_limiter

    # Get singleton instance
    rate_limiter = get_groq_rate_limiter()

    # Before making a request
    api_key = await rate_limiter.acquire(estimated_tokens=1000)

    # After request completes
    rate_limiter.record_usage(api_key, tokens_used=850)

    # Or use the context manager
    async with rate_limiter.request(estimated_tokens=1000) as api_key:
        response = await groq_client.chat.completions.create(...)
        rate_limiter.record_usage(api_key, response.usage.total_tokens)
"""

import asyncio
import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, AsyncIterator
from contextlib import asynccontextmanager
from collections import deque
import threading


logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 30
    tokens_per_minute: int = 6000
    buffer_percentage: float = 0.9  # Use 90% of limit as safety buffer
    min_wait_seconds: float = 0.1   # Minimum wait between requests
    max_wait_seconds: float = 60.0  # Maximum wait time

    @property
    def effective_rpm(self) -> int:
        """Effective requests per minute with buffer"""
        return int(self.requests_per_minute * self.buffer_percentage)

    @property
    def effective_tpm(self) -> int:
        """Effective tokens per minute with buffer"""
        return int(self.tokens_per_minute * self.buffer_percentage)


@dataclass
class KeyUsageStats:
    """Usage statistics for a single API key"""
    api_key: str
    request_timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    token_usage: deque = field(default_factory=lambda: deque(maxlen=1000))
    total_requests: int = 0
    total_tokens: int = 0
    throttle_count: int = 0
    last_request_time: float = 0.0

    def get_requests_in_window(self, window_seconds: float = 60.0) -> int:
        """Count requests in the sliding window"""
        now = time.time()
        cutoff = now - window_seconds
        count = sum(1 for ts in self.request_timestamps if ts > cutoff)
        return count

    def get_tokens_in_window(self, window_seconds: float = 60.0) -> int:
        """Count tokens used in the sliding window"""
        now = time.time()
        cutoff = now - window_seconds
        total = sum(tokens for ts, tokens in self.token_usage if ts > cutoff)
        return total

    def add_request(self):
        """Record a new request"""
        now = time.time()
        self.request_timestamps.append(now)
        self.last_request_time = now
        self.total_requests += 1

    def add_tokens(self, tokens: int):
        """Record token usage"""
        now = time.time()
        self.token_usage.append((now, tokens))
        self.total_tokens += tokens

    def time_until_request_available(self, config: RateLimitConfig) -> float:
        """Calculate time until a request slot is available"""
        current_requests = self.get_requests_in_window()

        if current_requests < config.effective_rpm:
            return 0.0

        # Find the oldest request in the window
        now = time.time()
        cutoff = now - 60.0

        for ts in self.request_timestamps:
            if ts > cutoff:
                # Wait until this request falls out of the window
                wait_time = (ts + 60.0) - now
                return max(0.0, wait_time)

        return 0.0

    def time_until_tokens_available(self, config: RateLimitConfig, required_tokens: int) -> float:
        """Calculate time until enough token budget is available"""
        current_tokens = self.get_tokens_in_window()
        available_tokens = config.effective_tpm - current_tokens

        if available_tokens >= required_tokens:
            return 0.0

        # Find when old tokens will fall out of the window
        now = time.time()
        cutoff = now - 60.0
        tokens_needed = required_tokens - available_tokens
        cumulative_freed = 0

        for ts, tokens in self.token_usage:
            if ts > cutoff:
                cumulative_freed += tokens
                if cumulative_freed >= tokens_needed:
                    wait_time = (ts + 60.0) - now
                    return max(0.0, wait_time)

        # Worst case: wait full minute
        return 60.0


class GroqRateLimiter:
    """
    Rate limiter for Groq API with multi-key rotation.

    Features:
    - Request per minute limiting per API key
    - Token per minute limiting per API key
    - Round-robin key rotation
    - Intelligent throttling with minimal wait times
    - Thread-safe and async-compatible

    Example:
        limiter = GroqRateLimiter()

        # Acquire a key and wait if necessary
        api_key = await limiter.acquire(estimated_tokens=500)

        # Make your API call...
        response = await client.chat.completions.create(...)

        # Record actual usage
        limiter.record_usage(api_key, response.usage.total_tokens)
    """

    _instance: Optional['GroqRateLimiter'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for consistent rate limiting across the application"""
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
        self._async_lock = asyncio.Lock()
        self._key_index = 0

        # Load configuration from environment
        self.config = RateLimitConfig(
            requests_per_minute=int(os.getenv("GROQ_REQUESTS_PER_MINUTE", "30")),
            tokens_per_minute=int(os.getenv("GROQ_TOKENS_PER_MINUTE", "6000")),
            buffer_percentage=float(os.getenv("GROQ_RATE_LIMIT_BUFFER", "0.9")),
        )

        # Load API keys
        keys_str = os.getenv("GROQ_API_KEYS", "")
        if keys_str:
            self._api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            # Fallback to single key
            single_key = os.getenv("GROQ_API_KEY", "")
            self._api_keys = [single_key] if single_key else []

        # Initialize usage stats for each key
        self._key_stats: Dict[str, KeyUsageStats] = {
            key: KeyUsageStats(api_key=key) for key in self._api_keys
        }

        # Log initialization
        key_count = len(self._api_keys)
        if key_count > 0:
            # Mask keys for logging
            masked_keys = [f"{k[:8]}...{k[-4:]}" if len(k) > 12 else "***" for k in self._api_keys]
            logger.info(f"[GROQ_RATE_LIMITER] Initialized with {key_count} API key(s): {masked_keys}")
            logger.info(f"[GROQ_RATE_LIMITER] Limits: {self.config.effective_rpm} req/min, "
                       f"{self.config.effective_tpm} tokens/min per key")
        else:
            logger.warning("[GROQ_RATE_LIMITER] No API keys configured! Set GROQ_API_KEYS or GROQ_API_KEY")

    @property
    def has_keys(self) -> bool:
        """Check if any API keys are configured"""
        return len(self._api_keys) > 0

    @property
    def key_count(self) -> int:
        """Number of API keys available"""
        return len(self._api_keys)

    def _get_next_key_index(self) -> int:
        """Get the next key index in round-robin fashion"""
        if not self._api_keys:
            raise ValueError("No API keys configured")

        index = self._key_index
        self._key_index = (self._key_index + 1) % len(self._api_keys)
        return index

    def _find_best_key(self, estimated_tokens: int) -> tuple[str, float]:
        """
        Find the best available key with minimum wait time.

        Returns:
            Tuple of (api_key, wait_time_seconds)
        """
        if not self._api_keys:
            raise ValueError("No API keys configured")

        best_key = None
        best_wait = float('inf')

        # Check all keys to find the one with shortest wait
        for key in self._api_keys:
            stats = self._key_stats[key]

            # Calculate wait for both requests and tokens
            request_wait = stats.time_until_request_available(self.config)
            token_wait = stats.time_until_tokens_available(self.config, estimated_tokens)

            # Total wait is the max of both constraints
            total_wait = max(request_wait, token_wait)

            if total_wait < best_wait:
                best_wait = total_wait
                best_key = key

                # Early exit if no wait needed
                if total_wait == 0:
                    break

        return best_key, best_wait

    async def acquire(self, estimated_tokens: int = 500) -> str:
        """
        Acquire an API key, waiting if necessary for rate limits.

        This method will:
        1. Find the key with the shortest wait time
        2. Wait if necessary to respect rate limits
        3. Record the request
        4. Return the API key to use

        Args:
            estimated_tokens: Estimated tokens for the request (for token limiting)

        Returns:
            API key to use for the request

        Raises:
            ValueError: If no API keys are configured
        """
        if not self.has_keys:
            raise ValueError("No Groq API keys configured. Set GROQ_API_KEYS or GROQ_API_KEY environment variable.")

        async with self._async_lock:
            api_key, wait_time = self._find_best_key(estimated_tokens)

            # Apply wait time if needed
            if wait_time > 0:
                # Cap wait time
                wait_time = min(wait_time, self.config.max_wait_seconds)

                # Log throttling
                stats = self._key_stats[api_key]
                stats.throttle_count += 1

                masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
                logger.info(f"[GROQ_RATE_LIMITER] Throttling: waiting {wait_time:.2f}s for key {masked_key} "
                           f"(requests: {stats.get_requests_in_window()}/{self.config.effective_rpm}, "
                           f"tokens: {stats.get_tokens_in_window()}/{self.config.effective_tpm})")

                await asyncio.sleep(wait_time)

            # Record the request
            self._key_stats[api_key].add_request()

            return api_key

    def acquire_sync(self, estimated_tokens: int = 500) -> str:
        """
        Synchronous version of acquire for non-async code.

        Args:
            estimated_tokens: Estimated tokens for the request

        Returns:
            API key to use for the request
        """
        if not self.has_keys:
            raise ValueError("No Groq API keys configured. Set GROQ_API_KEYS or GROQ_API_KEY environment variable.")

        with self._lock:
            api_key, wait_time = self._find_best_key(estimated_tokens)

            if wait_time > 0:
                wait_time = min(wait_time, self.config.max_wait_seconds)

                stats = self._key_stats[api_key]
                stats.throttle_count += 1

                masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
                logger.info(f"[GROQ_RATE_LIMITER] Throttling (sync): waiting {wait_time:.2f}s for key {masked_key}")

                time.sleep(wait_time)

            self._key_stats[api_key].add_request()

            return api_key

    def record_usage(self, api_key: str, tokens_used: int):
        """
        Record token usage after a request completes.

        Call this after receiving a response to update the token budget.

        Args:
            api_key: The API key that was used
            tokens_used: Number of tokens consumed (input + output)
        """
        if api_key in self._key_stats:
            self._key_stats[api_key].add_tokens(tokens_used)
        else:
            logger.warning(f"[GROQ_RATE_LIMITER] Unknown API key: {api_key[:8]}...")

    @asynccontextmanager
    async def request(self, estimated_tokens: int = 500) -> AsyncIterator[str]:
        """
        Context manager for making rate-limited requests.

        Usage:
            async with rate_limiter.request(estimated_tokens=1000) as api_key:
                response = await client.chat.completions.create(...)
                # Token recording should be done manually after getting response

        Args:
            estimated_tokens: Estimated tokens for the request

        Yields:
            API key to use for the request
        """
        api_key = await self.acquire(estimated_tokens)
        try:
            yield api_key
        finally:
            pass  # Token recording is manual

    def get_stats(self) -> Dict[str, Dict]:
        """
        Get current usage statistics for all keys.

        Returns:
            Dictionary with stats for each key
        """
        stats = {}
        for key, key_stats in self._key_stats.items():
            masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
            stats[masked_key] = {
                "requests_last_minute": key_stats.get_requests_in_window(),
                "tokens_last_minute": key_stats.get_tokens_in_window(),
                "total_requests": key_stats.total_requests,
                "total_tokens": key_stats.total_tokens,
                "throttle_count": key_stats.throttle_count,
                "available_requests": max(0, self.config.effective_rpm - key_stats.get_requests_in_window()),
                "available_tokens": max(0, self.config.effective_tpm - key_stats.get_tokens_in_window()),
            }
        return stats

    def get_total_stats(self) -> Dict:
        """
        Get aggregated statistics across all keys.

        Returns:
            Dictionary with total stats
        """
        total_requests = sum(s.total_requests for s in self._key_stats.values())
        total_tokens = sum(s.total_tokens for s in self._key_stats.values())
        total_throttles = sum(s.throttle_count for s in self._key_stats.values())

        return {
            "key_count": self.key_count,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_throttles": total_throttles,
            "config": {
                "requests_per_minute_per_key": self.config.requests_per_minute,
                "tokens_per_minute_per_key": self.config.tokens_per_minute,
                "effective_rpm_per_key": self.config.effective_rpm,
                "effective_tpm_per_key": self.config.effective_tpm,
                "total_effective_rpm": self.config.effective_rpm * self.key_count,
                "total_effective_tpm": self.config.effective_tpm * self.key_count,
            }
        }

    def reset(self):
        """Reset all usage statistics (useful for testing)"""
        with self._lock:
            for key in self._api_keys:
                self._key_stats[key] = KeyUsageStats(api_key=key)
            self._key_index = 0
            logger.info("[GROQ_RATE_LIMITER] Stats reset")


# Singleton accessor
def get_groq_rate_limiter() -> GroqRateLimiter:
    """Get the singleton GroqRateLimiter instance"""
    return GroqRateLimiter()


# Convenience function for simple usage
async def acquire_groq_key(estimated_tokens: int = 500) -> str:
    """
    Convenience function to acquire a rate-limited Groq API key.

    Args:
        estimated_tokens: Estimated tokens for the request

    Returns:
        API key to use
    """
    return await get_groq_rate_limiter().acquire(estimated_tokens)


def acquire_groq_key_sync(estimated_tokens: int = 500) -> str:
    """
    Synchronous convenience function to acquire a rate-limited Groq API key.

    Args:
        estimated_tokens: Estimated tokens for the request

    Returns:
        API key to use
    """
    return get_groq_rate_limiter().acquire_sync(estimated_tokens)


def record_groq_usage(api_key: str, tokens_used: int):
    """
    Convenience function to record Groq API token usage.

    Args:
        api_key: The API key that was used
        tokens_used: Number of tokens consumed
    """
    get_groq_rate_limiter().record_usage(api_key, tokens_used)
