"""
HTTP Client with Retry Logic

Provides a resilient HTTP client that handles transient network failures,
DNS resolution issues, and connection timeouts common in Docker environments.
"""

import asyncio
import httpx
from typing import Any, Dict, Optional
from functools import wraps


class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        retry_on_status: tuple = (502, 503, 504),
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_on_status = retry_on_status


DEFAULT_RETRY_CONFIG = RetryConfig()


class ResilientHTTPClient:
    """
    HTTP client with built-in retry logic for Docker network resilience.

    Handles:
    - DNS resolution failures (common after container restarts)
    - Connection timeouts
    - Temporary service unavailability
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _should_retry(self, exception: Exception) -> bool:
        """Determine if we should retry based on the exception"""
        # Network-related errors that are typically transient
        retryable_errors = (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            ConnectionRefusedError,
            OSError,  # Includes DNS resolution failures
        )
        return isinstance(exception, retryable_errors)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff"""
        delay = self.retry_config.initial_delay * (
            self.retry_config.exponential_base ** attempt
        )
        return min(delay, self.retry_config.max_delay)

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """Execute request with retry logic"""
        last_exception = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.request(method, path, **kwargs)

                # Check if we should retry based on status code
                if response.status_code in self.retry_config.retry_on_status:
                    if attempt < self.retry_config.max_retries:
                        delay = self._calculate_delay(attempt)
                        print(f"[HTTP_CLIENT] Status {response.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.retry_config.max_retries})", flush=True)
                        await asyncio.sleep(delay)
                        continue

                return response

            except Exception as e:
                last_exception = e

                if not self._should_retry(e):
                    raise

                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_delay(attempt)
                    error_type = type(e).__name__
                    print(f"[HTTP_CLIENT] {error_type}: {str(e)[:100]}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.retry_config.max_retries})", flush=True)

                    # Close and recreate client on connection errors
                    await self.close()
                    await asyncio.sleep(delay)
                else:
                    print(f"[HTTP_CLIENT] Max retries exceeded for {method} {path}", flush=True)
                    raise

        raise last_exception

    async def get(self, path: str, **kwargs) -> httpx.Response:
        """GET request with retry"""
        return await self._request_with_retry("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> httpx.Response:
        """POST request with retry"""
        return await self._request_with_retry("POST", path, **kwargs)

    async def put(self, path: str, **kwargs) -> httpx.Response:
        """PUT request with retry"""
        return await self._request_with_retry("PUT", path, **kwargs)

    async def delete(self, path: str, **kwargs) -> httpx.Response:
        """DELETE request with retry"""
        return await self._request_with_retry("DELETE", path, **kwargs)

    async def get_json(self, path: str, **kwargs) -> Dict[str, Any]:
        """GET request returning JSON"""
        response = await self.get(path, **kwargs)
        response.raise_for_status()
        return response.json()

    async def post_json(self, path: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """POST request with JSON body returning JSON"""
        response = await self.post(path, json=data, **kwargs)
        response.raise_for_status()
        return response.json()


# Singleton clients for common services
_clients: Dict[str, ResilientHTTPClient] = {}


def get_service_client(service_name: str, base_url: str) -> ResilientHTTPClient:
    """Get or create a resilient client for a service"""
    if service_name not in _clients:
        _clients[service_name] = ResilientHTTPClient(base_url)
    return _clients[service_name]


async def close_all_clients():
    """Close all HTTP clients (call on shutdown)"""
    for client in _clients.values():
        await client.close()
    _clients.clear()
