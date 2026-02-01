"""
Cache Metrics Module

Provides Prometheus metrics for cache hit/miss tracking across Viralify services.
Supports optional import - services gracefully degrade if prometheus-client is not installed.

Usage:
    from shared.cache_metrics import CacheMetrics

    metrics = CacheMetrics("redis_job_store", "presentation-generator")
    metrics.hit()   # Record a cache hit
    metrics.miss()  # Record a cache miss

    hit_rate = metrics.get_hit_rate()  # Returns 0.0 - 1.0
"""

import time
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# Only create metrics if prometheus-client is available
if PROMETHEUS_AVAILABLE:
    CACHE_HITS = Counter(
        'viralify_cache_hits_total',
        'Total number of cache hits',
        ['cache_type', 'service']
    )

    CACHE_MISSES = Counter(
        'viralify_cache_misses_total',
        'Total number of cache misses',
        ['cache_type', 'service']
    )

    CACHE_SIZE = Gauge(
        'viralify_cache_size',
        'Current size of cache (entries)',
        ['cache_type', 'service']
    )

    CACHE_LATENCY = Histogram(
        'viralify_cache_operation_seconds',
        'Cache operation latency in seconds',
        ['cache_type', 'service', 'operation'],
        buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0]
    )


class CacheMetrics:
    """
    Wrapper for cache metrics with convenient methods.

    Falls back to no-op counters if prometheus-client is not installed.
    """

    def __init__(self, cache_type: str, service: str):
        """
        Initialize cache metrics.

        Args:
            cache_type: Type of cache (e.g., "redis_job_store", "embedding_engine")
            service: Service name (e.g., "presentation-generator", "course-generator")
        """
        self.cache_type = cache_type
        self.service = service
        self._hits = 0
        self._misses = 0

    def hit(self):
        """Record a cache hit."""
        self._hits += 1
        if PROMETHEUS_AVAILABLE:
            CACHE_HITS.labels(cache_type=self.cache_type, service=self.service).inc()

    def miss(self):
        """Record a cache miss."""
        self._misses += 1
        if PROMETHEUS_AVAILABLE:
            CACHE_MISSES.labels(cache_type=self.cache_type, service=self.service).inc()

    def set_size(self, size: int):
        """Set the current cache size."""
        if PROMETHEUS_AVAILABLE:
            CACHE_SIZE.labels(cache_type=self.cache_type, service=self.service).set(size)

    def observe_latency(self, operation: str, seconds: float):
        """Record operation latency."""
        if PROMETHEUS_AVAILABLE:
            CACHE_LATENCY.labels(
                cache_type=self.cache_type,
                service=self.service,
                operation=operation
            ).observe(seconds)

    def timed_operation(self, operation: str):
        """Context manager for timing cache operations."""
        return _TimedOperation(self, operation)

    def get_hit_rate(self) -> float:
        """
        Calculate hit rate (hits / total).

        Returns:
            Float between 0.0 and 1.0, or 0.0 if no operations recorded.
        """
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Get current statistics."""
        total = self._hits + self._misses
        return {
            "cache_type": self.cache_type,
            "service": self.service,
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": f"{self.get_hit_rate() * 100:.1f}%",
            "prometheus_enabled": PROMETHEUS_AVAILABLE,
        }


class _TimedOperation:
    """Context manager for timing cache operations."""

    def __init__(self, metrics: CacheMetrics, operation: str):
        self.metrics = metrics
        self.operation = operation
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.metrics.observe_latency(self.operation, elapsed)
        return False


# Singleton instances for common caches (created on first import)
_instances: dict[str, CacheMetrics] = {}


def get_metrics(cache_type: str, service: str) -> CacheMetrics:
    """
    Get or create a CacheMetrics instance (singleton per cache_type+service).

    Args:
        cache_type: Type of cache
        service: Service name

    Returns:
        CacheMetrics instance
    """
    key = f"{cache_type}:{service}"
    if key not in _instances:
        _instances[key] = CacheMetrics(cache_type, service)
    return _instances[key]


def get_all_stats() -> list[dict]:
    """Get statistics for all registered cache metrics."""
    return [m.get_stats() for m in _instances.values()]
