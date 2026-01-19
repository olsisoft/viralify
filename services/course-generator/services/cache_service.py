"""
Cache Service for Course Generator

Simple caching layer with Redis backend and in-memory fallback.
Used to cache expensive AI operations like element suggestions.
"""
import hashlib
import json
import os
from typing import Any, Optional
from datetime import datetime, timedelta


class CacheService:
    """Simple cache service with Redis backend and memory fallback"""

    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600):
        """
        Initialize cache service.

        Args:
            redis_url: Redis connection URL (optional)
            default_ttl: Default TTL in seconds (1 hour)
        """
        self.default_ttl = default_ttl
        self.redis_client = None
        self._memory_cache: dict = {}
        self._memory_expiry: dict = {}

        # Try to connect to Redis
        redis_url = redis_url or os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                print("[CACHE] Connected to Redis", flush=True)
            except Exception as e:
                print(f"[CACHE] Redis not available, using memory cache: {e}", flush=True)
                self.redis_client = None
        else:
            print("[CACHE] No Redis URL configured, using memory cache", flush=True)

    def _make_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments"""
        # Create a hash of the arguments for consistent keys
        args_str = json.dumps(args, sort_keys=True, default=str)
        args_hash = hashlib.md5(args_str.encode()).hexdigest()[:12]
        return f"viralify:course:{prefix}:{args_hash}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # Memory cache with expiry check
                if key in self._memory_cache:
                    if datetime.utcnow() < self._memory_expiry.get(key, datetime.min):
                        return self._memory_cache[key]
                    else:
                        # Expired
                        del self._memory_cache[key]
                        del self._memory_expiry[key]
        except Exception as e:
            print(f"[CACHE] Get error: {e}", flush=True)
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        try:
            json_value = json.dumps(value, default=str)
            if self.redis_client:
                self.redis_client.setex(key, ttl, json_value)
            else:
                # Memory cache
                self._memory_cache[key] = value
                self._memory_expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)
            return True
        except Exception as e:
            print(f"[CACHE] Set error: {e}", flush=True)
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            elif key in self._memory_cache:
                del self._memory_cache[key]
                self._memory_expiry.pop(key, None)
            return True
        except Exception as e:
            print(f"[CACHE] Delete error: {e}", flush=True)
            return False

    def get_stats(self) -> dict:
        """Get cache statistics"""
        if self.redis_client:
            try:
                info = self.redis_client.info("memory")
                return {
                    "backend": "redis",
                    "memory_used": info.get("used_memory_human", "unknown"),
                }
            except:
                return {"backend": "redis", "status": "error"}
        else:
            # Clean expired entries
            now = datetime.utcnow()
            expired = [k for k, exp in self._memory_expiry.items() if exp < now]
            for k in expired:
                self._memory_cache.pop(k, None)
                self._memory_expiry.pop(k, None)

            return {
                "backend": "memory",
                "entries": len(self._memory_cache),
            }


# Global cache instance (singleton)
_cache_instance: Optional[CacheService] = None


def get_cache() -> CacheService:
    """Get the global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheService()
    return _cache_instance
