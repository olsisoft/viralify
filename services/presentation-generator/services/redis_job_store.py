"""
Redis Job Store for Presentation Generator

Provides shared job storage across multiple Uvicorn workers.
Replaces in-memory dictionaries that don't work with multiple workers.

Usage:
    from services.redis_job_store import job_store

    # Save a job
    await job_store.save("job-123", {"status": "processing", ...}, prefix="v3")

    # Get a job
    job = await job_store.get("job-123", prefix="v3")

    # List jobs
    jobs = await job_store.list(prefix="v3", limit=20)
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import redis.asyncio as redis


class RedisConnectionError(Exception):
    """Raised when Redis connection fails (timeout, connection refused, etc.)"""
    pass


class RedisJobStore:
    """
    Redis-based job storage for presentation generation jobs.

    Supports multiple job types via prefixes:
    - pres:v1: Legacy compositor jobs
    - pres:v2: LangGraph jobs
    - pres:v3: MultiAgent jobs (main)

    Uses connection pooling for better concurrency handling.
    """

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        # Check if already connected (without lock for fast path)
        if self._redis is not None and self._connected:
            return self._redis

        async with self._lock:
            # Double-check after acquiring lock
            if self._redis is not None and self._connected:
                return self._redis

            redis_host = os.getenv("REDIS_HOST", "redis")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_password = os.getenv("REDIS_PASSWORD")
            redis_db = int(os.getenv("REDIS_JOB_DB", "6"))

            print(f"[REDIS_JOB_STORE] Connecting to redis://{redis_host}:{redis_port}/db{redis_db}", flush=True)

            # Create Redis client - simple and robust
            self._redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=30,
                socket_timeout=30,
            )

            # Test connection
            try:
                await self._redis.ping()
                self._connected = True
                print(f"[REDIS_JOB_STORE] Connected to Redis successfully", flush=True)
            except Exception as e:
                print(f"[REDIS_JOB_STORE] Failed to connect to Redis: {e}", flush=True)
                self._connected = False
                if self._redis:
                    await self._redis.close()
                self._redis = None
                raise

        return self._redis

    async def _execute_with_retry(self, operation, *args, max_retries: int = 3, **kwargs):
        """Execute a Redis operation with retry logic."""
        last_error = None
        for attempt in range(max_retries):
            try:
                r = await self._get_redis()
                return await operation(r, *args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"[REDIS_JOB_STORE] Retry {attempt + 1}/{max_retries}: {e}", flush=True)
                    # Reset connection on error
                    self._connected = False
                    await asyncio.sleep(0.5 * (attempt + 1))  # Backoff: 0.5s, 1s, 1.5s
        raise last_error

    def _make_key(self, job_id: str, prefix: str = "v3") -> str:
        """Generate Redis key for a job."""
        return f"pres:{prefix}:{job_id}"

    def _make_index_key(self, prefix: str = "v3") -> str:
        """Generate Redis key for job index (sorted set by creation time)."""
        return f"pres:{prefix}:index"

    async def save(
        self,
        job_id: str,
        data: Dict[str, Any],
        prefix: str = "v3",
        ttl_seconds: int = 86400  # 24 hours default
    ) -> bool:
        """
        Save a job to Redis with automatic retry.

        Args:
            job_id: Unique job identifier
            data: Job data dictionary
            prefix: Job type prefix (v1, v2, v3)
            ttl_seconds: Time to live in seconds (default 24h)

        Returns:
            True if successful
        """
        async def _do_save(r, key, index_key, serialized_data, score, ttl):
            # Save job data
            await r.set(key, json.dumps(serialized_data), ex=ttl)
            # Add to index
            await r.zadd(index_key, {job_id: score})
            await r.expire(index_key, ttl)
            return True

        try:
            key = self._make_key(job_id, prefix)
            index_key = self._make_index_key(prefix)

            # Add timestamp if not present
            if "updated_at" not in data:
                data["updated_at"] = datetime.utcnow().isoformat()

            # Serialize datetime objects
            serialized_data = self._serialize_data(data)

            # Calculate score for index
            created_at = data.get("created_at", datetime.utcnow().isoformat())
            if isinstance(created_at, str):
                try:
                    score = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
                except:
                    score = datetime.utcnow().timestamp()
            else:
                score = created_at.timestamp() if hasattr(created_at, 'timestamp') else datetime.utcnow().timestamp()

            return await self._execute_with_retry(
                _do_save, key, index_key, serialized_data, score, ttl_seconds,
                max_retries=3
            )

        except Exception as e:
            print(f"[REDIS_JOB_STORE] Error saving job {job_id} after retries: {e}", flush=True)
            return False

    async def get(self, job_id: str, prefix: str = "v3", raise_on_error: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get a job from Redis with automatic retry.

        Args:
            job_id: Job identifier
            prefix: Job type prefix
            raise_on_error: If True, raises RedisConnectionError on Redis failures.
                           If False, returns None (legacy behavior).

        Returns:
            Job data dictionary or None if not found

        Raises:
            RedisConnectionError: When Redis is unavailable and raise_on_error=True
        """
        async def _do_get(r, key):
            data = await r.get(key)
            if data:
                return json.loads(data)
            return None

        try:
            key = self._make_key(job_id, prefix)
            return await self._execute_with_retry(_do_get, key, max_retries=3)

        except Exception as e:
            print(f"[REDIS_JOB_STORE] Error getting job {job_id} after retries: {e}", flush=True)
            if raise_on_error:
                raise RedisConnectionError(f"Redis unavailable: {e}")
            return None

    async def exists(self, job_id: str, prefix: str = "v3") -> bool:
        """Check if a job exists."""
        try:
            r = await self._get_redis()
            key = self._make_key(job_id, prefix)
            return await r.exists(key) > 0
        except Exception as e:
            print(f"[REDIS_JOB_STORE] Error checking job {job_id}: {e}", flush=True)
            return False

    async def delete(self, job_id: str, prefix: str = "v3") -> bool:
        """Delete a job from Redis."""
        try:
            r = await self._get_redis()
            key = self._make_key(job_id, prefix)
            index_key = self._make_index_key(prefix)

            await r.delete(key)
            await r.zrem(index_key, job_id)

            return True

        except Exception as e:
            print(f"[REDIS_JOB_STORE] Error deleting job {job_id}: {e}", flush=True)
            return False

    async def list(
        self,
        prefix: str = "v3",
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List jobs, sorted by creation time (newest first).

        Args:
            prefix: Job type prefix
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of job data dictionaries
        """
        try:
            r = await self._get_redis()
            index_key = self._make_index_key(prefix)

            # Get job IDs from sorted set (newest first)
            job_ids = await r.zrevrange(index_key, offset, offset + limit - 1)

            if not job_ids:
                return []

            # Get all job data
            jobs = []
            for job_id in job_ids:
                job_data = await self.get(job_id, prefix)
                if job_data:
                    jobs.append(job_data)

            return jobs

        except Exception as e:
            print(f"[REDIS_JOB_STORE] Error listing jobs: {e}", flush=True)
            return []

    async def update_field(
        self,
        job_id: str,
        field: str,
        value: Any,
        prefix: str = "v3"
    ) -> bool:
        """
        Update a single field in a job.

        More efficient than get/modify/save for single field updates.
        """
        try:
            job_data = await self.get(job_id, prefix)
            if job_data is None:
                return False

            job_data[field] = value
            job_data["updated_at"] = datetime.utcnow().isoformat()

            return await self.save(job_id, job_data, prefix)

        except Exception as e:
            print(f"[REDIS_JOB_STORE] Error updating field {field} for job {job_id}: {e}", flush=True)
            return False

    async def update_fields(
        self,
        job_id: str,
        updates: Dict[str, Any],
        prefix: str = "v3"
    ) -> bool:
        """
        Update multiple fields in a job.
        """
        try:
            job_data = await self.get(job_id, prefix)
            if job_data is None:
                return False

            job_data.update(updates)
            job_data["updated_at"] = datetime.utcnow().isoformat()

            return await self.save(job_id, job_data, prefix)

        except Exception as e:
            print(f"[REDIS_JOB_STORE] Error updating fields for job {job_id}: {e}", flush=True)
            return False

    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data for JSON storage (handle datetime objects)."""
        result = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = self._serialize_data(value)
            elif isinstance(value, list):
                result[key] = [
                    self._serialize_data(item) if isinstance(item, dict)
                    else item.isoformat() if isinstance(item, datetime)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
        self._redis = None
        self._connected = False
        print("[REDIS_JOB_STORE] Connection closed", flush=True)


# Global instance for easy import
job_store = RedisJobStore()
