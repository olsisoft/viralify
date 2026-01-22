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

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import redis.asyncio as redis


class RedisJobStore:
    """
    Redis-based job storage for presentation generation jobs.

    Supports multiple job types via prefixes:
    - pres:v1: Legacy compositor jobs
    - pres:v2: LangGraph jobs
    - pres:v3: MultiAgent jobs (main)
    """

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._connected = False

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None or not self._connected:
            redis_host = os.getenv("REDIS_HOST", "redis")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_password = os.getenv("REDIS_PASSWORD", None)
            redis_db = int(os.getenv("REDIS_JOB_DB", "1"))  # Use DB 1 for jobs

            # Debug: log connection params (password masked)
            print(f"[REDIS_JOB_STORE] Connecting with host={redis_host}, port={redis_port}, "
                  f"db={redis_db}, password={'***' if redis_password else 'None'}", flush=True)

            self._redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            try:
                await self._redis.ping()
                self._connected = True
                print(f"[REDIS_JOB_STORE] Connected to Redis at {redis_host}:{redis_port}/db{redis_db}", flush=True)
            except Exception as e:
                print(f"[REDIS_JOB_STORE] Failed to connect to Redis: {e}", flush=True)
                self._connected = False
                raise

        return self._redis

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
        Save a job to Redis.

        Args:
            job_id: Unique job identifier
            data: Job data dictionary
            prefix: Job type prefix (v1, v2, v3)
            ttl_seconds: Time to live in seconds (default 24h)

        Returns:
            True if successful
        """
        try:
            r = await self._get_redis()
            key = self._make_key(job_id, prefix)
            index_key = self._make_index_key(prefix)

            # Add timestamp if not present
            if "updated_at" not in data:
                data["updated_at"] = datetime.utcnow().isoformat()

            # Serialize datetime objects
            serialized_data = self._serialize_data(data)

            # Save job data
            await r.set(key, json.dumps(serialized_data), ex=ttl_seconds)

            # Add to index (sorted set by creation time for listing)
            created_at = data.get("created_at", datetime.utcnow().isoformat())
            if isinstance(created_at, str):
                try:
                    score = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
                except:
                    score = datetime.utcnow().timestamp()
            else:
                score = created_at.timestamp() if hasattr(created_at, 'timestamp') else datetime.utcnow().timestamp()

            await r.zadd(index_key, {job_id: score})
            await r.expire(index_key, ttl_seconds)

            return True

        except Exception as e:
            print(f"[REDIS_JOB_STORE] Error saving job {job_id}: {e}", flush=True)
            return False

    async def get(self, job_id: str, prefix: str = "v3") -> Optional[Dict[str, Any]]:
        """
        Get a job from Redis.

        Args:
            job_id: Job identifier
            prefix: Job type prefix

        Returns:
            Job data dictionary or None if not found
        """
        try:
            r = await self._get_redis()
            key = self._make_key(job_id, prefix)

            data = await r.get(key)
            if data:
                return json.loads(data)
            return None

        except Exception as e:
            print(f"[REDIS_JOB_STORE] Error getting job {job_id}: {e}", flush=True)
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
            self._connected = False


# Global instance for easy import
job_store = RedisJobStore()
