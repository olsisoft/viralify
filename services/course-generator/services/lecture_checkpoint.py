"""
Lecture Checkpoint Service

Tracks lecture generation progress in Redis to enable:
1. Skip already completed lectures on retry
2. Resume generation after OOM/crash
3. Partial success tracking

Usage:
    from services.lecture_checkpoint import checkpoint_service

    # Check if lecture already done
    if await checkpoint_service.is_completed(job_id, lecture_id):
        video_url = await checkpoint_service.get_video_url(job_id, lecture_id)
        return video_url  # Skip regeneration

    # After successful generation
    await checkpoint_service.mark_completed(job_id, lecture_id, video_url)
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import redis.asyncio as redis


class LectureCheckpointService:
    """
    Redis-based checkpoint storage for lecture generation.

    Key structure:
    - course:checkpoint:{job_id}:lectures -> Hash of lecture_id -> status JSON
    - course:checkpoint:{job_id}:meta -> Job metadata (timestamps, counts)

    Uses connection pooling for better concurrency handling.
    """

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection using from_url (more stable)."""
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
            redis_db = int(os.getenv("REDIS_CHECKPOINT_DB", "7"))

            # Build URL - from_url is more stable than Redis() with params
            if redis_password:
                redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
            else:
                redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

            print(f"[CHECKPOINT] Connecting to redis://{redis_host}:{redis_port}/db{redis_db}", flush=True)

            try:
                # Use from_url - avoids recursion issues in redis-py 5.x
                self._redis = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=30,
                    socket_timeout=30,
                )
                await self._redis.ping()
                self._connected = True
                print(f"[CHECKPOINT] Connected to Redis successfully", flush=True)
            except Exception as e:
                print(f"[CHECKPOINT] Failed to connect to Redis: {e}", flush=True)
                self._connected = False
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
                    print(f"[CHECKPOINT] Retry {attempt + 1}/{max_retries}: {e}", flush=True)
                    # Reset connection on error
                    self._connected = False
                    await asyncio.sleep(0.5 * (attempt + 1))  # Backoff: 0.5s, 1s, 1.5s
        raise last_error

    def _lecture_key(self, job_id: str) -> str:
        return f"course:checkpoint:{job_id}:lectures"

    def _meta_key(self, job_id: str) -> str:
        return f"course:checkpoint:{job_id}:meta"

    async def is_completed(self, job_id: str, lecture_id: str) -> bool:
        """Check if a lecture has already been successfully generated."""
        async def _do_check(r, key, lid):
            data = await r.hget(key, lid)
            if data:
                status = json.loads(data)
                return status.get("status") == "completed" and status.get("video_url")
            return False

        try:
            return await self._execute_with_retry(
                _do_check, self._lecture_key(job_id), lecture_id, max_retries=3
            )
        except Exception as e:
            print(f"[CHECKPOINT] Error checking completion for {lecture_id}: {e}", flush=True)
            return False

    async def get_video_url(self, job_id: str, lecture_id: str) -> Optional[str]:
        """Get the video URL for a completed lecture."""
        async def _do_get(r, key, lid):
            data = await r.hget(key, lid)
            if data:
                status = json.loads(data)
                return status.get("video_url")
            return None

        try:
            return await self._execute_with_retry(
                _do_get, self._lecture_key(job_id), lecture_id, max_retries=3
            )
        except Exception as e:
            print(f"[CHECKPOINT] Error getting video URL for {lecture_id}: {e}", flush=True)
            return None

    async def get_lecture_status(self, job_id: str, lecture_id: str) -> Optional[Dict[str, Any]]:
        """Get full status for a lecture."""
        async def _do_get(r, key, lid):
            data = await r.hget(key, lid)
            if data:
                return json.loads(data)
            return None

        try:
            return await self._execute_with_retry(
                _do_get, self._lecture_key(job_id), lecture_id, max_retries=3
            )
        except Exception as e:
            print(f"[CHECKPOINT] Error getting status for {lecture_id}: {e}", flush=True)
            return None

    async def mark_completed(
        self,
        job_id: str,
        lecture_id: str,
        video_url: str,
        duration_seconds: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Mark a lecture as successfully completed."""
        async def _do_mark(r, key, lid, status_json, meta_key):
            await r.hset(key, lid, status_json)
            await r.expire(key, 7 * 24 * 3600)
            await r.hincrby(meta_key, "completed_count", 1)
            await r.expire(meta_key, 7 * 24 * 3600)
            return True

        try:
            status = {
                "status": "completed",
                "video_url": video_url,
                "duration_seconds": duration_seconds,
                "completed_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }

            result = await self._execute_with_retry(
                _do_mark,
                self._lecture_key(job_id),
                lecture_id,
                json.dumps(status),
                self._meta_key(job_id),
                max_retries=3
            )

            print(f"[CHECKPOINT] Marked {lecture_id} as completed: {video_url[:50]}...", flush=True)
            return result

        except Exception as e:
            print(f"[CHECKPOINT] Error marking completion for {lecture_id}: {e}", flush=True)
            return False

    async def mark_failed(
        self,
        job_id: str,
        lecture_id: str,
        error: str,
        retry_count: int = 0
    ) -> bool:
        """Mark a lecture as failed."""
        async def _do_mark(r, key, lid, status_json, meta_key):
            await r.hset(key, lid, status_json)
            await r.expire(key, 7 * 24 * 3600)
            await r.hincrby(meta_key, "failed_count", 1)
            await r.expire(meta_key, 7 * 24 * 3600)
            return True

        try:
            status = {
                "status": "failed",
                "error": error,
                "retry_count": retry_count,
                "failed_at": datetime.utcnow().isoformat()
            }

            result = await self._execute_with_retry(
                _do_mark,
                self._lecture_key(job_id),
                lecture_id,
                json.dumps(status),
                self._meta_key(job_id),
                max_retries=3
            )

            print(f"[CHECKPOINT] Marked {lecture_id} as failed: {error[:50]}...", flush=True)
            return result

        except Exception as e:
            print(f"[CHECKPOINT] Error marking failure for {lecture_id}: {e}", flush=True)
            return False

    async def mark_in_progress(self, job_id: str, lecture_id: str) -> bool:
        """Mark a lecture as currently being generated."""
        async def _do_mark(r, key, lid, status_json):
            await r.hset(key, lid, status_json)
            await r.expire(key, 7 * 24 * 3600)
            return True

        try:
            status = {
                "status": "in_progress",
                "started_at": datetime.utcnow().isoformat()
            }

            return await self._execute_with_retry(
                _do_mark,
                self._lecture_key(job_id),
                lecture_id,
                json.dumps(status),
                max_retries=3
            )

        except Exception as e:
            print(f"[CHECKPOINT] Error marking in_progress for {lecture_id}: {e}", flush=True)
            return False

    async def get_all_statuses(self, job_id: str) -> Dict[str, Dict[str, Any]]:
        """Get status of all lectures for a job."""
        async def _do_get(r, key):
            data = await r.hgetall(key)
            return {
                lecture_id: json.loads(status_json)
                for lecture_id, status_json in data.items()
            }

        try:
            return await self._execute_with_retry(
                _do_get, self._lecture_key(job_id), max_retries=3
            )
        except Exception as e:
            print(f"[CHECKPOINT] Error getting all statuses: {e}", flush=True)
            return {}

    async def get_completed_lectures(self, job_id: str) -> List[str]:
        """Get list of completed lecture IDs."""
        statuses = await self.get_all_statuses(job_id)
        return [
            lecture_id
            for lecture_id, status in statuses.items()
            if status.get("status") == "completed"
        ]

    async def get_summary(self, job_id: str) -> Dict[str, Any]:
        """Get checkpoint summary for a job."""
        statuses = await self.get_all_statuses(job_id)

        completed = sum(1 for s in statuses.values() if s.get("status") == "completed")
        failed = sum(1 for s in statuses.values() if s.get("status") == "failed")
        in_progress = sum(1 for s in statuses.values() if s.get("status") == "in_progress")

        return {
            "total": len(statuses),
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": len(statuses) - completed - failed - in_progress
        }

    async def clear_job(self, job_id: str) -> bool:
        """Clear all checkpoints for a job (for fresh restart)."""
        async def _do_clear(r, lkey, mkey):
            await r.delete(lkey)
            await r.delete(mkey)
            return True

        try:
            result = await self._execute_with_retry(
                _do_clear,
                self._lecture_key(job_id),
                self._meta_key(job_id),
                max_retries=3
            )
            print(f"[CHECKPOINT] Cleared checkpoints for job {job_id}", flush=True)
            return result
        except Exception as e:
            print(f"[CHECKPOINT] Error clearing job {job_id}: {e}", flush=True)
            return False

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
        self._redis = None
        self._connected = False
        print("[CHECKPOINT] Connection closed", flush=True)


# Global singleton
checkpoint_service = LectureCheckpointService()
