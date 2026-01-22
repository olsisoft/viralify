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
            # Use same DB as course-generator (7)
            redis_db = int(os.getenv("REDIS_CHECKPOINT_DB", "7"))

            self._redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            try:
                await self._redis.ping()
                self._connected = True
                print(f"[CHECKPOINT] Connected to Redis at {redis_host}:{redis_port}/db{redis_db}", flush=True)
            except Exception as e:
                print(f"[CHECKPOINT] Failed to connect to Redis: {e}", flush=True)
                self._connected = False
                raise

        return self._redis

    def _lecture_key(self, job_id: str) -> str:
        return f"course:checkpoint:{job_id}:lectures"

    def _meta_key(self, job_id: str) -> str:
        return f"course:checkpoint:{job_id}:meta"

    async def is_completed(self, job_id: str, lecture_id: str) -> bool:
        """Check if a lecture has already been successfully generated."""
        try:
            r = await self._get_redis()
            data = await r.hget(self._lecture_key(job_id), lecture_id)

            if data:
                status = json.loads(data)
                return status.get("status") == "completed" and status.get("video_url")

            return False

        except Exception as e:
            print(f"[CHECKPOINT] Error checking completion for {lecture_id}: {e}", flush=True)
            return False

    async def get_video_url(self, job_id: str, lecture_id: str) -> Optional[str]:
        """Get the video URL for a completed lecture."""
        try:
            r = await self._get_redis()
            data = await r.hget(self._lecture_key(job_id), lecture_id)

            if data:
                status = json.loads(data)
                return status.get("video_url")

            return None

        except Exception as e:
            print(f"[CHECKPOINT] Error getting video URL for {lecture_id}: {e}", flush=True)
            return None

    async def get_lecture_status(self, job_id: str, lecture_id: str) -> Optional[Dict[str, Any]]:
        """Get full status for a lecture."""
        try:
            r = await self._get_redis()
            data = await r.hget(self._lecture_key(job_id), lecture_id)

            if data:
                return json.loads(data)

            return None

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
        try:
            r = await self._get_redis()

            status = {
                "status": "completed",
                "video_url": video_url,
                "duration_seconds": duration_seconds,
                "completed_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }

            await r.hset(self._lecture_key(job_id), lecture_id, json.dumps(status))
            # Set TTL: 7 days
            await r.expire(self._lecture_key(job_id), 7 * 24 * 3600)

            # Update meta
            await self._increment_meta(job_id, "completed_count")

            print(f"[CHECKPOINT] Marked {lecture_id} as completed: {video_url[:50]}...", flush=True)
            return True

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
        try:
            r = await self._get_redis()

            status = {
                "status": "failed",
                "error": error,
                "retry_count": retry_count,
                "failed_at": datetime.utcnow().isoformat()
            }

            await r.hset(self._lecture_key(job_id), lecture_id, json.dumps(status))
            await r.expire(self._lecture_key(job_id), 7 * 24 * 3600)

            # Update meta
            await self._increment_meta(job_id, "failed_count")

            print(f"[CHECKPOINT] Marked {lecture_id} as failed: {error[:50]}...", flush=True)
            return True

        except Exception as e:
            print(f"[CHECKPOINT] Error marking failure for {lecture_id}: {e}", flush=True)
            return False

    async def mark_in_progress(self, job_id: str, lecture_id: str) -> bool:
        """Mark a lecture as currently being generated."""
        try:
            r = await self._get_redis()

            status = {
                "status": "in_progress",
                "started_at": datetime.utcnow().isoformat()
            }

            await r.hset(self._lecture_key(job_id), lecture_id, json.dumps(status))
            await r.expire(self._lecture_key(job_id), 7 * 24 * 3600)

            return True

        except Exception as e:
            print(f"[CHECKPOINT] Error marking in_progress for {lecture_id}: {e}", flush=True)
            return False

    async def get_all_statuses(self, job_id: str) -> Dict[str, Dict[str, Any]]:
        """Get status of all lectures for a job."""
        try:
            r = await self._get_redis()
            data = await r.hgetall(self._lecture_key(job_id))

            return {
                lecture_id: json.loads(status_json)
                for lecture_id, status_json in data.items()
            }

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
        try:
            r = await self._get_redis()
            await r.delete(self._lecture_key(job_id))
            await r.delete(self._meta_key(job_id))
            print(f"[CHECKPOINT] Cleared checkpoints for job {job_id}", flush=True)
            return True
        except Exception as e:
            print(f"[CHECKPOINT] Error clearing job {job_id}: {e}", flush=True)
            return False

    async def _increment_meta(self, job_id: str, field: str):
        """Increment a counter in job metadata."""
        try:
            r = await self._get_redis()
            await r.hincrby(self._meta_key(job_id), field, 1)
            await r.expire(self._meta_key(job_id), 7 * 24 * 3600)
        except Exception:
            pass

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False


# Global singleton
checkpoint_service = LectureCheckpointService()
