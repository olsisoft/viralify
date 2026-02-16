"""
Finalization Queue Service

Redis-based queue for course finalization jobs.
Triggered when all lectures in a course are complete.
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Callable, Optional, Dict, Any
import aioredis

from models.queue_models import QueuedFinalizationJob


class FinalizationQueueService:
    """
    Redis Streams-based queue service for course finalization.

    Features:
    - Reliable message delivery with consumer groups
    - Triggered automatically when all lectures complete
    - Handles quiz generation, ZIP creation, video compilation
    """

    STREAM_NAME = "finalization_jobs"
    CONSUMER_GROUP = "finalization_workers"
    DLQ_STREAM = "finalization_jobs_dlq"

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/7"
        )
        self._redis: Optional[aioredis.Redis] = None
        self._is_consuming = False
        self._consumer_name: Optional[str] = None

    async def connect(self) -> None:
        """Establish connection to Redis"""
        if self._redis:
            return

        print(f"[FINALIZATION_QUEUE] Connecting to Redis...", flush=True)

        try:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._redis.ping()
            print(f"[FINALIZATION_QUEUE] Connected to Redis", flush=True)

            # Create consumer group if it doesn't exist
            try:
                await self._redis.xgroup_create(
                    self.STREAM_NAME,
                    self.CONSUMER_GROUP,
                    id='0',
                    mkstream=True
                )
                print(f"[FINALIZATION_QUEUE] Created consumer group: {self.CONSUMER_GROUP}", flush=True)
            except aioredis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

        except Exception as e:
            print(f"[FINALIZATION_QUEUE] Failed to connect: {e}", flush=True)
            raise

    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            print("[FINALIZATION_QUEUE] Disconnected from Redis", flush=True)

    async def publish(self, job: QueuedFinalizationJob) -> str:
        """
        Publish a finalization job to the queue.

        Returns the message ID.
        """
        if not self._redis:
            await self.connect()

        message_data = {
            'job_data': job.to_json(),
            'priority': str(job.priority),
            'created_at': job.created_at or datetime.utcnow().isoformat()
        }

        message_id = await self._redis.xadd(
            self.STREAM_NAME,
            message_data
        )

        print(f"[FINALIZATION_QUEUE] Published finalization job for course {job.course_job_id}", flush=True)
        return message_id

    async def consume(
        self,
        handler: Callable[[QueuedFinalizationJob], Any],
        consumer_name: str = None,
        block_ms: int = 5000
    ) -> None:
        """
        Start consuming finalization jobs from the queue.

        Args:
            handler: Async function to process each job
            consumer_name: Unique name for this consumer
            block_ms: How long to block waiting for messages
        """
        if not self._redis:
            await self.connect()

        self._consumer_name = consumer_name or f"finalizer_{os.getpid()}"
        self._is_consuming = True

        print(f"[FINALIZATION_QUEUE] Starting consumer: {self._consumer_name}", flush=True)

        while self._is_consuming:
            try:
                messages = await self._redis.xreadgroup(
                    self.CONSUMER_GROUP,
                    self._consumer_name,
                    {self.STREAM_NAME: '>'},
                    count=1,
                    block=block_ms
                )

                if not messages:
                    continue

                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        await self._process_message(message_id, data, handler)

            except asyncio.CancelledError:
                print(f"[FINALIZATION_QUEUE] Consumer cancelled", flush=True)
                break
            except Exception as e:
                print(f"[FINALIZATION_QUEUE] Consumer error: {e}", flush=True)
                await asyncio.sleep(1)

    async def _process_message(
        self,
        message_id: str,
        data: Dict[str, str],
        handler: Callable[[QueuedFinalizationJob], Any]
    ) -> None:
        """Process a single finalization message"""
        try:
            job_data = data.get('job_data')
            if not job_data:
                print(f"[FINALIZATION_QUEUE] Invalid message format: {message_id}", flush=True)
                await self._redis.xack(self.STREAM_NAME, self.CONSUMER_GROUP, message_id)
                return

            job = QueuedFinalizationJob.from_json(job_data)
            print(f"[FINALIZATION_QUEUE] Processing finalization for course {job.course_job_id}", flush=True)

            # Call the handler
            await handler(job)

            # Acknowledge successful processing
            await self._redis.xack(self.STREAM_NAME, self.CONSUMER_GROUP, message_id)
            print(f"[FINALIZATION_QUEUE] Completed finalization for course {job.course_job_id}", flush=True)

        except Exception as e:
            print(f"[FINALIZATION_QUEUE] Failed to process finalization: {e}", flush=True)
            # Move to DLQ on failure
            await self._move_to_dlq(message_id, data, str(e))
            await self._redis.xack(self.STREAM_NAME, self.CONSUMER_GROUP, message_id)

    async def _move_to_dlq(self, message_id: str, data: Dict, error: str) -> None:
        """Move failed message to dead letter queue"""
        dlq_data = {
            **data,
            'original_message_id': message_id,
            'error': error,
            'failed_at': datetime.utcnow().isoformat()
        }
        await self._redis.xadd(self.DLQ_STREAM, dlq_data)
        print(f"[FINALIZATION_QUEUE] Moved to DLQ: {message_id}", flush=True)

    async def stop_consuming(self) -> None:
        """Stop the consumer loop"""
        self._is_consuming = False

    async def get_queue_length(self) -> int:
        """Get the number of pending finalization jobs"""
        if not self._redis:
            await self.connect()

        info = await self._redis.xinfo_stream(self.STREAM_NAME)
        return info.get('length', 0)


# Global instance
_finalization_queue: Optional[FinalizationQueueService] = None


async def get_finalization_queue() -> FinalizationQueueService:
    """Get or create the global finalization queue instance"""
    global _finalization_queue
    if _finalization_queue is None:
        _finalization_queue = FinalizationQueueService()
        await _finalization_queue.connect()
    return _finalization_queue
