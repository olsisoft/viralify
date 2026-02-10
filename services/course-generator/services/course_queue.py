"""
Course Queue Service

Manages RabbitMQ queue for course generation jobs.
Provides reliable job queuing with retry support.
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Optional, Callable, Any
from dataclasses import dataclass, asdict

import aio_pika
from aio_pika import Message, DeliveryMode
from aio_pika.abc import AbstractRobustConnection, AbstractChannel, AbstractQueue


@dataclass
class QueuedCourseJob:
    """Represents a course job in the queue"""
    job_id: str
    topic: str
    num_sections: int
    lectures_per_section: int
    user_id: str
    difficulty_start: str = "beginner"
    difficulty_end: str = "intermediate"
    target_audience: str = "general"
    language: str = "en"
    category: str = "education"
    domain: Optional[str] = None
    selected_elements: Optional[list] = None
    quiz_config: Optional[dict] = None
    document_ids: Optional[list] = None
    source_ids: Optional[list] = None
    created_at: Optional[str] = None
    priority: int = 5  # 1-10, lower = higher priority

    def to_json(self) -> str:
        data = asdict(self)
        if not data.get('created_at'):
            data['created_at'] = datetime.utcnow().isoformat()
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'QueuedCourseJob':
        data = json.loads(json_str)
        return cls(**data)


class CourseQueueService:
    """
    RabbitMQ-based queue service for course generation.

    Features:
    - Reliable message delivery with acknowledgments
    - Priority queue support
    - Dead letter queue for failed jobs
    - Automatic reconnection
    """

    QUEUE_NAME = "course_generation_queue"
    DLQ_NAME = "course_generation_dlq"  # Dead Letter Queue
    MANUAL_QUEUE_NAME = "course_generation_manual"  # Manual intervention queue
    EXCHANGE_NAME = "course_exchange"
    MAX_RETRY_COUNT = 3  # Maximum retries before manual intervention

    def __init__(self, rabbitmq_url: str = None):
        self.rabbitmq_url = rabbitmq_url or os.getenv(
            "RABBITMQ_URL",
            "amqp://viralify:viralify_secure_2024@rabbitmq:5672/"
        )
        self._connection: Optional[AbstractRobustConnection] = None
        self._channel: Optional[AbstractChannel] = None
        self._queue: Optional[AbstractQueue] = None
        self._dlq: Optional[AbstractQueue] = None
        self._is_consuming = False

    async def connect(self) -> None:
        """Establish connection to RabbitMQ"""
        if self._connection and not self._connection.is_closed:
            return

        print(f"[QUEUE] Connecting to RabbitMQ...", flush=True)

        try:
            self._connection = await aio_pika.connect_robust(
                self.rabbitmq_url,
                timeout=30.0
            )
            self._channel = await self._connection.channel()

            # Set QoS - process one message at a time per worker
            # IMPORTANT: prefetch_count=1 ensures fair distribution across workers
            # This is set here and NOT overridden in consume() to ensure round-robin
            await self._channel.set_qos(prefetch_count=1, global_=False)

            # Declare dead letter queue first
            self._dlq = await self._channel.declare_queue(
                self.DLQ_NAME,
                durable=True
            )

            # Declare main queue with dead letter exchange
            self._queue = await self._channel.declare_queue(
                self.QUEUE_NAME,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": self.DLQ_NAME,
                    "x-max-priority": 10  # Enable priority queue
                }
            )

            print(f"[QUEUE] Connected to RabbitMQ", flush=True)
            print(f"[QUEUE] Queue '{self.QUEUE_NAME}' ready", flush=True)

        except Exception as e:
            print(f"[QUEUE] Connection error: {e}", flush=True)
            raise

    async def disconnect(self) -> None:
        """Close connection to RabbitMQ"""
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            print(f"[QUEUE] Disconnected from RabbitMQ", flush=True)

    async def publish(self, job: QueuedCourseJob) -> bool:
        """
        Publish a course generation job to the queue.

        Returns True if successful, False otherwise.
        """
        await self.connect()

        try:
            message = Message(
                body=job.to_json().encode(),
                delivery_mode=DeliveryMode.PERSISTENT,  # Survive broker restart
                priority=job.priority,
                message_id=job.job_id,
                timestamp=datetime.utcnow(),
                headers={
                    "job_id": job.job_id,
                    "topic": job.topic,
                    "user_id": job.user_id
                }
            )

            await self._channel.default_exchange.publish(
                message,
                routing_key=self.QUEUE_NAME
            )

            print(f"[QUEUE] Published job {job.job_id}: {job.topic[:50]}...", flush=True)
            return True

        except Exception as e:
            print(f"[QUEUE] Publish error: {e}", flush=True)
            return False

    async def get_queue_stats(self) -> dict:
        """Get current queue statistics"""
        await self.connect()

        # Re-declare to get current counts
        queue = await self._channel.declare_queue(
            self.QUEUE_NAME,
            durable=True,
            passive=True  # Don't create, just check
        )

        dlq = await self._channel.declare_queue(
            self.DLQ_NAME,
            durable=True,
            passive=True
        )

        return {
            "queue_name": self.QUEUE_NAME,
            "pending_jobs": queue.declaration_result.message_count,
            "consumers": queue.declaration_result.consumer_count,
            "failed_jobs": dlq.declaration_result.message_count
        }

    async def consume(
        self,
        callback: Callable[[QueuedCourseJob], Any],
        max_concurrent: int = 1
    ) -> None:
        """
        Start consuming jobs from the queue.

        Args:
            callback: Async function to process each job
            max_concurrent: Maximum concurrent jobs (default: 1)
        """
        await self.connect()

        # NOTE: prefetch_count is set to 1 in connect() for fair distribution
        # Do NOT override it here - we want round-robin across workers
        # max_concurrent is kept for logging/future use only

        self._is_consuming = True
        print(f"[QUEUE] Starting consumer (prefetch=1 for fair distribution)", flush=True)

        async def process_message(message: aio_pika.IncomingMessage):
            async with message.process(requeue=False):
                job_id = message.headers.get("job_id", "unknown")

                try:
                    job = QueuedCourseJob.from_json(message.body.decode())
                    print(f"[QUEUE] Processing job {job.job_id}: {job.topic[:50]}...", flush=True)

                    await callback(job)

                    print(f"[QUEUE] Completed job {job.job_id}", flush=True)

                except Exception as e:
                    print(f"[QUEUE] Job {job_id} failed: {e}", flush=True)
                    # Message will go to DLQ due to requeue=False and failure
                    raise

        await self._queue.consume(process_message)

        # Keep running until stopped
        while self._is_consuming:
            await asyncio.sleep(1)

    def stop_consuming(self) -> None:
        """Stop the consumer"""
        self._is_consuming = False
        print(f"[QUEUE] Stopping consumer", flush=True)

    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from Redis storage.

        Note: This removes the job data from Redis. It does NOT remove
        pending messages from the queue (those are handled by the consumer).
        """
        await self.connect()

        try:
            # Import redis to delete the job hash
            import aioredis
            import os

            redis_url = os.getenv("REDIS_URL", "redis://:redis_secure_2024@redis:6379/7")
            redis = await aioredis.from_url(redis_url)

            # Delete job data from Redis
            deleted = await redis.delete(f"course_job:{job_id}")
            await redis.close()

            if deleted:
                print(f"[QUEUE] Deleted job {job_id} from Redis", flush=True)
                return True
            else:
                print(f"[QUEUE] Job {job_id} not found in Redis", flush=True)
                return False

        except Exception as e:
            print(f"[QUEUE] Delete job error: {e}", flush=True)
            return False

    async def requeue_failed_job(self, job_id: str) -> bool:
        """
        Move a job from the DLQ back to the main queue for retry.
        """
        await self.connect()

        # Get message from DLQ
        message = await self._dlq.get(timeout=5)
        if message is None:
            return False

        if message.headers.get("job_id") == job_id:
            # Republish to main queue
            job = QueuedCourseJob.from_json(message.body.decode())
            await self.publish(job)
            await message.ack()
            print(f"[QUEUE] Requeued job {job_id}", flush=True)
            return True
        else:
            # Not the right message, reject it back
            await message.nack(requeue=True)
            return False

    async def publish_to_manual_queue(self, job: QueuedCourseJob, error: str, retry_count: int) -> bool:
        """
        Publish a job to the manual intervention queue after max retries.
        """
        await self.connect()

        # Declare manual queue if not exists
        manual_queue = await self._channel.declare_queue(
            self.MANUAL_QUEUE_NAME,
            durable=True
        )

        try:
            message = Message(
                body=job.to_json().encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                message_id=job.job_id,
                timestamp=datetime.utcnow(),
                headers={
                    "job_id": job.job_id,
                    "topic": job.topic,
                    "user_id": job.user_id,
                    "retry_count": retry_count,
                    "last_error": error[:500] if error else "Unknown error",
                    "failed_at": datetime.utcnow().isoformat()
                }
            )

            await self._channel.default_exchange.publish(
                message,
                routing_key=self.MANUAL_QUEUE_NAME
            )

            print(f"[QUEUE] Job {job.job_id} moved to manual queue after {retry_count} retries", flush=True)
            return True

        except Exception as e:
            print(f"[QUEUE] Failed to publish to manual queue: {e}", flush=True)
            return False

    async def consume_dlq(
        self,
        callback: Callable[[QueuedCourseJob], Any],
        retry_delay_seconds: int = 30
    ) -> None:
        """
        Consume jobs from the DLQ and retry them.

        Args:
            callback: Async function to process each job (same as main worker)
            retry_delay_seconds: Delay before retrying (default: 30s)
        """
        await self.connect()

        # Declare manual queue for jobs that exceed max retries
        await self._channel.declare_queue(
            self.MANUAL_QUEUE_NAME,
            durable=True
        )

        self._is_consuming = True
        print(f"[DLQ] Starting DLQ consumer (max_retries={self.MAX_RETRY_COUNT}, delay={retry_delay_seconds}s)", flush=True)

        async def process_dlq_message(message: aio_pika.IncomingMessage):
            job_id = message.headers.get("job_id", "unknown")

            # Get retry count from headers (x-death header set by RabbitMQ)
            retry_count = 0
            x_death = message.headers.get("x-death", [])
            if x_death and len(x_death) > 0:
                retry_count = x_death[0].get("count", 0) if isinstance(x_death[0], dict) else 0

            # Also check our custom retry header
            custom_retry = message.headers.get("retry_count", 0)
            retry_count = max(retry_count, custom_retry)

            print(f"[DLQ] Processing failed job {job_id} (retry #{retry_count + 1}/{self.MAX_RETRY_COUNT})", flush=True)

            try:
                job = QueuedCourseJob.from_json(message.body.decode())

                # Check if max retries exceeded
                if retry_count >= self.MAX_RETRY_COUNT:
                    print(f"[DLQ] Job {job_id} exceeded max retries, moving to manual queue", flush=True)
                    last_error = message.headers.get("last_error", "Max retries exceeded")
                    await self.publish_to_manual_queue(job, last_error, retry_count)
                    await message.ack()
                    return

                # Wait before retry
                print(f"[DLQ] Waiting {retry_delay_seconds}s before retry...", flush=True)
                await asyncio.sleep(retry_delay_seconds)

                # Try to process the job
                await callback(job)

                # Success - acknowledge the message
                await message.ack()
                print(f"[DLQ] Job {job_id} completed successfully on retry", flush=True)

            except Exception as e:
                error_msg = str(e)
                print(f"[DLQ] Job {job_id} failed again: {error_msg}", flush=True)

                # Increment retry count and republish to DLQ with updated headers
                new_retry_count = retry_count + 1

                if new_retry_count >= self.MAX_RETRY_COUNT:
                    # Max retries reached, move to manual queue
                    print(f"[DLQ] Job {job_id} moving to manual queue after {new_retry_count} retries", flush=True)
                    job = QueuedCourseJob.from_json(message.body.decode())
                    await self.publish_to_manual_queue(job, error_msg, new_retry_count)
                    await message.ack()
                else:
                    # Republish to DLQ with updated retry count
                    new_message = Message(
                        body=message.body,
                        delivery_mode=DeliveryMode.PERSISTENT,
                        message_id=message.message_id,
                        timestamp=datetime.utcnow(),
                        headers={
                            **dict(message.headers),
                            "retry_count": new_retry_count,
                            "last_error": error_msg[:500]
                        }
                    )
                    await self._channel.default_exchange.publish(
                        new_message,
                        routing_key=self.DLQ_NAME
                    )
                    await message.ack()
                    print(f"[DLQ] Job {job_id} requeued to DLQ (retry {new_retry_count}/{self.MAX_RETRY_COUNT})", flush=True)

        await self._dlq.consume(process_dlq_message)

        # Keep running until stopped
        while self._is_consuming:
            await asyncio.sleep(1)


# Singleton instance
_queue_service: Optional[CourseQueueService] = None


def get_queue_service() -> CourseQueueService:
    """Get the singleton queue service instance"""
    global _queue_service
    if _queue_service is None:
        _queue_service = CourseQueueService()
    return _queue_service
