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
    EXCHANGE_NAME = "course_exchange"

    def __init__(self, rabbitmq_url: str = None):
        self.rabbitmq_url = rabbitmq_url or os.getenv(
            "RABBITMQ_URL",
            "amqp://tiktok:rabbitmq_secure_2024@rabbitmq:5672/"
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
            await self._channel.set_qos(prefetch_count=1)

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

        # Update prefetch for concurrent processing
        await self._channel.set_qos(prefetch_count=max_concurrent)

        self._is_consuming = True
        print(f"[QUEUE] Starting consumer (max concurrent: {max_concurrent})", flush=True)

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


# Singleton instance
_queue_service: Optional[CourseQueueService] = None


def get_queue_service() -> CourseQueueService:
    """Get the singleton queue service instance"""
    global _queue_service
    if _queue_service is None:
        _queue_service = CourseQueueService()
    return _queue_service
